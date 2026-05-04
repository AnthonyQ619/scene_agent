import os
import numpy as np
import torch
from pathlib import Path
from mapanything.models import MapAnything
from mapanything.utils.image import load_images

def tensor_to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def invert_c2w_to_w2c_3x4(poses_c2w):
    """
    Convert cam-to-world 4x4 poses to world-to-camera 3x4 extrinsics.

    Parameters
    ----------
    poses_c2w:
        Array of shape (N, 4, 4)

    Returns
    -------
    extrinsics_w2c:
        Array of shape (N, 3, 4)

    cam_centers_world:
        Array of shape (N, 3)
    """
    poses_c2w = np.asarray(poses_c2w, dtype=np.float64)

    if poses_c2w.ndim != 3 or poses_c2w.shape[-2:] != (4, 4):
        raise ValueError(f"Expected poses_c2w shape (N,4,4), got {poses_c2w.shape}")

    R_c2w = poses_c2w[:, :3, :3]
    t_c2w = poses_c2w[:, :3, 3]

    R_w2c = np.transpose(R_c2w, (0, 2, 1))
    t_w2c = -np.einsum("nij,nj->ni", R_w2c, t_c2w)

    extrinsics_w2c = np.concatenate(
        [R_w2c, t_w2c[..., None]],
        axis=-1,
    )

    cam_centers_world = t_c2w

    return extrinsics_w2c, cam_centers_world

def extract_mapanything_points_from_predictions(
    predictions,
    use_mask=True,
    use_confidence_filter=True,
    confidence_percentile=10,
    max_points_per_view=None,
    save_colors=True,
):
    """
    Extract dense world-coordinate points from MapAnything predictions.

    Parameters
    ----------
    predictions:
        Output of model.infer(...)

    use_mask:
        If True, use pred["mask"] to remove invalid pixels.

    use_confidence_filter:
        If True, remove points below the confidence percentile per view.

    confidence_percentile:
        Percentile cutoff for pred["conf"]. For example, 10 removes the
        bottom 10% confidence points.

    max_points_per_view:
        Optional random subsampling count per view. Useful because dense
        MapAnything outputs can be very large.

    save_colors:
        If True, tries to attach RGB colors from pred["img"] or pred["image"]
        if available.

    Returns
    -------
    all_points:
        Array of shape (M, 3)

    all_colors:
        Array of shape (M, 3), or None
    """
    all_points = []
    all_colors = []

    rng = np.random.default_rng(0)

    for view_idx, pred in enumerate(predictions):
        if "pts3d" not in pred:
            raise KeyError(
                "MapAnything prediction does not contain key 'pts3d'. "
                f"Available keys: {list(pred.keys())}"
            )

        pts3d = tensor_to_numpy(pred["pts3d"])

        # Expected: (B, H, W, 3), usually B=1
        if pts3d.ndim == 4 and pts3d.shape[0] == 1:
            pts3d = pts3d[0]

        if pts3d.ndim != 3 or pts3d.shape[-1] != 3:
            raise ValueError(f"Expected pts3d shape (H,W,3), got {pts3d.shape}")

        H, W, _ = pts3d.shape

        valid = np.isfinite(pts3d).all(axis=-1)

        if use_mask and "mask" in pred:
            mask = tensor_to_numpy(pred["mask"])

            # Expected: (B, H, W, 1) or (B, H, W)
            if mask.ndim == 4 and mask.shape[0] == 1:
                mask = mask[0]
            if mask.ndim == 3 and mask.shape[-1] == 1:
                mask = mask[..., 0]

            if mask.shape != (H, W):
                raise ValueError(
                    f"Mask shape {mask.shape} does not match pts3d shape {(H, W)}"
                )

            valid &= mask.astype(bool)

        if use_confidence_filter and "conf" in pred:
            conf = tensor_to_numpy(pred["conf"])

            # Expected: (B, H, W)
            if conf.ndim == 3 and conf.shape[0] == 1:
                conf = conf[0]
            if conf.ndim == 4 and conf.shape[0] == 1 and conf.shape[-1] == 1:
                conf = conf[0, ..., 0]

            if conf.shape != (H, W):
                raise ValueError(
                    f"Confidence shape {conf.shape} does not match pts3d shape {(H, W)}"
                )

            valid_conf = np.isfinite(conf)
            if np.any(valid & valid_conf):
                cutoff = np.percentile(conf[valid & valid_conf], confidence_percentile)
                valid &= conf >= cutoff

        points = pts3d[valid]

        colors = None

        if save_colors:
            # MapAnything view/output key names may differ depending on version.
            # Try common possible keys.
            img = None
            for key in ["img", "image", "images"]:
                if key in pred:
                    img = tensor_to_numpy(pred[key])
                    break

            if img is not None:
                # Possible shapes:
                #   (B, H, W, 3)
                #   (H, W, 3)
                #   (B, 3, H, W)
                #   (3, H, W)
                if img.ndim == 4 and img.shape[0] == 1:
                    img = img[0]

                if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
                    img = np.transpose(img, (1, 2, 0))

                if img.ndim == 3 and img.shape[-1] == 3:
                    if img.shape[:2] == (H, W):
                        colors = img[valid]

                        # Convert [0,1] float images to [0,255].
                        if np.issubdtype(colors.dtype, np.floating) and colors.max() <= 1.0:
                            colors = colors * 255.0

        if max_points_per_view is not None and len(points) > max_points_per_view:
            idx = rng.choice(len(points), size=max_points_per_view, replace=False)
            points = points[idx]
            if colors is not None:
                colors = colors[idx]

        all_points.append(points.astype(np.float64))

        if colors is not None:
            all_colors.append(colors.astype(np.uint8))

    if len(all_points) == 0:
        raise RuntimeError("No points were extracted from MapAnything predictions.")

    all_points = np.concatenate(all_points, axis=0)

    if save_colors and len(all_colors) == len(predictions):
        all_colors = np.concatenate(all_colors, axis=0)
        if len(all_colors) != len(all_points):
            all_colors = None
    else:
        all_colors = None

    return all_points, all_colors

def write_ascii_ply(points, colors=None, output_path="points.ply"):
    """
    Write Nx3 points and optional Nx3 uint8 RGB colors to an ASCII PLY file.
    """
    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points shape (N,3), got {points.shape}")

    has_color = colors is not None

    if has_color:
        colors = np.asarray(colors)
        if colors.ndim != 2 or colors.shape[1] != 3:
            raise ValueError(f"Expected colors shape (N,3), got {colors.shape}")
        colors = np.clip(colors, 0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property double x\n")
        f.write("property double y\n")
        f.write("property double z\n")

        if has_color:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")

        f.write("end_header\n")

        if has_color:
            for p, c in zip(points, colors):
                f.write(
                    f"{p[0]} {p[1]} {p[2]} "
                    f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
                )
        else:
            for p in points:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")

def extract_and_save_mapanything_poses_and_ply(
    model,
    image_paths,
    pose_output_path,
    ply_output_path,
    device="cuda",
    image_names=None,
    image_size=None,
    extra_metadata=None,
    model_name="facebook/map-anything",
    use_mask=True,
    use_confidence_filter=True,
    confidence_percentile=10,
    max_points_per_view=100_000,
    save_colors=True,
    save_dense=False,
):
    """
    Runs MapAnything once, saves poses in a VGGT-like .npz, and exports
    dense world-coordinate points to a .ply.

    Notes
    -----
    - MapAnything's native dense points are pred["pts3d"], already in world coordinates.
    - MapAnything's native camera poses are pred["camera_poses"], cam2world 4x4.
    - This function also saves inverted world-to-camera 3x4 extrinsics for compatibility
      with your VGGT pose-eval format.
    """

    if ("cuda" in device) and not torch.cuda.is_available():
        device = "cpu"

    views = load_images(image_paths)

    with torch.no_grad():
        predictions = model.infer(
            views,
            memory_efficient_inference=True,
            minibatch_size=None,
            use_amp=("cuda" in device),
            amp_dtype="bf16",
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=False,
            confidence_percentile=confidence_percentile,
            use_multiview_confidence=False,
        )

    # -------------------------
    # Save poses
    # -------------------------
    poses_c2w = []

    for pred in predictions:
        pose = tensor_to_numpy(pred["camera_poses"])

        # Expected: (B, 4, 4), usually B=1
        if pose.ndim == 3 and pose.shape[0] == 1:
            pose = pose[0]

        if pose.shape != (4, 4):
            raise ValueError(f"Expected each MapAnything pose to be (4,4), got {pose.shape}")

        poses_c2w.append(pose.astype(np.float64))

    extrinsics_c2w = np.stack(poses_c2w, axis=0)
    extrinsics_w2c, cam_centers_world = invert_c2w_to_w2c_3x4(extrinsics_c2w)

    R_w2c = extrinsics_w2c[:, :3, :3]
    t_w2c = extrinsics_w2c[:, :3, 3]

    if image_names is None:
        image_names = [os.path.basename(p) for p in image_paths]
        print(image_names)

    if len(image_names) != len(extrinsics_c2w):
        raise ValueError(
            f"Number of image names ({len(image_names)}) does not match "
            f"number of poses ({len(extrinsics_c2w)})."
        )

    pose_save_dict = {
        "extrinsics_w2c": extrinsics_w2c.astype(np.float64),
        "R_w2c": R_w2c.astype(np.float64),
        "t_w2c": t_w2c.astype(np.float64),
        "extrinsics_c2w": extrinsics_c2w.astype(np.float64),
        "cam_centers_world": cam_centers_world.astype(np.float64),
        "image_names": np.asarray(image_names),
        "pose_convention": np.array(
            "MapAnything OpenCV camera-to-world native; also saved as world-to-camera / camera-from-world"
        ),
    }

    if image_size is not None:
        pose_save_dict["image_size"] = np.asarray(image_size, dtype=np.int32)

    if extra_metadata is not None:
        for k, v in extra_metadata.items():
            pose_save_dict[k] = np.asarray(v)

    os.makedirs(os.path.dirname(pose_output_path), exist_ok=True)
    np.savez_compressed(pose_output_path, **pose_save_dict)

    # -------------------------
    # Save dense point cloud
    # -------------------------
    points = []
    if save_dense:
        points, colors = extract_mapanything_points_from_predictions(
            predictions=predictions,
            use_mask=use_mask,
            use_confidence_filter=use_confidence_filter,
            confidence_percentile=confidence_percentile,
            max_points_per_view=max_points_per_view,
            save_colors=save_colors,
        )

        write_ascii_ply(
            points=points,
            colors=colors,
            output_path=ply_output_path,
        )

    return {
        "pose_output_path": pose_output_path,
        "ply_output_path": ply_output_path,
        "num_images": len(image_paths),
        "num_points": int(len(points)),
        "poses": pose_save_dict,
    }

log_folder = "/home/anthonyq/projects/scene_agent/breadth_agent/results/map_anything_pose_results"
gpu_num = "2"
# DTU RUN
# scene_list = ["scan1", "scan4", "scan9", "scan10", 
#               "scan11", "scan12", "scan13", "scan15", 
#               "scan23", "scan24", "scan29", "scan32", "scan33",
#               "scan34", "scan48", "scan49", "scan62", "scan75",
#               "scan77", "scan110", "scan114", "scan118"]

# # Change to dataset home location
# d_set = "DTU"
home_folder = f"/home/anthonyq/datasets/DTU"

# ETH RUN (Change homefolder to ETH location - Path commented below)
ETH_images = ["courtyard", "delivery_area", "electro", 
              "facade", "kicker", "meadow",
              "office", "pipes", "playground",
              "relief", "relief_2", "terrace", "terrains"]
## Home Folder/Image_path to use!

# Uncomment all vars!
# d_set = "ETH"
# homde_folder = "/home/anthonyq/datasets/ETH/ETH"
# image_path = home_folder + f"{ETH_images[i]}/images/dslr_images_undistorted" -> Used below, just comment out

# TT RUN 
# TT_images = ["barn_1_40", "barn_186_225", "barn_371_410",
#              "caterpillar_1_40", "caterpillar_173_212", "caterpillar_344_383",
#              "church_1_40", "church_235_274", "church_468_507",
#              "courthouse_1_40", "courthouse_534_573", "courthouse_1067_1106",
#              "ignatius_1_40", "ignatius_113_152", "ignatius_224_263",
#              "meetingroom_1_40", "meetingroom_167_206", "meetingroom_332_371",
#              "truck_1_40", "truck_107_146", "truck_212_251"]

# Co3Dv2 Test Run!
# img_postfix = "vggt_random_10" # Swap to Sequential String when ready
img_postfix = "middle_sequential_10"
co3d_images = ["apple/110_13051_23361", "apple/189_20393_38136",
               "ball/123_14363_28981", "ball/375_42693_85518",
               "bench/415_57112_110099", "bench/415_57121_110109",
               "book/119_13962_28926", "book/247_26469_51778", 
               "bowl/69_5465_12831", "bowl/70_5792_13401", 
               "broccoli/372_41112_81867", "broccoli/412_56288_108844",
               "cake/374_42274_84517", "cake/403_53094_103680", 
               "donut/391_47032_93657", "donut/403_52964_103416", 
               "hydrant/167_18184_34441", "hydrant/411_56064_108483",
               "mouse/107_12753_23606", "mouse/377_43416_86289", 
               "orange/374_42196_84367", "orange/385_45386_90752", 
               "plant/247_26441_50907", "plant/374_42005_84358",
               "remote/195_20989_41543", "remote/350_36761_68623", 
               "skateboard/245_26182_52130", "skateboard/366_39266_76077", 
               "suitcase/50_2928_8645", "suitcase/410_55734_107452",
               "teddybear/34_1479_4753", "teddybear/187_20215_38541", 
               "toaster/372_41229_82130", "toaster/416_57389_110765", 
               "toytrain/240_25394_51994", "toytrain/399_51323_100753",
               "toytruck/190_20494_39385", "toytruck/346_36113_66551", 
               "vase/374_41862_83720", "vase/380_44863_89631"]
d_set = "co3d"
# for i in range(len(co3d_images)):
#     img_seq = co3d_images[i]
#     c, seq = img_seq.split('/')
#     image_path = f"/home/anthonyq/datasets/co3d_v2/{img_seq}/{img_postfix}"

model = MapAnything.from_pretrained("facebook/map-anything").to(f"cuda:{gpu_num}")
model.eval()

for i in range(len(co3d_images)):
    img_seq = co3d_images[i]
    c, seq = img_seq.split('/')
    image_paths = f"/home/anthonyq/datasets/co3d_v2/{img_seq}/{img_postfix}"
    # image_paths = home_folder + f"/{scene_list[i]}"
    out_path_pose = os.path.join(log_folder, d_set, c, seq, f"mapanything_poses_{img_postfix}.npz")
    out_path_ply = os.path.join(log_folder, d_set, c, seq, "mapanything_dense_points.ply")

    image_names = sorted(p.name for p in Path(image_paths).glob("*"))
    result = extract_and_save_mapanything_poses_and_ply(
        model=model,
        image_paths=image_paths,
        pose_output_path=out_path_pose,
        ply_output_path=out_path_ply,
        device=f"cuda:{gpu_num}",
        image_names=image_names,
        image_size=None,
        extra_metadata={
            "model_name": "facebook/map-anything",
            "script_id": "1",
        },
        use_mask=True,
        use_confidence_filter=True,
        confidence_percentile=10,
        max_points_per_view=100_000,
        save_colors=True,
        save_dense=False
    )

    print("Saved poses:", result["pose_output_path"])
    # print("Saved PLY:", result["ply_output_path"])
    print("Num points:", result["num_points"])