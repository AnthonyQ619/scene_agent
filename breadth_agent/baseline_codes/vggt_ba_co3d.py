import pycolmap

from pathlib import Path
import random
import numpy as np
import glob
import os
import copy
import torch
import torch.nn.functional as F

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track, calculate_reproj_error

# from modules.visualize import VisualizeScene

# TODO: add support for masks
# TODO: add iterative BA
# TODO: add support for radial distortion, which needs extra_params
# TODO: test with more cases
# TODO: test different camera types


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Demo")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_ba", action="store_true", default=False, help="Use BA for reconstruction")
    ######### BA parameters #########
    parser.add_argument(
        "--max_reproj_error", type=float, default=8.0, help="Maximum reprojection error for reconstruction"
    )
    parser.add_argument("--shared_camera", action="store_true", default=False, help="Use shared camera for all images")
    parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE", help="Camera type for reconstruction")
    parser.add_argument("--vis_thresh", type=float, default=0.2, help="Visibility threshold for tracks")
    parser.add_argument("--query_frame_num", type=int, default=8, help="Number of frames to query")
    parser.add_argument("--max_query_pts", type=int, default=4096, help="Maximum number of query points")
    parser.add_argument(
        "--fine_tracking", action="store_true", default=True, help="Use fine tracking (slower but more accurate)"
    )
    parser.add_argument(
        "--conf_thres_value", type=float, default=5.0, help="Confidence threshold value for depth filtering (wo BA)"
    )
    return parser.parse_args()


def run_VGGT(model, images, dtype, resolution=518):
    # images: [B, 3, H, W]

    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # hard-coded to use 518 for VGGT
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    return extrinsic, intrinsic, depth_map, depth_conf
# Helper File to extract camera pose information
def get_image_pose_world_to_cam(image):
    """
    Returns R_world_to_cam, t_world_to_cam from a pycolmap Image.

    pycolmap versions differ slightly, so this tries the common APIs.
    """
    # Newer pycolmap versions
    if hasattr(image, "cam_from_world"):
        cam_from_world = image.cam_from_world()

        # Rigid3d-like object
        if hasattr(cam_from_world, "rotation") and hasattr(cam_from_world, "translation"):
            rot = cam_from_world.rotation
            t = np.asarray(cam_from_world.translation, dtype=np.float64)

            if hasattr(rot, "matrix"):
                R = np.asarray(rot.matrix(), dtype=np.float64)
            elif hasattr(rot, "to_matrix"):
                R = np.asarray(rot.to_matrix(), dtype=np.float64)
            else:
                R = np.asarray(rot, dtype=np.float64)

            return R, t

        # Sometimes transform matrix may be exposed
        if hasattr(cam_from_world, "matrix"):
            T = np.asarray(cam_from_world.matrix(), dtype=np.float64)
            return T[:3, :3], T[:3, 3]

    # Older pycolmap-style API
    if hasattr(image, "qvec") and hasattr(image, "tvec"):
        qvec = np.asarray(image.qvec, dtype=np.float64)
        t = np.asarray(image.tvec, dtype=np.float64)
        R = self.qvec_to_rotmat(qvec)
        return R, t

    raise RuntimeError(f"Could not extract pose for image {image.name}")


def qvec_to_rotmat(qvec):
    """
    COLMAP quaternion convention: q = [qw, qx, qy, qz].
    """
    qvec = np.asarray(qvec, dtype=np.float64)
    qw, qx, qy, qz = qvec

    return np.array([
        [
            1 - 2 * qy ** 2 - 2 * qz ** 2,
            2 * qx * qy - 2 * qz * qw,
            2 * qz * qx + 2 * qy * qw,
        ],
        [
            2 * qx * qy + 2 * qz * qw,
            1 - 2 * qx ** 2 - 2 * qz ** 2,
            2 * qy * qz - 2 * qx * qw,
        ],
        [
            2 * qz * qx - 2 * qy * qw,
            2 * qy * qz + 2 * qx * qw,
            1 - 2 * qx ** 2 - 2 * qy ** 2,
        ],
    ], dtype=np.float64)


def camera_to_K_and_dist(camera):
    """
    Returns a 3x3 calibration matrix K and raw COLMAP camera params.

    camera.params is model-dependent. We save both:
    - K for easy evaluation
    - raw params/model for exact reconstruction reference
    """
    if hasattr(camera, "calibration_matrix"):
        K = np.asarray(camera.calibration_matrix(), dtype=np.float64)
    else:
        # Fallback for common COLMAP models.
        params = np.asarray(camera.params, dtype=np.float64)
        model = str(camera.model)

        if "SIMPLE" in model:
            f, cx, cy = params[:3]
            fx, fy = f, f
        else:
            fx, fy, cx, cy = params[:4]

        K = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    params = np.asarray(camera.params, dtype=np.float64)

    return K, params

def store_extrinsics_information(recon, out_path) -> None:
    # out_path = os.path.join(self.cam_data.logging_dir, str(self.cam_data.script_id), f"cam_poses_log.npz")
    image_records = []

    image_items_sorted = sorted(
        recon.images.items(),
        key=lambda kv: kv[1].name
    )

    for image_id, image in image_items_sorted:
        has_pose = bool(getattr(image, "has_pose", False))
        if not has_pose:
            # print(f"[Warning] Image is not registered, no pose available: {image.name}")
            image_records.append(
                {
                    "image_id": int(image_id),
                    "image_name": image.name,
                    "camera_id": int(image.camera_id),
                    "width": int(camera.width),
                    "height": int(camera.height),
                    "camera_model": str(camera.model),
                    "camera_params": np.asarray(camera.params, dtype=np.float64),
                    "K": camera_to_K_and_dist(camera)[0],
                    "R_world_to_cam": np.full((3, 3), np.nan),
                    "t_world_to_cam": np.full(3, np.nan),
                    "camera_center_world": np.full(3, np.nan),
                    "pose_available": False,
                }
            )
            continue


        camera = recon.cameras[image.camera_id]

        R_wc, t_wc = get_image_pose_world_to_cam(image)
        C_w = -R_wc.T @ t_wc

        K, cam_params = camera_to_K_and_dist(camera)

        image_records.append(
            {
                "image_id": int(image_id),
                "image_name": image.name,
                "camera_id": int(image.camera_id),
                "width": int(camera.width),
                "height": int(camera.height),
                "camera_model": str(camera.model),
                "camera_params": cam_params,
                "K": K,
                "R_world_to_cam": R_wc,
                "t_world_to_cam": t_wc,
                "camera_center_world": C_w,
            }
        )

    # Sort by image name for deterministic ordering.
    image_records = sorted(image_records, key=lambda x: x["image_name"])

    image_ids = np.array([r["image_id"] for r in image_records], dtype=np.int64)
    image_names = np.array([r["image_name"] for r in image_records])
    camera_ids = np.array([r["camera_id"] for r in image_records], dtype=np.int64)

    widths = np.array([r["width"] for r in image_records], dtype=np.int64)
    heights = np.array([r["height"] for r in image_records], dtype=np.int64)
    camera_models = np.array([r["camera_model"] for r in image_records])

    K = np.stack([r["K"] for r in image_records], axis=0)
    R_world_to_cam = np.stack([r["R_world_to_cam"] for r in image_records], axis=0)
    t_world_to_cam = np.stack([r["t_world_to_cam"] for r in image_records], axis=0)
    camera_center_world = np.stack([r["camera_center_world"] for r in image_records], axis=0)

    # Camera params can be different lengths depending on model.
    # Store as object array.
    camera_params = np.array([r["camera_params"] for r in image_records], dtype=object)

    save_dict = {
        "image_ids": image_ids,
        "image_names": image_names,
        "camera_ids": camera_ids,
        "widths": widths,
        "heights": heights,
        "camera_models": camera_models,
        "camera_params": camera_params,
        "K": K,
        "R_world_to_cam": R_world_to_cam,
        "t_world_to_cam": t_world_to_cam,
        "camera_center_world": camera_center_world,
    }

    np.savez_compressed(out_path, **save_dict)

def demo_fn(gpu_num: str, image_dir:str, log_dir:str, log_file:str):
    # Print configuration
    # print("Arguments:", vars(args))
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    out_path = f"{log_dir}/{log_file}.npz"
    print(out_path)
    # Set seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(30)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)  # for multi-GPU
    print(f"Setting seed as: {42}")

    # Set device and dtype
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Run VGGT for camera and depth estimation
    model = VGGT()
    WEIGHT_MODULE = "/home/anthonyq/projects/scene_agent/breadth_agent/src/modules/models/sfm_models/vggt/weights/model.pt"
    model.load_state_dict(torch.load(WEIGHT_MODULE, weights_only=True))
    model.eval()
    model = model.to(device)
    print(f"Model loaded")

    # Get image paths and preprocess them
    # # SCENE_DIR_TT = "/home/anthonyq/datasets/tanks_and_temples/Lighthouse"
    # SCENE_DIR = "/home/anthonyq/datasets/ETH/ETH/office/images/dslr_images_undistorted"
    # # SCENE_DIR_ETH = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\ETH\\door_dslr_undistorted\\door\\images\\dslr_images_undistorted"
    # image_dir = SCENE_DIR

    image_path_list = glob.glob(os.path.join(image_dir, "*"))#[:frame_nums]
    # random.shuffle(image_path_list)
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")
    base_image_path_list = [os.path.basename(path) for path in image_path_list]

    # Load images and original coordinates
    # Load Image in 1024, while running VGGT with 518
    vggt_fixed_resolution = 518
    img_load_resolution = 1024

    images, original_coords = load_and_preprocess_images_square(image_path_list, img_load_resolution)
    print(images.shape)
    images = images.to(device)
    original_coords = original_coords.to(device)
    print(f"Loaded {len(images)} images from {image_dir}")
    frame_nums = len(images)

    # Run VGGT to estimate camera and depth
    # Run with 518x518 images
    extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, images, dtype, vggt_fixed_resolution)
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    image_size = np.array(images.shape[-2:])
    scale = img_load_resolution / vggt_fixed_resolution
    shared_camera = True # Was True

    with torch.cuda.amp.autocast(dtype=dtype):
        # Predicting Tracks
        # Using VGGSfM tracker instead of VGGT tracker for efficiency
        # VGGT tracker requires multiple backbone runs to query different frames (this is a problem caused by the training process)
        # Will be fixed in VGGT v2

        # You can also change the pred_tracks to tracks from any other methods
        # e.g., from COLMAP, from CoTracker, or by chaining 2D matches from Lightglue/LoFTR.
        pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
            images,
            conf=depth_conf,
            points_3d=points_3d,
            masks=None,
            max_query_pts=4096,
            query_frame_num=frame_nums,
            keypoint_extractor="sp",
            fine_tracking=False,
        )

        torch.cuda.empty_cache()

    # rescale the intrinsic matrix from 518 to 1024
    intrinsic[:, :2, :] *= scale
    # print(track_mask)
    track_mask = pred_vis_scores > 0.1

    print("POINTS", points_3d.shape)
    print(track_mask)

    
    # TODO: radial distortion, iterative BA, masks
    reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
        points_3d,
        extrinsic,
        intrinsic,
        pred_tracks,
        image_size,
        masks=track_mask,
        max_reproj_error=8.0,
        shared_camera=shared_camera,
        camera_type="SIMPLE_PINHOLE",
        points_rgb=points_rgb,
    )

    if reconstruction is None:
        raise ValueError("No reconstruction can be built with BA")

    print(points_3d.shape)
    print(pred_tracks.shape)
    print(valid_track_mask.shape)
    # Bundle Adjustment
    ba_options = pycolmap.BundleAdjustmentOptions()
    pycolmap.bundle_adjustment(reconstruction, ba_options)

    reconstruction.update_point_3d_errors()
    # Mean reprojection error in pixels (final cost)
    mean_error = reconstruction.compute_mean_reprojection_error()   

    # Store estimated extrinsics!
    store_extrinsics_information(reconstruction, out_path)
    torch.cuda.empty_cache()

    return mean_error


def rename_colmap_recons_and_rescale_camera(
    reconstruction, image_paths, original_coords, img_size, shift_point2d_to_original_res=False, shared_camera=False
):
    rescale_camera = True

    for pyimageid in reconstruction.images:
        # Reshaped the padded&resized image to the original size
        # Rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            # Rescale the camera parameters
            pred_params = copy.deepcopy(pycamera.params)

            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp  # center of the image

            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            # Also shift the point2D to original resolution
            top_left = original_coords[pyimageid - 1, :2]

            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            # If shared_camera, all images share the same camera
            # no need to rescale any more
            rescale_camera = False

    return reconstruction


# if __name__ == "__main__":
#     # args = parse_args()

# Log Folder
log_folder = "/home/anthonyq/projects/scene_agent/breadth_agent/results/vggt_sparse_results"
gpu_num = "7"

# # TT RUN 
# scene_list = ["barn_1_40", "barn_186_225", "barn_371_410",
#              "caterpillar_1_40", "caterpillar_173_212", "caterpillar_344_383",
#              "church_1_40", "church_235_274", "church_468_507",
#              "courthouse_1_40", "courthouse_534_573", "courthouse_1067_1106",
#              "ignatius_1_40", "ignatius_113_152", "ignatius_224_263",
#              "meetingroom_1_40", "meetingroom_167_206", "meetingroom_332_371",
#              "truck_1_40", "truck_107_146", "truck_212_251"]

# # Uncomment Both vars!
# d_set = "tanks_and_temples"
# home_folder = f"/home/anthonyq/datasets/tanks_and_temples"


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
               "hydrant/167_18184_34441", "hydrant/411_56064_108483"]
d_set = "co3d"
failed_runs = []

with torch.no_grad():
    errors = []
    with open(f"{log_folder}/{d_set}/mean_result.txt", "w") as f:
        for i in range(len(co3d_images)):
            img_seq = co3d_images[i]
            c, seq = img_seq.split('/')
            image_path = f"/home/anthonyq/datasets/co3d_v2/{img_seq}/{img_postfix}"
            # log_file = f"log_cam_poses_{img_seq}_{img_postfix}"
            # cal_path = f"/home/anthonyq/datasets/co3d_v2/{c}/calibration_new_{seq}.npz"
            # image_paths = home_folder + f"/{scene_list[i]}"
            # out_path_pose = os.path.join(log_folder, d_set, c, seq, f"mapanything_poses_{img_postfix}.npz")
            # out_path_ply = os.path.join(log_folder, d_set, c, seq, "mapanything_dense_points.ply")
            outpath = f"{log_folder}/{d_set}/{img_postfix}/{c}/{seq}"
            os.makedirs(f"{log_folder}/{d_set}/{img_postfix}/{c}", exist_ok=True)
            log_file = f"{img_seq}_{img_postfix}_pose_log"
            error = demo_fn(gpu_num, image_path, f"{log_folder}/{d_set}/{img_postfix}", log_file)
            print(f"{img_seq}_{img_postfix} Error: {error}")
            errors.append(error)
            f.write(f"{img_seq}_{img_postfix} Error: {error}\n")

        print("Reprojection Mean:", np.mean(errors))
        # with open(f"{log_folder}/{d_set}/mean_result.txt", "w") as f:
        f.write(f"Mean reprojection value: {np.mean(errors)}\n")

# with torch.no_grad():
#     errors = []
#     with open(f"{log_folder}/{d_set}/mean_result.txt", "w") as f:
#         for i in range(len(scene_list)):
#             image_path = home_folder + f"/{scene_list[i]}"
#             log_file = f"log_cam_poses_{scene_list[i]}"
#             # For ETH Run!
#             # image_path = home_folder + f"{ETH_images[i]}/images/dslr_images_undistorted"

#             error = demo_fn(gpu_num, image_path, f"{log_folder}/{d_set}", log_file)
#             print(f"/{scene_list[i]} Error: {error}")
#             errors.append(error)
#             f.write(f"/{scene_list[i]} Error: {error}\n")

#         print("Reprojection Mean:", np.mean(errors))
#     # with open(f"{log_folder}/{d_set}/mean_result.txt", "w") as f:
#         f.write(f"Mean reprojection value: {np.mean(errors)}\n")
#         # f.close()
