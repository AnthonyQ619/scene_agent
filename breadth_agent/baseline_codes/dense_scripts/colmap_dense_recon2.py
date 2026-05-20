from pathlib import Path
import json
import shutil
import pycolmap
import numpy as np
import os

def load_calibration(calibration_file):
    """
    Loads camera intrinsics from either:

    1. JSON format:
       {
           "camera_model": "PINHOLE",
           "width": 1920,
           "height": 1080,
           "params": [fx, fy, cx, cy]
       }

       or for SIMPLE_RADIAL:
       {
           "camera_model": "SIMPLE_RADIAL",
           "width": 1920,
           "height": 1080,
           "params": [f, cx, cy, k]
       }

    2. COLMAP cameras.txt style single camera line:
       CAMERA_ID MODEL WIDTH HEIGHT PARAMS...
       Example:
       1 PINHOLE 1920 1080 1000 1000 960 540
    """
    calibration_file = Path(calibration_file)

    if calibration_file.suffix.lower() == ".json":
        with open(calibration_file, "r") as f:
            data = json.load(f)

        camera_model = data["camera_model"]
        params = data["params"]

        return {
            "camera_model": camera_model,
            "camera_params": ",".join(map(str, params)),
        }

    # COLMAP cameras.txt-style parser
    with open(calibration_file, "r") as f:
        lines = f.readlines()

    valid_lines = [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]

    if len(valid_lines) == 0:
        raise ValueError(f"No valid camera line found in {calibration_file}")

    # Use the first valid camera line
    parts = valid_lines[0].split()

    if len(parts) < 5:
        raise ValueError(
            "Expected COLMAP camera format: CAMERA_ID MODEL WIDTH HEIGHT PARAMS..."
        )

    camera_model = parts[1]
    params = parts[4:]

    return {
        "camera_model": camera_model,
        "camera_params": ",".join(params),
    }

def load_calibration_npz(calibration_file):
    data = np.load(calibration_file)
    data.allow_pickle = True 
    full_cal_data = dict(data)

    # Keys are
    # - k_mats: (N, 3, 3) -> N = num of cameras
    # - dists: (N, 1, 5) -> N = num of cameras 
    # - baseline_ext: None or (3, 4) -> the baseline of stereo camera

    K = full_cal_data['k_mats'][0]
    dists = full_cal_data['dists']
    baseline_ext = full_cal_data['baseline_ext']

    fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])

    if dists is None:
        model = "PINHOLE" #pycolmap.CameraModelId.PINHOLE
        params = ",".join([str(fx), str(fy), str(cx), str(cy)]) #np.array([fx, fy, cx, cy], dtype=np.float64)
    else:
        # assume OpenCV 4-dist for monocular
        d = dists.ravel().astype(np.float64)
        model = "OPENCV" #pycolmap.CameraModelId.OPENCV
        params = ",".join([str(fx), str(fy), str(cx), str(cy), str(d[0]), str(d[1]), str(d[2]), str(d[3])]) #np.array([fx, fy, cx, cy, d[0], d[1], d[2], d[3]], dtype=np.float64)
    return model, params

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
        R = qvec_to_rotmat(qvec)
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

def store_extrinsics_information(recon, out_dir, pose_file_name) -> None:
    out_path = os.path.join(out_dir, pose_file_name)
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
                    "K": self.camera_to_K_and_dist(camera)[0],
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

def run_default_pycolmap_sparse_reconstruction(
    image_dir,
    calibration_file,
    output_dir,
    use_gpu=True,
    clean_output=True,
    matcher="exhaustive",
    pose_file_name="",
    run_dense=True,
):
    """
    Default COLMAP-style sparse reconstruction using pycolmap.

    Args:
        image_dir:
            Path to folder containing images.

        calibration_file:
            Path to calibration JSON or COLMAP cameras.txt-style file.

        output_dir:
            Where to write database and sparse reconstruction.

        use_gpu:
            Whether to use GPU for feature extraction / matching if available.

        clean_output:
            If True, removes existing output_dir before running.

        matcher:
            "exhaustive" or "sequential".
            Use "exhaustive" for unordered image sets.
            Use "sequential" for video-like ordered datasets.

    Returns:
        best_reconstruction:
            pycolmap.Reconstruction object.

        reconstructions:
            Dictionary of all reconstructed models returned by pycolmap.
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    if clean_output and output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    database_path = output_dir / "database.db"
    sparse_path = output_dir / "sparse"
    sparse_path.mkdir(parents=True, exist_ok=True)
    dense_path = output_dir / "dense"
    fused_ply_path = dense_path / "fused.ply"

    # calib = load_calibration(calibration_file)
    # device = _get_device(use_gpu)

    model, params = load_calibration_npz(calibration_file)

    reader_options = pycolmap.ImageReaderOptions(
        camera_model=model, #calib["camera_model"],
        camera_params=params, #calib["camera_params"],
    )

    extraction_options = pycolmap.FeatureExtractionOptions()
    # extraction_options.use_gpu = use_gpu

    matching_options = pycolmap.FeatureMatchingOptions()
    # matching_options.use_gpu = use_gpu

    # 1. Extract SIFT features and import images into database.
    # camera_mode=SINGLE means all images share the same camera intrinsics.
    pycolmap.extract_features(
        database_path=database_path,
        image_path=image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_options,
        extraction_options=extraction_options,
    )

    # 2. Match features.
    if matcher == "exhaustive":
        pycolmap.match_exhaustive(
            database_path=database_path,
            matching_options=matching_options,
        )
    elif matcher == "sequential":
        pycolmap.match_sequential(
            database_path=database_path,
            matching_options=matching_options,
        )
    else:
        raise ValueError("matcher must be either 'exhaustive' or 'sequential'")

    # 3. Sparse incremental mapping.
    mapper_options = pycolmap.IncrementalPipelineOptions()

    # Since calibration is provided, usually you do NOT want intrinsics drifting.
    # This keeps the reconstruction closer to your provided calibration.
    mapper_options.ba_refine_focal_length = False
    mapper_options.ba_refine_principal_point = False
    mapper_options.ba_refine_extra_params = False

    reconstructions = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_dir,
        output_path=sparse_path,
        options=mapper_options,
    )

    if len(reconstructions) == 0:
        raise RuntimeError("COLMAP failed to reconstruct any sparse model.")

    # Pick largest reconstruction by number of registered images.
    best_model_id = max(
        reconstructions.keys(),
        key=lambda k: reconstructions[k].num_reg_images(),
    )
    best_reconstruction = reconstructions[best_model_id]

    # 4. Optional final bundle adjustment.
    # This refines poses and points, while keeping intrinsics fixed.
    ba_options = pycolmap.BundleAdjustmentOptions()
    pycolmap.bundle_adjustment(best_reconstruction, ba_options)

    # 5. Save final selected sparse model.
    final_sparse_path = output_dir / "sparse_final"
    final_sparse_path.mkdir(parents=True, exist_ok=True)
    best_reconstruction.write(final_sparse_path)

    print("Reconstruction complete.")
    print(best_reconstruction.summary())
    print(f"Database: {database_path}")
    print(f"All sparse models: {sparse_path}")
    print(f"Final sparse model: {final_sparse_path}")
    with open(f"{output_dir}/sparse_result.txt", "w") as f:
        f.write(best_reconstruction.summary())
    
    # store_extrinsics_information(best_reconstruction, output_dir, pose_file_name)
    if run_dense:
        print("HERE IN DENSE")
        # if not use_gpu:
        #     raise RuntimeError(
        #         "pycolmap.patch_match_stereo requires CUDA. "
        #         "Set use_gpu=True or skip dense reconstruction."
        #     )

        dense_path.mkdir(parents=True, exist_ok=True)

        # 4a. Undistort images and prepare COLMAP MVS workspace.
        # This creates:
        # dense/images
        # dense/sparse
        # dense/stereo
        pycolmap.undistort_images(
            output_path=str(dense_path),
            input_path=str(final_sparse_path),
            image_path=str(image_dir),
            output_type="COLMAP",
        )
        print("HERE IN DENSE")
        # 4b. PatchMatch stereo.
        patch_match_options = pycolmap.PatchMatchOptions()

        # Default COLMAP dense behavior usually uses geometric consistency.
        # This is the standard choice for better fused clouds.
        patch_match_options.geom_consistency = True

        patch_match_options.gpu_index = '4,5,6'

        pycolmap.patch_match_stereo(
            workspace_path=str(dense_path),
            workspace_format="COLMAP",
            options=patch_match_options,
        )
        print("HERE IN DENSE")
        # 4c. Stereo fusion to produce dense point cloud.
        fusion_options = pycolmap.StereoFusionOptions()

        pycolmap.stereo_fusion(
            output_path=str(fused_ply_path),
            workspace_path=str(dense_path),
            workspace_format="COLMAP",
            input_type="geometric",
            options=fusion_options,
            output_type="ply",
        )

        if not fused_ply_path.exists():
            raise RuntimeError(
                f"Dense fusion finished, but fused point cloud was not found: {fused_ply_path}"
            )
        print("HERE IN DENSE")
        # result["dense_path"] = str(dense_path)
        # result["fused_ply_path"] = str(fused_ply_path)

        print("Dense reconstruction complete.")
        print(f"Dense workspace: {dense_path}")
        print(f"Fused dense point cloud: {fused_ply_path}")
    return best_reconstruction, reconstructions

# Log Folder
log_folder = "/home/anthonyq/projects/scene_agent/breadth_agent/results/colmap_dense_results"

# # DTU RUN
scene_list = ["scan34", "scan48",
             "scan49", "scan62", "scan75", "scan77", 
             "scan110", "scan114", "scan118"]

# Change to dataset home location
d_set = "DTU"
home_folder = f"/home/anthonyq/datasets/DTU"
cal_path = "/home/anthonyq/datasets/DTU/calibration_DTU_new.npz"
failed_runs = []

for i in range(len(scene_list)):
    image_path = home_folder + f"/{scene_list[i]}"
    outpath = f"{log_folder}/{d_set}/{scene_list[i]}"
    # image_paths = home_folder + f"/{scene_list[i]}"
    # out_path_pose = os.path.join(log_folder, d_set, c, seq, f"mapanything_poses_{img_postfix}.npz")
    # out_path_ply = os.path.join(log_folder, d_set, c, seq, "mapanything_dense_points.ply")
    # outpath = f"{log_folder}/{d_set}/{img_postfix}/{c}/{seq}"
    file_name = f"{scene_list[i]}_pose_log.npz"
    # failed_runs = []
    print(image_path)
    try:
        run_default_pycolmap_sparse_reconstruction(
            image_dir=image_path,
            calibration_file=cal_path,
            output_dir=outpath,
            use_gpu=True,
            clean_output=True,
            matcher="exhaustive",
            pose_file_name=file_name,
            run_dense=True
        )
    except:
        failed_runs.append(scene_list[i])

print(failed_runs)