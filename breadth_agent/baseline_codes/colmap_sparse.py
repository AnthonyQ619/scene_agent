from pathlib import Path
import json
import shutil
import pycolmap
import numpy as np

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

def run_default_pycolmap_sparse_reconstruction(
    image_dir,
    calibration_file,
    output_dir,
    use_gpu=True,
    clean_output=True,
    matcher="exhaustive",
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

    # calib = load_calibration(calibration_file)

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

    return best_reconstruction, reconstructions

# Log Folder
log_folder = "/home/anthonyq/projects/scene_agent/breadth_agent/results/colmap_sparse_results"

# # DTU RUN
# scene_list = ["scan32", "scan33",
#               "scan34", "scan48", "scan49", "scan62", "scan75",
#               "scan77", "scan110", "scan114", "scan118"]

# # Change to dataset home location
# d_set = "DTU"
# home_folder = f"/home/anthonyq/datasets/DTU"
# cal_path = "/home/anthonyq/datasets/DTU/calibration_DTU_new.npz"

## ETH RUN (Change homefolder to ETH location - Path commented below)
# scene_list = ["courtyard", "delivery_area", "electro", 
#               "facade", "kicker", "meadow",
#               "office", "pipes", "playground",
#               "relief", "relief_2", "terrace", "terrains"]
## Home Folder/Image_path to use!
## Uncomment all vars!
# d_set = "ETH"
# homde_folder = "/home/anthonyq/datasets/ETH/ETH"
## image_path/cal_path in the loop to uncomment!

# UNCOMMENT FOR TT RUN 
# scene_list = ["barn_1_40", "barn_186_225", "barn_371_410",
#              "caterpillar_1_40", "caterpillar_173_212", "caterpillar_344_383",
#              "church_1_40", "church_235_274", "church_468_507",
#              "courthouse_1_40", "courthouse_534_573", "courthouse_1067_1106",
#              "ignatius_1_40", "ignatius_113_152", "ignatius_224_263",
#              "meetingroom_1_40", "meetingroom_167_206", "meetingroom_332_371",
#              "truck_1_40", "truck_107_146", "truck_212_251"]
scene_list = ["meetingroom_167_206", "meetingroom_332_371",
             "truck_1_40", "truck_107_146", "truck_212_251"]

# Uncomment Both vars!
cal_path = "/home/anthonyq/datasets/tanks_and_temples/calibration_new_1920.npz"
d_set = "tanks_and_temples"
home_folder = f"/home/anthonyq/datasets/tanks_and_temples"

for i in range(len(scene_list)):
    # For ETH Run!
    # image_path = home_folder + f"{scene_list[i]}/images/dslr_images_undistorted"
    # cal_path = f"/home/anthonyq/datasets/ETH/ETH/{ETH_images[i]}/dslr_calibration_undistorted/calibration_ETH_new.npz"
    image_path = home_folder + f"/{scene_list[i]}"
    outpath = f"{log_folder}/{d_set}/{scene_list[i]}"
    run_default_pycolmap_sparse_reconstruction(
        image_dir=image_path,
        calibration_file=cal_path,
        output_dir=outpath,
        use_gpu=False,
        clean_output=True,
        matcher="exhaustive",
    )