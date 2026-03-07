
# Fix for VGGT square image requirement: pad images to square before running the pipeline
import os
import numpy as np
from PIL import Image

# Original paths
image_path = r"C:\Users\Anthony\Documents\Projects\datasets\sfm_dataset\Experiments\DTU\scan90\images"
calibration_path = r"C:\Users\Anthony\Documents\Projects\datasets\sfm_dataset\Experiments\DTU\calibration_DTU_new.npz"

# Create a squared image directory
square_image_path = os.path.join(os.path.dirname(image_path), os.path.basename(image_path) + "_square")
os.makedirs(square_image_path, exist_ok=True)

# Helper: pad an image to square (no distortion)
def pad_to_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = max(w, h)
    background = Image.new(mode=img.mode, size=(s, s), color=(0, 0, 0))
    left = (s - w) // 2
    top = (s - h) // 2
    background.paste(img, (left, top))
    return background

# Accepted image extensions
valid_ext = {{".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}}

# Process all images in the directory and save squared versions
for fname in sorted(os.listdir(image_path)):
    fpath = os.path.join(image_path, fname)
    if not os.path.isfile(fpath) or os.path.splitext(fname.lower())[1] not in valid_ext:
        continue
    img = Image.open(fpath).convert("RGB")
    img_sq = pad_to_square(img)
    img_sq.save(os.path.join(square_image_path, fname))

# Compute square padding offsets using the first image (DTU images are typically same size)
def get_square_padding_offsets(img_path: str):
    for fname in sorted(os.listdir(img_path)):
        if os.path.splitext(fname.lower())[1] in valid_ext:
            fpath = os.path.join(img_path, fname)
            with Image.open(fpath) as im:
                w, h = im.size
            s = max(w, h)
            left = (s - w) // 2
            top = (s - h) // 2
            return w, h, s, left, top
    raise RuntimeError("No valid images found to compute padding offsets.")

orig_w, orig_h, sq_size, pad_left, pad_top = get_square_padding_offsets(image_path)

# Adjust calibration intrinsics for square padding and save a new npz
def adjust_calibration_for_square(calib_npz_path: str,
                                  out_npz_path: str,
                                  pad_left: int,
                                  pad_top: int,
                                  sq_size: int):
    data = np.load(calib_npz_path)
    out = {{}}

    # Helper to shift K
    def shift_K(K):
        K = np.array(K, dtype=float).copy()
        # Assume standard pinhole K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        K[0, 2] = K[0, 2] + pad_left
        K[1, 2] = K[1, 2] + pad_top
        return K

    # Copy through unknown keys by default
    for k in data.files:
        out[k] = data[k]

    # Handle common intrinsics representations
    # Single K
    if "K" in data.files and data["K"].ndim == 2 and data["K"].shape == (3, 3):
        out["K"] = shift_K(data["K"])
    # Multiple Ks (per image)
    if "K" in data.files and data["K"].ndim == 3 and data["K"].shape[-2:] == (3, 3):
        Ks = data["K"]
        out["K"] = np.stack([shift_K(Ks[i]) for i in range(Ks.shape[0])], axis=0)

    # Some datasets might use 'Ks' or 'intrinsics'
    if "Ks" in data.files and data["Ks"].ndim == 3 and data["Ks"].shape[-2:] == (3, 3):
        Ks = data["Ks"]
        out["Ks"] = np.stack([shift_K(Ks[i]) for i in range(Ks.shape[0])], axis=0)

    if "intrinsics" in data.files:
        intr = data["intrinsics"]
        if intr.ndim == 2 and intr.shape == (3, 3):
            out["intrinsics"] = shift_K(intr)
        elif intr.ndim == 3 and intr.shape[-2:] == (3, 3):
            out["intrinsics"] = np.stack([shift_K(intr[i]) for i in range(intr.shape[0])], axis=0)

    # Scalar or vector parameters
    if "fx" in data.files:
        out["fx"] = data["fx"]
    if "fy" in data.files:
        out["fy"] = data["fy"]
    if "cx" in data.files:
        cx = data["cx"]
        out["cx"] = cx + pad_left
    if "cy" in data.files:
        cy = data["cy"]
        out["cy"] = cy + pad_top

    # Update image size fields if present
    for key in ["width", "W"]:
        if key in data.files:
            out[key] = np.array(sq_size, dtype=np.int64)
    for key in ["height", "H"]:
        if key in data.files:
            out[key] = np.array(sq_size, dtype=np.int64)
    if "image_size" in data.files:
        # image_size might be [W, H] or array of such; force square
        img_sz = data["image_size"]
        if img_sz.ndim == 1 and img_sz.shape[0] == 2:
            out["image_size"] = np.array([sq_size, sq_size], dtype=np.int64)
        elif img_sz.ndim == 2 and img_sz.shape[1] == 2:
            out["image_size"] = np.tile(np.array([sq_size, sq_size], dtype=np.int64), (img_sz.shape[0], 1))

    np.savez(out_npz_path, **out)
    return out_npz_path

adjusted_calibration_path = os.path.join(square_image_path, "calibration_square.npz")
adjust_calibration_for_square(calibration_path, adjusted_calibration_path, pad_left, pad_top, sq_size)

# STEP 1: Read in Camera Data (use squared images; provide adjusted intrinsics)
from modules.cameramanager import CameraDataManager

CDM = CameraDataManager(image_path=square_image_path,
                        calibration_path=adjusted_calibration_path)

camera_data = CDM.get_camera_data()

# STEP 2: Camera Pose Estimation with VGGT
from modules.camerapose import CamPoseEstimatorVGGTModel

pose_estimator = CamPoseEstimatorVGGTModel(cam_data=camera_data)
cam_poses = pose_estimator()

# STEP 3: Sparse 3D Reconstruction with VGGT (features are not required)
from modules.scenereconstruction import Sparse3DReconstructionVGGT

sparse_reconstruction = Sparse3DReconstructionVGGT(cam_data=camera_data)
sparse_scene = sparse_reconstruction(camera_poses=cam_poses)

# STEP 4: Global Bundle Adjustment (refine intrinsics; intrinsics are now valid for square images)
from modules.optimization import BundleAdjustmentOptimizerGlobal

optimizer_global = BundleAdjustmentOptimizerGlobal(cam_data=camera_data,
                                                   refine_focal_length=True,
                                                   refine_principal_point=True,
                                                   refine_extra_params=False,
                                                   max_num_iterations=180,
                                                   use_gpu=True,
                                                   robust_loss=True)

optimal_scene = optimizer_global(sparse_scene)
