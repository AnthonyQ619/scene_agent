
# Construct Modules with Initialized Arguments
image_path = "/home/anthonyq/datasets/DTU/DTU/scan10/images"
calibration_path = "/home/anthonyq/datasets/DTU/DTU/calibration_DTU_new.npz"

from modules.features import FeatureDetectionSIFT
from modules.featurematching import FeatureMatchBFPair, FeatureMatchBFTracking
from modules.camerapose import CamPoseEstimatorEssentialToPnP
from modules.scenereconstruction import Sparse3DReconstructionMono
from modules.optimization import BundleAdjustmentOptimizerGlobal, BundleAdjustmentOptimizerLocal
from modules.baseclass import SfMScene

# Step 1: Read in Calibration/Image Data
reconstructed_scene = SfMScene(
    id = 1,
    image_path=image_path,
    max_images=20,
    calibration_path=calibration_path
)

# Step 2: Detect Features
reconstructed_scene.FeatureDetectionSIFT(
    max_keypoints=12000,
    contrast_threshold=0.01,
    edge_threshold=12
)

"""
# Step 3: Detect Feature Pairs
reconstructed_scene.FeatureMatchBFPair(
    detector="sift",
    lowes_thresh=0.76,
    RANSAC_threshold=2.0, #0.02,
    RANSAC_homography=False
)

# Step 4: Detect/Estimate Camera Poses
reconstructed_scene.CamPoseEstimatorEssentialToPnP(
    reprojection_error=3.0,
    iteration_count=220,
    confidence=0.995,
    optimizer=("BundleAdjustmentOptimizerLocal", {
        "max_num_iterations": 25,
        "window_size": 8,
        "robust_loss": True,
        "use_gpu": False
    }),
)

# Step 5: Detect Feature Tracks
reconstructed_scene.FeatureMatchBFTracking(
    detector="sift",
    cross_check=True,
    lowes_thresh=0.72,
    RANSAC_threshold=1.0 #0.015
)

# Step 6: Estimate Sparse Reconstruction
reconstructed_scene.Sparse3DReconstructionMono(
    min_observe=4,
    min_angle=2.5,
    multi_view=True
)

# Step 7: Run Global Optimization
reconstructed_scene.BundleAdjustmentOptimizerGlobal(
    max_num_iterations=180,
    use_gpu=False
)
"""
