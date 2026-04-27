from modules.features import FeatureDetectionSP
from modules.featurematching import FeatureMatchSuperGlueTracking, FeatureMatchSuperGluePair
# from modules.camerapose import CamPoseEstimatorEssentialToPnP
from modules.optimization import BundleAdjustmentOptimizerLocal
from modules.scenereconstruction import Sparse3DReconstructionMono
from modules.visualize import VisualizeScene
import os
from modules.baseclass import SfMScene

# Construct Modules with Initialized Arguments
image_path = "/home/anthonyq/datasets/DTU/DTU/scan22"
calibration_path = "/home/anthonyq/datasets/DTU/DTU/calibration_DTU_new.npz"

## NEW SFM PIPELINE
# Step 1: Read in Calibration/Image Data
reconstructed_scene = SfMScene(image_path = image_path, 
                                max_images = 15,
                                calibration_path = calibration_path)

# Step 2: Detect Features
reconstructed_scene.FeatureDetectionSIFT(
    max_keypoints=10000,
    contrast_threshold=0.009,
    edge_threshold=20,
)
from modules.camerapose import CamPoseEstimatorEssentialToPnP
# Step 3: Detect Feature Pairs
reconstructed_scene.FeatureMatchFlannPair(detector="sift",
                                        k=2,
                                        lowes_thresh=0.78,
                                        RANSAC_homography=False,
                                        RANSAC_threshold=0.02,
                                        RANSAC_conf=0.999)

# FeatureMatchLightGluePair(
#     detector='sift',   
#     RANSAC_threshold=0.02,
#     RANSAC_conf=0.999
# )

# Step 4: Detect/Estimate Camera Poses
reconstructed_scene.CamPoseEstimatorEssentialToPnP(
    iteration_count=150,
    reprojection_error = 3.0,
    optimizer = ("BundleAdjustmentOptimizerLocal", {
        "max_num_iterations": 25,
        "robust_loss": True,
        "use_gpu": False
    }),
)

# Step 5: Detect Feature Tracks
reconstructed_scene.FeatureMatchBFTracking(
    detector = "sift",
    k=2,
    cross_check=False,
    lowes_thresh=0.70,
    RANSAC_threshold=0.015,
    RANSAC_conf=0.999
)

# Step 6: Estimate Sparse Reconstruction
reconstructed_scene.Sparse3DReconstructionMono(
    min_observe=3,
    min_angle=2.0,
    multi_view=True
)

# Step 7: Run Optimization
reconstructed_scene.BundleAdjustmentOptimizerGlobal(
    max_num_iterations=200,
    use_gpu=False
)