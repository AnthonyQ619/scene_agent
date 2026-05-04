# Construct Modules with Initialized Arguments
# image_path = "/home/anthonyq/datasets/ETH/ETH/living_room/images/dslr_images_undistorted" #"/home/anthonyq/datasets/DTU/scan6_illumination_change" #"C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\scan6_illumination_change"

from modules.features import FeatureDetectionSP
from modules.featurematching import FeatureMatchSuperGlueTracking, FeatureMatchLightGlueTracking
from modules.camerapose import CamPoseEstimatorVGGTModel
from modules.scenereconstruction import Sparse3DReconstructionVGGT, Dense3DReconstructionVGGT, Sparse3DReconstructionVGGTNoFeatures
from modules.optimization import BundleAdjustmentOptimizerGlobal
from modules.baseclass import SfMScene
"""
# Step 1: Read in Calibration/Image Data
reconstructed_scene = SfMScene(id=3,
                              gpu_num="5",
                              log_dir="/home/anthonyq/projects/scene_agent/breadth_agent/results/ETH/",
                                image_path=image_path,
                                max_images=30,
                                target_resolution=[1024, 1024]
)

# Step 2: Detect Features
# reconstructed_scene.FeatureDetectionSP(max_keypoints=6000)

# Step 3: Detect Feature Pairs
# Ignore since pose estimation is using VGGT as the backbone structure

# Step 4: Detect/Estimate Camera Poses
reconstructed_scene.CamPoseEstimatorVGGTModel()

# Step 5: Detect Feature Tracks
# reconstructed_scene.FeatureMatchLightGlueTracking(
#     detector='superpoint', 
#     RANSAC_threshold=2.0
# )

# Step 6: Estimate Sparse Reconstruction
reconstructed_scene.Sparse3DReconstructionVGGTNoFeatures()

# Step 7: Run Optimization
reconstructed_scene.BundleAdjustmentOptimizerGlobal(
    max_num_iterations=130,
)
"""

# Construct Modules with Initialized Arguments
image_path = "/home/anthonyq/datasets/ETH/ETH/office/images/dslr_images_undistorted"
calibration_path = "/home/anthonyq/datasets/ETH/ETH/office/dslr_calibration_undistorted/calibration_ETH_new.npz"

# Step 1: Read in Calibration/Image Data (VGGT requires square resizing)
reconstructed_scene = SfMScene(
    id=3,
    gpu_num="5",
    log_dir="/home/anthonyq/projects/scene_agent/breadth_agent/results/ETH/",
    image_path=image_path,
    max_images=20,
    calibration_path=calibration_path,
    target_resolution=[1024, 1024]
)

# Step 2: Detect Features
# Skipped (detector-free pipeline)

# Step 3: Detect Feature Pairs
# Skipped (VGGT pose estimation does not require pairwise matches)

# Step 4: Detect/Estimate Camera Poses (VGGT)
reconstructed_scene.CamPoseEstimatorVGGTModel()

# Step 5: Detect Feature Tracks
# Skipped (no-feature path)

# Step 6: Estimate Sparse Reconstruction (VGGT No-Features)
reconstructed_scene.Sparse3DReconstructionVGGTNoFeatures()

# Step 7: Run Global Optimization
reconstructed_scene.BundleAdjustmentOptimizerGlobal(
    refine_focal_length=False,
    refine_principal_point=False,
    refine_extra_params=False,
    max_num_iterations=200,
    # robust_loss=True
)

