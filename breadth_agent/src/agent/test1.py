# Construct Modules with Initialized Arguments
# image_path = "/work/dataset/scan14"
calibration_path = "/work/dataset/DTU/calibration_DTU_new.npz"

# Construct Modules with Initialized Arguments
image_path = "/work/dataset/scan14"

from modules.features import FeatureDetectionSP
from modules.featurematching import FeatureMatchSuperGlueTracking
from modules.camerapose import CamPoseEstimatorVGGTModel
from modules.scenereconstruction import Sparse3DReconstructionVGGT
from modules.optimization import BundleAdjustmentOptimizerGlobal
from modules.baseclass import SfMScene

# Step 1: Read in Calibration/Image Data
reconstructed_scene = SfMScene(image_path=image_path,
                                max_images=5,
                                target_resolution=[1024, 1024]
)

# Step 2: Detect Features
reconstructed_scene.FeatureDetectionSP(max_keypoints=5000)

# Step 3: Detect Feature Pairs
# Ignore since pose estimation is using VGGT as the backbone structure

# Step 4: Detect/Estimate Camera Poses
reconstructed_scene.CamPoseEstimatorVGGTModel()

# Step 5: Detect Feature Tracks
reconstructed_scene.FeatureMatchSuperGlueTracking(
    detector='superpoint', 
    RANSAC_threshold=0.01,
    setting="indoor"
)

# Step 6: Estimate Sparse Reconstruction
reconstructed_scene.Sparse3DReconstructionVGGT(min_observe=4)

# Step 7: Run Optimization
reconstructed_scene.BundleAdjustmentOptimizerGlobal(
    max_num_iterations=60,
    use_gpu=False
)
