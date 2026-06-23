from modules.features import (FeatureDetectionSIFT, FeatureDetectionSP, FeatureDetectionORB)
from modules.featurematching import (FeatureMatchFlannTracking, FeatureMatchBFPair, FeatureMatchFlannPair, FeatureMatchBFTracking, FeatureMatchLightGlueTracking, FeatureMatchSuperGlueTracking, FeatureMatchLightGluePair, FeatureMatchSuperGluePair)
from modules.camerapose import (CamPoseEstimatorEssentialToPnP, CamPoseEstimatorVGGTModel)
from modules.scenereconstruction import (Sparse3DReconstructionMono, Sparse3DReconstructionVGGT, Dense3DReconstructionVGGT, Dense3DReconstructionMono)
from modules.optimization import (BundleAdjustmentOptimizerLocal, BundleAdjustmentOptimizerGlobal)
from modules.baseclass import SfMScene
ID = "1"
log_dir = "/home/anthonyq/projects/scene_agent/breadth_agent/results/co3d/apple_110_13051_23361_vggt_random_10"
gpu_num = "5"



# Construct Modules with Initialized Arguments
image_path = "/home/anthonyq/datasets/co3d_v2/apple/110_13051_23361/vggt_random_10"

# Step 1: Read in Image Data (VGGT pipeline requires square target resolution and no calibration input)
reconstructed_scene = SfMScene(
ID,
log_dir=log_dir,
gpu_num=gpu_num,
image_path=image_path,
target_resolution=[1024, 1024]
)

# Step 2: Detect Features
# Skipped (detector-free pipeline)

# Step 3: Detect Feature Pairs
# Skipped (VGGT pose initializer does not require pairwise matches)

# Step 4: Detect/Estimate Camera Poses (VGGT feed-forward initializer)
reconstructed_scene.CamPoseEstimatorVGGTModel()

# Step 5: Estimate Sparse Reconstruction (detector-free)
reconstructed_scene.Sparse3DReconstructionVGGTNoFeatures()

# Step 6: Global Bundle Adjustment with fixed intrinsics (pose-only refinement)
reconstructed_scene.BundleAdjustmentOptimizerGlobal(
refine_focal_length=False,
refine_principal_point=False,
refine_extra_params=False,
robust_loss=False,
max_num_iterations=220
)