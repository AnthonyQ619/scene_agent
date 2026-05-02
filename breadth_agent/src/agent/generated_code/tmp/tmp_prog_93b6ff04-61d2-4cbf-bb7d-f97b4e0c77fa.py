
from modules.features import (FeatureDetectionSIFT, FeatureDetectionSP, FeatureDetectionORB)
from modules.featurematching import (FeatureMatchFlannTracking, FeatureMatchBFPair, FeatureMatchFlannPair, FeatureMatchBFTracking, FeatureMatchLightGlueTracking, FeatureMatchSuperGlueTracking, FeatureMatchLightGluePair, FeatureMatchSuperGluePair)
from modules.camerapose import (CamPoseEstimatorEssentialToPnP, CamPoseEstimatorVGGTModel)
from modules.scenereconstruction import (Sparse3DReconstructionMono, Sparse3DReconstructionVGGT, Dense3DReconstructionVGGT, Dense3DReconstructionMono)
from modules.optimization import (BundleAdjustmentOptimizerLocal, BundleAdjustmentOptimizerGlobal)
from modules.baseclass import SfMScene
ID = "93b6ff04-61d2-4cbf-bb7d-f97b4e0c77fa"


# Construct Modules with Initialized Arguments
image_path = "/home/anthonyq/datasets/ETH/ETH/living_room/images/dslr_images_undistorted"
calibration_path = "/home/anthonyq/datasets/ETH/ETH/office/dslr_calibration_undistorted/calibration_ETH_new.npz"

# Step 1: Read in Calibration/Image Data
reconstructed_scene = SfMScene(ID,
                               image_path=image_path,
                               max_images=20,
                               calibration_path=calibration_path)

# Step 2: Detect Features
reconstructed_scene.FeatureDetectionSP(
    max_keypoints=6000
)

# Step 3: Skip pairwise matching (handled by not calling any pairwise matcher)

# Step 4: Estimate Camera Poses (VGGT)
reconstructed_scene.CamPoseEstimatorVGGTModel()

# Step 5: Track Features (SuperGlue)
reconstructed_scene.FeatureMatchSuperGlueTracking(
    detector="superpoint",
    setting="indoor",
    RANSAC_homography=False,
    RANSAC_threshold=2.5,
    RANSAC_conf=0.999
)

# Step 6: Sparse Reconstruction using VGGT poses
reconstructed_scene.Sparse3DReconstructionVGGT(
    min_observe=4
)

# Step 7: Global Bundle Adjustment (calibrated intrinsics kept fixed)
reconstructed_scene.BundleAdjustmentOptimizerGlobal(
    refine_focal_length=False,
    refine_principal_point=False,
    refine_extra_params=False,
    max_num_iterations=300,
    robust_loss=True
)
