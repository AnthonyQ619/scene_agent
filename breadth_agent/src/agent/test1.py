from modules.features import (FeatureDetectionSIFT, FeatureDetectionSP, FeatureDetectionORB)
from modules.featurematching import (FeatureMatchFlannTracking, FeatureMatchBFPair, FeatureMatchFlannPair, FeatureMatchBFTracking, FeatureMatchLightGlueTracking, FeatureMatchSuperGlueTracking, FeatureMatchLightGluePair, FeatureMatchSuperGluePair)
from modules.camerapose import (CamPoseEstimatorEssentialToPnP, CamPoseEstimatorVGGTModel)
from modules.scenereconstruction import (Sparse3DReconstructionMono, Sparse3DReconstructionVGGT, Dense3DReconstructionVGGT, Dense3DReconstructionMono)
from modules.optimization import (BundleAdjustmentOptimizerLocal, BundleAdjustmentOptimizerGlobal)
from modules.baseclass import SfMScene
ID = "b03290c8-ab09-46b1-800f-b735736be54c"


# Construct Modules with Initialized Arguments
image_path = "/home/anthonyq/datasets/ETH/ETH/living_room/images/dslr_images_undistorted"
calibration_path = "/home/anthonyq/datasets/ETH/ETH/office/dslr_calibration_undistorted/calibration_ETH_new.npz"

# Step 1: Read in Calibration/Image Data
reconstructed_scene = SfMScene(
    ID,
    image_path=image_path,
    max_images=20,
    calibration_path=calibration_path
)

# Step 2: Detect Features
reconstructed_scene.FeatureDetectionSP(
    max_keypoints=6000
)

# Step 3: Detect Pairwise Feature Matches
reconstructed_scene.FeatureMatchSuperGluePair(
    detector="superpoint",
    setting="indoor",
    sinkhorn_iterations=20,
    match_threshold=0.2,
    RANSAC_homography=False,
    RANSAC_threshold=1.0,
    RANSAC_conf=0.999
)

# Step 4: Detect/Estimate Camera Poses with Local BA
reconstructed_scene.CamPoseEstimatorEssentialToPnP(
    iteration_count=300,
    reprojection_error=3.0,
    confidence=0.99,
    ba_per_frame=4,
    optimizer=("BundleAdjustmentOptimizerLocal", {
        "max_num_iterations": 40,
        "robust_loss": True,
        "use_gpu": False,
        "window_size": 10,
        "min_track_len": 3,
        "refine_focal_length": False,
        "refine_principal_point": False,
        "refine_extra_params": False
    })
)

# Step 5: Detect Feature Tracks
reconstructed_scene.FeatureMatchSuperGlueTracking(
    detector="superpoint",
    setting="indoor",
    RANSAC_homography=False,
    RANSAC_threshold=2.0,
    RANSAC_conf=0.999
)

# Step 6: Estimate Sparse Reconstruction
reconstructed_scene.Sparse3DReconstructionMono(
    min_observe=4,
    min_angle=2.5,
    reproj_error=3.0,
    multi_view=True
)

# Step 7: Run Global Optimization
reconstructed_scene.BundleAdjustmentOptimizerGlobal(
    refine_focal_length=False,
    refine_principal_point=False,
    refine_extra_params=False,
    max_num_iterations=300,
    robust_loss=True,
    use_gpu=False
)
