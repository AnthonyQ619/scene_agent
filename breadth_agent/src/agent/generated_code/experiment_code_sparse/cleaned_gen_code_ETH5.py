
# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Experiments\\ETH\\office\\images\\dslr_images_undistorted"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Experiments\\ETH\\office\\dslr_calibration_undistorted\\calibration_ETH_new.npz"

# STEP 1: Read in Camera Data
from modules.cameramanager import CameraDataManager

CDM = CameraDataManager(image_path=image_path,
                        calibration_path=calibration_path)

# Get Camera Data
camera_data = CDM.get_camera_data()

# STEP 2: Estimate Features per Image
from modules.features import FeatureDetectionSP
# Feature Module Initialization
feature_detector = FeatureDetectionSP(cam_data=camera_data,
                                      max_keypoints=6000)
# Detect Features for all Images
features = feature_detector()

# STEP 3: Match Features Per Image (Pairwise for Pose Initialization)
from modules.featurematching import FeatureMatchSuperGluePair, FeatureMatchLightGluePair

# Try SuperGlue first with less aggressive filtering; fallback to LightGlue if needed
try:
    feature_matcher = FeatureMatchSuperGluePair(cam_data=camera_data,
                                                detector='superpoint',
                                                setting='indoor',
                                                match_threshold=0.1,       # admit more tentative matches
                                                RANSAC=True,
                                                RANSAC_threshold=0.5,      # less strict for normalized coords
                                                RANSAC_conf=0.995)
    matched_features = feature_matcher(features=features)
except Exception:
    # Fallback: LightGlue (more forgiving and faster)
    feature_matcher = FeatureMatchLightGluePair(cam_data=camera_data,
                                                detector='SuperPoint',
                                                filter_threshold=0.1,
                                                RANSAC=True,
                                                RANSAC_threshold=0.5,
                                                RANSAC_conf=0.995)
    matched_features = feature_matcher(features=features)

# Safety: drop any pairs with < 4 correspondences if returned by the matcher
if hasattr(matched_features, "pairwise_matches") and isinstance(matched_features.pairwise_matches, list):
    matched_features.pairwise_matches = [pm for pm in matched_features.pairwise_matches if pm is not None and pm.shape[0] >= 4]

# STEP 4: Estimate Camera Poses with Local Bundle Adjustment
from modules.optimization import BundleAdjustmentOptimizerLocal

# Build Local Optimizer
optimizer_local = BundleAdjustmentOptimizerLocal(cam_data=camera_data,
                                                 refine_focal_length=False,
                                                 refine_principal_point=False,
                                                 refine_extra_params=False,
                                                 max_num_iterations=40,
                                                 window_size=10,
                                                 robust_loss=True,
                                                 min_track_len=3)

from modules.camerapose import CamPoseEstimatorEssentialToPnP
# Camera Pose Module Initialization (pass optimizer_local here)
pose_estimator = CamPoseEstimatorEssentialToPnP(cam_data=camera_data,
                                                iteration_count=300,
                                                reprojection_error=3.0,
                                                confidence=0.995,
                                                optimizer=optimizer_local)

# From estimated feature pairs, estimate the camera poses for all image frames
cam_poses = pose_estimator(feature_pairs=matched_features)

# STEP 5: Estimate Feature Tracks (Multi-View Tracking)
from modules.featurematching import FeatureMatchSuperGlueTracking
# Feature Tracking Module Initialization (less strict thresholds)
feature_tracker = FeatureMatchSuperGlueTracking(cam_data=camera_data,
                                                detector='superpoint',
                                                setting='indoor',
                                                match_threshold=0.1,
                                                RANSAC_threshold=0.5,
                                                RANSAC_conf=0.99)

# From estimated features, track features across multiple images
tracked_features = feature_tracker(features=features)

# STEP 6: Sparse 3D Reconstruction (Multi-View Triangulation)
from modules.scenereconstruction import Sparse3DReconstructionMono
# Scene Reconstruction Module Initialization
sparse_reconstruction = Sparse3DReconstructionMono(cam_data=camera_data,
                                                   view="multi",
                                                   min_observe=4,
                                                   min_angle=2.0,
                                                   reproj_error=2.0)

# Estimate sparse 3D scene from tracked features and camera poses
sparse_scene = sparse_reconstruction(tracked_features, cam_poses)

# STEP 7: Global Bundle Adjustment (Lock Intrinsics)
from modules.optimization import BundleAdjustmentOptimizerGlobal

# Build Global Optimizer
optimizer = BundleAdjustmentOptimizerGlobal(cam_data=camera_data,
                                            refine_focal_length=False,
                                            refine_principal_point=False,
                                            refine_extra_params=False,
                                            max_num_iterations=200,
                                            use_gpu=True,
                                            gpu_index=0,
                                            robust_loss=True)

# Run Global Optimization
optimal_scene = optimizer(sparse_scene)
