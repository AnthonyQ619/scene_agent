
# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Experiments\\ETH\\door\\images\\dslr_images_undistorted"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Experiments\\ETH\\door\\dslr_calibration_undistorted\\calibration_new.npz"

# STEP 1: Read in Camera Data
from modules.cameramanager import CameraDataManager

CDM = CameraDataManager(image_path=image_path,
                        calibration_path=calibration_path)

# Get Camera Data
camera_data = CDM.get_camera_data()

# STEP 2: Estimate Features per Image
from modules.features import FeatureDetectionSIFT
# Feature Module Initialization
feature_detector = FeatureDetectionSIFT(cam_data=camera_data,
                                        max_keypoints=15000,
                                        contrast_threshold=0.01,
                                        edge_threshold=12,
                                        n_octave_layers=3,
                                        sigma=1.6)

# Detect Features for all Images
features = feature_detector()

# STEP 3: Match Features Per Image (Pairwise for Pose Initialization)
from modules.featurematching import FeatureMatchBFPair
# Pairwise Feature Matching Module Initialization
feature_matcher = FeatureMatchBFPair(detector="sift",
                                     cam_data=camera_data,
                                     k=2,
                                     cross_check=False,
                                     lowes_thresh=0.70,
                                     RANSAC_threshold=0.02,
                                     RANSAC_conf=0.99)

# Detect Image Pair Correspondences for Pose Estimation
feature_pairs = feature_matcher(features=features)

# STEP 4: Estimate Camera Poses of Scene with Local Bundle Adjustment Activated
from modules.optimization import BundleAdjustmentOptimizerLocal

# Build Local Optimizer (Intrinsics kept fixed)
optimizer_local = BundleAdjustmentOptimizerLocal(cam_data=camera_data,
                                                 refine_focal_length=False,
                                                 refine_principal_point=False,
                                                 refine_extra_params=False,
                                                 max_num_iterations=40,
                                                 window_size=8,
                                                 robust_loss=True,
                                                 min_track_len=3,
                                                 use_gpu=True,
                                                 gpu_index=0)

from modules.camerapose import CamPoseEstimatorEssentialToPnP
# Camera Pose Module Initialization
pose_estimator = CamPoseEstimatorEssentialToPnP(cam_data=camera_data,
                                                iteration_count=300,
                                                reprojection_error=3.0,
                                                confidence=0.995,
                                                optimizer=optimizer_local)

# From estimated features, estimate the camera poses for all image frames
cam_poses = pose_estimator(feature_pairs=feature_pairs)

# STEP 5: Track Features across Multiple Images
from modules.featurematching import FeatureMatchBFTracking
# Feature Tracking Module Initialization
feature_tracker = FeatureMatchBFTracking(detector="sift",
                                         cam_data=camera_data,
                                         k=2,
                                         cross_check=False,
                                         lowes_thresh=0.72,
                                         RANSAC_threshold=0.5,   # Relaxed to increase track length
                                         RANSAC_conf=0.99)

# From estimated features, tracked features across multiple images
tracked_features = feature_tracker(features=features)

# STEP 6: Sparse 3D Reconstruction (Multi-View)
from modules.scenereconstruction import Sparse3DReconstructionMono
# Scene Reconstruction Module Initialization (relaxed triangulation requirements)
sparse_reconstruction = Sparse3DReconstructionMono(cam_data=camera_data,
                                                   view="multi",
                                                   min_observe=3,
                                                   min_angle=1.0,
                                                   reproj_error=3.0)

# Estimate sparse 3D scene from tracked features and camera poses
sparse_scene = sparse_reconstruction(tracked_features, cam_poses)

# STEP 7: Global Bundle Adjustment
from modules.optimization import BundleAdjustmentOptimizerGlobal

# Build Global Optimizer (Intrinsics kept fixed)
optimizer_global = BundleAdjustmentOptimizerGlobal(cam_data=camera_data,
                                                   refine_focal_length=False,
                                                   refine_principal_point=False,
                                                   refine_extra_params=False,
                                                   max_num_iterations=200,
                                                   robust_loss=True,
                                                   use_gpu=True,
                                                   gpu_index=0)

# Run Optimizer
optimal_scene = optimizer_global(sparse_scene)
