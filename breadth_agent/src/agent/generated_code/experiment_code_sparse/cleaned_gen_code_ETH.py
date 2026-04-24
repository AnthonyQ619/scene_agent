
# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Experiments\\ETH\\observatory\\images\\dslr_images_undistorted"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Experiments\\ETH\\observatory\\dslr_calibration_undistorted\\calibration_ETH_new.npz"

# STEP 1: Read in Camera Data
from modules.cameramanager import CameraDataManager

CDM = CameraDataManager(image_path=image_path,
                        max_images=100,
                        calibration_path=calibration_path)

# Get Camera Data
camera_data = CDM.get_camera_data()

# STEP 2: Estimate Features per Image (SIFT)
from modules.features import FeatureDetectionSIFT
# Feature Module Initialization
feature_detector = FeatureDetectionSIFT(cam_data=camera_data, 
                                        max_keypoints=15000,
                                        contrast_threshold=0.01,
                                        edge_threshold=10,
                                        sigma=1.6)

# Detect Features for all Images
features = feature_detector()

# STEP 3: Match Features Per Image (Pairwise BF for initialization)
from modules.featurematching import FeatureMatchBFPair
# Pairwise Feature Matching Module Initialization
feature_matcher = FeatureMatchBFPair(detector="sift", 
                                     cam_data=camera_data,
                                     k=2,
                                     cross_check=False,
                                     RANSAC=True,
                                     RANSAC_threshold=0.02,
                                     RANSAC_conf=0.99,
                                     lowes_thresh=0.70)

# Detect Image Pair Correspondences for Pose Estimation
feature_pairs = feature_matcher(features=features)

# STEP 4: Estimate Camera Poses with Local Bundle Adjustment
from modules.optimization import BundleAdjustmentOptimizerLocal

# Build Local Optimizer (keep intrinsics fixed)
optimizer_local = BundleAdjustmentOptimizerLocal(cam_data=camera_data,
                                                 refine_focal_length=False,
                                                 refine_principal_point=False,
                                                 refine_extra_params=False,
                                                 max_num_iterations=50,
                                                 use_gpu=True,
                                                 gpu_index=0,
                                                 robust_loss=True,
                                                 window_size=8)

from modules.camerapose import CamPoseEstimatorEssentialToPnP
# Camera Pose Module Initialization with Local BA attached
pose_estimator = CamPoseEstimatorEssentialToPnP(cam_data=camera_data,
                                                iteration_count=300,
                                                reprojection_error=3.0,
                                                confidence=0.995,
                                                optimizer=optimizer_local)

# From estimated features, estimate the camera poses for all image frames
cam_poses = pose_estimator(feature_pairs=feature_pairs)

# STEP 5: Track Features across Multiple Images (BF Tracking)
from modules.featurematching import FeatureMatchBFTracking
# Feature Tracking Module Initialization
feature_tracker = FeatureMatchBFTracking(detector="sift", 
                                         cam_data=camera_data,
                                         k=2,
                                         cross_check=False,
                                         RANSAC_threshold=0.02,
                                         RANSAC_conf=0.99,
                                         lowes_thresh=0.75)

# From estimated features, track features across multiple images
tracked_features = feature_tracker(features=features)

# STEP 6: Sparse 3D Reconstruction (Multi-view Triangulation)
from modules.scenereconstruction import Sparse3DReconstructionMono
# Scene Reconstruction Module Initialization
sparse_reconstruction = Sparse3DReconstructionMono(cam_data=camera_data,
                                                   view="multi",
                                                   min_observe=4,
                                                   min_angle=2.0,
                                                   reproj_error=2.0)

# Estimate sparse 3D scene from tracked features and camera poses
sparse_scene = sparse_reconstruction(tracked_features=tracked_features, cam_poses=cam_poses)

# STEP 7: Global Bundle Adjustment (Final Optimization)
from modules.optimization import BundleAdjustmentOptimizerGlobal

# Build Global Optimizer (keep intrinsics fixed)
optimizer = BundleAdjustmentOptimizerGlobal(cam_data=camera_data,
                                            refine_focal_length=False,
                                            refine_principal_point=False,
                                            refine_extra_params=False,
                                            max_num_iterations=200,
                                            use_gpu=True,
                                            gpu_index=0,
                                            robust_loss=True)

# Run Optimizer
optimal_scene = optimizer(sparse_scene)
