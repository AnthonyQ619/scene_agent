
# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Experiments\\tanks_and_temples\\Lighthouse"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Experiments\\tanks_and_temples\\calibration_new_2048.npz"

# STEP 1: Read in Camera Data
from modules.cameramanager import CameraDataManager

CDM = CameraDataManager(image_path=image_path,
                        max_images=100,
                        calibration_path=calibration_path)

# Get Camera Data
cam_data = CDM.get_camera_data()

# STEP 2: Estimate Features per Image
from modules.features import FeatureDetectionSIFT
# Feature Module Initialization
feature_detector = FeatureDetectionSIFT(cam_data=cam_data,
                                        max_keypoints=15000,
                                        contrast_threshold=0.01,
                                        edge_threshold=12,
                                        sigma=1.6)

# Detect Features for all Images
features = feature_detector()

# STEP 3: Match Features Per Image (Pairwise for Pose Initialization)
from modules.featurematching import FeatureMatchLightGluePair
# Pairwise Feature Matching Module Initialization
feature_matcher = FeatureMatchLightGluePair(cam_data=cam_data,
                                            detector='sift',
                                            n_layers=9,
                                            flash=True,
                                            mp=False,
                                            depth_confidence=0.90,
                                            width_confidence=0.99,
                                            filter_threshold=0.15,
                                            RANSAC=True,
                                            RANSAC_threshold=0.03,
                                            RANSAC_conf=0.999)

# Detect Image Pair Correspondences for Pose Estimation
feature_pairs = feature_matcher(features=features)

# STEP 4: Estimate Camera Poses of Scene with Local Bundle Adjustment Activated
from modules.optimization import BundleAdjustmentOptimizerLocal

# Build Local Optimizer
optimizer_local = BundleAdjustmentOptimizerLocal(cam_data=cam_data,
                                                 refine_focal_length=False,
                                                 refine_principal_point=False,
                                                 refine_extra_params=False,
                                                 max_num_iterations=60,
                                                 use_gpu=True,
                                                 gpu_index=0,
                                                 robust_loss=True,
                                                 window_size=8,
                                                 min_track_len=3)

from modules.camerapose import CamPoseEstimatorEssentialToPnP
# Camera Pose Module Initialization
pose_estimator = CamPoseEstimatorEssentialToPnP(cam_data=cam_data,
                                                iteration_count=300,
                                                reprojection_error=3.0,
                                                confidence=0.995,
                                                optimizer=optimizer_local)

# From estimated features, estimate the camera poses for all image frames
cam_poses = pose_estimator(feature_pairs=feature_pairs)

# STEP 5: Track Features Across Multiple Images (Multi-View Tracks)
from modules.featurematching import FeatureMatchLightGlueTracking
# Feature Tracking Module Initialization
feature_tracker = FeatureMatchLightGlueTracking(cam_data=cam_data,
                                                detector='sift',
                                                n_layers=7,
                                                flash=True,
                                                mp=False,
                                                depth_confidence=0.90,
                                                width_confidence=0.98,
                                                filter_threshold=0.15,
                                                RANSAC_threshold=0.03,
                                                RANSAC_conf=0.99)

# From estimated features, track features across multiple images
tracked_features = feature_tracker(features=features)

# STEP 6: Reconstruct Sparse 3D Scene (Classical Triangulation)
from modules.scenereconstruction import Sparse3DReconstructionMono
# Scene Reconstruction Module Initialization
sparse_reconstruction = Sparse3DReconstructionMono(cam_data=cam_data,
                                                   view="multi",
                                                   min_observe=4,
                                                   min_angle=2.0,
                                                   reproj_error=2.0)

# Estimate sparse 3D scene from tracked features and camera poses
sparse_scene = sparse_reconstruction(tracked_features, cam_poses)

# STEP 7: Global Bundle Adjustment (Intrinsics Fixed)
from modules.optimization import BundleAdjustmentOptimizerGlobal

# Build Global Optimizer
optimizer_global = BundleAdjustmentOptimizerGlobal(cam_data=cam_data,
                                                   refine_focal_length=False,
                                                   refine_principal_point=False,
                                                   refine_extra_params=False,
                                                   max_num_iterations=200,
                                                   use_gpu=True,
                                                   gpu_index=0,
                                                   robust_loss=True)

# Run Global Optimization
optimal_scene = optimizer_global(sparse_scene)
