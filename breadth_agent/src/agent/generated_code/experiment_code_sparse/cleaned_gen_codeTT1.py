
# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Experiments\\tanks_and_temples\\Train"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Experiments\\tanks_and_temples\\calibration_new_1920.npz"

# STEP 1: Read in Camera Data
from modules.cameramanager import CameraDataManager

CDM = CameraDataManager(image_path=image_path,
                        max_images=100,
                        calibration_path=calibration_path)

# Get Camera Data
camera_data = CDM.get_camera_data()

# STEP 2: Estimate Features per Image
from modules.features import FeatureDetectionSIFT
# Feature Module Initialization
feature_detector = FeatureDetectionSIFT(cam_data=camera_data, 
                                        max_keypoints=15000,
                                        contrast_threshold=0.01,
                                        edge_threshold=12)

# Detect Features for all Images
features = feature_detector()

# STEP 3: Match Features Per Image (Pairwise bootstrapping for PnP)
from modules.featurematching import FeatureMatchLightGluePair
# Pairwise Feature Matching Module Initialization
feature_matcher = FeatureMatchLightGluePair(cam_data=camera_data,
                                            detector='sift',
                                            filter_threshold=0.15,
                                            RANSAC_threshold=0.02,
                                            RANSAC_conf=0.995)

# Detect Image Pair Correspondences for Pose Estimation
feature_pairs = feature_matcher(features=features)

# STEP 4: Estimate Camera Poses of Scene with Local Bundle Adjustment Activated
from modules.optimization import BundleAdjustmentOptimizerLocal

# Build Optimizer
optimizer_local = BundleAdjustmentOptimizerLocal(cam_data=camera_data,
                                                 max_num_iterations=60,
                                                 window_size=8,
                                                 robust_loss=True,
                                                 refine_focal_length=False,
                                                 refine_principal_point=False,
                                                 refine_extra_params=False)

from modules.camerapose import CamPoseEstimatorEssentialToPnP
# Camera Pose Module Initialization
pose_estimator = CamPoseEstimatorEssentialToPnP(cam_data=camera_data,
                                                reprojection_error=3.0,
                                                iteration_count=300,
                                                confidence=0.995,
                                                optimizer=optimizer_local)

# From estimated features, estimate the camera poses for all image frames
cam_poses = pose_estimator(feature_pairs=feature_pairs)

# STEP 5: Estimate Feature Tracks (Multi-view tracking)
from modules.featurematching import FeatureMatchLightGlueTracking
# Feature Tracking Module Initialization
feature_tracker = FeatureMatchLightGlueTracking(cam_data=camera_data,
                                                detector='sift',
                                                filter_threshold=0.10,
                                                RANSAC_threshold=0.04,
                                                RANSAC_conf=0.99)

# From estimated features, tracked features across multiple images
tracked_features = feature_tracker(features=features)

# STEP 6: Reconstruct Sparse 3D Scene (Classical multi-view triangulation)
from modules.scenereconstruction import Sparse3DReconstructionMono
# Scene Reconstruction Module Initialization (move view to constructor)
sparse_reconstruction = Sparse3DReconstructionMono(cam_data=camera_data,
                                                   view='multi',
                                                   reproj_error=2.0,
                                                   min_observe=4,
                                                   min_angle=2.0)

# Estimate sparse 3D scene from tracked features and camera poses (no view in call)
sparse_scene = sparse_reconstruction(tracked_features=tracked_features, cam_poses=cam_poses)

# STEP 7: Global Bundle Adjustment for Consistent Final Sparse Model
from modules.optimization import BundleAdjustmentOptimizerGlobal

# Build Optimizer
optimizer_global = BundleAdjustmentOptimizerGlobal(cam_data=camera_data,
                                                   max_num_iterations=200,
                                                   robust_loss=True,
                                                   refine_focal_length=False,
                                                   refine_principal_point=False,
                                                   refine_extra_params=False)

# Run Optimizer
optimal_scene = optimizer_global(sparse_scene)
