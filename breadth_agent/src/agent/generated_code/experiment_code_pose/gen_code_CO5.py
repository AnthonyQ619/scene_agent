
# Construct Modules with Initialized Arguments
image_path = r"C:\Users\Anthony\Documents\Projects\datasets\sfm_dataset\Experiments\co3dv2\orange\374_42196_84367\images"
calibration_path = r"C:\Users\Anthony\Documents\Projects\datasets\sfm_dataset\Experiments\co3dv2\orange\calibration_new_374_42196_84367.npz"

# STEP 1: Read in Camera Data
from modules.cameramanager import CameraDataManager

CDM = CameraDataManager(image_path=image_path,
                        calibration_path=calibration_path,
                        target_resolution=[1024, 1024])

# Get Camera Data
camera_data = CDM.get_camera_data()

# STEP 2: Estimate Features per Image
from modules.features import FeatureDetectionSP
# Feature Module Initialization
feature_detector = FeatureDetectionSP(cam_data=camera_data,
                                      max_keypoints=6000)

# Detect Features for all Images
features = feature_detector()

# STEP 3: Match Features Per Image (Pairwise)
from modules.featurematching import FeatureMatchSuperGluePair
# Pairwise Feature Matching Module Initialization
feature_matcher = FeatureMatchSuperGluePair(cam_data=camera_data,
                                            detector='superpoint',
                                            setting='indoor',
                                            RANSAC_threshold=0.03)

# Detect Image Pair Correspondences for Pose Estimation
feature_pairs = feature_matcher(features=features)

# STEP 4: Estimate Camera Poses of Scene with Feature Pairs with Local Bundle Adjustment Activated
from modules.optimization import BundleAdjustmentOptimizerLocal

# Build Local Optimizer
optimizer_local = BundleAdjustmentOptimizerLocal(cam_data=camera_data,
                                                 refine_focal_length=False,
                                                 refine_principal_point=False,
                                                 refine_extra_params=False,
                                                 max_num_iterations=30,
                                                 use_gpu=True,
                                                 robust_loss=True,
                                                 window_size=12)

from modules.camerapose import CamPoseEstimatorEssentialToPnP
# Camera Pose Module Initialization
pose_estimator = CamPoseEstimatorEssentialToPnP(cam_data=camera_data,
                                                reprojection_error=3.0,
                                                iteration_count=300,
                                                confidence=0.995,
                                                optimizer=optimizer_local)

# From estimated features, estimate the camera poses for all image frames
cam_poses = pose_estimator(feature_pairs=feature_pairs)

# STEP 5: Estimate Feature Tracks (Multi-View)
from modules.featurematching import FeatureMatchLightGlueTracking
# Feature Tracking Module Initialization
feature_tracker = FeatureMatchLightGlueTracking(detector='superpoint',
                                                cam_data=camera_data,
                                                RANSAC_threshold=0.05)

# From estimated features, track features across multiple images
tracked_features = feature_tracker(features=features)

# STEP 6: Reconstruct Lightweight Sparse Scene (Multi-View Triangulation)
from modules.scenereconstruction import Sparse3DReconstructionMono
# Scene Reconstruction Module Initialization
sparse_reconstruction = Sparse3DReconstructionMono(cam_data=camera_data,
                                                   view="multi",
                                                   min_observe=3,
                                                   min_angle=1.5,
                                                   reproj_error=3.0)

# Estimate sparse 3D scene from tracked features and camera poses
sparse_scene = sparse_reconstruction(tracked_features, cam_poses)

# STEP 7: Run Global Bundle Adjustment
from modules.optimization import BundleAdjustmentOptimizerGlobal

# Build Global Optimizer
optimizer_global = BundleAdjustmentOptimizerGlobal(cam_data=camera_data,
                                                   refine_focal_length=False,
                                                   refine_principal_point=False,
                                                   refine_extra_params=False,
                                                   max_num_iterations=150,
                                                   use_gpu=True,
                                                   robust_loss=True)

# Run Global Optimization
optimal_scene = optimizer_global(sparse_scene)

# STEP 8: Output Optimized Camera Extrinsics (Optional Visualization)
optimal_scene.cam_poses
