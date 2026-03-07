
# Construct Modules with Initialized Arguments
image_path = r"C:\Users\Anthony\Documents\Projects\datasets\sfm_dataset\Experiments\co3dv2\mouse\107_12753_23606\images"
calibration_path = r"C:\Users\Anthony\Documents\Projects\datasets\sfm_dataset\Experiments\co3dv2\mouse\calibration_new_107_12753_23606.npz"

# STEP 1: Read in Camera Data
from modules.cameramanager import CameraDataManager

CDM = CameraDataManager(image_path=image_path,
                        max_images=20,
                        calibration_path=calibration_path)

# Get Camera Data
camera_data = CDM.get_camera_data()

# STEP 2: Estimate Features per Image
from modules.features import FeatureDetectionSP
# Feature Module Initialization
feature_detector = FeatureDetectionSP(cam_data=camera_data,
                                      max_keypoints=5000)

# Detect Features for all Images
features = feature_detector()

# STEP 3: Match Features Per Image (Pairwise for E-matrix init)
from modules.featurematching import FeatureMatchSuperGluePair
# Pairwise Feature Matching Module Initialization
feature_matcher = FeatureMatchSuperGluePair(cam_data=camera_data,
                                            detector='superpoint',
                                            setting='indoor',
                                            RANSAC_threshold=0.03,
                                            RANSAC_conf=0.99)

# Detect Image Pair Correspondences for Pose Estimation
feature_pairs = feature_matcher(features=features)

# STEP 4: Estimate Camera Poses with Local BA Drift Control
from modules.optimization import BundleAdjustmentOptimizerLocal

# Build Local Optimizer
optimizer_local = BundleAdjustmentOptimizerLocal(cam_data=camera_data,
                                                 max_num_iterations=30,
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

# From estimated pairwise matches, estimate the camera poses for all image frames
cam_poses = pose_estimator(feature_pairs=feature_pairs)

# STEP 5: Track Features across Multiple Images (Lightweight multi-view tracks)
from modules.featurematching import FeatureMatchLightGlueTracking
# Feature Tracking Module Initialization
feature_tracker = FeatureMatchLightGlueTracking(cam_data=camera_data,
                                                detector='SuperPoint',
                                                n_layers=7,
                                                flash=True,
                                                mp=True,
                                                RANSAC_threshold=0.05,
                                                RANSAC_conf=0.99)

# From estimated features, track features across multiple images
tracked_features = feature_tracker(features=features)

# STEP 6: Build Minimal Sparse Structure for Pose Stabilization
from modules.scenereconstruction import Sparse3DReconstructionMono
# Scene Reconstruction Module Initialization
sparse_reconstruction = Sparse3DReconstructionMono(cam_data=camera_data,
                                                   view="multi",
                                                   min_observe=3,
                                                   min_angle=1.5,
                                                   reproj_error=2.5)

# Estimate sparse 3D scene from tracked features and camera poses
sparse_scene = sparse_reconstruction(tracked_features, cam_poses)

# STEP 7: Global Bundle Adjustment for Consistent Trajectory
from modules.optimization import BundleAdjustmentOptimizerGlobal

# Build Global Optimizer
optimizer_global = BundleAdjustmentOptimizerGlobal(cam_data=camera_data,
                                                   max_num_iterations=150,
                                                   robust_loss=True,
                                                   refine_focal_length=False,
                                                   refine_principal_point=False,
                                                   refine_extra_params=False)

# Run Global Optimization
optimal_scene = optimizer_global(sparse_scene)

# STEP 8: Export Camera Trajectory (list of 3x4 [R|T] per frame)
camera_trajectory = optimal_scene.cam_poses
