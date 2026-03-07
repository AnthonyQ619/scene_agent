
# Construct Modules with Initialized Arguments
image_path = r"C:\Users\Anthony\Documents\Projects\datasets\sfm_dataset\Experiments\DTU\scan10\images"
calibration_path = r"C:\Users\Anthony\Documents\Projects\datasets\sfm_dataset\Experiments\DTU\calibration_DTU_new.npz"

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
                                        n_octave_layers=3,
                                        contrast_threshold=0.01,
                                        edge_threshold=12,
                                        sigma=1.6)

# Detect Features for all Images
features = feature_detector()

# STEP 3: Pairwise Feature Matching for Pose Bootstrapping
from modules.featurematching import FeatureMatchFlannPair
# Pairwise Feature Matching Module Initialization
feature_matcher = FeatureMatchFlannPair(cam_data=camera_data,
                                        detector="sift",
                                        k=2,
                                        RANSAC=True,
                                        RANSAC_threshold=0.02,
                                        RANSAC_conf=0.995,
                                        lowes_thresh=0.78)

# Detect Image Pair Correspondences for Pose Estimation
feature_pairs = feature_matcher(features=features)

# STEP 4: Estimate Camera Poses with Local Bundle Adjustment Activated
from modules.optimization import BundleAdjustmentOptimizerLocal

# Build Local Optimizer
optimizer_local = BundleAdjustmentOptimizerLocal(cam_data=camera_data,
                                                 refine_focal_length=False,
                                                 refine_principal_point=False,
                                                 refine_extra_params=False,
                                                 max_num_iterations=40,
                                                 robust_loss=True,
                                                 window_size=8,
                                                 min_track_len=3)

from modules.camerapose import CamPoseEstimatorEssentialToPnP
# Camera Pose Module Initialization
pose_estimator = CamPoseEstimatorEssentialToPnP(cam_data=camera_data,
                                                iteration_count=300,
                                                reprojection_error=3.0,
                                                confidence=0.995,
                                                optimizer=optimizer_local)

# From estimated pairwise matches, estimate the camera poses for all image frames
cam_poses = pose_estimator(feature_pairs=feature_pairs)

# STEP 5: Multi-view Feature Tracking
from modules.featurematching import FeatureMatchBFTracking
# Feature Tracking Module Initialization
feature_tracker = FeatureMatchBFTracking(cam_data=camera_data,
                                         detector="sift",
                                         k=2,
                                         cross_check=False,
                                         RANSAC_threshold=0.015,
                                         RANSAC_conf=0.995,
                                         lowes_thresh=0.72)

# From detected features, track features across multiple images
tracked_features = feature_tracker(features=features)

# STEP 6: Sparse 3D Reconstruction (Classical Triangulation)
from modules.scenereconstruction import Sparse3DReconstructionMono
# Scene Reconstruction Module Initialization
sparse_reconstruction = Sparse3DReconstructionMono(cam_data=camera_data,
                                                   view="multi",
                                                   min_observe=4,
                                                   min_angle=2.0,
                                                   reproj_error=2.0)

# Estimate sparse 3D scene from tracked features and camera poses
sparse_scene = sparse_reconstruction(tracked_features, cam_poses)

# STEP 7: Global Bundle Adjustment
from modules.optimization import BundleAdjustmentOptimizerGlobal

# Build Global Optimizer
optimizer_global = BundleAdjustmentOptimizerGlobal(cam_data=camera_data,
                                                   refine_focal_length=False,
                                                   refine_principal_point=False,
                                                   refine_extra_params=False,
                                                   max_num_iterations=200,
                                                   robust_loss=True)

# Run Global Optimization
optimal_scene = optimizer_global(sparse_scene)
