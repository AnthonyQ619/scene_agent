# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\scan21_normal_lighting"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\calibration_DTU_new.npz"

# STEP 1: Read in Camera Data
from modules.cameramanager import CameraDataManager

CDM = CameraDataManager(image_path=image_path,
                        calibration_path=calibration_path)

# Get Camera Data
camera_data = CDM.get_camera_data()

# STEP 2: Detect Features (SIFT)
from modules.features import FeatureDetectionSIFT

feature_detector = FeatureDetectionSIFT(cam_data=camera_data,
                                        max_keypoints=18000,
                                        contrast_threshold=0.005,
                                        edge_threshold=12,
                                        sigma=1.6)

# Detect Features for all Images
features = feature_detector()

# STEP 3: Pairwise Feature Matching for Initial Pose (FLANN)
from modules.featurematching import FeatureMatchFlannPair

feature_matcher_pair = FeatureMatchFlannPair(cam_data=camera_data,
                                             detector='sift',
                                             k=2,
                                             lowes_thresh=0.75,
                                             RANSAC=True,
                                             RANSAC_threshold=0.3,
                                             RANSAC_conf=0.99)

# Compute Pairwise Correspondences
feature_pairs = feature_matcher_pair(features=features)

# STEP 4: Camera Pose Estimation (Essential -> PnP)
from modules.camerapose import CamPoseEstimatorEssentialToPnP

pose_estimator = CamPoseEstimatorEssentialToPnP(cam_data=camera_data,
                                                iteration_count=200,
                                                reprojection_error=3.0,
                                                confidence=0.995,
                                                optimizer=None)

# Estimate Camera Poses
cam_poses = pose_estimator(feature_pairs=feature_pairs)

# STEP 5: Multi-view Feature Tracking (FLANN)
from modules.featurematching import FeatureMatchFlannTracking

feature_tracker = FeatureMatchFlannTracking(cam_data=camera_data,
                                            detector='sift',
                                            k=2,
                                            lowes_thresh=0.75,
                                            RANSAC_threshold=0.2,
                                            RANSAC_conf=0.99)

# Track Features Across Images
tracked_features = feature_tracker(features=features)

# STEP 6: Sparse 3D Reconstruction (Mono, Multi-view)
from modules.scenereconstruction import Sparse3DReconstructionMono

sparse_reconstruction = Sparse3DReconstructionMono(cam_data=camera_data,
                                                   view='multi',
                                                   min_observe=3,
                                                   min_angle=2.0,
                                                   reproj_error=3.0)

# Estimate Sparse 3D Scene (use keyword args to avoid ordering issues)
sparse_scene = sparse_reconstruction(cam_poses=cam_poses, tracked_features=tracked_features)

# STEP 7: Global Bundle Adjustment (Keep Intrinsics Fixed)
from modules.optimization import BundleAdjustmentOptimizerGlobal

optimizer_global = BundleAdjustmentOptimizerGlobal(cam_data=camera_data,
                                                   refine_focal_length=False,
                                                   refine_principal_point=False,
                                                   refine_extra_params=False,
                                                   max_num_iterations=80,
                                                   use_gpu=True,
                                                   gpu_index=0,
                                                   robust_loss=True)

# Optimize Sparse Scene
optimized_scene = optimizer_global(sparse_scene)