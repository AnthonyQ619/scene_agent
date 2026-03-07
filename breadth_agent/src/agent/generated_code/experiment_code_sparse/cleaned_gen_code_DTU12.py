
# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Experiments\\DTU\\scan120\\images"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Experiments\\DTU\\calibration_DTU_new.npz"

# STEP 1: Read in Camera Data
from modules.cameramanager import CameraDataManager

CDM = CameraDataManager(image_path=image_path,
                        calibration_path=calibration_path)

# Get Camera Data
camera_data = CDM.get_camera_data()

# STEP 2: Detect Features (SuperPoint)
from modules.features import FeatureDetectionSP

feature_detector = FeatureDetectionSP(cam_data=camera_data,
                                      max_keypoints=6000)

# Detect Features for all Images
features = feature_detector()

# STEP 3: Pairwise Feature Matching (SuperGlue Pair)
from modules.featurematching import FeatureMatchSuperGluePair

feature_matcher = FeatureMatchSuperGluePair(cam_data=camera_data,
                                            detector='superpoint',
                                            setting='indoor',
                                            sinkhorn_iterations=20,
                                            match_threshold=0.2,
                                            RANSAC=True,
                                            RANSAC_threshold=0.03,
                                            RANSAC_conf=0.99)

# Detect Image Pair Correspondences for Pose Estimation
feature_pairs = feature_matcher(features=features)

# STEP 4: Camera Pose Estimation with Local BA (Essential -> PnP)
from modules.optimization import BundleAdjustmentOptimizerLocal

optimizer_local = BundleAdjustmentOptimizerLocal(cam_data=camera_data,
                                                 max_num_iterations=30,
                                                 window_size=10,
                                                 robust_loss=True,
                                                 use_gpu=True,
                                                 gpu_index=0,
                                                 refine_focal_length=False,
                                                 refine_principal_point=False,
                                                 refine_extra_params=False)

from modules.camerapose import CamPoseEstimatorEssentialToPnP

pose_estimator = CamPoseEstimatorEssentialToPnP(cam_data=camera_data,
                                                iteration_count=300,
                                                reprojection_error=3.0,
                                                confidence=0.995,
                                                optimizer=optimizer_local)

# Estimate camera poses for all frames
cam_poses = pose_estimator(feature_pairs=feature_pairs)

# STEP 5: Multi-View Feature Tracking (SuperGlue Tracking)
from modules.featurematching import FeatureMatchSuperGlueTracking

feature_tracker = FeatureMatchSuperGlueTracking(cam_data=camera_data,
                                                detector='superpoint',
                                                setting='indoor',
                                                sinkhorn_iterations=20,
                                                match_threshold=0.2,
                                                RANSAC_threshold=0.05,
                                                RANSAC_conf=0.99)

# Track features across multiple images
tracked_features = feature_tracker(features=features)

# STEP 6: Sparse 3D Reconstruction (Classical Triangulation, Mono, Multi-View)
from modules.scenereconstruction import Sparse3DReconstructionMono

sparse_reconstruction = Sparse3DReconstructionMono(cam_data=camera_data,
                                                   view="multi",
                                                   min_observe=4,
                                                   min_angle=2.0,
                                                   reproj_error=3.0)

# Use keyword arguments to match documented parameter names/order
sparse_scene = sparse_reconstruction(cam_poses=cam_poses, tracked_features=tracked_features)

# STEP 7: Global Bundle Adjustment
from modules.optimization import BundleAdjustmentOptimizerGlobal

optimizer_global = BundleAdjustmentOptimizerGlobal(cam_data=camera_data,
                                                   max_num_iterations=220,
                                                   robust_loss=True,
                                                   use_gpu=True,
                                                   gpu_index=0,
                                                   refine_focal_length=False,
                                                   refine_principal_point=False,
                                                   refine_extra_params=False)

optimal_scene = optimizer_global(sparse_scene)
