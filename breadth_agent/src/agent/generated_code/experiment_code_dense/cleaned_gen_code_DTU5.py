
# STEP 1: Read in Camera Data
image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Experiments\\DTU\\scan50\\images"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Experiments\\DTU\\calibration_DTU_new.npz"

from modules.cameramanager import CameraDataManager

CDM = CameraDataManager(image_path=image_path,
                        calibration_path=calibration_path)

camera_data = CDM.get_camera_data()

# STEP 2: Detect Features (SuperPoint)
from modules.features import FeatureDetectionSP

feature_detector = FeatureDetectionSP(cam_data=camera_data,
                                      max_keypoints=6000)

features = feature_detector()

# STEP 3: Pairwise Feature Matching (SuperGlue)
from modules.featurematching import FeatureMatchSuperGluePair

feature_matcher = FeatureMatchSuperGluePair(cam_data=camera_data,
                                            detector='superpoint',
                                            setting='indoor',
                                            RANSAC_threshold=0.03)

feature_pairs = feature_matcher(features=features)

# STEP 4: Camera Pose Estimation with Local BA
from modules.optimization import BundleAdjustmentOptimizerLocal

optimizer_local = BundleAdjustmentOptimizerLocal(cam_data=camera_data,
                                                 refine_focal_length=False,
                                                 refine_principal_point=False,
                                                 refine_extra_params=False,
                                                 max_num_iterations=30,
                                                 use_gpu=True,
                                                 gpu_index=0,
                                                 robust_loss=True,
                                                 window_size=10,
                                                 min_track_len=3)

from modules.camerapose import CamPoseEstimatorEssentialToPnP

pose_estimator = CamPoseEstimatorEssentialToPnP(cam_data=camera_data,
                                                iteration_count=300,
                                                reprojection_error=3.0,
                                                confidence=0.995,
                                                optimizer=optimizer_local)

cam_poses = pose_estimator(feature_pairs=feature_pairs)

# STEP 5: Multi-view Feature Tracking (LightGlue)
from modules.featurematching import FeatureMatchLightGlueTracking

feature_tracker = FeatureMatchLightGlueTracking(cam_data=camera_data,
                                                detector='superpoint',
                                                RANSAC_threshold=0.05)

tracked_features = feature_tracker(features=features)

# STEP 6: Sparse 3D Reconstruction (Multi-view)
from modules.scenereconstruction import Sparse3DReconstructionMono

sparse_reconstruction = Sparse3DReconstructionMono(cam_data=camera_data,
                                                   view="multi",
                                                   min_observe=3,
                                                   min_angle=1.2,
                                                   reproj_error=2.0)

sparse_scene = sparse_reconstruction(tracked_features=tracked_features, cam_poses=cam_poses)

# STEP 7: Global Bundle Adjustment (fixed intrinsics)
from modules.optimization import BundleAdjustmentOptimizerGlobal

optimizer_global = BundleAdjustmentOptimizerGlobal(cam_data=camera_data,
                                                   refine_focal_length=False,
                                                   refine_principal_point=False,
                                                   refine_extra_params=False,
                                                   max_num_iterations=180,
                                                   use_gpu=True,
                                                   gpu_index=0,
                                                   robust_loss=True)

optimal_scene = optimizer_global(sparse_scene)

# STEP 8: Dense Reconstruction
from modules.scenereconstruction import Dense3DReconstructionMono

dense_reconstruction = Dense3DReconstructionMono(cam_data=camera_data)
dense_scene = dense_reconstruction()
