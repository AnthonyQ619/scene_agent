
# STEP 1: Read in Camera Data
image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Experiments\\DTU\\scan10\\images"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Experiments\\DTU\\calibration_DTU_new.npz"

from modules.cameramanager import CameraDataManager

CDM = CameraDataManager(image_path=image_path,
                        calibration_path=calibration_path)

camera_data = CDM.get_camera_data()

# STEP 2: Detect Features (SIFT)
from modules.features import FeatureDetectionSIFT

feature_detector = FeatureDetectionSIFT(cam_data=camera_data,
                                        max_keypoints=15000,
                                        n_octave_layers=3,
                                        contrast_threshold=0.01,
                                        edge_threshold=12,
                                        sigma=1.6)

features = feature_detector()

# STEP 3: Pairwise Feature Matching for initialization (FLANN)
from modules.featurematching import FeatureMatchFlannPair

feature_matcher = FeatureMatchFlannPair(cam_data=camera_data,
                                        detector='sift',
                                        k=2,
                                        RANSAC=True,
                                        RANSAC_threshold=0.02,
                                        RANSAC_conf=0.995,
                                        lowes_thresh=0.78)

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
                                                 window_size=8,
                                                 min_track_len=3)

from modules.camerapose import CamPoseEstimatorEssentialToPnP

pose_estimator = CamPoseEstimatorEssentialToPnP(cam_data=camera_data,
                                                iteration_count=250,
                                                reprojection_error=2.5,
                                                confidence=0.995,
                                                optimizer=optimizer_local)

cam_poses = pose_estimator(feature_pairs=feature_pairs)

# STEP 5: Multi-view Feature Tracking (BF with cross-check)
from modules.featurematching import FeatureMatchBFTracking

feature_tracker = FeatureMatchBFTracking(cam_data=camera_data,
                                         detector='sift',
                                         k=2,
                                         cross_check=True,
                                         RANSAC_threshold=0.015,
                                         RANSAC_conf=0.995,
                                         lowes_thresh=0.70)

tracked_features = feature_tracker(features=features)

# STEP 6: Sparse 3D Reconstruction (multi-view triangulation)
from modules.scenereconstruction import Sparse3DReconstructionMono

sparse_reconstruction = Sparse3DReconstructionMono(cam_data=camera_data,
                                                   view="multi",
                                                   min_observe=4,
                                                   min_angle=2.0,
                                                   reproj_error=2.0)

sparse_scene = sparse_reconstruction(tracked_features, cam_poses)

# STEP 7: Global Bundle Adjustment (final sparse optimization)
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

# STEP 8: Dense Reconstruction (MVS and fusion)
from modules.scenereconstruction import Dense3DReconstructionMono

# Note: Dense3DReconstructionMono uses the COLMAP workspace created during sparse reconstruction.
dense_reconstruction = Dense3DReconstructionMono(cam_data=camera_data)

# Do not pass the scene; the module uses the workspace directly.
dense_scene = dense_reconstruction()
