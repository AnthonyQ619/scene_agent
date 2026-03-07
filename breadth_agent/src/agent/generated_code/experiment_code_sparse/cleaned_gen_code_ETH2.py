
# STEP 1: Read in Camera Data
image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Experiments\\ETH\\bridge\\images\\dslr_images_undistorted"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Experiments\\ETH\\bridge\\dslr_calibration_undistorted\\calibration_ETH_new.npz"

from modules.cameramanager import CameraDataManager

CDM = CameraDataManager(image_path=image_path,
                        calibration_path=calibration_path)

camera_data = CDM.get_camera_data()

# STEP 2: Detect Features (SIFT)
from modules.features import FeatureDetectionSIFT

feature_detector = FeatureDetectionSIFT(cam_data=camera_data,
                                        max_keypoints=12000,
                                        n_octave_layers=3,
                                        contrast_threshold=0.01,
                                        edge_threshold=12,
                                        sigma=1.6)

features = feature_detector()

# STEP 3: Pairwise Feature Matching (LightGlue)
from modules.featurematching import FeatureMatchLightGluePair

feature_matcher = FeatureMatchLightGluePair(cam_data=camera_data,
                                            detector='sift',
                                            n_layers=7,
                                            flash=True,
                                            mp=True,
                                            depth_confidence=0.95,
                                            width_confidence=0.99,
                                            filter_threshold=0.15,
                                            RANSAC=True,
                                            RANSAC_threshold=0.02,
                                            RANSAC_conf=0.99)

feature_pairs = feature_matcher(features=features)

# STEP 4: Camera Pose Estimation with Local BA
from modules.optimization import BundleAdjustmentOptimizerLocal
from modules.camerapose import CamPoseEstimatorEssentialToPnP

optimizer_local = BundleAdjustmentOptimizerLocal(cam_data=camera_data,
                                                 refine_focal_length=False,
                                                 refine_principal_point=False,
                                                 refine_extra_params=False,
                                                 max_num_iterations=60,
                                                 use_gpu=True,
                                                 gpu_index=0,
                                                 robust_loss=True,
                                                 window_size=8,
                                                 min_track_len=3)

pose_estimator = CamPoseEstimatorEssentialToPnP(cam_data=camera_data,
                                                iteration_count=300,
                                                reprojection_error=3.0,
                                                confidence=0.995,
                                                optimizer=optimizer_local)

cam_poses = pose_estimator(feature_pairs=feature_pairs)

# STEP 5: Multi-view Feature Tracking (LightGlue)
from modules.featurematching import FeatureMatchLightGlueTracking

feature_tracker = FeatureMatchLightGlueTracking(cam_data=camera_data,
                                                detector='sift',
                                                n_layers=7,
                                                flash=True,
                                                mp=True,
                                                depth_confidence=0.95,
                                                width_confidence=0.99,
                                                filter_threshold=0.10,
                                                RANSAC_threshold=0.03,
                                                RANSAC_conf=0.99)

tracked_features = feature_tracker(features=features)

# STEP 6: Sparse 3D Reconstruction (Multi-view) - Loosen thresholds
from modules.scenereconstruction import Sparse3DReconstructionMono

sparse_reconstruction = Sparse3DReconstructionMono(cam_data=camera_data,
                                                   view="multi",
                                                   min_observe=3,
                                                   min_angle=1.0,
                                                   reproj_error=3.0)

sparse_scene = sparse_reconstruction(tracked_features, cam_poses)

# STEP 7: Global Bundle Adjustment
from modules.optimization import BundleAdjustmentOptimizerGlobal

optimizer_global = BundleAdjustmentOptimizerGlobal(cam_data=camera_data,
                                                   refine_focal_length=False,
                                                   refine_principal_point=False,
                                                   refine_extra_params=False,
                                                   max_num_iterations=220,
                                                   use_gpu=True,
                                                   gpu_index=0,
                                                   robust_loss=True)

optimal_scene = optimizer_global(sparse_scene)
