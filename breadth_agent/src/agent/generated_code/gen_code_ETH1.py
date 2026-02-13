# GOAL RECONSTRUCTION: Sparse Reconstruction

# STEP 1: Read in Camera Data
image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\ETH\\door_dslr_undistorted\\door\\images\\dslr_images_undistorted"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\ETH\\door_dslr_undistorted\\door\\dslr_calibration_undistorted\\calibration_DTU_new.npz"


from modules.cameramanager import CameraDataManager

CDM = CameraDataManager(image_path=image_path,
                        calibration_path=calibration_path)

# Get Camera Data
camera_data = CDM.get_camera_data()

# STEP 2: Detect Features (SIFT)
from modules.features import FeatureDetectionSIFT

feature_detector = FeatureDetectionSIFT(cam_data=camera_data,
                                        max_keypoints=15000,
                                        n_octave_layers=3,
                                        contrast_threshold=0.01,
                                        edge_threshold=10,
                                        sigma=1.6)

features = feature_detector()

# STEP 3: Pairwise Feature Matching (FLANN)
from modules.featurematching import FeatureMatchFlannPair

feature_matcher = FeatureMatchFlannPair(cam_data=camera_data,
                                        detector='sift',
                                        k=2,
                                        RANSAC=True,
                                        RANSAC_threshold=0.2,
                                        RANSAC_conf=0.995,
                                        lowes_thresh=0.70)

feature_pairs = feature_matcher(features=features)

# STEP 4: Camera Pose Estimation (Essential -> PnP) with Local BA
from modules.optimization import BundleAdjustmentOptimizerLocal
from modules.camerapose import CamPoseEstimatorEssentialToPnP

local_optimizer = BundleAdjustmentOptimizerLocal(cam_data=camera_data,
                                                 refine_focal_length=False,
                                                 refine_principal_point=False,
                                                 refine_extra_params=False,
                                                 max_num_iterations=30,
                                                 use_gpu=True,
                                                 gpu_index=0,
                                                 robust_loss=True,
                                                 window_size=8,
                                                 min_track_len=4)

pose_estimator = CamPoseEstimatorEssentialToPnP(cam_data=camera_data,
                                                iteration_count=300,
                                                reprojection_error=2.5,
                                                confidence=0.995,
                                                optimizer=local_optimizer)

cam_poses = pose_estimator(feature_pairs=feature_pairs)

# STEP 5: Multi-View Feature Tracking (FLANN)
from modules.featurematching import FeatureMatchFlannTracking

feature_tracker = FeatureMatchFlannTracking(cam_data=camera_data,
                                            detector='sift',
                                            k=2,
                                            RANSAC_threshold=0.20,
                                            RANSAC_conf=0.995,
                                            lowes_thresh=0.75)

tracked_features = feature_tracker(features=features)

# STEP 6: Sparse 3D Reconstruction (Multi-View)
from modules.scenereconstruction import Sparse3DReconstructionMono

sparse_reconstruction = Sparse3DReconstructionMono(cam_data=camera_data,
                                                   view='multi',
                                                   min_observe=3,
                                                   min_angle=1.0,#2.5,
                                                   reproj_error=2.5)

sparse_scene = sparse_reconstruction(tracked_features, cam_poses)

# STEP 7: Global Bundle Adjustment (Intrinsics Fixed)
from modules.optimization import BundleAdjustmentOptimizerGlobal

global_optimizer = BundleAdjustmentOptimizerGlobal(cam_data=camera_data,
                                                   refine_focal_length=False,
                                                   refine_principal_point=False,
                                                   refine_extra_params=False,
                                                   max_num_iterations=80,
                                                   use_gpu=True,
                                                   gpu_index=0,
                                                   robust_loss=True)

optimal_scene = global_optimizer(sparse_scene)
