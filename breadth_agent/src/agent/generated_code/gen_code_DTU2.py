# GOAL RECONSTRUCTION: Sparse Reconstruction

# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\scan93_normal_lighting"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\calibration_DTU_new.npz"

# STEP 1: Read in Camera Data
from modules.cameramanager import CameraDataManager

CDM = CameraDataManager(image_path=image_path,
                        calibration_path=calibration_path)

# Get Camera Data
camera_data = CDM.get_camera_data()

# STEP 2: Detect robust features (SIFT)
from modules.features import FeatureDetectionSIFT

feature_detector = FeatureDetectionSIFT(cam_data=camera_data,
                                        max_keypoints=15000,
                                        n_octave_layers=3,
                                        contrast_threshold=0.01,
                                        edge_threshold=12,
                                        sigma=1.6)

features = feature_detector()

# STEP 3: Pairwise matching for incremental pose initialization
from modules.featurematching import FeatureMatchFlannPair

feature_matcher = FeatureMatchFlannPair(cam_data=camera_data,
                                        detector='sift',
                                        k=2,
                                        RANSAC=True,
                                        RANSAC_threshold=0.20,
                                        RANSAC_conf=0.99,
                                        lowes_thresh=0.70)

feature_pairs = feature_matcher(features=features)

# STEP 4: Calibrated incremental pose estimation with local BA
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
                                                 min_track_len=3)

pose_estimator = CamPoseEstimatorEssentialToPnP(cam_data=camera_data,
                                                iteration_count=300,
                                                reprojection_error=3.0,
                                                confidence=0.995,
                                                optimizer=local_optimizer)

cam_poses = pose_estimator(feature_pairs=feature_pairs)

# STEP 5: Multi-view feature tracking
from modules.featurematching import FeatureMatchFlannTracking

feature_tracker = FeatureMatchFlannTracking(cam_data=camera_data,
                                            detector='sift',
                                            k=2,
                                            RANSAC_threshold=0.15,
                                            RANSAC_conf=0.99,
                                            lowes_thresh=0.70)

tracked_features = feature_tracker(features=features)

# STEP 6: Sparse 3D reconstruction (multi-view)
from modules.scenereconstruction import Sparse3DReconstructionMono

sparse_reconstruction = Sparse3DReconstructionMono(cam_data=camera_data,
                                                   view='multi',
                                                   min_observe=4,
                                                   min_angle=3.0,
                                                   reproj_error=2.5)

sparse_scene = sparse_reconstruction(tracked_features, cam_poses)

# STEP 7: Global bundle adjustment for consistency
from modules.optimization import BundleAdjustmentOptimizerGlobal

global_optimizer = BundleAdjustmentOptimizerGlobal(cam_data=camera_data,
                                                   refine_focal_length=False,
                                                   refine_principal_point=False,
                                                   refine_extra_params=False,
                                                   max_num_iterations=80,
                                                   use_gpu=True,
                                                   gpu_index=0,
                                                   robust_loss=True)

optimized_sparse_scene = global_optimizer(sparse_scene)
