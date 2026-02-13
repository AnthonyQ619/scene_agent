# GOAL RECONSTRUCTION: Sparse Reconstruction

# STEP 1: Read in Camera Data
image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\scan_125_normal_lighting"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\calibration_DTU_new.npz"

from modules.cameramanager import CameraDataManager

CDM = CameraDataManager(image_path=image_path,
                        calibration_path=calibration_path)

# Get Camera Data
cam_data = CDM.get_camera_data()

# STEP 2: Detect Features (SIFT)
from modules.features import FeatureDetectionSIFT

feature_detector = FeatureDetectionSIFT(cam_data=cam_data,
                                        max_keypoints=16000,
                                        n_octave_layers=3,
                                        contrast_threshold=0.01,
                                        edge_threshold=12,
                                        sigma=1.6)

features = feature_detector()

# STEP 3: Pairwise Matching for Pose Initialization (FLANN)
from modules.featurematching import FeatureMatchFlannPair

feature_matcher = FeatureMatchFlannPair(cam_data=cam_data,
                                        detector='sift',
                                        k=2,
                                        RANSAC=True,
                                        RANSAC_threshold=0.3,
                                        RANSAC_conf=0.995,
                                        lowes_thresh=0.75)

feature_pairs = feature_matcher(features=features)

# STEP 4: Camera Pose Estimation (Essential + PnP with Local BA)
from modules.optimization import BundleAdjustmentOptimizerLocal
from modules.camerapose import CamPoseEstimatorEssentialToPnP

local_optimizer = BundleAdjustmentOptimizerLocal(cam_data=cam_data,
                                                 refine_focal_length=False,
                                                 refine_principal_point=False,
                                                 refine_extra_params=False,
                                                 max_num_iterations=20,
                                                 use_gpu=True,
                                                 gpu_index=0,
                                                 robust_loss=True,
                                                 window_size=8,
                                                 min_track_len=3)

pose_estimator = CamPoseEstimatorEssentialToPnP(cam_data=cam_data,
                                                iteration_count=250,
                                                reprojection_error=3.0,
                                                confidence=0.995,
                                                optimizer=local_optimizer)

cam_poses = pose_estimator(feature_pairs=feature_pairs)

# STEP 5: Multi-View Feature Tracking (FLANN)
from modules.featurematching import FeatureMatchFlannTracking

feature_tracker = FeatureMatchFlannTracking(cam_data=cam_data,
                                            detector='sift',
                                            k=2,
                                            RANSAC_threshold=0.2,
                                            RANSAC_conf=0.99,
                                            lowes_thresh=0.75)

tracked_features = feature_tracker(features=features)

# STEP 6: Sparse 3D Reconstruction (Monocular, Multi-View Triangulation)
from modules.scenereconstruction import Sparse3DReconstructionMono

sparse_reconstruction = Sparse3DReconstructionMono(cam_data=cam_data,
                                                   view='multi',
                                                   min_observe=4,
                                                   min_angle=2.0,
                                                   reproj_error=3.0)

sparse_scene = sparse_reconstruction(tracked_features, cam_poses)

# STEP 7: Global Bundle Adjustment (Finalize Sparse Scene)
from modules.optimization import BundleAdjustmentOptimizerGlobal

global_optimizer = BundleAdjustmentOptimizerGlobal(cam_data=cam_data,
                                                   refine_focal_length=False,
                                                   refine_principal_point=False,
                                                   refine_extra_params=False,
                                                   max_num_iterations=60,
                                                   use_gpu=True,
                                                   gpu_index=0,
                                                   robust_loss=True)

optimal_scene = global_optimizer(sparse_scene)
