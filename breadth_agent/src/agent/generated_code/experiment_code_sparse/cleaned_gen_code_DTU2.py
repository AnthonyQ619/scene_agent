
# -*- coding: utf-8 -*-

# STEP 1: Read in Camera Data
image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Experiments\\DTU\\scan20\\images"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Experiments\\DTU\\calibration_DTU_new.npz"

from modules.cameramanager import CameraDataManager

CDM = CameraDataManager(image_path=image_path,
                        calibration_path=calibration_path)

camera_data = CDM.get_camera_data()

# STEP 2: Detect Features with SIFT
from modules.features import FeatureDetectionSIFT

feature_detector = FeatureDetectionSIFT(cam_data=camera_data,
                                        max_keypoints=12000,
                                        contrast_threshold=0.01,
                                        edge_threshold=10,
                                        sigma=1.6)

features = feature_detector()

# STEP 3: Pairwise Feature Matching with FLANN (SIFT)
from modules.featurematching import FeatureMatchFlannPair

feature_matcher = FeatureMatchFlannPair(cam_data=camera_data,
                                        detector="sift",
                                        k=2,
                                        lowes_thresh=0.78,
                                        RANSAC=True,
                                        RANSAC_threshold=0.02,
                                        RANSAC_conf=0.999)

feature_pairs = feature_matcher(features=features)

# STEP 4: Pose Estimation with Essential->PnP + Local BA
from modules.optimization import BundleAdjustmentOptimizerLocal
from modules.camerapose import CamPoseEstimatorEssentialToPnP

optimizer_local = BundleAdjustmentOptimizerLocal(cam_data=camera_data,
                                                 refine_focal_length=False,
                                                 refine_principal_point=False,
                                                 refine_extra_params=False,
                                                 max_num_iterations=30,
                                                 use_gpu=True,
                                                 robust_loss=True,
                                                 window_size=8,
                                                 min_track_len=3)

pose_estimator = CamPoseEstimatorEssentialToPnP(cam_data=camera_data,
                                                reprojection_error=3.0,
                                                iteration_count=300,
                                                confidence=0.995,
                                                optimizer=optimizer_local)

cam_poses = pose_estimator(feature_pairs=feature_pairs)

# STEP 5: Multi-View Feature Tracking with BF (SIFT)
from modules.featurematching import FeatureMatchBFTracking

feature_tracker = FeatureMatchBFTracking(cam_data=camera_data,
                                         detector="sift",
                                         k=2,
                                         cross_check=False,
                                         lowes_thresh=0.70,
                                         RANSAC_threshold=0.015,
                                         RANSAC_conf=0.999)

tracked_features = feature_tracker(features=features)

# STEP 6: Sparse 3D Reconstruction (Mono, Multi-View)
from modules.scenereconstruction import Sparse3DReconstructionMono

sparse_reconstruction = Sparse3DReconstructionMono(cam_data=camera_data,
                                                   view="multi",
                                                   min_observe=4,
                                                   min_angle=2.0,
                                                   reproj_error=3.0)

# Correct call order: (tracked_features, cam_poses) - use keywords to avoid ambiguity
sparse_scene = sparse_reconstruction(tracked_features=tracked_features,
                                     cam_poses=cam_poses)

# STEP 7: Global Bundle Adjustment (fixed intrinsics)
from modules.optimization import BundleAdjustmentOptimizerGlobal

optimizer_global = BundleAdjustmentOptimizerGlobal(cam_data=camera_data,
                                                   refine_focal_length=False,
                                                   refine_principal_point=False,
                                                   refine_extra_params=False,
                                                   max_num_iterations=200,
                                                   use_gpu=True,
                                                   robust_loss=True)

optimal_scene = optimizer_global(sparse_scene)
