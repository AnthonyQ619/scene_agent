
# STEP 1: Read in Camera Data
from modules.cameramanager import CameraDataManager

image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Experiments\\DTU\\scan30\\images"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Experiments\\DTU\\calibration_DTU_new.npz"

CDM = CameraDataManager(image_path=image_path,
                        calibration_path=calibration_path)

camera_data = CDM.get_camera_data()

# STEP 2: Detect Features (SIFT)
from modules.features import FeatureDetectionSIFT

feature_detector = FeatureDetectionSIFT(cam_data=camera_data,
                                        max_keypoints=12000,
                                        contrast_threshold=0.01,
                                        edge_threshold=12,
                                        sigma=1.6)

features = feature_detector()

# STEP 3: Pairwise Feature Matching (FLANN)
from modules.featurematching import FeatureMatchFlannPair

feature_matcher = FeatureMatchFlannPair(cam_data=camera_data,
                                        detector='sift',
                                        k=2,
                                        RANSAC=True,
                                        RANSAC_threshold=0.02,
                                        RANSAC_conf=0.995,
                                        lowes_thresh=0.75)

feature_pairs = feature_matcher(features=features)

# STEP 4: Camera Pose Estimation with Local BA
from modules.optimization import BundleAdjustmentOptimizerLocal
from modules.camerapose import CamPoseEstimatorEssentialToPnP

optimizer_local = BundleAdjustmentOptimizerLocal(cam_data=camera_data,
                                                 max_num_iterations=30,
                                                 window_size=8,
                                                 robust_loss=True,
                                                 refine_focal_length=False,
                                                 refine_principal_point=False)

pose_estimator = CamPoseEstimatorEssentialToPnP(cam_data=camera_data,
                                                iteration_count=300,
                                                reprojection_error=3.0,
                                                confidence=0.995,
                                                optimizer=optimizer_local)

cam_poses = pose_estimator(feature_pairs=feature_pairs)

# STEP 5: Multi-view Feature Tracking (FLANN)
from modules.featurematching import FeatureMatchFlannTracking

feature_tracker = FeatureMatchFlannTracking(cam_data=camera_data,
                                            detector='sift',
                                            k=2,
                                            RANSAC_threshold=0.015,
                                            RANSAC_conf=0.995,
                                            lowes_thresh=0.70)

tracked_features = feature_tracker(features=features)

# STEP 6: Sparse 3D Reconstruction (Multi-view Triangulation)
from modules.scenereconstruction import Sparse3DReconstructionMono

sparse_reconstruction = Sparse3DReconstructionMono(cam_data=camera_data,
                                                   view="multi",
                                                   reproj_error=2.0,
                                                   min_observe=4,
                                                   min_angle=2.0)

sparse_scene = sparse_reconstruction(tracked_features, cam_poses)

# STEP 7: Global Bundle Adjustment (Final Refinement)
from modules.optimization import BundleAdjustmentOptimizerGlobal

optimizer_global = BundleAdjustmentOptimizerGlobal(cam_data=camera_data,
                                                   max_num_iterations=200,
                                                   robust_loss=True,
                                                   refine_focal_length=True,
                                                   refine_principal_point=False,
                                                   refine_extra_params=False)

optimal_scene = optimizer_global(sparse_scene)
