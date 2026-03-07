
# STEP 1: Read in Camera Data
from modules.cameramanager import CameraDataManager

image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Experiments\\co3dv2\\bench\\415_57112_110099\\images"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Experiments\\co3dv2\\bench\\calibration_new_415_57112_110099.npz"

CDM = CameraDataManager(image_path=image_path,
                        calibration_path=calibration_path)

camera_data = CDM.get_camera_data()

# STEP 2: Detect Features with SIFT
from modules.features import FeatureDetectionSIFT

feature_detector = FeatureDetectionSIFT(cam_data=camera_data,
                                        max_keypoints=10000,
                                        contrast_threshold=0.01,
                                        edge_threshold=12)

features = feature_detector()

# STEP 3: Pairwise Feature Matching (BF + RANSAC)
from modules.featurematching import FeatureMatchBFPair

feature_matcher = FeatureMatchBFPair(detector="sift",
                                     cam_data=camera_data,
                                     cross_check=False,
                                     RANSAC_threshold=0.02,
                                     RANSAC_conf=0.995,
                                     lowes_thresh=0.70)

feature_pairs = feature_matcher(features=features)

# STEP 4: Pose Estimation with Local BA
from modules.optimization import BundleAdjustmentOptimizerLocal

optimizer_local = BundleAdjustmentOptimizerLocal(cam_data=camera_data,
                                                 refine_focal_length=False,
                                                 refine_principal_point=False,
                                                 refine_extra_params=False,
                                                 max_num_iterations=40,
                                                 use_gpu=True,
                                                 gpu_index=0,
                                                 robust_loss=True,
                                                 window_size=8,
                                                 min_track_len=3)

from modules.camerapose import CamPoseEstimatorEssentialToPnP

pose_estimator = CamPoseEstimatorEssentialToPnP(cam_data=camera_data,
                                                reprojection_error=3.0,
                                                iteration_count=300,
                                                confidence=0.995,
                                                optimizer=optimizer_local)

cam_poses = pose_estimator(feature_pairs=feature_pairs)

# STEP 5: Multi-View Feature Tracking (BF)
from modules.featurematching import FeatureMatchBFTracking

feature_tracker = FeatureMatchBFTracking(cam_data=camera_data,
                                         detector="sift",
                                         cross_check=False,
                                         RANSAC_threshold=0.02,
                                         RANSAC_conf=0.99,
                                         lowes_thresh=0.75)

tracked_features = feature_tracker(features=features)

# STEP 6: Sparse 3D Reconstruction (Multi-view triangulation)
from modules.scenereconstruction import Sparse3DReconstructionMono

sparse_reconstruction = Sparse3DReconstructionMono(cam_data=camera_data,
                                                   view="multi",
                                                   min_observe=3,
                                                   min_angle=2.0,
                                                   reproj_error=3.0)

sparse_scene = sparse_reconstruction(tracked_features, cam_poses)

# STEP 7: Global Bundle Adjustment (Extrinsics + 3D points, fixed intrinsics)
from modules.optimization import BundleAdjustmentOptimizerGlobal

optimizer = BundleAdjustmentOptimizerGlobal(cam_data=camera_data,
                                            refine_focal_length=False,
                                            refine_principal_point=False,
                                            refine_extra_params=False,
                                            max_num_iterations=200,
                                            use_gpu=True,
                                            gpu_index=0,
                                            robust_loss=True)

optimal_scene = optimizer(sparse_scene)

# STEP 8: Deliver final camera poses (pose-only result)
final_cam_poses = optimal_scene.cam_poses
