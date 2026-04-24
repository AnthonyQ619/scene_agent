
# STEP 1: Read in Camera Data
from modules.cameramanager import CameraDataManager

image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Experiments\\co3dv2\\donut\\391_47032_93657\\images"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Experiments\\co3dv2\\donut\\calibration_new_391_47032_93657.npz"

CDM = CameraDataManager(image_path=image_path,
                        calibration_path=calibration_path)

camera_data = CDM.get_camera_data()

# STEP 2: Detect Features (SuperPoint)
from modules.features import FeatureDetectionSP

feature_detector = FeatureDetectionSP(cam_data=camera_data,
                                      max_keypoints=6000)

features = feature_detector()

# STEP 3: Pairwise Feature Matching (LightGlue Pair)
from modules.featurematching import FeatureMatchLightGluePair

feature_matcher = FeatureMatchLightGluePair(cam_data=camera_data,
                                            detector="SuperPoint",
                                            n_layers=7,
                                            flash=True,
                                            mp=True,
                                            depth_confidence=0.90,
                                            width_confidence=0.95,
                                            filter_threshold=0.20,
                                            RANSAC=True,
                                            RANSAC_threshold=0.03,
                                            RANSAC_conf=0.999)

feature_pairs = feature_matcher(features=features)

# STEP 4: Pose Estimation with Essential + PnP and Local BA
from modules.optimization import BundleAdjustmentOptimizerLocal

optimizer_local = BundleAdjustmentOptimizerLocal(cam_data=camera_data,
                                                 refine_focal_length=False,
                                                 refine_principal_point=False,
                                                 refine_extra_params=False,
                                                 max_num_iterations=40,
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

# STEP 5: Multi-View Feature Tracking (LightGlue Tracking)
from modules.featurematching import FeatureMatchLightGlueTracking

feature_tracker = FeatureMatchLightGlueTracking(cam_data=camera_data,
                                                detector="SuperPoint",
                                                n_layers=7,
                                                flash=True,
                                                mp=True,
                                                depth_confidence=0.90,
                                                width_confidence=0.95,
                                                filter_threshold=0.10,
                                                RANSAC_threshold=0.05,
                                                RANSAC_conf=0.995)

tracked_features = feature_tracker(features=features)

# STEP 6: Minimal Sparse Reconstruction for Consistency
from modules.scenereconstruction import Sparse3DReconstructionMono

sparse_reconstruction = Sparse3DReconstructionMono(cam_data=camera_data,
                                                   view="multi",
                                                   min_observe=4,
                                                   min_angle=2.0,
                                                   reproj_error=3.0)

sparse_scene = sparse_reconstruction(tracked_features, cam_poses)

# STEP 7: Global Bundle Adjustment (Extrinsics Only)
from modules.optimization import BundleAdjustmentOptimizerGlobal

optimizer_global = BundleAdjustmentOptimizerGlobal(cam_data=camera_data,
                                                   refine_focal_length=False,
                                                   refine_principal_point=False,
                                                   refine_extra_params=False,
                                                   max_num_iterations=150,
                                                   use_gpu=True,
                                                   gpu_index=0,
                                                   robust_loss=True)

optimal_scene = optimizer_global(sparse_scene)

# Final globally consistent camera extrinsics (3x4 [R|T] per frame)
final_cam_poses = optimal_scene.cam_poses
