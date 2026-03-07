
# STEP 1: Read and prepare camera data
image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Experiments\\tanks_and_temples\\Panther"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Experiments\\tanks_and_temples\\calibration_new_2048.npz"

from modules.cameramanager import CameraDataManager

CDM = CameraDataManager(image_path=image_path,
                        calibration_path=calibration_path)

camera_data = CDM.get_camera_data()

# STEP 2: Detect features (SIFT)
from modules.features import FeatureDetectionSIFT

feature_detector = FeatureDetectionSIFT(cam_data=camera_data,
                                        max_keypoints=15000,
                                        n_octave_layers=3,
                                        contrast_threshold=0.01,
                                        edge_threshold=12,
                                        sigma=1.6)

features = feature_detector()

# STEP 3: Pairwise feature matching for pose initialization (LightGlue + SIFT)
from modules.featurematching import FeatureMatchLightGluePair

feature_matcher = FeatureMatchLightGluePair(cam_data=camera_data,
                                            detector='sift',
                                            n_layers=9,
                                            flash=True,
                                            mp=True,
                                            depth_confidence=0.95,
                                            width_confidence=0.99,
                                            filter_threshold=0.2,
                                            RANSAC=True,
                                            RANSAC_threshold=0.02,
                                            RANSAC_conf=0.995)

feature_pairs = feature_matcher(features=features)

# STEP 4: Estimate all camera poses with local BA correction
from modules.optimization import BundleAdjustmentOptimizerLocal

optimizer_local = BundleAdjustmentOptimizerLocal(cam_data=camera_data,
                                                 refine_focal_length=False,
                                                 refine_principal_point=False,
                                                 refine_extra_params=False,
                                                 max_num_iterations=60,
                                                 use_gpu=True,
                                                 gpu_index=0,
                                                 robust_loss=True,
                                                 window_size=12,
                                                 min_track_len=3)

from modules.camerapose import CamPoseEstimatorEssentialToPnP

pose_estimator = CamPoseEstimatorEssentialToPnP(cam_data=camera_data,
                                                iteration_count=350,
                                                reprojection_error=3.0,
                                                confidence=0.995,
                                                optimizer=optimizer_local)

cam_poses = pose_estimator(feature_pairs=feature_pairs)

# STEP 5: Build multi-view tracks for robust triangulation (LightGlue tracking + SIFT)
from modules.featurematching import FeatureMatchLightGlueTracking

feature_tracker = FeatureMatchLightGlueTracking(cam_data=camera_data,
                                                detector='sift',
                                                n_layers=9,
                                                flash=True,
                                                mp=True,
                                                depth_confidence=0.95,
                                                width_confidence=0.99,
                                                filter_threshold=0.1,
                                                RANSAC_threshold=0.03,
                                                RANSAC_conf=0.995)

tracked_features = feature_tracker(features=features)

# STEP 6: Sparse 3D reconstruction via multi-view triangulation
from modules.scenereconstruction import Sparse3DReconstructionMono

sparse_reconstruction = Sparse3DReconstructionMono(cam_data=camera_data,
                                                   view="multi",
                                                   min_observe=4,
                                                   min_angle=2.5,
                                                   reproj_error=2.0)

sparse_scene = sparse_reconstruction(tracked_features, cam_poses)

# STEP 7: Global bundle adjustment for final consistency (fixed intrinsics)
from modules.optimization import BundleAdjustmentOptimizerGlobal

optimizer = BundleAdjustmentOptimizerGlobal(cam_data=camera_data,
                                            refine_focal_length=False,
                                            refine_principal_point=False,
                                            refine_extra_params=False,
                                            max_num_iterations=250,
                                            use_gpu=True,
                                            gpu_index=0,
                                            robust_loss=True)

optimal_scene = optimizer(sparse_scene)
