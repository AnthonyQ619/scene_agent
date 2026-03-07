
# STEP 1: Read in Camera Data
from modules.cameramanager import CameraDataManager

image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Experiments\\tanks_and_temples\\Francis"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Experiments\\tanks_and_temples\\calibration_new_1920.npz"

CDM = CameraDataManager(image_path=image_path,
                        calibration_path=calibration_path)

# Get Camera Data
camera_data = CDM.get_camera_data()

# STEP 2: Detect Features (SIFT)
from modules.features import FeatureDetectionSIFT

feature_detector = FeatureDetectionSIFT(cam_data=camera_data,
                                        max_keypoints=15000,
                                        contrast_threshold=0.010,
                                        edge_threshold=10,
                                        sigma=1.6)

features = feature_detector()

# STEP 3: Detect Feature Matches (Pairwise, BF)
from modules.featurematching import FeatureMatchBFPair

feature_matcher = FeatureMatchBFPair(cam_data=camera_data,
                                     detector="sift",
                                     k=2,
                                     cross_check=False,
                                     RANSAC=True,
                                     RANSAC_threshold=0.02,
                                     RANSAC_conf=0.995,
                                     lowes_thresh=0.70)

feature_pairs = feature_matcher(features=features)

# STEP 4: Estimate Camera Poses with Local BA (Essential -> PnP)
from modules.optimization import BundleAdjustmentOptimizerLocal

optimizer_local = BundleAdjustmentOptimizerLocal(cam_data=camera_data,
                                                 max_num_iterations=60,
                                                 window_size=8,
                                                 robust_loss=True,
                                                 refine_focal_length=False,
                                                 refine_principal_point=False,
                                                 refine_extra_params=False)

from modules.camerapose import CamPoseEstimatorEssentialToPnP

pose_estimator = CamPoseEstimatorEssentialToPnP(cam_data=camera_data,
                                                iteration_count=300,
                                                reprojection_error=3.0,
                                                confidence=0.995,
                                                optimizer=optimizer_local)

cam_poses = pose_estimator(feature_pairs=feature_pairs)

# STEP 5: Track Features across Multiple Images (BF Tracking)
from modules.featurematching import FeatureMatchBFTracking

feature_tracker = FeatureMatchBFTracking(cam_data=camera_data,
                                         detector="sift",
                                         k=2,
                                         cross_check=False,
                                         RANSAC_threshold=0.02,
                                         RANSAC_conf=0.995,
                                         lowes_thresh=0.75)

tracked_features = feature_tracker(features=features)

# STEP 6: Sparse 3D Reconstruction (Mono, Multi-View)
from modules.scenereconstruction import Sparse3DReconstructionMono

sparse_reconstruction = Sparse3DReconstructionMono(cam_data=camera_data,
                                                   view="multi",
                                                   min_observe=4,
                                                   min_angle=2.0,
                                                   reproj_error=2.0)

sparse_scene = sparse_reconstruction(tracked_features, cam_poses)

# STEP 7: Global Bundle Adjustment
from modules.optimization import BundleAdjustmentOptimizerGlobal

optimizer_global = BundleAdjustmentOptimizerGlobal(cam_data=camera_data,
                                                   max_num_iterations=200,
                                                   robust_loss=True,
                                                   refine_focal_length=False,
                                                   refine_principal_point=False,
                                                   refine_extra_params=False)

optimal_scene = optimizer_global(sparse_scene)
