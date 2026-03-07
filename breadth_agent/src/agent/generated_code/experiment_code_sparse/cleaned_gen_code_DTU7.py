
# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Experiments\\DTU\\scan70\\images"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Experiments\\DTU\\calibration_DTU_new.npz"

# STEP 1: Read in Camera Data
from modules.cameramanager import CameraDataManager

CDM = CameraDataManager(image_path=image_path,
                        calibration_path=calibration_path)

# Get Camera Data
camera_data = CDM.get_camera_data()

# STEP 2: Detect Features (SuperPoint)
from modules.features import FeatureDetectionSP
feature_detector = FeatureDetectionSP(cam_data=camera_data,
                                      max_keypoints=6000)
features = feature_detector()

# STEP 3: Match Features Per Image (SuperGlue Pairwise)
from modules.featurematching import FeatureMatchSuperGluePair
feature_matcher = FeatureMatchSuperGluePair(cam_data=camera_data,
                                            detector='superpoint',
                                            setting='indoor',
                                            RANSAC_threshold=0.03)
feature_pairs = feature_matcher(features=features)

# STEP 4: Estimate Camera Poses with Local Bundle Adjustment
from modules.optimization import BundleAdjustmentOptimizerLocal
optimizer_local = BundleAdjustmentOptimizerLocal(cam_data=camera_data,
                                                 max_num_iterations=30,
                                                 window_size=12,
                                                 refine_focal_length=False,
                                                 refine_principal_point=False,
                                                 refine_extra_params=False,
                                                 robust_loss=True,
                                                 use_gpu=True)

from modules.camerapose import CamPoseEstimatorEssentialToPnP
pose_estimator = CamPoseEstimatorEssentialToPnP(cam_data=camera_data,
                                                iteration_count=300,
                                                reprojection_error=3.0,
                                                confidence=0.995,
                                                optimizer=optimizer_local)
cam_poses = pose_estimator(feature_pairs=feature_pairs)

# STEP 5: Track Features Across Multiple Images (SuperGlue Tracking)
from modules.featurematching import FeatureMatchSuperGlueTracking
feature_tracker = FeatureMatchSuperGlueTracking(cam_data=camera_data,
                                                detector='superpoint',
                                                setting='indoor',
                                                RANSAC_threshold=0.05)
tracked_features = feature_tracker(features=features)

# STEP 6: Sparse 3D Reconstruction (Multi-view Triangulation)
from modules.scenereconstruction import Sparse3DReconstructionMono
sparse_reconstruction = Sparse3DReconstructionMono(cam_data=camera_data,
                                                   view="multi",
                                                   min_observe=4,
                                                   min_angle=2.0,
                                                   reproj_error=2.0)
sparse_scene = sparse_reconstruction(tracked_features, cam_poses)

# STEP 7: Global Bundle Adjustment (Intrinsics Fixed)
from modules.optimization import BundleAdjustmentOptimizerGlobal
optimizer_global = BundleAdjustmentOptimizerGlobal(cam_data=camera_data,
                                                   refine_focal_length=False,
                                                   refine_principal_point=False,
                                                   refine_extra_params=False,
                                                   max_num_iterations=200,
                                                   robust_loss=True,
                                                   use_gpu=True)
optimal_scene = optimizer_global(sparse_scene)
