# Construct Modules with Initialized Arguments
image_path = r"C:\Users\Anthony\Documents\Projects\datasets\sfm_dataset\DTU\scan22_normal_lighting"
calibration_path = r"C:\Users\Anthony\Documents\Projects\datasets\sfm_dataset\DTU\calibration_DTU_new.npz"

# STEP 1: Read in Camera Data (Keep native resolution; CameraDataManager handles any scaling + intrinsic updates if resizing occurs)
from modules.cameramanager import CameraDataManager

CDM = CameraDataManager(image_path=image_path,
                        calibration_path=calibration_path)

# Get Camera Data
camera_data = CDM.get_camera_data()

# STEP 2: Feature Detection (SIFT for rich textures, stable lighting)
from modules.features import FeatureDetectionSIFT

feature_detector = FeatureDetectionSIFT(cam_data=camera_data,
                                        max_keypoints=18000,
                                        n_octave_layers=3,
                                        contrast_threshold=0.006,
                                        edge_threshold=12,
                                        sigma=1.6)

features = feature_detector()

# STEP 3: Pairwise Feature Matching for Initialization (FLANN)
from modules.featurematching import FeatureMatchFlannPair

feature_matcher = FeatureMatchFlannPair(cam_data=camera_data,
                                        detector='sift',
                                        k=2,
                                        RANSAC=True,
                                        RANSAC_threshold=0.3,
                                        RANSAC_conf=0.995,
                                        lowes_thresh=0.75)

feature_pairs = feature_matcher(features=features)

# STEP 4: Camera Pose Estimation (Essential + PnP)
from modules.camerapose import CamPoseEstimatorEssentialToPnP

pose_estimator = CamPoseEstimatorEssentialToPnP(cam_data=camera_data,
                                                iteration_count=200,
                                                reprojection_error=3.0,
                                                confidence=0.995,
                                                optimizer=None)

cam_poses = pose_estimator(feature_pairs=feature_pairs)

# Optional: If drift observed, enable Local BA during pose estimation
# from modules.optimization import BundleAdjustmentOptimizerLocal
# local_optimizer = BundleAdjustmentOptimizerLocal(cam_data=camera_data,
#                                                  max_num_iterations=20,
#                                                  window_size=8,
#                                                  robust_loss=True)
# pose_estimator_refined = CamPoseEstimatorEssentialToPnP(cam_data=camera_data,
#                                                         iteration_count=200,
#                                                         reprojection_error=3.0,
#                                                         confidence=0.995,
#                                                         optimizer=local_optimizer)
# cam_poses = pose_estimator_refined(feature_pairs=feature_pairs)

# STEP 5: Multi-View Feature Tracking (FLANN Tracking)
from modules.featuretracking import FeatureMatchFlannTracking

feature_tracker = FeatureMatchFlannTracking(cam_data=camera_data,
                                            detector='sift',
                                            k=2,
                                            RANSAC_threshold=0.2,
                                            RANSAC_conf=0.99,
                                            lowes_thresh=0.75)

tracked_features = feature_tracker(features=features)

# STEP 6: Sparse 3D Reconstruction (Multi-View Triangulation)
from modules.scenereconstruction import Sparse3DReconstructionMono

sparse_reconstruction = Sparse3DReconstructionMono(cam_data=camera_data,
                                                   view='multi',
                                                   min_observe=5,
                                                   min_angle=3.0,
                                                   reproj_error=3.0)

sparse_scene = sparse_reconstruction(tracked_features, cam_poses)

# STEP 7: Global Bundle Adjustment (Refinement)
from modules.optimization import BundleAdjustmentOptimizerGlobal

optimizer = BundleAdjustmentOptimizerGlobal(cam_data=camera_data,
                                            refine_focal_length=False,
                                            refine_principal_point=False,
                                            refine_extra_params=False,
                                            max_num_iterations=80,
                                            use_gpu=True,
                                            gpu_index=0,
                                            robust_loss=True)

optimal_scene = optimizer(sparse_scene)

# Optional Visualization (Sparse)
from modules.visualize import VisualizeScene
visualizer = VisualizeScene()
visualizer(optimal_scene)