from modules.features import FeatureDetectionSIFT, FeatureDetectionSP, FeatureDetectionORB
from modules.featurematching import (FeatureMatchFlannPair, 
                                     FeatureMatchLoftrPair, 
                                     FeatureMatchBFPair,
                                     FeatureMatchLightGluePair, 
                                     FeatureMatchLightGlueTracking,
                                     FeatureMatchSuperGluePair,
                                     FeatureMatchRoMAPair)
from modules.camerapose import CamPoseEstimatorEssentialToPnP, CamPoseEstimatorVGGTModel
from modules.visualize import VisualizeScene
from modules.cameramanager import CameraDataManager
from modules.scenereconstruction import Dense3DReconstructionMono

# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\scan14_normal_lighting"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\calibration_DTU_new.npz"


# STEP 1: Read in Camera Data
from modules.cameramanager import CameraDataManager

CDM = CameraDataManager(image_path=image_path,
                        calibration_path=calibration_path)

# Get Camera Data
camera_data = CDM.get_camera_data()

# STEP 2: Estimate Features per Image
from modules.features import FeatureDetectionSIFT
# Feature Module Initialization
feature_detector = FeatureDetectionSIFT(cam_data=camera_data, 
                                        max_keypoints=15000,
                                        contrast_threshold=0.01,
                                        edge_threshold=10)

# Detect Features for all Images
features = feature_detector()

# STEP 3: Match Features Per Image 
from modules.featurematching import FeatureMatchFlannPair
# Pairwise Feature Matching Module Initialization
feature_matcher = FeatureMatchFlannPair(detector='sift', 
                                        cam_data=camera_data,
                                        RANSAC_threshold=0.2,
                                        lowes_thresh=0.6)

# Detect Image Pair Correspondences for Pose Estimation
feature_pairs = feature_matcher(features=features)

# STEP 4: Estimate Camera Poses of Scene with Feature Pairs
from modules.camerapose import CamPoseEstimatorEssentialToPnP
# Camera Pose Module Initialization
pose_estimator = CamPoseEstimatorEssentialToPnP(cam_data=camera_data,
                                                reprojection_error=4.0,
                                                iteration_count=200,
                                                confidence=0.995)

# From estimated features, estimate the camera poses for all image frames
cam_poses = pose_estimator(feature_pairs=feature_pairs)

# STEP 5: Estimate Feature Tracks:
from modules.featurematching import FeatureMatchFlannTracking
# Feature Tracking Module Initialization
feature_tracker = FeatureMatchFlannTracking(detector='sift', 
                                            cam_data=camera_data,
                                            RANSAC_threshold=0.3)

# From estimated features, tracked features across multiple images
tracked_features = feature_tracker(features=features)

# STEP 6: Reconstruct Scene in Full
from modules.scenereconstruction import Sparse3DReconstructionMono
# Scene Reconstruction Module Initialization
sparse_reconstruction = Sparse3DReconstructionMono(cam_data=camera_data,
                                                   min_observe=4,
                                                   min_angle=1.0,
                                                   view="multi")

# Estimate sparse 3D scene from tracked features and camera poses
sparse_scene = sparse_reconstruction(tracked_features, cam_poses)

# STEP 7: Run Optimization Algorithm
from modules.optimization import BundleAdjustmentOptimizerGlobal
# Build Optimizer
optimizer = BundleAdjustmentOptimizerGlobal(max_num_iterations=30)

# Run Optimizer
optimal_scene, _ = optimizer.optimize(sparse_scene, 
                                      cam_data=camera_data)

# Conduct Dense Reconstruction
sparse_reconstruction = Dense3DReconstructionMono(cam_data=camera_data,
                                                  reproj_error=3.0,
                                                  min_triangulation_angle=1.0,
                                                  num_samples=15,
                                                  num_iterations=3)

# Estimate sparse 3D scene from tracked features and camera poses
sparse_scene = sparse_reconstruction(sparse_scene=optimal_scene)