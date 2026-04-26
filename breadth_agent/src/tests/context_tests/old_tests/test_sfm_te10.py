"""
This script solves the problem of creating a sparse 3D representation of a scene featuring an object as the central point 
using Structure-from-Motion as the algorithm solution. Using a monocular camera that is calibrated, the scene was captured to 
exhibit consistent lighting, high textured regions, and recorded with incremental movement across sequential images taken from
a video feed. The scene was an indoor well-lit scene with a highly textured object in the scene as the focal point. 
The goal of this script was to utilize the Structure-from-Motion (SfM) techniques and certain features of the scene to 
invoke the correct set of tools to properly execute the SfM algorithm with high accuracy and and computation speed in mind
with the preference of ORB over SIFT.
"""

# ==#$#==

# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\datasets\\Structure-from-Motion\\sfm_dataset"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\Structure-from-Motion\\calibration_new.npz"
bal_path = "C:\\Users\\Anthony\\Documents\\Projects\\scene_agent\\breadth_agent\\results\\scene_data\\bal_data.txt"


# STEP 1: Read in Camera Data
from modules.cameramanager import CameraDataManager

CDM = CameraDataManager(image_path=image_path,
                        calibration_path=calibration_path,
                        target_resolution=[1024, 1024])
# Any image pre-processing steps are ran here
# ...
# Get Camera Data
camera_data = CDM.get_camera_data()
print("IMAGE SHAPE:", camera_data.image_list[0].shape)
print("INTRINSICS:", camera_data.get_K(0))
print("INTRINSICS Second Call:", camera_data.get_K(0))
print("SCALE", camera_data.image_scale)
print("OLD SHAPE", camera_data.image_shape_old)
# STEP 2: Estimate Features per Image
from modules.features import FeatureDetectionSIFT
# Feature Module Initialization
feature_detector = FeatureDetectionSIFT(cam_data=camera_data)

# Detect Features for all Images
features = feature_detector()

print("INTRINSICS Features:", camera_data.get_K(0))

# STEP 3: Match Features Per Image 
from modules.featurematching import FeatureMatchFlannPair
feature_matcher = FeatureMatchFlannPair(cam_data=camera_data,
                                        detector="sift",
                                        RANSAC_threshold=0.5)
matched_features = feature_matcher(features)
print("INTRINSICS Matching:", camera_data.get_K(0))

# STEP 4: Estimate Camera Poses of Scene with Feature Pairs
from modules.camerapose import CamPoseEstimatorEssentialToPnP
# Camera Pose Module Initialization
pose_estimator = CamPoseEstimatorEssentialToPnP(cam_data=camera_data)

# From estimated features, estimate the camera poses for all image frames
cam_poses = pose_estimator(feature_pairs=matched_features)

print("INTRINSICS Pose Estimation:", camera_data.get_K(0))

# STEP 5: Estimate Feature Tracks:
from modules.featurematching import FeatureMatchFlannTracking
# Feature Tracking Module Initialization
feature_tracker = FeatureMatchFlannTracking(detector='sift', 
                                            cam_data=camera_data,
                                            RANSAC_threshold=0.2)

# From estimated features, tracked features across multiple images
tracked_features = feature_tracker(features=features)
print("INTRINSICS Tracking:", camera_data.get_K(0))

# STEP 6: Reconstruct Scene in Full
from modules.scenereconstruction import Sparse3DReconstructionMapAnything
# Scene Reconstruction Module Initialization
sparse_reconstruction = Sparse3DReconstructionMapAnything(cam_data=camera_data,
                                                          min_observe=4,
                                                          update_intrinsics=False)

# Estimate sparse 3D scene from tracked features and camera poses
sparse_scene = sparse_reconstruction(tracked_features, cam_poses=cam_poses)

# STEP 7: Run Optimization Algorithm
from modules.optimization import BundleAdjustmentOptimizerLeastSquares
# # Build Optimizer
optimizer = BundleAdjustmentOptimizerLeastSquares(cam_data=camera_data,
                                                  max_iterations=30, 
                                                  num_epochs=1, 
                                                  step_size=0.1,
                                                  optimizer_cls="LevenbergMarquardt")

# Run Optimizer
optimal_scene = optimizer(scene=sparse_scene)

# Optional Visualization
from modules.visualize import VisualizeScene
visualizer = VisualizeScene()

visualizer(optimal_scene)