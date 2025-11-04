"""
This script solves the problem of creating a sparse 3D representation of a scene featuring an object as the focal point 
using Structure-from-Motion as the algorithm solution. Using a monocular camera that wasn't pre-calibrated, the scene was captured to 
exhibit consistent lighting, no low textured regions, and recorded with incremental movement across sequential images. The goal of 
this script was to utilize the Structure-from-Motion (SfM) techniques and certain features of the scene to invoke the correct set 
of tools to properly execute the SfM algorithm as close to scale as possible with a monocular camera while also calibrating the camera 
to build an accurate sparse 3D scene. 
"""

# ==#$#==

from modules.utilities.utilities import CalibrationReader
from modules.features import FeatureDetectionSP
from modules.featurematching import FeatureMatchLightGlueTracking, FeatureMatchLoftrPair
from modules.camerapose import CamPoseEstimatorVGGTModel
from modules.scenereconstruction import Sparse3DReconstructionMono
from modules.visualize import VisualizeScene

# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\datasets\\Structure-from-Motion\\sfm_dataset"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\Structure-from-Motion\\calibration.npz"

# Camera Pose Module Initialization
calibration_data = CalibrationReader(calibration_path).get_calibration()
pose_estimator = CamPoseEstimatorVGGTModel(image_path=image_path, calibration=calibration_data)

# Feature Module Initialization
feature_detector = FeatureDetectionSP(image_path=image_path)

# Feature Tracking Module Initialization
feature_tracker = FeatureMatchLightGlueTracking(detector="superpoint")

# Scene Reconstruction Module Initialization
sparse_reconstruction = Sparse3DReconstructionMono(calibration=calibration_data, image_path=image_path)

# Visualization
visualizer = VisualizeScene()

# Solution Pipeline 
# From estimated features, estimate the camera poses for all image frames
cam_poses = pose_estimator()

# Detect Features for all Images
features = feature_detector()

# From estimated features, tracked features across multiple images
tracked_features = feature_tracker(features=features)

# Estimate sparse 3D scene from tracked features and camera poses
sparse_scene = sparse_reconstruction(tracked_features, cam_poses, view="multi")

visualizer(sparse_scene)