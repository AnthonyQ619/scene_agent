"""
This script solves the problem of creating a sparse 3D representation of an indoor scene featuring an object as the central point 
using Structure-from-Motion as the algorithm choice of solution. Using a monocular camera that is calibrated, the scene was captured within 
a low-lighting enviornment, decently textured regions, and recorded with consistent incremental movement across the scene. The goal of 
this script was to utilize the Structure-from-Motion (SfM) techniques and features of the scene, such as low-lighting, to invoke the correct set 
of tools to properly execute the SfM algorithm with great detail and as many as  possible for the sparse reconstruction due to the low-lit envionrmnet.
"""

# ==#$#==

from modules.utilities.utilities import CalibrationReader
from modules.features import FeatureDetectionSP
from modules.featurematching import FeatureMatchLightGlueTracking, FeatureMatchLightGluePair
from modules.camerapose import CamPoseEstimatorEssentialToPnP
from modules.scenereconstruction import Sparse3DReconstructionMono
from modules.visualize import VisualizeScene

# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\datasets\\Structure-from-Motion\\sfm_dataset"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\Structure-from-Motion\\calibration.npz"

# Feature Module Initialization
calibration_data = CalibrationReader(calibration_path).get_calibration()
feature_detector = FeatureDetectionSP(image_path=image_path)

# Camera Pose Module Initialization
pose_estimator = CamPoseEstimatorEssentialToPnP(calibration=calibration_data, image_path=image_path)

# Pairwise Feature Matching Module Initialization
feature_matcher = FeatureMatchLightGluePair(detector='superpoint')

# Feature Tracking Module Initialization
feature_tracker = FeatureMatchLightGlueTracking(detector='superpoint')

# Scene Reconstruction Module Initialization
sparse_reconstruction = Sparse3DReconstructionMono(calibration=calibration_data, image_path=image_path)

# Visualization
visualizer = VisualizeScene()

# Solution Pipeline 

# Detect Features for all Images
features = feature_detector()

# Detect Image Pair Correspondences for Pose Estimation
feature_pairs = feature_matcher(features=features)

# From estimated features, estimate the camera poses for all image frames
cam_poses = pose_estimator(features_pairs=feature_pairs)

# From estimated features, tracked features across multiple images
tracked_features = feature_tracker(features=features)

# Estimate sparse 3D scene from tracked features and camera poses
sparse_scene = sparse_reconstruction(tracked_features, cam_poses, view="multi")

visualizer(sparse_scene)
