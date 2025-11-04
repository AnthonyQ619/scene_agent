"""
This script solves the problem of creating a sparse 3D representation of a scene featuring an object as the central point 
using Structure-from-Motion as the algorithm solution. Using a monocular camera that is calibrated, the scene was captured to 
exhibit consistent lighting, highly textured objects, and recorded with incremental movement across sequential images taken from
a video feed. The scene was an indoor consistently lit scene with a highly textured object in the scene as the focal point. 
The goal of this script was to utilize the Structure-from-Motion (SfM) techniques and certain features of the scene to 
invoke the correct set of tools to properly execute the SfM algorithm with high accuracy and and computation speed in mind
with the preference of ORB over SIFT.
"""

# ==#$#==

from modules.utilities.utilities import CalibrationReader
from modules.features import FeatureDetectionSIFT
from modules.featurematching import FeatureMatchFlannTracking, FeatureMatchFlannPair
from modules.camerapose import CamPoseEstimatorEssentialToPnP, CamPoseEstimatorVGGTModel
from modules.scenereconstruction import Sparse3DReconstructionMono
from modules.optimization import BundleAdjustmentOptimizer
from modules.visualize import VisualizeScene

# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\datasets\\Structure-from-Motion\\sfm_dataset"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\Structure-from-Motion\\calibration.npz"
bal_path = "C:\\Users\\Anthony\\Documents\\Projects\\scene_agent\\breadth_agent\\results\\scene_data\\bal_data.txt"

# Feature Module Initialization
calibration_data = CalibrationReader(calibration_path).get_calibration()
feature_detector = FeatureDetectionSIFT(image_path=image_path)

# Camera Pose Module Initialization
pose_estimator = CamPoseEstimatorEssentialToPnP(calibration=calibration_data, image_path=image_path)

# Pairwise Feature Matching Module Initialization
feature_matcher = FeatureMatchFlannPair(detector='sift', calibration=calibration_data)

# Feature Tracking Module Initialization
feature_tracker = FeatureMatchFlannTracking(detector='sift', calibration=calibration_data)

# Scene Reconstruction Module Initialization
sparse_reconstruction = Sparse3DReconstructionMono(calibration=calibration_data, 
                                                   image_path=image_path,
                                                   min_observe=3,
                                                   min_angle=1.0)

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

# Establish and Execute Optimizer
optimizer = BundleAdjustmentOptimizer(scene=sparse_scene, calibration=calibration_data)
optimizer.prep_optimizer(ratio_known_cameras=0.0, max_iterations=40)

optimal_scene = optimizer(bal_path)

visualizer(sparse_scene)