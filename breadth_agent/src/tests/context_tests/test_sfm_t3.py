"""
This script solves the problem of creating a sparse 3D representation of a scene featuring an indoor environment
using Structure-from-Motion as the algorithm solution. Using a monocular camera that is calibrated, the scene was captured to 
exhibit consistent lighting in an indoor enviornment and recorded with incremental and large movement between sets of images frames
taken from camera shots, not a video. The scene was an indoor well-lit scene with varying objects in the scene as the focal point. 
The goal of this script was to utilize the Structure-from-Motion (SfM) techniques and certain features of the scene to 
invoke the correct set of tools to properly execute the SfM algorithm with high accuracy over computation speed.
"""

# ==#$#==

from modules.utilities.utilities import CalibrationReader
from modules.features import FeatureDetectionSIFT
from modules.featurematching import FeatureMatchFlannTracking
from modules.camerapose import CamPoseEstimatorVGGTModel
from modules.scenereconstruction import Sparse3DReconstructionMono
from modules.optimization import BundleAdjustmentOptimizer
from modules.visualize import VisualizeScene

# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\ETH\\living_room_dslr_undistorted\\living_room\\images\\dslr_images_undistorted"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\ETH\\living_room_dslr_undistorted\\living_room\\dslr_calibration_undistorted\\calibration_eth_lr.npz"
bal_path = "C:\\Users\\Anthony\\Documents\\Projects\\scene_agent\\breadth_agent\\results\\scene_data\\bal_data.txt"

# Feature Module Initialization
calibration_data = CalibrationReader(calibration_path).get_calibration()
feature_detector = FeatureDetectionSIFT(image_path=image_path)

# Camera Pose Module Initialization
pose_estimator = CamPoseEstimatorVGGTModel(image_path=image_path, calibration=calibration_data)


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
# From estimated features, estimate the camera poses for all image frames
cam_poses = pose_estimator()

# Detect Features for all Images
features = feature_detector()

# From estimated features, tracked features across multiple images
tracked_features = feature_tracker(features=features)

# Estimate sparse 3D scene from tracked features and camera poses
sparse_scene = sparse_reconstruction(tracked_features, cam_poses, view="multi")

# Establish Optimizer
optimizer = BundleAdjustmentOptimizer(scene=sparse_scene, calibration=calibration_data)
optimizer.prep_optimizer(ratio_known_cameras=0.0)

optimal_scene = optimizer(bal_path)

visualizer(optimal_scene)