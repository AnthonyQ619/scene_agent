"""
This script solves the problem of creating a sparse 3D representation of an indoor scene featuring an object as the focal point 
using Structure-from-Motion as the algorithm solution. Using a monocular camera that was calibrated, the scene was captured with 
illumination changes across images, no low textured regions, low-lit images, and recorded with incremental movement across sequential images. 
The goal of this script was to utilize the Structure-from-Motion (SfM) techniques and certain features of the scene to invoke the correct set 
of tools to properly execute the SfM algorithm with high-fidelity and as many points as possible for a sparse reconstruction. 
"""

# ==#$#==

from modules.utilities.utilities import CalibrationReader
from modules.featurematching import FeatureMatchRoMAPair
from modules.camerapose import CamPoseEstimatorEssentialToPnP
from modules.scenereconstruction import Sparse3DReconstructionMono
from modules.visualize import VisualizeScene

# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\datasets\\Structure-from-Motion\\sfm_dataset"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\Structure-from-Motion\\calibration.npz"

# Feature Module Initialization
calibration_data = CalibrationReader(calibration_path).get_calibration()

# Pairwise Feature Matching Module Initialization
feature_matcher = FeatureMatchRoMAPair(img_path=image_path)

# Camera Pose Module Initialization
pose_estimator = CamPoseEstimatorEssentialToPnP(calibration=calibration_data, image_path=image_path)

# Scene Reconstruction Module Initialization
sparse_reconstruction = Sparse3DReconstructionMono(calibration=calibration_data, image_path=image_path)

# Visualization
visualizer = VisualizeScene()

# Solution Pipeline 

# Detect Image Pair Correspondences for Pose Estimation
feature_pairs = feature_matcher()

# From estimated features, estimate the camera poses for all image frames
cam_poses = pose_estimator(features_pairs=feature_pairs)

# Estimate sparse 3D scene from tracked features and camera poses
sparse_scene = sparse_reconstruction(feature_pairs , cam_poses, view="two")

visualizer(sparse_scene)