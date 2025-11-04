"""
This script solves the problem of creating a sparse 3D representation of an indoor scene featuring an object as the central point 
using Structure-from-Motion as the algorithm choice of solution. Using a monocular camera that is calibrated, the scene was captured to 
exhibit consistent lighting, low-textured object, an object with specular lighting, and consistent incremental movement across scenes. 
The goal of this script was to utilize the Structure-from-Motion (SfM) techniques and specific features of the scene, such as a low-textured
object, to invoke the correct set of tools that excel at those features to properly execute the SfM algorithm with high detail and 
as many points in reconstruction as possible for a sparse 3D build.
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