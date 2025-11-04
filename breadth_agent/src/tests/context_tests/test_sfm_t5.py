"""
This script solves the problem of creating a sparse 3D representation of an outdoor scene with low-lighting conditions 
using Structure-from-Motion as the algorithm solution. Using a monocular camera that is calibrated, the scene captured an outdoor
scene with low-lighting conditions causing textured regions to be less pronounced, and recorded with varying degrees of view changes
across sequential images. The goal of this script was to utilize the Structure-from-Motion (SfM) models and certain 
features of the scene to invoke the correct set of tools to properly execute the SfM algorithm with high accuracy.
"""

# ==#$#==

from modules.utilities.utilities import CalibrationReader
from modules.camerapose import CamPoseEstimatorVGGTModel
from modules.scenereconstruction import Sparse3DReconstructionVGGT
from modules.optimization import BundleAdjustmentOptimizer
from modules.visualize import VisualizeScene

# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\ETH\\office\\images\\dslr_images_undistorted"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\ETH\\office\\dslr_calibration_undistorted\\calibration_eth_office.npz"
bal_path = "C:\\Users\\Anthony\\Documents\\Projects\\scene_agent\\breadth_agent\\results\\scene_data\\bal_data.txt"

# Feature Module Initialization
calibration_data = CalibrationReader(calibration_path).get_calibration()

# Camera Pose Module Initialization
pose_estimator = CamPoseEstimatorVGGTModel(image_path=image_path, calibration=calibration_data)

# Scene Reconstruction Module Initialization
sparse_reconstruction = Sparse3DReconstructionVGGT(calibration=calibration_data, 
                                                   image_path=image_path)

# Visualization
visualizer = VisualizeScene()

# Solution Pipeline 
# From estimated features, estimate the camera poses for all image frames
cam_poses = pose_estimator()

# Estimate sparse 3D scene from tracked features and camera poses
sparse_scene = sparse_reconstruction(camera_poses=cam_poses)

visualizer(sparse_scene)