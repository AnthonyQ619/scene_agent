"""
This script solves the problem of creating a sparse 3D representation of an indoor scene featuring an object as the central point 
using Structure-from-Motion as the algorithm choice of solution. Using a monocular camera that is calibrated, the scene was captured to 
exhibit consistent lighting, low-textured object with specular lighting, and consistent incremental movement across scenes. 
The goal of this script was to utilize the Structure-from-Motion (SfM) techniques and specific features of the scene, such as a low-textured
object, to invoke the correct set of tools that excel at those features to properly execute the SfM algorithm with high detail and 
as many points in reconstruction as possible for a sparse 3D build.
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
image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\scan8_normal_lighting"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\calibration_DTU.npz"
bal_path = "C:\\Users\\Anthony\\Documents\\Projects\\scene_agent\\breadth_agent\\results\\scene_data\\bal_data.txt"

# Feature Module Initialization
calibration_data = CalibrationReader(calibration_path).get_calibration()
feature_detector = FeatureDetectionSIFT(image_path=image_path)

# Pairwise Feature Matching Module Initialization
feature_tracker = FeatureMatchFlannTracking(img_path=image_path, calibration=calibration_data)

# Camera Pose Module Initialization
pose_estimator = CamPoseEstimatorVGGTModel(calibration=calibration_data, image_path=image_path)

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

# Establish Optimizer
optimizer = BundleAdjustmentOptimizer(scene=sparse_scene, calibration=calibration_data)
optimizer.prep_optimizer(ratio_known_cameras=0.0)

optimal_scene = optimizer(bal_path)

visualizer(optimal_scene)