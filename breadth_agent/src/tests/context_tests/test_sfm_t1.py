"""
This script solves the problem of creating a sparse 3D representation of a scene featuring an object as the central point 
using Structure-from-Motion as the algorithm solution. Using a monocular camera that is calibrated, the scene was captured to 
exhibit consistent lighting, high textured regions, and recorded with incremental movement across sequential images taken from
a video feed. The scene was an outdoor scene during the day with highly textured objects in the scene. The goal of this script 
was to utilize the Structure-from-Motion (SfM) techniques and certain features of the scene to invoke the correct set 
of tools to properly execute the SfM algorithm with high accuracy and computation speed.
"""

# ==#$#==

# from modules.cameramanager import CameraDataManager
# from modules.features import FeatureDetectionSIFT
# from modules.featurematching import FeatureMatchFlannTracking, FeatureMatchFlannPair
# from modules.camerapose import CamPoseEstimatorEssentialToPnP, CamPoseEstimatorVGGTModel
# from modules.scenereconstruction import Sparse3DReconstructionMono
# from modules.optimization import BundleAdjustmentOptimizer
# from modules.visualize import VisualizeScene

# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\datasets\\Structure-from-Motion\\sfm_dataset"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\Structure-from-Motion\\calibration_new.npz"
bal_path = "C:\\Users\\Anthony\\Documents\\Projects\\scene_agent\\breadth_agent\\results\\scene_data\\bal_data.txt"


# STEP 1: Read in Camera Data
from modules.cameramanager import CameraDataManager

CDM = CameraDataManager(image_path=image_path,
                        calibration_path=calibration_path)
# Any image pre-processing steps are ran here
# ...
# Get Camera Data
camera_data = CDM.get_camera_data()

# STEP 2: Estimate Features per Image
from modules.features import FeatureDetectionSIFT
# Feature Module Initialization
feature_detector = FeatureDetectionSIFT(cam_data=camera_data)

# Detect Features for all Images
features = feature_detector()

# STEP 3: Match Features Per Image 
from modules.featurematching import FeatureMatchFlannPair
# Pairwise Feature Matching Module Initialization
feature_matcher = FeatureMatchFlannPair(detector='sift', 
                                        cam_data=camera_data)

# Detect Image Pair Correspondences for Pose Estimation
feature_pairs = feature_matcher(features=features)

# STEP 4: Estimate Camera Poses of Scene with Feature Pairs
from modules.camerapose import CamPoseEstimatorEssentialToPnP
# Camera Pose Module Initialization
pose_estimator = CamPoseEstimatorEssentialToPnP(cam_data=camera_data)

# From estimated features, estimate the camera poses for all image frames
cam_poses = pose_estimator(feature_pairs=feature_pairs)

# STEP 5: Estimate Feature Tracks:
from modules.featurematching import FeatureMatchFlannTracking
# Feature Tracking Module Initialization
feature_tracker = FeatureMatchFlannTracking(detector='sift', 
                                            cam_data=camera_data)

# From estimated features, tracked features across multiple images
tracked_features = feature_tracker(features=features)

# STEP 6: Reconstruct Scene in Full
from modules.scenereconstruction import Sparse3DReconstructionMono
# Scene Reconstruction Module Initialization
sparse_reconstruction = Sparse3DReconstructionMono(cam_data=camera_data,
                                                   min_observe=3,
                                                   min_angle=1.0,
                                                   view="multi")

# Estimate sparse 3D scene from tracked features and camera poses
sparse_scene = sparse_reconstruction(tracked_features, cam_poses)

# STEP 7: Run Optimization Algorithm
# from modules.optimization import BundleAdjustmentOptimizer
# # Build Optimizer
# optimizer = BundleAdjustmentOptimizer(scene=sparse_scene, cam_data=camera_data)
# optimizer.prep_optimizer(ratio_known_cameras=0.0, max_iterations=40)

# # Run Optimizer
# optimal_scene = optimizer(bal_path)

# Optional Visualization
from modules.visualize import VisualizeScene
visualizer = VisualizeScene()

visualizer(sparse_scene)



