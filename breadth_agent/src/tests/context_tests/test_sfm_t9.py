"""
This script solves the problem of creating a sparse 3D representation of a scene featuring an object as the central point 
using Structure-from-Motion as the algorithm solution. Using a monocular camera that is calibrated, the scene was captured to 
exhibit changing lighting conditions, containing a highly textured object, and recorded with incremental movement across 
sequential images taken from a video feed. The scene was an outdoor scene during the day with highly textured objects in the scene
but increasing lighting during scene changes causes illumination changes between scene frames. The goal of this script 
was to utilize the Structure-from-Motion (SfM) techniques and certain features of the scene to invoke the correct set 
of tools to properly execute the SfM algorithm with high accuracy and computation speed.
"""

# ==#$#==

# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\ETH\\meadow_dslr_undistorted\\meadow\\images\\dslr_images_undistorted"

# STEP 1: Read in Camera Data
from modules.cameramanager import CameraDataManager

CDM = CameraDataManager(image_path=image_path,
                        target_resolution=[1024, 1024])

# Any image pre-processing steps are ran here
# ...
# Get Camera Data
camera_data = CDM.get_camera_data()

# STEP 2: Estimate Features per Image
from modules.features import FeatureDetectionSP
# Feature Module Initialization
feature_detector = FeatureDetectionSP(cam_data=camera_data,
                                      max_keypoints=10000)

# Detect Features for all Images
features = feature_detector()

# STEP 3: Match Features Per Image 
# Ignore

# STEP 4: Estimate Camera Poses of Scene with Feature Pairs
from modules.camerapose import CamPoseEstimatorVGGTModel
# Camera Pose Module Initialization
pose_estimator = CamPoseEstimatorVGGTModel(cam_data=camera_data)

# From estimated features, estimate the camera poses for all image frames
cam_poses = pose_estimator()

# STEP 5: Estimate Feature Tracks:
from modules.featurematching import FeatureMatchSuperGlueTracking
# Feature Tracking Module Initialization
feature_tracker = FeatureMatchSuperGlueTracking(detector='superpoint', 
                                            cam_data=camera_data,
                                            RANSAC_threshold=0.08)

# From estimated features, tracked features across multiple images
tracked_features = feature_tracker(features=features)

# STEP 6: Reconstruct Scene in Full
from modules.scenereconstruction import Sparse3DReconstructionVGGT
# Scene Reconstruction Module Initialization
sparse_reconstruction = Sparse3DReconstructionVGGT(cam_data=camera_data,
                                                   min_observe=4)

# Estimate sparse 3D scene from tracked features and camera poses
sparse_scene = sparse_reconstruction(tracked_features, cam_poses)

# STEP 7: Run Optimization Algorithm
# from modules.optimization import BundleAdjustmentOptimizerLeastSquares
# # # Build Optimizer
# optimizer = BundleAdjustmentOptimizerLeastSquares(cam_data=camera_data,
#                                                   max_iterations=30, 
#                                                   num_epochs=1, 
#                                                   step_size=0.1,
#                                                   optimizer_cls="LevenbergMarquardt")

# # Run Optimizer
# optimal_scene = optimizer(scene=sparse_scene)


# Optional Visualization
from modules.visualize import VisualizeScene
visualizer = VisualizeScene()

visualizer(sparse_scene)