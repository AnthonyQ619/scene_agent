"""
This script solves the problem of creating a sparse 3D representation of an indoor scene with varying lighting conditions 
using Structure-from-Motion as the algorithm solution. Using a monocular camera that is calibrated, the scene captures an Indoors 
museum/stairwell-like interior with arches, balustrade and stone walls. Artificial indoor lighting appears consistent across the images; 
only minor exposure/position changes of the ceiling lamp, no significant illumination change on the statue. Mixture of textures as the 
stone statue and carved fruit details are highly textured; surrounding walls, pillars, steps and floor are mostly smooth/low-textured.
A single stone statue mounted on a base is the persistent object of interest in all frames. Background architectural elements 
(arches, staircase, panels) remain similar. The lighting is soft/diffuse from overhead fixtures, producing mostly matte appearance 
on the stone. The sequence suggests small positional/rotational changes of the camera around the statue without leaving the 
immediate area; the object stays centered and visible the whole time. Gradual, incremental viewpoint shifts (slight translations 
and small rotations); no extreme or dramatic viewpoint changes.


Scene Summary: An indoor museum/stairwell setting with arches, a balustrade, and stone walls under stable artificial lighting. 
A central stone statue with richly textured carved details is consistently visible across three images, while surrounding walls, 
pillars, steps, and floors are relatively smooth and low-textured. The camera undergoes small translations and rotations around 
the statue, maintaining the object near the center with limited baseline changes and minimal illumination variation.
"""

# ==#$#==

# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\ETH\\statue\\images\\dslr_images_undistorted"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\ETH\\statue\\dslr_calibration_undistorted\\calibration_new.npz"
# bal_path = "C:\\Users\\Anthony\\Documents\\Projects\\scene_agent\\breadth_agent\\results\\scene_data\\bal_data.txt"


# STEP 1: Read in Camera Data
from modules.cameramanager import CameraDataManager

CDM = CameraDataManager(image_path=image_path,
                        calibration_path=calibration_path)

# Get Camera Data
camera_data = CDM.get_camera_data()

# STEP 2: Estimate Features per Image
from modules.features import FeatureDetectionSIFT
# Feature Module Initialization
feature_detector = FeatureDetectionSIFT(cam_data=camera_data, 
                                        max_keypoints=6000,
                                        contrast_threshold=0.02,
                                        edge_threshold=50)

# Detect Features for all Images
features = feature_detector()

# STEP 3: Match Features Per Image 
from modules.featurematching import FeatureMatchFlannPair
# Pairwise Feature Matching Module Initialization
feature_matcher = FeatureMatchFlannPair(detector='sift', 
                                        cam_data=camera_data,
                                        RANSAC_threshold=0.6,
                                        lowes_thresh=0.7)

# Detect Image Pair Correspondences for Pose Estimation
feature_pairs = feature_matcher(features=features)

# STEP 4: Estimate Camera Poses of Scene with Feature Pairs
from modules.camerapose import CamPoseEstimatorEssentialToPnP
# Camera Pose Module Initialization
pose_estimator = CamPoseEstimatorEssentialToPnP(cam_data=camera_data,
                                                reprojection_error=4.0,
                                                iteration_count=200,
                                                confidence=0.995)

# From estimated features, estimate the camera poses for all image frames
cam_poses = pose_estimator(feature_pairs=feature_pairs)

# STEP 5: Estimate Feature Tracks:
from modules.featurematching import FeatureMatchFlannTracking
# Feature Tracking Module Initialization
feature_tracker = FeatureMatchFlannTracking(detector='sift', 
                                            cam_data=camera_data,
                                            RANSAC_threshold=0.2)

# From estimated features, tracked features across multiple images
tracked_features = feature_tracker(features=features)

# STEP 6: Reconstruct Scene in Full
from modules.scenereconstruction import Sparse3DReconstructionMono
# Scene Reconstruction Module Initialization
sparse_reconstruction = Sparse3DReconstructionMono(cam_data=camera_data,
                                                   min_observe=4,
                                                   min_angle=2.0,
                                                   view="multi")

# Estimate sparse 3D scene from tracked features and camera poses
sparse_scene = sparse_reconstruction(tracked_features, cam_poses)

# STEP 7: Run Optimization Algorithm
from modules.optimization import BundleAdjustmentOptimizerLeastSquares
# # Build Optimizer
optimizer = BundleAdjustmentOptimizerLeastSquares(cam_data=camera_data,
                                                  max_iterations=10, 
                                                  num_epochs=1, 
                                                  step_size=0.1,
                                                  optimizer_cls="GaussNewton")

# Run Optimizer
optimal_scene = optimizer(scene=sparse_scene)


# Optional Visualization
from modules.visualize import VisualizeScene
visualizer = VisualizeScene()

visualizer(sparse_scene)