"""
This script solves the problem of creating a sparse 3D representation of a scene featuring an object as the central point 
using Structure-from-Motion as the algorithm solution. Using a monocular camera that is calibrated, the scene was captured to 
exhibit consistent lighting, high textured regions, and recorded with incremental movement across sequential images taken from
a video feed. The scene was an outdoor scene during the day with highly textured objects in the scene. The goal of this script 
was to utilize the Structure-from-Motion (SfM) techniques and certain features of the scene to invoke the correct set 
of tools to properly execute the SfM algorithm with high accuracy and computation speed.
"""

# ==#$#==

"""
GOAL RECONSTRUCTION: Sparse Reconstruction

STEP 1: Read in Camera data
- Set the image path to the provided directory of images to the CameraDataManager to read, resize, and pre-process images for reconstruction
- Set the calibration path to the provided calibration file if one is provided. Since it is provided, we have a calibrated camera and 
  activate the parameter
- Finally, call get_camera_data() function to have access to the camera data of images and calibration

STEP 2: 
- Scene: Highly-Textured (high textured regions) and consistent lighting with high textured objects (USE ORB)
- Select the ORB module since the scene featuring the object is well lit, in an indoor setting, and the object is of a brick house that is 
  will contain many corner and edge points, resulting in less points needed for an accurate sparse reconstruction.
    - Set max features to detect to 6000, to ensure we detect as much features as possible since we also want to detect corners due to the 
      object having many set shapes (house) that involves corners and edges.
    - Since the scene is very desirable for ORB usage, an object with many corner and edges while also being well-lit, we don't need to tune 
      any of the other parameters. Main goal is that max points is equivalent to the detected features. 

STEP 3: 
- Select the Brute Force feature matcher to utilize as a quick matcher for ORB, since we don't have complex descriptors, we can get away with
a more greedy approach as long as outlier rejection is tighter.
    - Set RANSAC_threshold to 0.01, for much more strict outlier rejection since we have an abundance of points in a well-lit scene
    - Set lowes_threshold to 0.6, since our ransac algorithm is strict, we want to enable more matching pairs to pass since any higher 
      threshold will remove usable inliers for matches, even if they are slightly incorrect.

STEP 4:
- Estimate the camera pose using matching feature pairs of the scene
    - Since we are using a more inaccurate feature detector (One of SuperPoint or ORB), we must add an extra layer of optimization
    - Initialize the Local Optimization module "BundleAdjustmentOptimizerLocal"
        - Set iteration count for Bundle Adjustment to only 20
        - Set window size to 8 (Default) since camera movement is minimal and better pose correction.
    - Set reprojection_error=3.0, since we are including corner points as well to ensure as many 3D points are included as possible for a 
      and more accurate bundle adjustment
    - Set iteration_count=300, higher setting for iteration count of the Levenberg-Marquardt algorithm during pose estimation. Since ORB is 
      inherently inaccurate compared to other detectors, the increased iteration count for optimization will improve initial poses for 
      a well-lit and textured scene for optimal initial pose estimation
    - Set confidence=0.995, Since the scene is well-lit and textured, we have a higher confidence of the proposed solutions 

STEP 5:
- Track Features across multiple images using the same feature matching tool
    - Set RANSAC_threshold to 0.008, as we want to be more strict on the feature tracks for more accurate estimate of 3D points in the scene
      and remove the chance for any outliers.
    - Set cross_check to False to enable nearest neighbor search with the default k value of 2 (choose best of 2 matches being more accurate)

STEP 6:
- Reconstruct the Scene now that we have the estimated Camera Poses and Tracked features for multi-view estimation
    - Set min_observe=3, since we are using ORB, the feature tracks will naturally be shorter since we have much stricter outlier rejection. 
    - Set min_angle=3.0, to offset the smaller observation count and utilize a larger minimum angle to ensure 3D point estimates are more acccurate.
    - Set view="multi", we are tracking features, so the estimation method is multi-view (Should be default)

STEP 7:
- Apply Global Bundle Adjustment to the scene for optimal reconstruction
    - Set max_num_iterations=80, since initial 2D points will not be as accurate due to leniency in constraints in previous methods despite
      optimal scene lighting and texture being for SIFT features.
"""

# ==#$#==

# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\scan21_normal_lighting"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\calibration_DTU_new.npz"

# STEP 1: Read in Camera Data
from modules.cameramanager import CameraDataManager

CDM = CameraDataManager(image_path=image_path,
                        calibration_path=calibration_path)

# Get Camera Data
camera_data = CDM.get_camera_data()

# STEP 2: Estimate Features per Image
from modules.features import FeatureDetectionORB
# Feature Module Initialization
feature_detector = FeatureDetectionORB(cam_data=camera_data, 
                                        max_keypoints=6000)

# Detect Features for all Images
features = feature_detector()

# STEP 3: Match Features Per Image 
from modules.featurematching import FeatureMatchBFPair
# Pairwise Feature Matching Module Initialization
feature_matcher = FeatureMatchBFPair(detector='orb', 
                                     cam_data=camera_data,
                                     RANSAC_threshold=0.01,
                                     lowes_thresh=0.6)

# Detect Image Pair Correspondences for Pose Estimation
feature_pairs = feature_matcher(features=features)

# STEP 4: Estimate Camera Poses of Scene with Feature Pairs with Local Bundle Adjustment Activated
from modules.optimization import BundleAdjustmentOptimizerLocal

# Build Optimizer
optimizer_local = BundleAdjustmentOptimizerLocal(max_num_iterations=20,
                                                 window_size=8,
                                                 cam_data=camera_data)

from modules.camerapose import CamPoseEstimatorEssentialToPnP
# Camera Pose Module Initialization
pose_estimator = CamPoseEstimatorEssentialToPnP(cam_data=camera_data,
                                                reprojection_error=3.0,
                                                iteration_count=300,
                                                confidence=0.995,
                                                optimizer=optimizer_local)

# From estimated features, estimate the camera poses for all image frames
cam_poses = pose_estimator(feature_pairs=feature_pairs)

# STEP 5: Estimate Feature Tracks:
from modules.featurematching import FeatureMatchBFTracking
# Feature Tracking Module Initialization
feature_tracker = FeatureMatchBFTracking(detector='orb', 
                                         cam_data=camera_data,
                                         RANSAC_threshold=0.008,
                                         cross_check=False)

# From estimated features, tracked features across multiple images
tracked_features = feature_tracker(features=features)

# STEP 6: Reconstruct Scene in Full
from modules.scenereconstruction import Sparse3DReconstructionMono
# Scene Reconstruction Module Initialization
sparse_reconstruction = Sparse3DReconstructionMono(cam_data=camera_data,
                                                   min_observe=3,
                                                   min_angle=3.0,
                                                   view="multi")

# Estimate sparse 3D scene from tracked features and camera poses
sparse_scene = sparse_reconstruction(tracked_features, cam_poses)

# STEP 7: Run Optimization Algorithm
from modules.optimization import BundleAdjustmentOptimizerGlobal

# Build Optimizer
optimizer = BundleAdjustmentOptimizerGlobal(max_num_iterations=70,
                                            cam_data=camera_data)

# Run Optimizer
optimal_scene = optimizer(sparse_scene)

# Optional Visualization
from modules.visualize import VisualizeScene
visualizer = VisualizeScene()

visualizer(optimal_scene)
