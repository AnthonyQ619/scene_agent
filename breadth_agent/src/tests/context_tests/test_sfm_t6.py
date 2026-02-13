"""
Scene Setting: Outdoors (campus courtyard with buildings, palm tree, bench, walkway).
Scene Lighting Effects: Consistent soft daylight across all three images; no noticeable 
illumination change, diffuse light with mild shadows.
Scene Object Textures: High-textured primary object (bronze statue with detailed folds 
and surface patina) and brick paving; also some low-texture areas such as sky and smooth 
building walls.
Scene Key Features: A bronze statue on a pedestal remains the central object of interest 
in every image. The scene includes a bench, lamp post, palm tree, shrubs, and campus 
buildings. Lighting is diffuse and even. The camera keeps the statue centered while moving 
slightly to the right, giving mild background parallax. The statue is highly textured; 
surroundings feature both textured (brick, foliage, palm trunk) and smoother regions.
Scene Camera Movement: Small, incremental viewpoint shifts 
(slight pan/short orbit to the right); no extreme changes.


Scene Summary: An outdoor courtyard with a bronze statue centered in each view, surrounded 
by brick paving, a bench, a lamp post, a palm tree with foliage, shrubs, and campus buildings. 
Illumination is diffuse and even, producing mild shadows. Textures vary: the statue, bricks, 
foliage, and palm trunk are richly textured, while the sky and some building walls are smooth. 
The camera performs slight rightward orbit/pan movements, yielding mild background parallax 
while keeping the statue central.
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
- Primary object is High-textured primary object with brick paving with scene having consistent lighting. USE SIFT FEATURES.
- Select the SIFT module since the scene featuring the object is well lit, in an indoor setting, and the object is of a brick house that is 
  well textured for using SIFT reliably.
    - Set max features to detect to 15000, to ensure we detect as much features as possible since we also want to detect corners due to the 
      object having many set shapes (house)
    - Set contrast threshold to 0.01, with default being 0.04, the lower the threshold, more features are produced
    - Set edge_threshold to 10, the higher the threshold, the more edge-like features are produced, which we need for this scene.

STEP 3: 
- Select the Flann feature matcher to utilize a nearest neighbor matcher, as we do not have a constraint of time to enforce the brute force 
  algorithm
    - Set RANSAC_threshold to 0.3, for default behavior in outlier rejection since we have a well-lit scene for SIFT
    - Set lowes_threshold to 0.75, to enable slightly more strict rejection in matching pairs to pass since any higher threshold to remove any 
      stray outliers in the scene.

STEP 4:
- Estimate the camera pose using matching feature pairs of the scene
    - Since we are using SIFT feature detector in a more difficult and less ideal setting (Outdoor but highly textured), we need further refinment
      in pose estimation for more accurate scene reconsturction
        - Initialize the Local Optimization module "BundleAdjustmentOptimizerLocal"
            - Set iteration count for Bundle Adjustment to only 20
            - Set window size to 8 (Default) since camera movement is minimal and better pose correction.
    - Set reprojection_error= 3.0, since we are including corner points as well to ensure as the 3D points that are included are more 
      more accurate for better optimized scene after bundle adjustment.
    - Set iteration_count=300, higher setting for iteration count of the Levenberg-Marquardt algorithm during pose estimation as the setting
      is outdoors, requiring for better pose estimation.
    - Set confidence=0.995, Since the scene is well-lit and textured, we have a higher confidence of the proposed solutions 

STEP 5:
- Track Features across multiple images using the same feature matching tool
    - Set RANSAC_threshold to 0.2, as we want to be more strict on the feature tracks for more accurate estimate of 3D points in the scene

STEP 6:
- Reconstruct the Scene now that we have the estimated Camera Poses and Tracked features for multi-view estimation
    - Set min_observe=5, to increase the minimum number of observations needed to estimate a 3D point of 4 features, since this scene has 
      minimal camera movement
    - Set min_angle=3.0, utilize a larger minimum angle to ensure 3D point estimates are more acccurate.
    - Set view="multi", we are tracking features, so the estimation method is multi-view

STEP 7:
- Apply Global Bundle Adjustment to the scene for optimal reconstruction
    - Set max_num_iterations=80, since initial scene is not optimal geometrically despite the scene lighting and texture being optimal for SIFT features.
"""

# ==#$#==

# Construct Modules with Initialized Arguments
# image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Tanks_and_Temples\\Family"
# calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Tanks_and_Temples\\calibration_new_2048.npz"
image_path = r"C:\Users\Anthony\Documents\Projects\datasets\CO3D\apple\189_20393_38136\images"
calibration_path = r"C:\Users\Anthony\Documents\Projects\datasets\CO3D\apple\calibration_new_189.npz"

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
                                        max_keypoints=15000,
                                        contrast_threshold=0.01,
                                        edge_threshold=10)

# Detect Features for all Images
features = feature_detector()

# STEP 3: Match Features Per Image 
from modules.featurematching import FeatureMatchFlannPair
# Pairwise Feature Matching Module Initialization
feature_matcher = FeatureMatchFlannPair(detector='sift', 
                                        cam_data=camera_data,
                                        RANSAC_threshold=0.3,
                                        lowes_thresh=0.75)

# Detect Image Pair Correspondences for Pose Estimation
feature_pairs = feature_matcher(features=features)

# STEP 4: Estimate Camera Poses of Scene with Feature Pairs
# from modules.camerapose import CamPoseEstimatorEssentialToPnP

from modules.optimization import BundleAdjustmentOptimizerLocal

# Build Optimizer
optimizer_local = BundleAdjustmentOptimizerLocal(max_num_iterations=30,
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
                                                   min_observe=3,
                                                   min_angle=1.0,
                                                   view="multi")

# Estimate sparse 3D scene from tracked features and camera poses
sparse_scene = sparse_reconstruction(tracked_features, cam_poses)

# STEP 7: Run Optimization Algorithm
from modules.optimization import BundleAdjustmentOptimizerGlobal

# Build Optimizer
optimizer = BundleAdjustmentOptimizerGlobal(max_num_iterations=80,
                                            cam_data=camera_data)

# Run Optimizer
optimal_scene = optimizer(sparse_scene)


# Optional Visualization
from modules.visualize import VisualizeScene
visualizer = VisualizeScene()

visualizer(optimal_scene)