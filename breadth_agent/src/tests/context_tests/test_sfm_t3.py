"""
This script solves the problem of creating a sparse 3D representation of a scene featuring an object as the central point 
using Structure-from-Motion as the algorithm solution. Using a monocular camera that is calibrated, the scene captured is
Indoors of a tabletop/studio setup with a model house placed on bricks over a white surface. There are illumination changes
the first image is darker with stronger shadows; subsequent images are brighter and more evenly lit. Then back to dark.
Lighting direction is consistent and mostly diffuse with a few specular highlights. Predominantly high-textured (brick surfaces, 
roof shingles, façade details). Some low-texture areas exist in the plain white background. The scene centers on a small 
model building resting on painted, rough bricks. The object of interest (the model house) is present in all images.
"""

# ==#$#==

"""
GOAL RECONSTRUCTION: Sparse Reconstruction

STEP 1: Read in Camera data
- Set the image path to the provided directory of images to the CameraDataManager to read, resize, and pre-process images for reconstruction
- Since no calibration path is provided, we ignore this parameter (This is a strong indication to utilize a tool that does calibration simultaneously)
- Since no calibration calls for the VGGT pipeline, resize images into a square format for better VGGT optimization, and convert the size to (1024,1024)
- Finally, call get_camera_data() function to have access to the camera data of images and calibration

STEP 2: 
- Scene: stronger shadows, but subsequent images are brighter (Illumination changes) with diffuse lighting. Predominantly high-textured but low texture areas 
  (MIXED TEXTURED). USE SUPERPOINT DUE TO LIGHTING CHANGES AND MIXED TEXTURE.
- Select the SuerPoint module since the scene featuring the object is causing illumination changes across the set of images, and even though it is in an 
  indoor setting, and the object is of a brick house that is well textured, the illumination changes causes weak correspondence in traditional detectors.
  Thus, we opt for SuperPoint, and ML detector robust to illumination or dull lighting enviornments.
    - Set max features to detect to 5000, to ensure we detect as much features as possible for SuperPoint.

STEP 3: 
- Since we are using the VGGT pipeline, we don't need feature points to be matcher pairwise as VGGT estimates pose with images directly.

STEP 4:
- Estimate the camera pose using using the VGGT pipeline as we have no camera calibration,
    - Set the cam_data parameter to the camera_data variable holding our pre-processed image set.

STEP 5:
- Track Features across multiple images using the SuperGlue Feature Tracking tool
    - Set RANSAC_threshold to 0.3, default setting as we want to build longer feature tracks for more accurate estimate of 3D points optimization in the scene
    - Set the detector to "superpoint" as that is the detector we are using.
    - Set the setting to "Indoor" to utilize the indoor weights of the model ("Outdoor" is for outdoor scenes)

STEP 6:
- Reconstruct the Scene now that we have the estimated Camera Poses and Tracked features for multi-view estimation
    - Set min_observe=3, since we are using SuperPoint, the feature tracks might be shorter since we have less points due to the inconsistent lighting. 
    - Since VGGT estimates 3D points just using the camera pose, we don't need other features besides image data from cam_data

STEP 7:
- Apply Global Bundle Adjustment to the scene for optimal reconstruction
    - Set max_num_iterations=50, since initial 2D points will not be as accurate due to leniency in 3D point to point map matching from VGGT calculation
      in previous methods despite superpoint being more robust to illumination changing environments.
"""

# ==#$#==

# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\scan6_illumination_change"

# STEP 1: Read in Camera Data
from modules.cameramanager import CameraDataManager

CDM = CameraDataManager(image_path=image_path,
                        target_resolution=[1024, 1024])

# Get Camera Data
camera_data = CDM.get_camera_data()

# STEP 2: Estimate Features per Image
from modules.features import FeatureDetectionSP
# Feature Module Initialization
feature_detector = FeatureDetectionSP(cam_data=camera_data,
                                        max_keypoints=5000)

# Detect Features for all Images
features = feature_detector()

# STEP 3: Match Features Per Image 
# Ignore since pose estimation is using VGGT as the backbone structure

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
                                            RANSAC_threshold=0.3,
                                            setting="indoor")


# From estimated features, tracked features across multiple images
tracked_features = feature_tracker(features=features)

# STEP 6: Reconstruct Scene in Full
from modules.scenereconstruction import Sparse3DReconstructionVGGT
# Scene Reconstruction Module Initialization
sparse_reconstruction = Sparse3DReconstructionVGGT(cam_data=camera_data,
                                                   min_observe=3)

# Estimate sparse 3D scene from tracked features and camera poses
sparse_scene = sparse_reconstruction(tracked_features, cam_poses)

# STEP 7: Run Optimization Algorithm
from modules.optimization import BundleAdjustmentOptimizerGlobal
# Build Optimizer
optimizer = BundleAdjustmentOptimizerGlobal(max_num_iterations=60,
                                            cam_data=camera_data)

# Run Optimizer
optimal_scene = optimizer(sparse_scene)


# Optional Visualization
from modules.visualize import VisualizeScene
visualizer = VisualizeScene()

visualizer(optimal_scene)


