"""
GOAL RECONSTRUCTION: Sparse Reconstruction

STEP 1: Read in Camera data
- Set the image path to the provided directory of images to the CameraDataManager to read, resize, and pre-process images for reconstruction
    - image_path = C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\scan6_illumination_change
- Set max_images=20 (Never above 40), we want to only use the first 20 images when evaluating the feasibility of the workflow.
- Since no calibration path is provided, we ignore this parameter (This is a strong indication to utilize a tool that does calibration simultaneously)
- Since no calibration calls for the VGGT pipeline, resize images into a square format for better VGGT optimization, and convert the size to (1024,1024)
- Finally, call get_camera_data() function to have access to the camera data of images and calibration

STEP 2: Detect Features
- Select the SuerPoint module since the scene featuring the object is causing illumination changes across the set of images, and even though it is in an 
  indoor setting, and the object is of a brick house that is well textured, the illumination changes causes weak correspondence in traditional detectors.
  Thus, we opt for SuperPoint, and ML detector robust to illumination or dull lighting enviornments.
    - Set max_keypoints = 5000
        - Reasoning: to ensure we detect as much features as possible from SuperPoint. SuperPoint does not 

STEP 3: Do Not Detect Feature Matches 
- Reasoning: Since we are using the VGGT pipeline, we don't need feature points to be matcher pairwise as VGGT estimates pose with images directly.

STEP 4: Estimate the camera pose using image data directly
- Estimate the camera pose using using the VGGT pipeline as we have no camera calibration and the scene having consistent changes to illumination will lead 
  to errors when taking the classical approach. Thus, we opt for the VGGT pose estimation module
    - Set the cam_data parameter to the camera_data variable holding our pre-processed image set.

STEP 5: Track Features across multiple images to create feature tracks for 3D point estimation
- Track Features across multiple images using the SuperGlue Feature Tracking tool. Since the images of the scene have illumination changes, we need a detector
  more robust to lighting changes, and in this case it is either superglue or lightglue. We opt for SuperGlue for more accurate matches in tracking points.
    - Set RANSAC_threshold = 0.3 (Default 0.3)
        - Reasoning: Default setting as we want to build longer feature tracks for more accurate estimate of 3D points optimization in the scene
    - Set detector = "superpoint" as that is the detector we are using.
    - Set setting = "Indoor" to utilize the indoor weights of the model ("Outdoor" is for outdoor scenes)
        - Reasoning: Scene is in an indoor setting, so we use the weights of the model trained for indoor environments.

STEP 6: Reconstruct the Scene now that we have the estimated Camera Poses and Tracked features for multi-view estimation
- Reconstruct the Scene now that we have the estimated Camera Poses and Tracked features for multi-view estimation
    - Set min_observe=4
        - Reasoning: Since we are using SuperPoint with SuperGlue, we will have longer tracks with decent accuracy as the feature correspondences are more robust 
          to lighting changes due to using SuperPoint instead of classical detectors.
    - Since VGGT estimates 3D points just using the camera pose, we don't need other features besides image data from cam_data

STEP 7: Apply Global Bundle Adjustment to the scene for optimal reconstruction
- To ensure scene is geometrically correct, we want to set 
    - Set max_num_iterations=100
      - Reasoning: Since initial 2D points will not be as accurate due to leniency in 3D point to point map matching from VGGT calculation
      in previous methods despite superpoint being more robust to illumination changing environments.
"""

# ==#$#==

# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\scan6_illumination_change"

# STEP 1: Read in Camera Data
from modules.cameramanager import CameraDataManager

CDM = CameraDataManager(image_path=image_path,
                        max_images=20,
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
                                            RANSAC_threshold=0.01,
                                            setting="indoor")


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


