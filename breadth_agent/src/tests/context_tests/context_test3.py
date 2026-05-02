"""
GOAL RECONSTRUCTION: Sparse Reconstruction

STEP 1: Initialize the scene through the object SfMScene and read in Camera data
- Initialize the scene as SfMScene(...)
- Set the image path to the provided directory of images to be read, resize, and pre-process images for reconstruction
    - image_path = C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\scan6_illumination_change
- Set max_images=20 (Never above 40), we want to only use the first 20 images when evaluating the feasibility of the workflow.
- Since no calibration path is provided, we ignore this parameter (This is a strong indication to utilize a tool that does calibration simultaneously)
- Since no calibration calls for the VGGT pipeline, resize images into a square format for better VGGT optimization, and convert the size to (1024,1024)

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
  - No Parameters needed as all data is stored in the SfMScene object

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
    - Set max_num_iterations=60
      - Reasoning: Since initial 2D points will not be as accurate due to leniency in 3D point to point map matching from VGGT calculation
      in previous methods despite superpoint being more robust to illumination changing environments.
    - Set GPU = False,
      - Reasoning: Global Bundle Adjustment does not need GPU to run effeciently.
"""

# ==#$#==

# Construct Modules with Initialized Arguments
image_path = "/home/anthonyq/datasets/ETH/ETH/living_room/images/dslr_images_undistorted" #"/home/anthonyq/datasets/DTU/scan6_illumination_change" #"C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\scan6_illumination_change"

from modules.features import FeatureDetectionSP
from modules.featurematching import FeatureMatchSuperGlueTracking
from modules.camerapose import CamPoseEstimatorVGGTModel
from modules.scenereconstruction import Sparse3DReconstructionVGGT, Dense3DReconstructionVGGT
from modules.optimization import BundleAdjustmentOptimizerGlobal
from modules.baseclass import SfMScene

# Step 1: Read in Calibration/Image Data
reconstructed_scene = SfMScene(id=3,
                              log_dir="/home/anthonyq/projects/scene_agent/breadth_agent/results/ETH/eth_living_room",
                                image_path=image_path,
                                max_images=25,
                                target_resolution=[1024, 1024]
)

# Step 2: Detect Features
reconstructed_scene.FeatureDetectionSP(max_keypoints=6000)

# Step 3: Detect Feature Pairs
# Ignore since pose estimation is using VGGT as the backbone structure

# Step 4: Detect/Estimate Camera Poses
reconstructed_scene.CamPoseEstimatorVGGTModel()

# Step 5: Detect Feature Tracks
reconstructed_scene.FeatureMatchLightGlueTracking(
    detector='superpoint', 
    RANSAC_threshold=2.0,
    # setting="indoor"
)

# Step 6: Estimate Sparse Reconstruction
reconstructed_scene.Sparse3DReconstructionVGGT(min_observe=4)

# Step 7: Run Optimization
reconstructed_scene.BundleAdjustmentOptimizerGlobal(
    max_num_iterations=130,
)
