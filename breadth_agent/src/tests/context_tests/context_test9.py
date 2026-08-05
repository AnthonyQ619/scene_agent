"""
GOAL RECONSTRUCTION: Camera Pose Reconstruction

STEP 1: Initialize the scene through the object SfMScene and read in Camera data
- Initialize the scene as SfMScene(...)
- Set the image path to the provided directory of images to be read, resize, and pre-process images for reconstruction
    - image_path = /work/dataset/ETH/living_room/images/dslr_images_undistorted
- Set max_images=20 (Never above 40), we want to only use the first 20 images when evaluating the feasibility of the workflow.
- Since no calibration path is provided, we ignore this parameter (This is a strong indication to utilize a tool that does calibration simultaneously)
- Since no calibration calls for the VGGT pipeline, resize images into a square format for better VGGT optimization, and convert the size to (1024,1024)

STEP 2: Do not Detect Features
- Reasoning: Since features can be detected, but have trouble building good correspondences or tracks, it's better to approach this problem detector free instead
  and entirely forgo feature detection for the camera pose estimation!

STEP 3: Do Not Detect Feature Matches 
- Reasoning: Since we are using the VGGT pipeline, we don't need feature points to be matcher pairwise as VGGT estimates pose with images directly.

STEP 4: Estimate the camera pose using image data directly
- Estimate the camera pose using using the VGGT pipeline as we have no camera calibration and the scene having consistent changes to illumination will lead 
  to errors when taking the classical approach. Thus, we opt for the VGGT pose estimation module
  - No Parameters needed as all data is stored in the SfMScene object
  - Don't need to do global optimization to store camera poses, VGGT pose module stores the estimated poses directly so we can utilize them for our needs!
"""

# ==#$#==

# Construct Modules with Initialized Arguments
image_path = "/home/anthonyq/datasets/ETH/ETH/office/images/dslr_images_undistorted" #"/home/anthonyq/datasets/DTU/scan6_illumination_change" #"C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\scan6_illumination_change"

from sfmcore.camerapose import CamPoseEstimatorVGGTModel
from sfmcore.optimization import BundleAdjustmentOptimizerGlobal
from sfmcore.baseclass import SfMScene

# Step 1: Read in Calibration/Image Data
reconstructed_scene = SfMScene(id=9,
                                gpu_num="5",
                                log_dir="/home/anthonyq/projects/scene_agent/breadth_agent/results/ETH/office",
                                image_path=image_path,
                                max_images=25,
                                target_resolution=[1024, 1024]
)

# Step 2: Detect Features
# Ignore since Pose estimation from VGGT doesn't utilize features (good in scenes that can't detect enough features or have large baselines)

# Step 3: Detect Feature Pairs
# Ignore since pose estimation is using VGGT as the backbone structure

# Step 4: Detect/Estimate Camera Poses
reconstructed_scene.CamPoseEstimatorVGGTModel()