"""
GOAL RECONSTRUCTION: Sparse Reconstruction

STEP 1: Initialize the scene through the object SfMScene and read in Camera data
- Initialize the scene as SfMScene(...)
- Set the image path to the provided directory of images to be read, resize, and pre-process images for reconstruction
    - image_path = /work/dataset/ETH/living_room/images/dslr_images_undistorted
- Set max_images=25 (Never above 40), we want to only use the first 25 images when evaluating the feasibility of the workflow.
- Since no calibration path is provided, we ignore this parameter (This is a strong indication to utilize a tool that does calibration simultaneously)
- Since no calibration calls for the VGGT pipeline, resize images into a square format for better VGGT optimization, and convert the size to (1024,1024)

STEP 2: Do Not Detect Feature
- Reasoning: This specific scene has very large view point changes and low/inconsistent lighting, so features are inaccurate, which will lead to any 
  matching correspondences or tracks to be very inaccuarte. So we opt for a detector free pipeline in this scenario!

STEP 3: Do Not Detect Feature Matches 
- Reasoning: Since we are using the VGGT pipeline, we don't need feature points to be matcher pairwise as VGGT estimates pose with images directly.

STEP 4: Estimate the camera pose using image data directly
- Estimate the camera pose using using the VGGT pipeline as we have no camera calibration and the scene having consistent changes to illumination will lead 
  to errors when taking the classical approach. Thus, we opt for the VGGT pose estimation module
  - No Parameters needed as all data is stored in the SfMScene object

STEP 5: Track Features across multiple images to create feature tracks for global pose optimization
- Reasoning: This specific scene has many feature points, but not abundant or accurate enough for traditional feature matching to apply union
  find tracking to enable multi-view tracking extensively. Utilize an ML feature tracker on the found feature points to generate robust feature tracks.
  In this case, use the FeatureTrackingVGGSfM module.

STEP 6: Reconstruct the Scene now that we have the estimated Camera Poses with corresponding feature tracks. We will use the 
Sparse3DReconstructionVGGT Module in this scenario
- Reconstruct the Scene now that we have the estimated Camera Poses:
    - Since VGGT estimates 3D points just using the camera pose, we can avoid detecting features, and utilize the 3D points to enable VGGTSfM Feature Tracking 
    to garner more accurate tracks in this case!

STEP 7: Apply Global Bundle Adjustment to the scene for optimal reconstruction
- To ensure scene is geometrically correct, we want to set 
    - Set max_num_iterations=130
      - Reasoning: Since initial 2D points will not be as accurate due to leniency in 3D point to point map matching from VGGT calculation
      in previous methods despite superpoint being more robust to illumination changing environments.
"""

# ==#$#==

from sfmcore.features import FeatureDetectionSP
from sfmcore.featurematching import FeatureMatchSuperGluePair
from sfmcore.camerapose import CamPoseEstimatorVGGTModel
from sfmcore.featuretracking import FeatureTrackingVGGSfM
from sfmcore.scenereconstruction import Sparse3DReconstructionVGGT, Dense3DReconstructionVGGT
from sfmcore.optimization import BundleAdjustmentOptimizerGlobal
from sfmcore.baseclass import SfMScene


# Construct Modules with Initialized Arguments
image_path = "/home/anthonyq/datasets/ETH/ETH/office/images/dslr_images_undistorted"
calibration_path = "/home/anthonyq/datasets/ETH/ETH/office/dslr_calibration_undistorted/calibration_ETH_new.npz"

# Step 1: Read in Calibration/Image Data (VGGT requires square resizing)
reconstructed_scene = SfMScene(
    id=3,
    gpu_num="5",
    log_dir="/home/anthonyq/projects/scene_agent/breadth_agent/results/ETH/",
    image_path=image_path,
    max_images=20,
    calibration_path=calibration_path,
    target_resolution=[1024, 1024]
)

# Step 2: Detect Features
reconstructed_scene.FeatureDetectionSP(
    max_keypoints=6000
)

# Step 3: Detect Feature Pairs
# reconstructed_scene.FeatureMatchSuperGluePair(
#     detector='superpoint',
#     RANSAC_threshold=3.0
# )

# Step 4: Detect/Estimate Camera Poses (VGGT)
reconstructed_scene.CamPoseEstimatorVGGTModel()

# Step 5: Detect Feature Tracks
reconstructed_scene.FeatureTrackingVGGSfM()

# Step 6: Estimate Sparse Reconstruction (VGGT No-Features)
reconstructed_scene.Sparse3DReconstructionVGGT()

# Step 7: Run Global Optimization
reconstructed_scene.BundleAdjustmentOptimizerGlobal(
    refine_focal_length=False,
    refine_principal_point=False,
    refine_extra_params=False,
    max_num_iterations=200,
    # robust_loss=True
)

