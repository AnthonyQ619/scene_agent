# Construct Modules with Initialized Arguments
image_path = "/home/anthonyq/datasets/ETH/ETH/office/images/dslr_images_undistorted" #"/home/anthonyq/datasets/DTU/scan6_illumination_change" #"C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\scan6_illumination_change"

from modules.featurematching import FeatureMatchSuperGlueTracking
from modules.camerapose import CamPoseEstimatorVGGTModel
from modules.scenereconstruction import Sparse3DReconstructionVGGT, Dense3DReconstructionVGGT
from modules.optimization import BundleAdjustmentOptimizerGlobal
from modules.baseclass import SfMScene

# Step 1: Read in Calibration/Image Data
reconstructed_scene = SfMScene(id=9,
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