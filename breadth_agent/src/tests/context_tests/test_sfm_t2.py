"""
This script solves the problem of creating a sparse 3D representation of a scene featuring an object as the central point 
using Structure-from-Motion as the algorithm solution. Using a monocular camera that is calibrated, the scene captured is
Indoors of a tabletop/studio setup with a model house placed on bricks over a white surface. There are illumination changes
the first image is darker with stronger shadows; subsequent images are brighter and more evenly lit. Then back to dark.
Lighting direction is consistent and mostly diffuse with a few specular highlights. Predominantly high-textured (brick surfaces, 
roof shingles, façade details). Some low-texture areas exist in the plain white background. The scene centers on a small 
model building resting on painted, rough bricks. The object of interest (the model house) is present in all images. 
The camera appears to move/rotate around the object, maintaining focus on the same subject. The object and bricks provide 
rich texture; lighting is largely diffuse with minor highlights. Gradual viewpoint changes with moderate rotation and 
small translations around the object; no extreme viewpoint jumps.


Scene Summary: Indoor tabletop/studio scene featuring a small model house resting on painted, rough bricks over a white 
background. Surfaces of interest (bricks, shingles, façade) are highly textured; the background is largely textureless. 
Lighting is mostly diffuse with moderate inter-image exposure differences and minor specular highlights. 
The camera moves smoothly around the object with moderate rotations and small translations, keeping the model house centered 
and consistently visible.
"""

# ==#$#==

# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\scan6_illumination_change"

# STEP 1: Read in Camera Data
from modules.cameramanager import CameraDataManager

CDM = CameraDataManager(image_path=image_path,
                        #calibration_path=calibration_path,
                        target_resolution=[1024, 1024])
# Any image pre-processing steps are ran here
# ...
# Get Camera Data
camera_data = CDM.get_camera_data()

# STEP 2: Estimate Features per Image
from modules.features import FeatureDetectionSIFT
# Feature Module Initialization
feature_detector = FeatureDetectionSIFT(cam_data=camera_data)

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
from modules.featurematching import FeatureMatchFlannTracking
# Feature Tracking Module Initialization
feature_tracker = FeatureMatchFlannTracking(detector='sift', 
                                            cam_data=camera_data,
                                            RANSAC_threshold=0.2)

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
from modules.optimization import BundleAdjustmentOptimizerLeastSquares
# # Build Optimizer
optimizer = BundleAdjustmentOptimizerLeastSquares(cam_data=camera_data,
                                                  max_iterations=30, 
                                                  num_epochs=1, 
                                                  step_size=0.1,
                                                  optimizer_cls="LevenbergMarquardt")

# Run Optimizer
optimal_scene = optimizer(scene=sparse_scene)


# Optional Visualization
from modules.visualize import VisualizeScene
visualizer = VisualizeScene()

visualizer(optimal_scene)