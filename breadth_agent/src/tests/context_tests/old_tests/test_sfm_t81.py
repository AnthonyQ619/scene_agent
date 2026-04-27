"""
This script solves the problem of creating a sparse 3D representation of a scene featuring an indoor environment
using Structure-from-Motion as the algorithm solution. Using a monocular camera that is calibrated, the scene captures an Indoors, 
tabletop/studio scene with a miniature building placed on bricks against a white background. Consistent, even diffuse lighting 
across all three images; only minor shading changes from viewpoint/object rotation, no significant illumination change. Mostly 
high-texture elements (gritty brick surfaces, roof shingles, façade details). Some low-texture regions are present (smooth painted 
walls and the white background). A small model building is the persistent object of interest in all frames, resting on textured bricks. 
The sequence appears to orbit/rotate around the model (or the model rotates on a turntable). The object is fairly highly textured, and 
lighting is diffuse with soft shadows, aiding feature visibility. Viewpoint changes are incremental and moderate—small translations and 
rotations/orbit around the object, with no extreme perspective or scale changes.


Scene Summary: Indoor tabletop/studio setting with a miniature building resting on textured bricks against a plain white background. 
Lighting is uniform and diffuse with soft shadows, producing consistent appearance across views. The building and bricks provide rich, 
fine-scale texture; the white background is largely textureless. The viewpoints change moderately, effectively orbiting the object 
(or the object rotates), yielding small to moderate parallax without extreme perspective changes.
"""

# ==#$#==

"""
STEP 1: Read in Camera data
- Set the image path to the provided directory of images to the CameraDataManager to read, resize, and pre-process images for reconstruction
- Set the calibration path to the provided calibration file if one is provided. Since it is provided, we have a calibrated camera and activate the parameter
- Finally, call get_camera_data() function to have access to the camera data of images and calibration

STEP 2: 
- Select the SIFT module since the scene featuring the object is well lit, in an indoor setting, and the object is of a brick house that is well textured for using
SIFT reliably.
    - Since the object is well-textured and in an indoor setting, we simply set the max_features parameter to 5000 as this detects enough features for 

"""

# ==#$#==

# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\datasets\\Structure-from-Motion\\sfm_dataset"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\Structure-from-Motion\\calibration_new.npz"

# STEP 1: Read in Camera Data
from modules.cameramanager import CameraDataManager

CDM = CameraDataManager(image_path=image_path,
                        calibration_path=calibration_path)

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
from modules.featurematching import FeatureMatchSuperGluePair
# Pairwise Feature Matching Module Initialization
feature_matcher = FeatureMatchSuperGluePair(cam_data=camera_data,
                                            RANSAC_threshold=0.03,
                                            detector='superpoint',
                                            setting="outdoor")

# Detect Image Pair Correspondences for Pose Estimation
feature_pairs = feature_matcher(features=features)

# STEP 4: Estimate Camera Poses of Scene with Feature Pairs with Local Bundle Adjustment Activated
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
from modules.featurematching import FeatureMatchSuperGlueTracking
# Feature Tracking Module Initialization
feature_tracker = FeatureMatchSuperGlueTracking(detector='superpoint', 
                                            cam_data=camera_data,
                                            RANSAC_threshold=0.4,
                                            setting="outdoor")


# From estimated features, tracked features across multiple images
tracked_features = feature_tracker(features=features)

# STEP 6: Reconstruct Scene in Full
from modules.scenereconstruction import Sparse3DReconstructionMono
# Scene Reconstruction Module Initialization
sparse_reconstruction = Sparse3DReconstructionMono(cam_data=camera_data,
                                                   min_observe=3,
                                                   min_angle=2.0,
                                                   view="multi")

# Estimate sparse 3D scene from tracked features and camera poses
sparse_scene = sparse_reconstruction(tracked_features, cam_poses)

# STEP 7: Run Optimization Algorithm
from modules.optimization import BundleAdjustmentOptimizerGlobal

# Build Optimizer
optimizer = BundleAdjustmentOptimizerGlobal(max_num_iterations=45,
                                            cam_data=camera_data)

# Run Optimizer
optimal_scene = optimizer(sparse_scene)

# Optional Visualization
from modules.visualize import VisualizeScene
visualizer = VisualizeScene()

visualizer(optimal_scene)