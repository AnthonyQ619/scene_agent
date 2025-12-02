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

# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\scan6_normal_lighting"
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
                                        max_keypoints=5000)

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

# STEP 4: Estimate Camera Poses of Scene with Feature Pairs
from modules.camerapose import CamPoseEstimatorEssentialToPnP
# Camera Pose Module Initialization
pose_estimator = CamPoseEstimatorEssentialToPnP(cam_data=camera_data,
                                                reprojection_error=4.0,
                                                iteration_count=340,
                                                confidence=0.995)

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
                                                   min_angle=2.0,
                                                   view="multi")

# Estimate sparse 3D scene from tracked features and camera poses
sparse_scene = sparse_reconstruction(tracked_features, cam_poses)

# STEP 7: Run Optimization Algorithm
from modules.optimization import BundleAdjustmentOptimizerLeastSquares
# # Build Optimizer
optimizer = BundleAdjustmentOptimizerLeastSquares(cam_data=camera_data,
                                                  max_iterations=10, 
                                                  num_epochs=1, 
                                                  step_size=0.1,
                                                  optimizer_cls="GaussNewton")

# Run Optimizer
optimal_scene = optimizer(scene=sparse_scene)


# Optional Visualization
from modules.visualize import VisualizeScene
visualizer = VisualizeScene()

visualizer(sparse_scene)