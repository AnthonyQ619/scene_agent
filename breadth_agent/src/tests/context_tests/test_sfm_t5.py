"""
This script solves the problem of creating a sparse 3D representation of an Indoors (tabletop/studio shot of a miniature street scene).
It is captured on a monocular camera that is calibrated. Consistent, diffuse studio lighting with no significant illumination changes 
across the set of images. The model buildings (brick walls, roof shingles, windows, awnings) are highly textured, while the white 
background areas are low texture. Images show a scale-model block of buildings viewed from an oblique top‑down angle. The corner with 
red and gray roofs and yellow shop awnings stays present in all frames. Lighting is soft/diffuse, minimal specular highlights. 
The subject is richly textured (brick, shingles, facades) against a plain background. Camera motion is gradual between frames—small 
rotations and slight translations around the building corner rather than extreme viewpoint changes.


Scene Summary: A controlled indoor tabletop/studio scene featuring a miniature block of buildings with richly textured façades, brickwork, 
shingles, windows, and yellow awnings. The camera views the subject from oblique, slightly top‑down angles with small rotational and 
translational changes around a persistent building corner. Lighting is soft and uniform, producing stable appearances without specular 
artifacts. The subject is set against a largely white, low‑texture background, ensuring most reliable features originate on the model itself.
"""

# ==#$#==

# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\scan14_normal_lighting"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\calibration_DTU_new.npz"


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
                                        max_keypoints=6000,
                                        contrast_threshold=0.02,
                                        edge_threshold=50)

# Detect Features for all Images
features = feature_detector()

# STEP 3: Match Features Per Image 
from modules.featurematching import FeatureMatchFlannPair
# Pairwise Feature Matching Module Initialization
feature_matcher = FeatureMatchFlannPair(detector='sift', 
                                        cam_data=camera_data,
                                        RANSAC_threshold=0.3,
                                        lowes_thresh=0.7)

# Detect Image Pair Correspondences for Pose Estimation
feature_pairs = feature_matcher(features=features)

# STEP 4: Estimate Camera Poses of Scene with Feature Pairs
from modules.camerapose import CamPoseEstimatorEssentialToPnP
# Camera Pose Module Initialization
pose_estimator = CamPoseEstimatorEssentialToPnP(cam_data=camera_data,
                                                reprojection_error=4.0,
                                                iteration_count=200,
                                                confidence=0.995)

# From estimated features, estimate the camera poses for all image frames
cam_poses = pose_estimator(feature_pairs=feature_pairs)

# STEP 5: Estimate Feature Tracks:
from modules.featurematching import FeatureMatchFlannTracking
# Feature Tracking Module Initialization
feature_tracker = FeatureMatchFlannTracking(detector='sift', 
                                            cam_data=camera_data,
                                            RANSAC_threshold=0.3)

# From estimated features, tracked features across multiple images
tracked_features = feature_tracker(features=features)

# STEP 6: Reconstruct Scene in Full
from modules.scenereconstruction import Sparse3DReconstructionMono
# Scene Reconstruction Module Initialization
sparse_reconstruction = Sparse3DReconstructionMono(cam_data=camera_data,
                                                   min_observe=4,
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