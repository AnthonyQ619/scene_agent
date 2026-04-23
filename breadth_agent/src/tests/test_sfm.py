from modules.utilities.utilities import CalibrationReader
from modules.features import FeatureDetectionSP
from modules.featurematching import FeatureMatchSuperGlueTracking, FeatureMatchSuperGluePair
# from modules.camerapose import CamPoseEstimatorEssentialToPnP
from modules.optimization import BundleAdjustmentOptimizerLocal
from modules.scenereconstruction import Sparse3DReconstructionMono
from modules.visualize import VisualizeScene
import os

# Construct Modules with Initialized Arguments
image_path = "/home/anthonyq/datasets/DTU/DTU/scan22"
calibration_path = "/home/anthonyq/datasets/DTU/DTU/calibration_DTU_new.npz"

# STEP 1: Read in Camera Data
from modules.cameramanager import CameraDataManager

CDM = CameraDataManager(image_path=image_path,
                        # max_images=10,
                        calibration_path=calibration_path)

# Get Camera Data
camera_data = CDM.get_camera_data()

# STEP 2: Estimate Features per Image
from modules.features import FeatureDetectionSIFT
# Feature Module Initialization
feature_detector = FeatureDetectionSIFT(cam_data=camera_data, 
                                        max_keypoints=10000,
                                        contrast_threshold=0.009,
                                        edge_threshold=20)

# Detect Features for all Images
features = feature_detector()

# STEP 3: Match Features Per Image 
from modules.featurematching import FeatureMatchLightGluePair
# Pairwise Feature Matching Module Initialization
feature_matcher = FeatureMatchLightGluePair(detector='sift', 
                                     cam_data=camera_data,
                                     RANSAC_threshold=0.02,
                                     RANSAC_conf=0.999)

# Detect Image Pair Correspondences for Pose Estimation
feature_pairs = feature_matcher(features=features)

# STEP 4: Estimate Camera Poses of Scene with Feature Pairs
from modules.camerapose import CamPoseEstimatorEssentialToPnP

# Camera Pose Module Initialization
pose_estimator = CamPoseEstimatorEssentialToPnP(cam_data=camera_data,
                                                reprojection_error=3.0,
                                                iteration_count=200,
                                                confidence=0.995)

# From estimated features, estimate the camera poses for all image frames
cam_poses = pose_estimator(feature_pairs=feature_pairs)

# STEP 5: Estimate Feature Tracks:
from modules.featurematching import FeatureMatchBFTracking
# Feature Tracking Module Initialization
feature_tracker = FeatureMatchBFTracking(cam_data=camera_data,
                                         detector="sift",
                                         RANSAC_threshold=0.008,
                                         lowes_thresh=0.65)

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
from modules.optimization import BundleAdjustmentOptimizerGlobal

# Build Optimizer
optimizer = BundleAdjustmentOptimizerGlobal(max_num_iterations=200,
                                            cam_data=camera_data,
                                            use_gpu=False)

# Run Optimizer
optimal_scene = optimizer(sparse_scene)

# Optional Visualization
from modules.visualize import VisualizeScene
visualizer = VisualizeScene(server=True, img_path = os.path.dirname(os.path.abspath(__file__)) + '/feature_images/point_cloud.png')

visualizer(optimal_scene)


## NEW SFM PIPELINE

# Step 1: Read in Calibration/Image Data
reconstructed_scene = Scene(image_path = image_path, 
                            calibration_path = calibration_path)

# Step 2: Detect Features
scene.FeatureDetectionSIFT(
    max_keypoints=9000,
    contrast_threshold=0.02,
    edge_threshold=20,
)

# Step 3: Detect Feature Matches/Correspondences
scene.FeatureMatchBFTracking(RANSAC_threshold=0.008,
                             lowes_thresh=0.65)
