from modules.utilities.utilities import CalibrationReader
from modules.features import FeatureDetectionSIFT
from modules.featurematching import FeatureMatchFlannTracking
from modules.camerapose import CamPoseEstimatorEssentialToPnP
from modules.scenereconstruction import Sparse3DReconstruction
from modules.visualize import VisualizeScene

# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\datasets\\Structure-from-Motion\\sfm_dataset"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\Structure-from-Motion\\calibration.npz"

# Feature Module Initialization
calibration_data = CalibrationReader(calibration_path).get_calibration()
feature_detector = FeatureDetectionSIFT(image_path=image_path)

# Camera Pose Module Initialization
pose_estimator = CamPoseEstimatorEssentialToPnP(calibration=calibration_data, image_path=image_path, detector="sift")

# Feature Tracking Module Initialization
feature_tracker = FeatureMatchFlannTracking(detector='sift')

# Scene Reconstruction Module Initialization
sparse_reconstruction = Sparse3DReconstruction(calibration=calibration_data, image_path=image_path)

# Visualization
visualizer = VisualizeScene()

# Solution Pipeline 

# Detect Features for all Images
features = feature_detector()

# From estimated features, estimate the camera poses for all image frames
cam_poses = pose_estimator(features=features)

# From estimated features, tracked features across multiple images
tracked_features = feature_tracker(features=features)

# Estimate sparse 3D scene from tracked features and camera poses
sparse_scene = sparse_reconstruction(tracked_features, cam_poses, view="multi")

visualizer(sparse_scene)


