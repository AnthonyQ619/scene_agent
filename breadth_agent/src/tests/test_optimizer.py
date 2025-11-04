from modules.utilities.utilities import CalibrationReader
from modules.features import FeatureDetectionSIFT
from modules.featurematching import FeatureMatchFlannTracking, FeatureMatchFlannPair
from modules.camerapose import CamPoseEstimatorEssentialToPnP
from modules.scenereconstruction import Sparse3DReconstructionMono
from modules.optimization import BundleAdjustmentOptimizer
from modules.visualize import VisualizeScene

# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\datasets\\Structure-from-Motion\\sfm_dataset"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\Structure-from-Motion\\calibration.npz"
bal_path = "C:\\Users\\Anthony\\Documents\\Projects\\scene_agent\\breadth_agent\\results\\scene_data\\bal_data.txt"

# Feature Module Initialization
calibration_data = CalibrationReader(calibration_path).get_calibration()
feature_detector = FeatureDetectionSIFT(image_path=image_path)

# Camera Pose Module Initialization
pose_estimator = CamPoseEstimatorEssentialToPnP(calibration=calibration_data, image_path=image_path)

# Pairwise Feature Matching Module Initialization
feature_matcher = FeatureMatchFlannPair(detector='sift', calibration=calibration_data)

# Feature Tracking Module Initialization
feature_tracker = FeatureMatchFlannTracking(detector='sift', calibration=calibration_data)

# Scene Reconstruction Module Initialization
sparse_reconstruction = Sparse3DReconstructionMono(calibration=calibration_data, image_path=image_path)

# Visualization
# visualizer = VisualizeScene()

# # Solution Pipeline 

# # Detect Features for all Images
# features = feature_detector()

# # Detect Image Pair Correspondences for Pose Estimation
# feature_pairs = feature_matcher(features=features)

# # From estimated features, estimate the camera poses for all image frames
# cam_poses = pose_estimator(features_pairs=feature_pairs)

# # From estimated features, tracked features across multiple images
# tracked_features = feature_tracker(features=features)

# # Estimate sparse 3D scene from tracked features and camera poses
# sparse_scene = sparse_reconstruction(tracked_features, cam_poses, view="multi")

# Establish Optimizer
optimizer = BundleAdjustmentOptimizer()#(scene=sparse_scene, calibration=calibration_data)
optimizer.prep_optimizer(ratio_known_cameras=0.0,
                         max_iterations=2,
                         num_epochs=1)

opt_scene = optimizer(bal_path)

visualizer = VisualizeScene()
print(len(opt_scene.cam_poses))
print(opt_scene.points3D.points3D.shape)
print(opt_scene.points3D.points3D)
visualizer(opt_scene)