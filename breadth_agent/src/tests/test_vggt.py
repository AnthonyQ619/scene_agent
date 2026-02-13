# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\ETH\\door_dslr_undistorted\\door\\images\\dslr_images_undistorted"

# STEP 1: Read in Camera Data
from modules.cameramanager import CameraDataManager

CDM = CameraDataManager(image_path=image_path,
                        target_resolution=[1024, 1024])

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
# Ignore since pose estimation is using VGGT as the backbone structure

# STEP 4: Estimate Camera Poses of Scene with Feature Pairs
from modules.camerapose import CamPoseEstimatorVGGTModel
# Camera Pose Module Initialization
pose_estimator = CamPoseEstimatorVGGTModel(cam_data=camera_data)

# From estimated features, estimate the camera poses for all image frames
cam_poses = pose_estimator()

# STEP 5: Estimate Feature Tracks:
from modules.featurematching import FeatureMatchSuperGlueTracking
# Feature Tracking Module Initialization
feature_tracker = FeatureMatchSuperGlueTracking(detector='superpoint', 
                                            cam_data=camera_data,
                                            RANSAC_threshold=0.3,
                                            setting="indoor")


# From estimated features, tracked features across multiple images
tracked_features = feature_tracker(features=features)

# STEP 6: Reconstruct Scene in Full
from modules.scenereconstruction import Sparse3DReconstructionVGGT
# Scene Reconstruction Module Initialization
sparse_reconstruction = Sparse3DReconstructionVGGT(cam_data=camera_data,
                                                   min_observe=3)

# Estimate sparse 3D scene from tracked features and camera poses
sparse_scene = sparse_reconstruction(tracked_features, cam_poses)

# STEP 7: Run Optimization Algorithm
from modules.optimization import BundleAdjustmentOptimizerGlobal
# Build Optimizer
optimizer = BundleAdjustmentOptimizerGlobal(max_num_iterations=60,
                                            cam_data=camera_data)

# Run Optimizer
optimal_scene = optimizer(sparse_scene)


# Optional Visualization
from modules.visualize import VisualizeScene
visualizer = VisualizeScene()

visualizer(optimal_scene)