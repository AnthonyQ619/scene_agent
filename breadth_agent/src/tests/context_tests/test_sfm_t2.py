"""
GOAL RECONSTRUCTION: Sparse Reconstruction

STEP 1: Read in Camera data
- Set the image path to the provided directory of images to the CameraDataManager to read, resize, and pre-process images for reconstruction
  - image_path = C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\scan21_normal_lighting
- Set max_images=20 (Never above 40), we want to only use the first 20 images when evaluating the feasibility of the workflow.
- Set the calibration path to the provided calibration file if one is provided. Since it is provided, we have a calibrated camera and 
  activate the parameter
  - calibration_path = C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\calibration_DTU_new.npz
- Finally, call get_camera_data() function to have access to the camera data of images and calibration

STEP 2: Detect Features in the scene
- Select the ORB module since the scene featuring the object is well lit, in an indoor setting, and the object is of a brick house that
  contains many corner and edge points through a large amount of windows on each of the buildings.
    - Set max_keypoints = 8000 (Default 1024)
      - Reasoning: to ensure we detect as much features as possible since we also want to detect many corners and edges due to the given scene
        object having many set shapes (house) that involves corners and edges.
    - Set set_nms = True (Default False)
      - Reasoning: Since object will contain many corners and edges in clusters, we want to ensure sparsity for proper Pose and Track estimation.
    - Set set_nms_allowed_points = 4000 (Default 3000, ensure this value is at most half of max_keypoints setting)
      - Reasoning: Ideally we want to detect 4000-6000 points with ORB since many points beyond this threshold will be noisy.
    - Since the scene is very desirable for ORB usage, an object with many corner and edges while also being well-lit, we don't need to tune 
      any of the other parameters. Main goal is that max points is equivalent to the detected features. 

STEP 3: Detect Feature Matches
- Select the Brute Force feature matcher to utilize as a robust matcher for ORB, since we don't have complex descriptors, we can get away with
a more approximate approach of flann even if it grants more matches.
    - Set RANSAC_threshold to 0.01
      - Reasoning: Even though the scene is captured is well suited for ORB, detected features are still noisy, so we need a much more strict outlier rejection 
        since we have an abundance of points.
    - Set lowes_threshold to 0.6 (Default 0.75) 
      - Reasoning: Since features are still noisy, and ransac is not completely robust to outliers, we utilize a strict lowes threshold to ensure
        most points we keep are inliers with the utmost confidence.

STEP 4: Estimate the camera pose using detected matching feature pairs.
- Opt to use the Camera Pose Estimation module EssentialToPnP since we have detected features and calibration for the classical approach
    - Since we are using a more inaccurate feature detector (One of SuperPoint or ORB), we must add an extra layer of optimization
    - Initialize the Local Optimization module "BundleAdjustmentOptimizerLocal"
        - Set iteration count for Bundle Adjustment to only 20
        - Set window size to 8 (Default) since camera movement is minimal and better pose correction.
        - Reasoning: We don't need to converge here, just reduce pose error to below 0.5 pixel reprojection error to ensure inital build allows 
          global bundle adjustment to converge
    - Set reprojection_error = 3.0
      - Reasoning: since we are including corner points as well to ensure as many 3D points are included as possible for a 
        and more accurate bundle adjustment
    - Set iteration_count=300
      - Reasoning: higher setting for iteration count of the Levenberg-Marquardt algorithm during pose estimation. Since ORB is 
        inherently inaccurate compared to other detectors, the increased iteration count for optimization will improve initial poses for 
        a well-lit and textured scene for optimal initial pose estimation
    - Set confidence=0.99 (Default) 
      - Reasoning: Since the scene is ORB points are inherently noisy, we leave the default setting here. 

STEP 5: Track Features across multiple images to create feature tracks for 3D point estimation
-  For sparse reconstruction, we want as many 3D points as possible with as long possible track lengths. The more accurate matcher here is the Brute-Force 
   Tracking tool, so use the BFTracking module since it's more accurate and robust to trickling outliers.
    - Set RANSAC_threshold = 0.02 (Default 3.0, Points are normalized, so we base reprojection threshold on normalized coordinates)
      - Reasoning: Although the scene is well-lit and good for ORB features, we want to be aggressive in outlier removal since we are completing
        feature tracks, thus we opt for a 0.02 ransac threshold, but more relaxed than pairwise matching. 
    - Set lowes_thresh = 0.8 (Default 0.75)
      - Reasoning: We want to be more strict on the feature tracks for more accurate estimate of 3D points in the scene
        we are more aggressive in lowes threshold to ensure the removal of most outliers, even if we remove inliers in the process.

STEP 6: Reconstruct the Scene now that we have the estimated Camera Poses and Tracked features for multi-view estimation
- We are usng the Sparse Reconstruction module, specifically the classical approach with a monocular camera, so opt for the Sparse3DReconstruction module
    - Set min_observe=3 (Default 3)
      - Reasoning: ORB will naturally make it more difficult for longer tracks, even if they are accurate, we will default for the minimum for this case.
    - Set min_angle=3.0 (Default 1.0) 
      - Reasoning: utilize a larger minimum angle to ensure 3D point estimates are more acccurate, and since we have longer tracks, this is very feasible for this scene.
    - Set view="multi"
      - Reasoning: we are tracking features, so the estimation method is multi-view (Which is most, if not all, cases for reconstruction)

STEP 7: Apply Global Bundle Adjustment to the scene for optimal reconstruction
- To ensure scene is geometrically correct, we want to set 
    - Set max_num_iterations=200, 
      - Reasoning: Since initial 2D points will not be as accurate due to inaccuracy of certain matches and tracks in previous methods despite
        optimal scene lighting and texture being for ORB features, we have to set interations to 200 to ensure convergence of scene for optimal reconstruction.
      
"""

# ==#$#==

# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\scan21_normal_lighting"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\calibration_DTU_new.npz"

# STEP 1: Read in Camera Data
from modules.cameramanager import CameraDataManager

CDM = CameraDataManager(image_path=image_path,
                        max_images=20,
                        calibration_path=calibration_path)

# Get Camera Data
camera_data = CDM.get_camera_data()

# STEP 2: Estimate Features per Image
from modules.features import FeatureDetectionORB
# Feature Module Initialization
feature_detector = FeatureDetectionORB(cam_data=camera_data, 
                                        max_keypoints=8000,
                                        set_nms=True,
                                        set_nms_allowed_points=4000)

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

# STEP 4: Estimate Camera Poses of Scene with Feature Pairs with Local Bundle Adjustment Activated
from modules.optimization import BundleAdjustmentOptimizerLocal

# Build Optimizer
optimizer_local = BundleAdjustmentOptimizerLocal(max_num_iterations=20,
                                                 window_size=8,
                                                 cam_data=camera_data)

from modules.camerapose import CamPoseEstimatorEssentialToPnP
# Camera Pose Module Initialization
pose_estimator = CamPoseEstimatorEssentialToPnP(cam_data=camera_data,
                                                reprojection_error=3.0,
                                                iteration_count=300,
                                                confidence=0.99,
                                                optimizer=optimizer_local)

# From estimated features, estimate the camera poses for all image frames
cam_poses = pose_estimator(feature_pairs=feature_pairs)

# STEP 5: Estimate Feature Tracks:
from modules.featurematching import FeatureMatchBFTracking
# Feature Tracking Module Initialization
feature_tracker = FeatureMatchBFTracking(detector='orb', 
                                         cam_data=camera_data,
                                         RANSAC_threshold=0.02,
                                         lowes_thresh=0.8,
                                         cross_check=False)

# From estimated features, tracked features across multiple images
tracked_features = feature_tracker(features=features)

# STEP 6: Reconstruct Scene in Full
from modules.scenereconstruction import Sparse3DReconstructionMono
# Scene Reconstruction Module Initialization
sparse_reconstruction = Sparse3DReconstructionMono(cam_data=camera_data,
                                                   min_observe=3,
                                                   min_angle=3.0,
                                                   view="multi")

# Estimate sparse 3D scene from tracked features and camera poses
sparse_scene = sparse_reconstruction(tracked_features, cam_poses)

# STEP 7: Run Optimization Algorithm
from modules.optimization import BundleAdjustmentOptimizerGlobal

# Build Optimizer
optimizer = BundleAdjustmentOptimizerGlobal(max_num_iterations=200,
                                            cam_data=camera_data)

# Run Optimizer
optimal_scene = optimizer(sparse_scene)

# Optional Visualization
from modules.visualize import VisualizeScene
visualizer = VisualizeScene()

visualizer(optimal_scene)
