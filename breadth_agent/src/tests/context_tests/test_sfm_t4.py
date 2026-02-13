"""
Scene Setting: Indoors (studio-like setup with object on a plain brown surface).
Scene Lighting Effects: Consistent across images; soft, diffuse lighting with similar shadows and no noticeable illumination changes.
Scene Object Textures: Mix of textures: the stone-like statue is highly textured; the background surface is mostly low-textured/uniform.
Scene Key Features: A single sculpted figurine (stone-like) remains in view in all frames on a plain background. The sequence appears to move 
slightly around or alongside the object while keeping it the main focus. The object is highly textured; lighting is diffuse with soft shadows 
and minimal specular highlights.
Scene Camera Movement: Small, incremental viewpoint shifts (slight lateral/orbital movement). No extreme viewpoint changes.


Scene Summary: Indoor studio-like setup with a single, highly textured stone-like figurine placed on a plain brown surface. Lighting is soft, 
diffuse, and consistent, producing stable appearances with minimal specular effects. The camera undergoes small, incremental lateral/orbital 
motions around the object; the figurine remains the central focus while the background is largely uniform and low-textured.
"""

# ==#$#==
"""
GOAL RECONSTRUCTION: Sparse Reconstruction
- Scene: the stone-like statue is highly textured; the background surface is mostly low-textured/uniform (MIXED TEXTURED). and 
  soft, diffuse lighting with similar shadows. USE SUPERPOINT DUE TO LIGHTING CHANGES AND MIXED TEXTURE.
STEP 1: Read in Camera data
- Set the image path to the provided directory of images to the CameraDataManager to read, resize, and pre-process images for reconstruction
- Set the calibration path to the provided calibration file if one is provided. Since it is provided, we have a calibrated camera and 
  activate the parameter
- Finally, call get_camera_data() function to have access to the camera data of images and calibration

STEP 2: 
- Scene: stronger shadows, but subsequent images are brighter (Illumination changes) with diffuse lighting. Predominantly high-textured but low texture areas 
  (MIXED TEXTURED). USE SUPERPOINT DUE TO LIGHTING CHANGES AND MIXED TEXTURE.
- Select the SuerPoint module since the scene featuring the object creates shadows with diffuse lighting, and even though it is in an 
  indoor setting with good texture, the illumination changes on the object causes weak correspondence in traditional detectors for tracking, which is crucial in pose
  estimation and scene reconstruction.
  Thus, we opt for SuperPoint, and ML detector robust to illumination or dull lighting enviornments.
    - Set max features to detect to 5000, to ensure we detect as much features as possible for SuperPoint.

STEP 3: 
- Select the LightGlue feature matcher for image pair matching to utilize as a quick matcher for SuperPoint for simple pair wise matching, where 
  we can be less strict in extremely accurate matches for more optimized computation speed.
    - Set RANSAC_threshold to 0.03, for much more strict outlier rejection since we need to ensure we remove all outlier correspondences

STEP 4:
- Estimate the camera pose using matching feature pairs of the scene
    - Since we are using a more inaccurate feature detector (One of SuperPoint or ORB), we must add an extra layer of optimization
    - Initialize the Local Optimization module "BundleAdjustmentOptimizerLocal"
        - Set iteration count for Bundle Adjustment to 30 since we are utilizing an ML detector, which is inherently less geometrically correct.
        - Set window size to 8 (Default) since camera movement is minimal and better pose correction.
    - Set reprojection_error=3.0, since we are including corner points as well to ensure as many 3D points are included as possible for a 
      and more accurate bundle adjustment
    - Set iteration_count=300, higher setting for iteration count of the Levenberg-Marquardt algorithm during pose estimation. Since ORB is 
      inherently inaccurate compared to other detectors, the increased iteration count for optimization will improve initial poses for 
      a well-lit and textured scene for optimal initial pose estimation
    - Set confidence=0.995, Since the scene is well-lit and textured, we have a higher confidence of the proposed solutions 

STEP 5:
- Track Features across multiple images using the SuperGlue Feature Tracking tool
    - Set RANSAC_threshold to 0.4, less strict since we are using SuperGlue, which is more accurate and  we want to build longer feature tracks 
      for more accurate estimate of 3D points optimization in the scene
    - Set the detector to "superpoint" as that is the detector we are using.
    - Set the setting to "Indoor" to utilize the indoor weights of the model ("Outdoor" is for outdoor scenes)

STEP 6:
- Reconstruct the Scene now that we have the estimated Camera Poses and Tracked features for multi-view estimation
    - Set min_observe=4, since we are using SuperPoint with SuperGlue, we have accessibility to large tracks as the lighting changes are not extreme.
    - Set min_angle=3.0, to utilize a larger minimum angle to ensure 3D point estimates are more acccurate.
    - Set view="multi", we are tracking features, so the estimation method is multi-view (Should be default)

STEP 7:
- Apply Global Bundle Adjustment to the scene for optimal reconstruction
    - Set max_num_iterations=80, since initial 2D points will not be as accurate due to leniency in constraints in previous methods despite
      optimal scene lighting and texture being for SIFT features.
"""

# ==#$#==

# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\scan118_normal_lighting"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\calibration_DTU_new.npz"


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
from modules.featurematching import FeatureMatchLightGluePair
# Pairwise Feature Matching Module Initialization
feature_matcher = FeatureMatchLightGluePair(cam_data=camera_data,
                                            RANSAC_threshold=0.03)

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
                                            RANSAC_threshold=0.4)


# From estimated features, tracked features across multiple images
tracked_features = feature_tracker(features=features)

# STEP 6: Reconstruct Scene in Full
from modules.scenereconstruction import Sparse3DReconstructionMono
# Scene Reconstruction Module Initialization
sparse_reconstruction = Sparse3DReconstructionMono(cam_data=camera_data,
                                                   min_observe=4,
                                                   min_angle=3.0,
                                                   view="multi")

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