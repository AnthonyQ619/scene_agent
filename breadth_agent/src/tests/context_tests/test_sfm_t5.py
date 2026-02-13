"""
This script solves the problem of creating a sparse 3D representation of an indoor scene with varying lighting conditions 
using Structure-from-Motion as the algorithm solution. Using a monocular camera that is calibrated, the scene captures an Indoors 
museum/stairwell-like interior with arches, balustrade and stone walls. Artificial indoor lighting appears consistent across the images; 
only minor exposure/position changes of the ceiling lamp, no significant illumination change on the statue. Mixture of textures as the 
stone statue and carved fruit details are highly textured; surrounding walls, pillars, steps and floor are mostly smooth/low-textured.
A single stone statue mounted on a base is the persistent object of interest in all frames. Background architectural elements 
(arches, staircase, panels) remain similar. The lighting is soft/diffuse from overhead fixtures, producing mostly matte appearance 
on the stone. The sequence suggests small positional/rotational changes of the camera around the statue without leaving the 
immediate area; the object stays centered and visible the whole time. Gradual, incremental viewpoint shifts (slight translations 
and small rotations); no extreme or dramatic viewpoint changes.

Scene Summary: An indoor museum/stairwell setting with arches, a balustrade, and stone walls under stable artificial lighting. 
A central stone statue with richly textured carved details is consistently visible across three images, while surrounding walls, 
pillars, steps, and floors are relatively smooth and low-textured. The camera undergoes small translations and rotations around 
the statue, maintaining the object near the center with limited baseline changes and minimal illumination variation.
"""

# ==#$#==

"""
GOAL RECONSTRUCTION: Dense Reconstruction

STEP 1: Read in Camera data
- Set the image path to the provided directory of images to the CameraDataManager to read, resize, and pre-process images for reconstruction
- Set the calibration path to the provided calibration file if one is provided. Since it is provided, we have a calibrated camera and 
  activate the parameter
- Finally, call get_camera_data() function to have access to the camera data of images and calibration

STEP 2: 
- Mixture of textures as the stone statue and carved fruit details are highly textured; surrounding walls, pillars, steps and floor are 
  mostly smooth/low-textured. For more points, USE SUPERPOINT DUE TO MIXTURE OF TEXTURE.
- Select the SuerPoint module since the scene featuring the scene contains a mixture of textures despite having consistent lighting, the 
  production of a mostly matte appearance on the stone leaves overall less texture.
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
        - Set window size to 5 (Default) since we are using super point, we will need better correction due to drift and better pose correction.
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
- Reconstruct the Sparse Scene now that we have the estimated Camera Poses and Tracked features for multi-view estimation
    - Set min_observe=4, since we are using SuperPoint with SuperGlue, we have accessibility to large tracks as the lighting changes are not extreme.
    - Set min_angle=3.0, to utilize a larger minimum angle to ensure 3D point estimates are more acccurate.
    - Set view="multi", we are tracking features, so the estimation method is multi-view (Should be default)

STEP 7:
- Apply Global Bundle Adjustment to the scene for optimal reconstruction
    - Set max_num_iterations=80, since initial 2D points will not be as accurate due to leniency in constraints in previous methods despite
      optimal scene lighting and texture being for SIFT features.

STEP 8:
- Apply Dense Reconstruction Module for dense scene reconstruction. Since we have a Mono camera, and not using VGGT, we opt for the traditonal approach
    - Set reproj_error=3.0, Default setting to ensure the maximum reprojection error of a given 3D point is only 3.0 pixels
    - Set min_triangulation_angle=1.0, After bundle adjustment of the sparse scene, we can be less cautious and enable more 3D points by lowering the 
      the minimum triangulation angle to just 1.0.
    - Set num_iterations=3, to speed up computation time, we set the number of coordinate descent iterations to just 3 since our build is already
      geometrically optimal at this point.
"""

# ==#$#==

# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\ETH\\statue\\images\\dslr_images_undistorted"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\ETH\\statue\\dslr_calibration_undistorted\\calibration_new.npz"


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
                                            detector='superpoint',
                                            RANSAC_threshold=0.03)

# Detect Image Pair Correspondences for Pose Estimation
feature_pairs = feature_matcher(features=features)

# STEP 4: Estimate Camera Poses of Scene with Feature Pairs with Local Bundle Adjustment Activated
from modules.optimization import BundleAdjustmentOptimizerLocal

# Build Optimizer
optimizer_local = BundleAdjustmentOptimizerLocal(max_num_iterations=30,
                                                 cam_data=camera_data,
                                                 window_size=5)

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

# STEP 6: Reconstruct Sparse Scene in Full
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
optimizer = BundleAdjustmentOptimizerGlobal(max_num_iterations=80,
                                            cam_data=camera_data)

# Run Optimizer
optimal_scene = optimizer(sparse_scene)

# STEP 8: Run Rense Reconstruction Algorithm
from modules.scenereconstruction import Dense3DReconstructionMono

# Conduct Dense Reconstruction Module
dense_reconstruction = Dense3DReconstructionMono(cam_data=camera_data,
                                                  reproj_error=3.0,
                                                  min_triangulation_angle=1.0,
                                                  num_samples=15,
                                                  num_iterations=3)

# Estimate sparse 3D scene from tracked features and camera poses
dense_scene = dense_reconstruction(sparse_scene=optimal_scene)
