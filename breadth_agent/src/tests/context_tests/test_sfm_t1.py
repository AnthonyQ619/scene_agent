"""
GOAL RECONSTRUCTION: Dense Reconstruction

STEP 1: Read in Camera data
- Set the image path to the provided directory of images to the CameraDataManager to read, resize, and pre-process images for reconstruction
  - image_path = C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\scan14_normal_lighting
- Set max_images=20 (Never above 40), we want to only use the first 20 images when evaluating the feasibility of the workflow.
- Set the calibration path to the provided calibration file if one is provided. Since it is provided, we have a calibrated camera and 
  activate the parameter
  - calibration_path = C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\calibration_DTU_new.npz
- Finally, call get_camera_data() function to have access to the camera data of images and calibration

STEP 2: Detect Features
- Select the SIFT module since the scene featuring the object is well lit, in an indoor setting, and the object is of a brick house that is 
  well textured for using SIFT reliably.
    - Set max_keypoints = 15000, to ensure we detect as much features as possible since we also want to detect corners due to the 
      object having many set shapes (house) that involves corners and edges.
      - Reasoning: Since the scene is well-lit with consistent lighting across buildings/objects, most points detected will likely be 
        reasonably accurate, so increasing maximum number of detected points enables more possible inliers to be detected.
    - Set contrast_threshold = 0.005, with default being 0.04, the lower the threshold, more features are produced, which is what we desire 
      for this scenario.
      - Reasoning: Since the scene is well-lit, has consistent lighting across buildings, allowing more features will enable more inliers to be 
        detected since the scene is good for the SIFT detector.
    - Set edge_threshold = 12, the higher the threshold, the more edge-like features are produced, which we need for this scene.
      - Reasoning: Slightly higher threshold to allow more edge-like features to be detected since some of the objects are model buildings, which
        have windows and edges that will be good for detected features. 

STEP 3: Detect Feature Matches
- Select the Flann feature matcher to utilize a nearest neighbor matcher, as we only need approximate matches here that are sparse for PnP
    - Set RANSAC_threshold = 0.02 (Default 0.3, Points are normalized, so we base reprojection threshold on normalized coordinates)
      - Reasoning: Scene has consistent lighting, and we want to be aggressive in outlier removal since PnP algorithms are not as robust to outliers, 
        but keep enough points in total for PnP to work accurately with enough points across the entire image.
    - Set lowes_threshold = 0.8 (Default is 0.75)
      - Reasoning: to enable more matching pairs to pass since any higher threshold will remove usable inliers for matches, that will be more clustered 
        in high textured/accurate areas. 

STEP 4: Estimate the camera pose using detected matching feature pairs.
- Opt to use the Camera Pose Estimation module EssentialToPnP since we have detected features and calibration for the classical approach
    - Set reprojection_error = 3.0 (Default is 3.0)
      - Reasoning: Since we are using SIFT and have slightly aggressive outlier rejection parameters in pair matching module, we keep the default
        reprojection error to ensure we don't remove any more points than necessary for pose esitmation.
    - Set iteration_count=200, default setting for iteration count of the Levenberg-Marquardt algorithm during pose estimation. 
      - Reasoning: Enough for iterations for a well-lit and textured scene for optimal initial pose estimation.
    - Set confidence=0.995 (Default 0.99)
      - Reasoning: Since the scene is well-lit and textured, we have a higher confidence of the proposed solutions to be accurate

STEP 5: Track Features across multiple images to create feature tracks for 3D point estimation
-  For sparse reconstruction, we want as many 3D points as possible with as long possible track lengths. The more accurate matcher here is the Brute-Force 
   Tracking tool, so use the BFTracking module since it's more accurate and robust to trickling outliers.
    - Set RANSAC_threshold = 0.008 (Default 3.0, Points are normalized, so we base reprojection threshold on normalized coordinates)
      - Reasoning: Although the scene is well-lit and good for SIFT features, we want to be extremely aggressive in outlier removal since we are completing
        feature tracks, thus we opt for a 0.008 ransac threshold instead of the 0.02 we did for pairwise matching 
    - Set lowes_thresh = 0.65 (Default 0.75)
      - Reasoning: as we want to be more strict on the feature tracks for more accurate estimate of 3D points in the scene
        we are more aggressive in lowes threshold to ensure the removal of most outliers, even if we remove inliers in the process.

STEP 6: Reconstruct the Scene now that we have the estimated Camera Poses and Tracked features for multi-view estimation
- We are usng the Sparse Reconstruction module, specifically the classical approach with a monocular camera, so opt for the Sparse3DReconstruction module
    - Set min_observe=4 (Default 3)
      - Reasoning: Since we are using the Brute Force Matcher for tracking features, we will have longer features to use for 3D point estimation, since this scene has 
        minimal camera movement, we need a wider baseline as well, so we use 4 2D points minimum to estimate an acceptable 3D point for sparse reconstruction. 
    - Set min_angle=2.0 (Default 1.0) 
      - Reasoning: utilize a larger minimum angle to ensure 3D point estimates are more acccurate, and since we have longer tracks, this is very feasible for this scene.
    - Set view="multi"
      - Reasoning: we are tracking features, so the estimation method is multi-view (Which is most, if not all, cases for reconstruction)

STEP 7: Apply Global Bundle Adjustment to the scene for optimal reconstruction
- To ensure scene is geometrically correct, we want to set 
    - Set max_num_iterations=200, 
      - Reasoning: Since initial 2D points will not be as accurate due to inaccuracy of certain matches and tracks in previous methods despite
        optimal scene lighting and texture being for SIFT features, we have to set interations to 200 to ensure convergence of scene for optimal reconstruction.
      
STEP 8: Apply Dense Reconstruction Module for dense scene reconstruction. 
- Since we have a Mono camera, and not using VGGT, we opt for the traditonal approach. Use Dense3DReconstructionMono module.
    - Set reproj_error=3.0 (Default 3.0, in Pixel Coordinates)
      - Reasoning: Default setting to ensure the maximum reprojection error of a given 3D point is only 3.0 pixels
    - Set min_triangulation_angle=1.0 (Default )
      - Reasoning: After bundle adjustment of the sparse scene, we can be less cautious and enable more 3D points by lowering the 
      the minimum triangulation angle to just 1.0.
    - Set num_iterations=3 (Default )
      - Reasoning: to speed up computation time, we set the number of coordinate descent iterations to just 3 since our build is already
      geometrically optimal at this point.
"""

# ==#$#==

# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\scan14_normal_lighting"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\calibration_DTU_new.npz"

# STEP 1: Read in Camera Data
from modules.cameramanager import CameraDataManager

CDM = CameraDataManager(image_path=image_path,
                        max_images=20,
                        calibration_path=calibration_path)

# Get Camera Data
camera_data = CDM.get_camera_data()

# STEP 2: Estimate Features per Image
from modules.features import FeatureDetectionSIFT
# Feature Module Initialization
feature_detector = FeatureDetectionSIFT(cam_data=camera_data, 
                                        max_keypoints=15000,
                                        contrast_threshold=0.009,
                                        edge_threshold=20)

# Detect Features for all Images
features = feature_detector()

# STEP 3: Match Features Per Image 
from modules.featurematching import FeatureMatchFlannPair
# Pairwise Feature Matching Module Initialization
feature_matcher = FeatureMatchFlannPair(detector="sift", 
                                        cam_data=camera_data,
                                        RANSAC_threshold=0.02,
                                        lowes_thresh=0.8)

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
dense_scene = dense_reconstruction.build_dense(sparse_scene=optimal_scene)

# Optional Visualization
from modules.visualize import VisualizeScene
visualizer = VisualizeScene()

visualizer(dense_scene)



