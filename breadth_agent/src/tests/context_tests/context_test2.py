"""
GOAL RECONSTRUCTION: Sparse Reconstruction

STEP 1: Initialize the scene through the object SfMScene and read in Camera data
- Initialize the scene as SfMScene(...)
- Set the image path to the provided directory of images to be read, resize, and pre-process images for reconstruction
  - image_path = C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\scan21_normal_lighting
- Set max_images=20 (Never above 40), we want to only use the first 20 images when evaluating the feasibility of the workflow.
- Set the calibration path to the provided calibration file if one is provided. Since it is provided, we have a calibrated camera and 
  activate the parameter
  - calibration_path = C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\calibration_DTU_new.npz

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
- Select the Brute Force feature matcher to utilize as a robust matcher for ORB, since we don't have complex descriptors, we must utilize a stronger
  matcher for more accurate matches across the corresponding points.
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
        - Set GPU = False,
            - Reasoning: Global Bundle Adjustment does not need GPU to run effeciently.
        - Set robust_lost = True,
            - Reasoniong: Since we are using less accurate features, we need a robust loss function to be resistent to outlier if they are left.
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
    - Set cross_check = False (Default True)
      - Reasoning: Since we are applying false, this will apply default BFMatcher behaviour when it finds the k nearest neighbors for each query descriptor,
        which will be more exhaustive, but ensure we find the most accurate matches/tracks across multiple images.

STEP 6: Reconstruct the Scene now that we have the estimated Camera Poses and Tracked features for multi-view estimation
- We are usng the Sparse Reconstruction module, specifically the classical approach with a monocular camera, so opt for the Sparse3DReconstruction module
    - Set min_observe=3 (Default 3)
      - Reasoning: ORB will naturally make it more difficult for longer tracks, even if they are accurate, we will default for the minimum for this case.
    - Set min_angle=3.0 (Default 1.0) 
      - Reasoning: utilize a larger minimum angle to ensure 3D point estimates are more acccurate, and since we have longer tracks, this is very feasible for this scene.
    - Set multi_view = True (Default True)
      - Reasoning: we are tracking features, so the estimation method is multi-view (Which is most, if not all, cases for reconstruction)

STEP 7: Apply Global Bundle Adjustment to the scene for optimal reconstruction
- To ensure scene is geometrically correct, we want to set 
    - Set max_num_iterations=200, 
      - Reasoning: Since initial 2D points will not be as accurate due to inaccuracy of certain matches and tracks in previous methods despite
        optimal scene lighting and texture being for ORB features, we have to set interations to 200 to ensure convergence of scene for optimal reconstruction.
    - Set GPU = False,
      - Reasoning: Global Bundle Adjustment does not need GPU to run effeciently.
"""

# ==#$#==

# Construct Modules with Initialized Arguments
image_path = "/home/anthonyq/datasets/DTU/DTU/scan21" #"C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\scan21_normal_lighting"
calibration_path = "/home/anthonyq/datasets/DTU/DTU/calibration_DTU_new.npz" #"C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\calibration_DTU_new.npz"

from modules.features import FeatureDetectionORB
from modules.featurematching import FeatureMatchBFPair, FeatureMatchBFTracking
from modules.camerapose import CamPoseEstimatorEssentialToPnP
from modules.optimization import BundleAdjustmentOptimizerLocal
from modules.scenereconstruction import Sparse3DReconstructionMono
from modules.optimization import BundleAdjustmentOptimizerGlobal, BundleAdjustmentOptimizerLocal
from modules.baseclass import SfMScene

# Step 1: Read in Calibration/Image Data
reconstructed_scene = SfMScene(id=2,
                                image_path = image_path, 
                                max_images = 20,
                                calibration_path = calibration_path)

# Step 2: Detect Features
reconstructed_scene.FeatureDetectionORB(
    max_keypoints=10000,
    set_nms=True,
    set_nms_allowed_points=4000
)

# Step 3: Detect Feature Pairs
reconstructed_scene.FeatureMatchBFPair(
    detector='orb', 
    RANSAC_threshold=3.0,
    lowes_thresh=0.8
)

# Step 4: Detect/Estimate Camera Poses
reconstructed_scene.CamPoseEstimatorEssentialToPnP(
    reprojection_error=3.0,
    iteration_count=300,
    confidence=0.99,
    optimizer = ("BundleAdjustmentOptimizerLocal", {
        "max_num_iterations": 20,
        "window_size": 8,
        "robust_loss": True,
        "use_gpu": False
    }),
)

# Step 5: Detect Feature Tracks
reconstructed_scene.FeatureMatchBFTracking(
    detector="orb",
    RANSAC_threshold=3.0,
    lowes_thresh=0.8,
    cross_check=False
)

# Step 6: Estimate Sparse Reconstruction
reconstructed_scene.Sparse3DReconstructionMono(
    min_observe=3,
    min_angle=3.0,
    multi_view=True
)

# Step 7: Run Optimization
reconstructed_scene.BundleAdjustmentOptimizerGlobal(
    max_num_iterations=400,
    use_gpu=False
)