"""
GOAL RECONSTRUCTION: Sparse Reconstruction

STEP 1: Initialize the scene through the object SfMScene and read in Camera data
- Initialize the scene as SfMScene(...)
- Set the image path to the provided directory of images to be read, resize, and pre-process images for reconstruction
  - image_path = D:\\aquir\\Documents\\Datasets\\CO3Dv2_DATASET\\apple\\189_20393_38136\\images
- Set max_images=20 (Never above 40), we want to only use the first 20 images when evaluating the feasibility of the workflow.
- Set the calibration path to the provided calibration file if one is provided. Since it is provided, we have a calibrated camera and 
  activate the parameter
  - calibration_path = D:\\aquir\\Documents\\Datasets\\CO3Dv2_DATASET\\apple\\calibration_new_189_20393_38136.npz

STEP 2: Detect Features
- Select the SuperPoint module since the scene featuring the object creates shadows with diffuse lighting, and even though it is in an 
  indoor setting with good texture, the illumination changes on the object causes weak correspondence in traditional detectors for tracking, which is crucial in pose
  estimation and scene reconstruction. To build keypoints robust to inconsistent lighting, we use an ML Detector.
  Thus, we opt for SuperPoint, and ML detector robust to illumination or dull lighting enviornments.
    - Set max_keypoints = 5000, to ensure we detect as much features as possible for SuperPoint.
      - Reasoning: SuperPoint typically maxes out the detected points around 5000 in settings it was well-trained in, so in difficult environments
        we set the max setting to detect as many points as possible.

STEP 3: Detect Feature Matches
- Select the SuperGlue feature matcher for image pair matching to utilize as a robust matcher for SuperPoint for point correspondence. We opt for SuperGlue 
  instead of lightglue here for more accuracy, as estimating camera pose using the PnP algorithm is not robust to outliers, and there's intrinsic error in superpoint
  keypoints, so we need very accurate correspondences to offset the baseline error of the detector here.
    - Set RANSAC_threshold to 0.03 (Default 0.3, Points are normalized, so we base reprojection threshold on normalized coordinates)
      Reasoning: for much more strict outlier rejection since we need to ensure we remove all outlier correspondences possible, since inlier correspondences will 
      inherently contain some errors, we need to have only inliers to reduce the compounding error in pose estimation as much as possible.

STEP 4: Estimate the camera pose using detected matching feature pairs.
- Opt to use the Camera Pose Estimation module EssentialToPnP since we have detected features and calibration for the classical approach
    - Since we are using a more inaccurate feature detector (One of SuperPoint or ORB), we must add an extra layer of optimization
    - Initialize the Local Optimization module "BundleAdjustmentOptimizerLocal"
      - Reasoning: This helps ensure the compounding error we endure when estimating the camera poses using superpoint features is reduced with 
        the constant local optimization of the poses every set frame based on window size.
        - set max_num_iterations = 21,
          - Reasoning: This number of iterations for bundle adjustment will not converge, but it will allow the poses to reduce compunding errors and keep poses on track.
        - Set window_size = 12 (Default is 8)
          - Reasoning: Since camera movement is minimal and we want to ensure we have enough data for better pose correction, we opt for a larger window size. 
            We also do not want Bundle adjustment to not over correct, so a larger window size ensures enough data is collected for optimization and limits over correction.
        - Set GPU = False,
            - Reasoning: Global Bundle Adjustment does not need GPU to run effeciently.
    - Set reprojection_error=3.0 (Pixel Coordinates)
      - Reasoning: since we have inherently less accurate key points (SuperPoint), we leave slightly larger error in pose estimation that will be handled in bundle adjustment
        both global and local optimization.
    - Set iteration_count = 300 (Default 200)
      - Reasoning: Higher setting for iteration count of the Levenberg-Marquardt algorithm during pose estimation. Since SuperPoint is 
        inherently inaccurate compared to other detectors, the increased iteration count for optimization will improve initial pose estimates for scene reconstruction.
    - Set confidence=0.99
      - Reasoning: Since we are applying local optimization, the proposed frame we estimate we have higher confidence to be correct.

STEP 5: Track Features across multiple images to create feature tracks for 3D point estimation
- Track Features across multiple images using the SuperGlue Feature Tracking tool
    - Set RANSAC_threshold = 0.05
      - Reasoning: less strict since we are using SuperGlue with SuperPoint features (which are inherently inaccuarate), 
        which is more accurate than lightglue and we want to build longer feature tracks 
        for more accurate estimate of 3D points optimization in the scene
    - Set the detector = "superpoint" as that is the detector we are using.
    - Set the setting = "Indoor" 
      - Reasoning: utilize the indoor weights of the model ("Outdoor" is for outdoor scenes)

STEP 6: Reconstruct the Scene now that we have the estimated Camera Poses and Tracked features for multi-view estimation
- Reconstruct the Scene now that we have the estimated Camera Poses and Tracked features for multi-view estimation
    - Set min_observe = 4 (Default 3), 
      - Reasoning: since we are using SuperPoint with SuperGlue, we have accessibility to large tracks as the lighting changes are not extreme enough
        for SuperPoint features to be extremely inaccurate.
    - Set min_angle = 3.0 (Default 1.0)
      - Reasoning: utilize a larger minimum angle to ensure 3D point estimates are more acccurate, and since we have longer tracks, this is very feasible for this scene.
    - Set multi_view = True (Default True)
      - Reasoning: we are tracking features, so the estimation method is multi-view (Which is most, if not all, cases for reconstruction)

STEP 7: Apply Global Bundle Adjustment to the scene for optimal reconstruction
- To ensure scene is geometrically correct, we want to set 
    - Set max_num_iterations=450, 
      - Reasoning: Since initial 2D points will not be as accurate due to inherent inaccauracy of detected features from SuperPoint along with the ML feature matchers,
         we have to set interations to 450 to ensure convergence of scene for optimal reconstruction. This is primarily due to the SuperPoint detect for scenes with lighting
         inconsistency and smoother textured surfaces lacking detectable points from classical detectors.
    - Set GPU = False,
      - Reasoning: Global Bundle Adjustment does not need GPU to run effeciently.
"""

# ==#$#==

# Construct Modules with Initialized Arguments
image_path = "/home/anthonyq/datasets/co3d_v2/apple/189_20393_38136/images" #"D:\\aquir\\Documents\\Datasets\\CO3Dv2_DATASET\\apple\\189_20393_38136\\images"
calibration_path = "/home/anthonyq/datasets/co3d_v2/apple/calibration_new_189_20393_38136.npz" #"D:\\aquir\\Documents\\Datasets\\CO3Dv2_DATASET\\apple\\calibration_new_189_20393_38136.npz"

from modules.features import FeatureDetectionSP
from modules.featurematching import FeatureMatchSuperGluePair, FeatureMatchLightGlueTracking
from modules.camerapose import CamPoseEstimatorEssentialToPnP
from modules.optimization import BundleAdjustmentOptimizerLocal
from modules.scenereconstruction import Sparse3DReconstructionMono
from modules.optimization import BundleAdjustmentOptimizerGlobal, BundleAdjustmentOptimizerLocal
from modules.baseclass import SfMScene

# Step 1: Read in Calibration/Image Data
reconstructed_scene = SfMScene(id=4,
                                image_path = image_path, 
                                max_images = 20,
                                calibration_path = calibration_path)

# Step 2: Detect Features
reconstructed_scene.FeatureDetectionSP(
    max_keypoints=6000
)

# Step 3: Detect Feature Pairs
reconstructed_scene.FeatureMatchSuperGluePair(
    RANSAC_threshold=3.0
)

# Step 4: Detect/Estimate Camera Poses
reconstructed_scene.CamPoseEstimatorEssentialToPnP(
    reprojection_error=3.0,
    iteration_count=300,
    confidence=0.99,
    optimizer = ("BundleAdjustmentOptimizerLocal", {
        "max_num_iterations": 21,
        "window_size": 12,
        "use_gpu": False
    }),
)

# Step 5: Detect Feature Tracks
reconstructed_scene.FeatureMatchLightGlueTracking(
    detector="superpoint",
    RANSAC_threshold=2.5
)

# Step 6: Estimate Sparse Reconstruction
reconstructed_scene.Sparse3DReconstructionMono(
    min_observe=4,
    min_angle=0.1,
    multi_view=True
)

# Step 7: Run Optimization
reconstructed_scene.BundleAdjustmentOptimizerGlobal(
    max_num_iterations=450,
    use_gpu=False
)