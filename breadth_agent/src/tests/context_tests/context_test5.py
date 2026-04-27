"""
GOAL RECONSTRUCTION: Dense Reconstruction

STEP 1: Initialize the scene through the object SfMScene and read in Camera data
- Initialize the scene as SfMScene(...)
- Set the image path to the provided directory of images to be read, resize, and pre-process images for reconstruction
  - image_path = C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\scan8_normal_lighting
- Set max_images=20 (Never above 40), we want to only use the first 20 images when evaluating the feasibility of the workflow.
- Set the calibration path to the provided calibration file if one is provided. Since it is provided, we have a calibrated camera and 
  activate the parameter
  - calibration_path = C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\calibration_DTU_new.npz

STEP 2: Detect Features
- Select the SuerPoint module since the scene featuring the object is smooth with a lack of rigid texture, and even though it is in an 
  indoor setting with good lighting, the lack of rigid texture, or corners, on the object causes weak correspondence in traditional detectors for tracking, 
  which is crucial in pose estimation and scene reconstruction. To build keypoints robust to inconsistent lighting, we use an ML Detector.
  Thus, we opt for SuperPoint, and ML detector robust to smooth objects with diffuse, inconsistent lighting to detect enough features for reconstruction.
    - Set max_keypoints = 6000, to ensure we detect as much features as possible for SuperPoint.
      - Reasoning: SuperPoint typically maxes out the detected points around 5000 in settings it was well-trained in, so in difficult environments
        we set the max setting to detect as many points as possible.

STEP 3: Detect Feature Matches
- Select the SuperGlue feature matcher for image pair matching to utilize as a robust matcher for SuperPoint for point correspondence. We opt for SuperGlue 
  instead of lightglue here for more accuracy, as estimating camera pose using the PnP algorithm is not robust to outliers, and there's intrinsic error in superpoint
  keypoints, so we need very accurate correspondences to offset the baseline error of the detector here. Since the object is a smooth surface, superglue is more 
  robust for detecting point correspondences in this environment setting.
    - Set RANSAC_threshold to 0.03 (Default 0.3, Points are normalized, so we base reprojection threshold on normalized coordinates)
      Reasoning: for much more strict outlier rejection since we need to ensure we remove all outlier correspondences possible, since inlier correspondences will 
      inherently contain some errors, especially in this case where the environment contains objects that are smooth and inconsistnet lighting, so point correspondences
      will inherently have some errors.

STEP 4: Estimate the camera pose using detected matching feature pairs.
- Opt to use the Camera Pose Estimation module EssentialToPnP since we have detected features and calibration for the classical approach
    - Since we are using a more inaccurate feature detector (One of SuperPoint or ORB), we must add an extra layer of optimization
    - Initialize the Local Optimization module "BundleAdjustmentOptimizerLocal"
      - Reasoning: This helps ensure the compounding error we endure when estimating the camera poses using superpoint features is reduced with 
        the constant local optimization of the poses every set frame based on window size.
        - set max_num_iterations = 30,
          - Reasoning: This number of iterations for bundle adjustment will not converge, but it will allow the poses to reduce compunding errors enough and keep poses on track.
        - Set window_size = 5 (Default is 8)
          - Reasoning: Since camera movement is non-trivial or minimal, slightly jerky, we opt for a smaller window size since the baseline between frames is slightly larger. 
            We also do not want Bundle adjustment to not over correct, so a larger window size ensures enough data is collected for optimization and limits over correction.
        - Set GPU = False,
            - Reasoning: Global Bundle Adjustment does not need GPU to run effeciently.
    - Set reprojection_error = 3.0 (Pixel Coordinates)
      - Reasoning: since we have inherently less accurate key points (SuperPoint), we leave slightly larger error in pose estimation that will be handled in bundle adjustment
        both global and local optimization.
    - Set iteration_count = 300 (Default 200)
      - Reasoning: Higher setting for iteration count of the Levenberg-Marquardt algorithm during pose estimation. Since SuperPoint is 
        inherently inaccurate compared to other detectors, the increased iteration count for optimization will improve initial pose estimates for scene reconstruction.
    - Set confidence=0.995
      - Reasoning: Since we are applying local optimization, the proposed frame we estimate we have higher confidence to be correct.

STEP 5: Track Features across multiple images to create feature tracks for 3D point estimation
- Track Features across multiple images using the LightGlue Feature Tracking tool
    - Set RANSAC_threshold = 0.05
      - Reasoning: less strict since we are using LightGlue with SuperPoint features (which are inherently inaccuarate), 
        since the scene is more difficult for detecting features, we need more leniency in point tracking/correspondences since we need to ensure
        we have enough tracks, and long enough, to properly estimate 3D points with good intialization for global optimization to accurately build the scene.
    - Set the detector = "superpoint" as that is the detector we are using.

STEP 6: Reconstruct the Scene now that we have the estimated Camera Poses and Tracked features for multi-view estimation
- Reconstruct the Scene now that we have the estimated Camera Poses and Tracked features for multi-view estimation
    - Set min_observe = 3 (Default 3), 
      - Reasoning: since we are using SuperPoint with LightGlue on a scene containing smooth surfaces with less texture, we do not have accessibility to 
      large tracks as the lighting changes and surface of scene objects leads to higher error keypoint correspondences. Thus, we use the lowest observation
      count for 3D point estimation to allow enough 3D points in the sparse reconstruction.
    - Set min_angle=1.0 (Default 1.0)
      - Reasoning: utilize a smaller minimum angle to ensure are able to utilize enough 3D point estimation for global optimization. Initial points will be inaccurate,
        but due to compounding errors of detected keypoints and correspondences, having a higher angle will lead to inlier 3D points to possibly be removed.
    - Set multi_view = True (Default True)
      - Reasoning: we are tracking features, so the estimation method is multi-view (Which is most, if not all, cases for reconstruction)

STEP 7: Apply Global Bundle Adjustment to the scene for optimal reconstruction
- To ensure scene is geometrically correct, we want to set 
    - Set max_num_iterations=170, 
      - Reasoning: Since initial 2D points will not be as accurate due to inherent inaccauracy of detected features from SuperPoint along with the ML feature matchers,
        we have to set interations to 170 to ensure convergence of scene for optimal reconstruction. This is primarily due to the SuperPoint detect for scenes with lighting
        inconsistency and smoother textured surfaces lacking detectable points from classical detectors.
    - Set GPU = False,
      - Reasoning: Global Bundle Adjustment does not need GPU to run effeciently.

STEP 8:  Apply Dense Reconstruction Module for dense scene reconstruction. 
- Since we have a Mono camera, and not using VGGT, we opt for the traditonal approach. Use Dense3DReconstructionMono module.
    - Set reproj_error = 3.0 (Default 3.0, in Pixel Coordinates)
      - Reasoning: Default setting to ensure the maximum reprojection error of a given 3D point is only 3.0 pixels
    - Set min_triangulation_angle = 1.0 (Default 1.0)
      - Reasoning: After bundle adjustment of the sparse scene, we can be less cautious and enable more 3D points by lowering the 
        the minimum triangulation angle to just 1.0.
    - Set num_iterations=3 (Default 5)
      - Reasoning: to speed up computation time, we set the number of coordinate descent iterations to just 3 since our build is already
        geometrically optimal at this point.
"""

# ==#$#==

# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\scan8_normal_lighting" # Scan 21 came out really good!
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\calibration_DTU_new.npz"

from modules.features import FeatureDetectionSP
from modules.featurematching import FeatureMatchSuperGluePair, FeatureMatchLightGlueTracking
from modules.camerapose import CamPoseEstimatorEssentialToPnP
from modules.optimization import BundleAdjustmentOptimizerLocal
from modules.scenereconstruction import Sparse3DReconstructionMono
from modules.optimization import BundleAdjustmentOptimizerGlobal, BundleAdjustmentOptimizerLocal
from modules.baseclass import SfMScene

# Step 1: Read in Calibration/Image Data
reconstructed_scene = SfMScene(image_path = image_path, 
                                max_images = 20,
                                calibration_path = calibration_path)

# Step 2: Detect Features
reconstructed_scene.FeatureDetectionSP(
    max_keypoints=6000
)

# Step 3: Detect Feature Pairs
reconstructed_scene.FeatureMatchSuperGluePair(
    detector='superpoint',
    RANSAC_threshold=0.03
)

# Step 4: Detect/Estimate Camera Poses
reconstructed_scene.CamPoseEstimatorEssentialToPnP(
    reprojection_error=3.0,
    iteration_count=300,
    confidence=0.995,
    optimizer = ("BundleAdjustmentOptimizerLocal", {
        "max_num_iterations": 301,
        "window_size": 5,
        "use_gpu": False
    }),
)

# Step 5: Detect Feature Tracks
reconstructed_scene.FeatureMatchLightGlueTracking(
    detector="superpoint",
    RANSAC_threshold=0.05
)

# Step 6: Estimate Sparse Reconstruction
reconstructed_scene.Sparse3DReconstructionMono(
    min_observe = 3,
    min_angle = 0.1,
    multi_view = True
)

# Step 7: Run Optimization
reconstructed_scene.BundleAdjustmentOptimizerGlobal(
    max_num_iterations=170,
    use_gpu=False
)

# STEP 8: Run Rense Reconstruction Algorithm
reconstructed_scene.Dense3DReconstructionMono(
    reproj_error=3.0,
    min_triangulation_angle=1.0,
    num_samples=15,
    num_iterations=3
)