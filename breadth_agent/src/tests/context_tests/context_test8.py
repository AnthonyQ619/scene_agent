"""
GOAL RECONSTRUCTION: Sparse Reconstruction

STEP 1: Initialize the scene through the object SfMScene and read in Camera data
- Initialize the scene as SfMScene(...)
- Set the image path to the provided directory of images to be read, resize, and pre-process images for reconstruction
    - image_path = C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\ETH\\statue\\images\\dslr_images_undistorted
- Set max_images=20 (Never above 40), we want to only use the first 20 images when evaluating the feasibility of the workflow.
- Set the calibration path to the provided calibration file if one is provided. Since it is provided, we have a calibrated camera and 
  activate the parameter
  - calibration_path = C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\ETH\\statue\\dslr_calibration_undistorted\\calibration_new.npz

STEP 2: Detect Features
- Select the SIFT module since the scene featuring the object is high-textured rigid surface and stone textured statue, with the scene 
  having consistent lighting, which is well-suited for using SIFT reliably.
    - Set max_keypoints = 15000, to ensure we detect as much features as possible since we also want to detect corners due to the 
      object having many set shapes (house)
        - Reasoning: Since the scene is well-lit with consistent lighting across objects with rigid textures, most points detected will likely be 
          reasonably accurate, so increasing maximum number of detected points enables more possible inliers to be detected.
    - Set contrast_threshold = 0.009, with default being 0.04, the lower the threshold, more features are produced, which is what we desire 
      for this scenario.
      - Reasoning: Since the scene is well-lit, has consistent lighting across the fire hydrant, allowing more features will enable more inliers to be 
        detected since the scene is good for the SIFT detector.
    - Set edge_threshold = 12, the higher the threshold, the more edge-like features are produced, which we need for this scene.
      - Reasoning: Slightly higher threshold to allow more edge-like features to be detected since some of the objects are quite defined in the images, which
        have ridges and edges that will be good for detected features. 

STEP 3: Detect Feature Matches
- Select the Brute-Force feature matcher to utilize a nearest neighbor matcher, since it's more accurate and robust to trickling outliers. This is 
  important in this case since we use corresponding points to estimate camera poses, which are not robust to outliers, and in the outside setting, detected 
  features are much less accurate, so having a robust matcher is necessary.
    - Set RANSAC_threshold = 0.02 (Default 0.3, Points are normalized, so we base reprojection threshold on normalized coordinates)
      - Reasoning: Scene has consistent lighting, and we want to be aggressive in outlier removal since PnP algorithms are not as robust to outliers, 
        but keep enough points in total for PnP to work accurately with enough points across the entire image.
    - Set lowes_threshold to 0.70 (Default 0.75) 
      - Reasoning: to enable slightly more strict rejection in matching pairs to pass since any higher threshold to remove any 
        stray outliers in the scene.

STEP 4: Estimate the camera pose using detected matching feature pairs.
- Estimate the camera pose using matching feature pairs of the scene
    - Since we are using SIFT feature detector in a more difficult and less ideal setting (Outdoor but highly textured), we need further refinment
      in pose estimation for more accurate scene reconsturction
        - Initialize the Local Optimization module "BundleAdjustmentOptimizerLocal"
            - Reasoning: This helps ensure the compounding error we endure when estimating the camera poses using is reduced with the constant local 
              optimization of the poses every set frame based on window size, as initial poses will drift due to detected points in the outdoor setting 
              will be noisy due to the lighting changes, and SIFT not being as robust to illumination fluctuations.
          - set max_num_iterations = 60,
            - Reasoning: This number of iterations for bundle adjustment will not converge, but it will allow the poses to reduce compunding errors and keep poses on track.
              Setting the number to 60 iterations ensures poses are optimized close to, or at convergence, of the bundle adjustment algorithm for the first poses
              estimated, ensuring compounding errors of pose estimation is kept to a minimum.
          - Set window_size = 8 (Default is 8)
          - Set GPU = False,
            - Reasoning: Global Bundle Adjustment does not need GPU to run effeciently.
    - Set reprojection_error=3.0 (Pixel Coordinates)
      - Reasoning: since we have inherently less accurate key points due to scene environment, we leave slightly larger error in pose estimation that will be handled 
        in bundle adjustment both global and local optimization.
    - Set iteration_count = 300 (Default 200)
      - Reasoning: Higher setting for iteration count of the Levenberg-Marquardt algorithm during pose estimation. Since the detected points will be less consistent due to
        the scene being outdoors with less consistent lighting than indoors, the increased iteration count for optimization will improve initial pose estimates for scene 
        reconstruction.
    - Set confidence=0.995
      - Reasoning: Since we are applying local optimization, the proposed frame we estimate we have higher confidence to be correct.

STEP 5: Track Features across multiple images to create feature tracks for 3D point estimation
-  For sparse reconstruction, we want as many 3D points as possible with as long possible track lengths. The more accurate matcher here is the Brute-Force 
   Tracking tool, so use the BFTracking module since it's more accurate and robust to trickling outliers.
    - Set RANSAC_threshold = 0.02 (Default 3.0, Points are normalized, so we base reprojection threshold on normalized coordinates)
      - Reasoning: Although the scene is well-lit and good for SIFT features, we want to be extremely aggressive in outlier removal since we are completing
        feature tracks, thus we opt for the same threshold as step 3 of the 0.02 we did for pairwise matching 
    - Set lowes_thresh = 0.75 (Default 0.75)
      
TEP 6: Reconstruct the Scene now that we have the estimated Camera Poses and Tracked features for multi-view estimation
- We are usng the Sparse Reconstruction module, specifically the classical approach with a monocular camera, so opt for the Sparse3DReconstruction module
    - Set min_observe=3 (Default 3)
    - Set min_angle=2.0 (Default 1.0) 
      - Reasoning: utilize a larger minimum angle to ensure 3D point estimates are more acccurate, and since we have longer tracks, this is very feasible for this scene.
    - Set multi_view = True (Default True)
      - Reasoning: we are tracking features, so the estimation method is multi-view (Which is most, if not all, cases for reconstruction)

STEP 7: Apply Global Bundle Adjustment to the scene for optimal reconstruction
- To ensure scene is geometrically correct, we want to set 
    - Set max_num_iterations=330, 
      - Reasoning: Since initial 2D points will not be as accurate due to inaccuracy of certain matches and tracks in previous methods despite
        optimal scene lighting and texture being for SIFT features, we have to set interations to 250 to ensure convergence of scene for optimal reconstruction.
    - Set GPU = False,
      - Reasoning: Global Bundle Adjustment does not need GPU to run effeciently.
"""

# ==#$#==

# Construct Modules with Initialized Arguments
image_path ="/home/anthonyq/datasets/ETH/ETH/statue/images/dslr_images_undistorted" #"C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\ETH\\statue\\images\\dslr_images_undistorted"
calibration_path = "/home/anthonyq/datasets/ETH/ETH/statue/dslr_calibration_undistorted/calibration_new.npz" #"C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\ETH\\statue\\dslr_calibration_undistorted\\calibration_new.npz"

from modules.features import FeatureDetectionSIFT
from modules.featurematching import FeatureMatchBFPair, FeatureMatchBFTracking
from modules.camerapose import CamPoseEstimatorEssentialToPnP
from modules.optimization import BundleAdjustmentOptimizerLocal
from modules.scenereconstruction import Sparse3DReconstructionMono
from modules.optimization import BundleAdjustmentOptimizerGlobal
from modules.baseclass import SfMScene

# Step 1: Read in Calibration/Image Data
reconstructed_scene = SfMScene(id=8,
                                image_path = image_path, 
                                max_images = 20,
                                calibration_path = calibration_path)

# Step 2: Detect Features
reconstructed_scene.FeatureDetectionSIFT(
    max_keypoints=15000,
    contrast_threshold=0.009,
    edge_threshold=20
)

# Step 3: Detect Feature Pairs
reconstructed_scene.FeatureMatchBFPair(
    detector="sift",
    cross_check=False,
    RANSAC_threshold=1.0,
    lowes_thresh=0.70
)

# Step 4: Detect/Estimate Camera Poses
reconstructed_scene.CamPoseEstimatorEssentialToPnP(
    reprojection_error=3.0,
    iteration_count=300,
    confidence=0.99,
    optimizer = ("BundleAdjustmentOptimizerLocal", {
        "max_num_iterations": 60,
        "use_gpu": False
    }),
)

# Step 5: Detect Feature Tracks
reconstructed_scene.FeatureMatchBFTracking(
    detector="sift",
    RANSAC_threshold=1.0,
    lowes_thresh=0.75
)

# Step 6: Estimate Sparse Reconstruction
reconstructed_scene.Sparse3DReconstructionMono(
    min_observe=3,
    min_angle=2.0,
    multi_view=True
)

# Step 7: Run Optimization
reconstructed_scene.BundleAdjustmentOptimizerGlobal(
    max_num_iterations=330,
    use_gpu=False
)