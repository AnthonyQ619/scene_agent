"""
GOAL RECONSTRUCTION: Camera Pose Reconstruction

STEP 1: Read in Camera data
- Initialize the scene as SfMScene(...)
- Set the image path to the provided directory of images to be read, resize, and pre-process images for reconstruction
  - image_path = D:\\aquir\\Documents\\Datasets\\CO3Dv2_DATASET\\hydrant\\167_18184_34441\\images
- Set max_images=20 (Never above 40), we want to only use the first 20 images when evaluating the feasibility of the workflow.
- Set the calibration path to the provided calibration file if one is provided. Since it is provided, we have a calibrated camera and 
  activate the parameter
  - calibration_path = D:\\aquir\\Documents\\Datasets\\CO3Dv2_DATASET\\hydrant\\calibration_new_167_18184_34441.npz

STEP 2: Detect Features
- Select the SIFT module since the scene featuring an outdoor setting, and the object is of a fire hydrant that had consistent lighting and
  well textured for using SIFT reliably.
    - Set max_keypoints = 15000, to ensure we detect as much features as possible since we also want to detect corners due to the 
      object having many set shapes that involves corners and edges.
      - Reasoning: Since the scene is well-lit with consistent lighting across the object, most points detected will likely be 
        reasonably accurate, so increasing maximum number of detected points enables more possible inliers to be detected.
    - Set contrast_threshold = 0.009, with default being 0.04, the lower the threshold, more features are produced, which is what we desire 
      for this scenario.
      - Reasoning: Since the scene is well-lit, has consistent lighting across the fire hydrant, allowing more features will enable more inliers to be 
        detected since the scene is good for the SIFT detector.
    - Set edge_threshold = 12, the higher the threshold, the more edge-like features are produced, which we need for this scene.
      - Reasoning: Slightly higher threshold to allow more edge-like features to be detected since some of the objects are quite defined in the images, which
        have windows and edges that will be good for detected features. 


STEP 3: Detect Feature Matches
- Select the Flann feature matcher to utilize a nearest neighbor matcher, as we only need approximate matches here that are sparse for PnP
    - Set RANSAC_threshold = 0.02 (Default 0.3, Points are normalized, so we base reprojection threshold on normalized coordinates)
      - Reasoning: Scene has consistent lighting, and we want to be aggressive in outlier removal since PnP algorithms are not as robust to outliers, 
        but keep enough points in total for PnP to work accurately with enough points across the entire image.
    - Set lowes_threshold = 0.8 (Default is 0.75)
      - Reasoning: to enable more matching pairs to pass since any higher threshold will remove usable inliers for matches, that will be more clustered 
        in high textured/accurate areas. 

STEP 4:
- Estimate the camera pose using matching feature pairs of the scene
    - Set reprojection_error=3.0, since we are including corner points as well to ensure as many 3D points are included as possible for a 
      and more accurate bundle adjustment
    - Set iteration_count=200, default setting for iteration count of the Levenberg-Marquardt algorithm during pose estimation. Enough for 
      a well-lit and textured scene for optimal initial pose estimation
    - Set confidence=0.995, Since the scene is well-lit and textured, we have a higher confidence of the proposed solutions 

STEP 4:Estimate the camera pose using detected matching feature pairs.
- Estimate the camera pose using matching feature pairs of the scene
    - Since we are using SIFT feature detector in a more difficult and less ideal setting (Outdoor but highly textured), we need further refinment
      in pose estimation for more accurate scene reconsturction
        - Initialize the Local Optimization module "BundleAdjustmentOptimizerLocal" inside Pose estimation
            - Reasoning: This helps ensure the compounding error we endure when estimating the camera poses using is reduced with the constant local 
              optimization of the poses every set frame based on window size, as initial poses will drift due to detected points in the outdoor setting 
              will be noisy due to the lighting changes, and SIFT not being as robust to illumination fluctuations.
        - set max_num_iterations = 20,
          - Reasoning: This number of iterations for bundle adjustment will not converge, but it will allow the poses to reduce compunding errors enough and keep poses on track.
        - Set window_size = 5 (Default is 8)
          - Reasoning: Since camera movement is non-trivial or minimal, slightly jerky, we opt for a smaller window size since the baseline between frames is slightly larger. 
            We also do not want Bundle adjustment to not over correct, so a larger window size ensures enough data is collected for optimization and limits over correction.
        - Set GPU = False,
          - Reasoning: Global Bundle Adjustment does not need GPU to run effeciently.    
    - Set reprojection_error=3.0 (Pixel Coordinates)
      - Reasoning: since we have inherently less accurate key points due to scene environment, we leave slightly larger error in pose estimation that will be handled 
        in bundle adjustment both global and local optimization.
    - Set iteration_count = 300 (Default 200)
      - Reasoning: Higher setting for iteration count of the Levenberg-Marquardt algorithm during pose estimation. Since the detected points will be less consistent due to
        the scene being outdoors with less consistent lighting, the increased iteration count for optimization will improve initial pose estimates for scene reconstruction.
    - Set confidence=0.995
      - Reasoning: Since we are applying local optimization, the proposed frame we estimate we have higher confidence to be correct.
"""

# ==#$#==

# Construct Modules with Initialized Arguments
image_path = "/home/anthonyq/datasets/co3d_v2/hydrant/167_18184_34441/images" #"D:\\aquir\\Documents\\Datasets\\CO3Dv2_DATASET\\hydrant\\167_18184_34441\\images"
calibration_path = "/home/anthonyq/datasets/co3d_v2/hydrant/calibration_new_167_18184_34441.npz" #"D:\\aquir\\Documents\\Datasets\\CO3Dv2_DATASET\\hydrant\\calibration_new_167_18184_34441.npz"

from modules.features import FeatureDetectionSIFT
from modules.featurematching import FeatureMatchFlannPair
from modules.camerapose import CamPoseEstimatorEssentialToPnP
from modules.optimization import BundleAdjustmentOptimizerLocal

from modules.baseclass import SfMScene

# Step 1: Read in Calibration/Image Data
reconstructed_scene = SfMScene(id=7, image_path = image_path, 
                                max_images = 20,
                                calibration_path = calibration_path)

# Step 2: Detect Features
reconstructed_scene.FeatureDetectionSIFT(
    max_keypoints=15000,
    contrast_threshold=0.009,
    edge_threshold=20
)

# Step 3: Detect Feature Pairs
reconstructed_scene.FeatureMatchFlannPair(
    detector="sift", 
    RANSAC_threshold=1.0,
    lowes_thresh=0.8
)

# Step 4: Detect/Estimate Camera Poses
reconstructed_scene.CamPoseEstimatorEssentialToPnP(
    reprojection_error=3.0,
    iteration_count=300,
    confidence=0.99,
    optimizer = ("BundleAdjustmentOptimizerLocal", {
        "max_num_iterations": 20,
        "window_size": 5,
        "use_gpu": False
    }),
)