import cv2
import math
import numpy as np
import glob
import json
import sys

from tqdm import tqdm
from lightglue import ALIKED 

from modules.DataTypes.pointDT import Points2D
from modules.DataTypes.cameraDT import CameraData

from modules.baseclass import FeatureClass
from modules.models.features import SuperPoint, load_image, numpy_image_to_torch 
import torch

############################################ HELPER FUNCTIONS ############################################

def anms_ssc(keypoints: list,  
             cols: int, 
             rows: int,
             num_ret_points: int = 1000, 
             tolerance: float = 0.1):
    
    exp1 = rows + cols + 2 * num_ret_points
    exp2 = (
        4 * cols
        + 4 * num_ret_points
        + 4 * rows * num_ret_points
        + rows * rows
        + cols * cols
        - 2 * rows * cols
        + 4 * rows * cols * num_ret_points
    )
    exp3 = math.sqrt(exp2)
    exp4 = num_ret_points - 1

    sol1 = -round(float(exp1 + exp3) / exp4)  # first solution
    sol2 = -round(float(exp1 - exp3) / exp4)  # second solution

    high = (
        sol1 if (sol1 > sol2) else sol2
    )  # binary search range initialization with positive solution
    low = math.floor(math.sqrt(len(keypoints) / num_ret_points))

    prev_width = -1
    selected_keypoints = []
    result_list = []
    result = []
    complete = False
    k = num_ret_points
    k_min = round(k - (k * tolerance))
    k_max = round(k + (k * tolerance))

    while not complete:
        width = low + (high - low) / 2
        if (
            width == prev_width or low > high
        ):  # needed to reassure the same radius is not repeated again
            result_list = result  # return the keypoints from the previous iteration
            break

        c = width / 2  # initializing Grid
        num_cell_cols = int(math.floor(cols / c))
        num_cell_rows = int(math.floor(rows / c))
        covered_vec = [
            [False for _ in range(num_cell_cols + 1)] for _ in range(num_cell_rows + 1)
        ]
        result = []

        for i in range(len(keypoints)):
            row = int(
                math.floor(keypoints[i].pt[1] / c)
            )  # get position of the cell current point is located at
            col = int(math.floor(keypoints[i].pt[0] / c))
            if not covered_vec[row][col]:  # if the cell is not covered
                result.append(i)
                # get range which current radius is covering
                row_min = int(
                    (row - math.floor(width / c))
                    if ((row - math.floor(width / c)) >= 0)
                    else 0
                )
                row_max = int(
                    (row + math.floor(width / c))
                    if ((row + math.floor(width / c)) <= num_cell_rows)
                    else num_cell_rows
                )
                col_min = int(
                    (col - math.floor(width / c))
                    if ((col - math.floor(width / c)) >= 0)
                    else 0
                )
                col_max = int(
                    (col + math.floor(width / c))
                    if ((col + math.floor(width / c)) <= num_cell_cols)
                    else num_cell_cols
                )
                for row_to_cover in range(row_min, row_max + 1):
                    for col_to_cover in range(col_min, col_max + 1):
                        if not covered_vec[row_to_cover][col_to_cover]:
                            # cover cells within the square bounding box with width w
                            covered_vec[row_to_cover][col_to_cover] = True

        if k_min <= len(result) <= k_max:  # solution found
            result_list = result
            complete = True
        elif len(result) < k_min:
            high = width - 1  # update binary search range
        else:
            low = width + 1
        prev_width = width

    # for i in range(len(result_list)):
    #     selected_keypoints.append(keypoints[result_list[i]])

    return result_list # List of indices of chosen points

##########################################################################################################

class FeatureDetectionSIFT(FeatureClass):
    def __init__(self, #image_path:str | None,
                 cam_data: CameraData,
                 max_keypoints: int = 1024, 
                 n_octave_layers: int = 3,
                 contrast_threshold: float = 0.04,
                 edge_threshold: int = 10,
                 sigma: float = 1.6):

        """
        Detect Features (Keypoints and Descriptors) using the SIFT algorithm

        Input: Path to image list
        Output: 
            list[Points2D]:
                Points2D (Detected Features per Scene):
                    points2D:       [N x 2] np.float32
                    descriptors:    [N x 128] np.float32
                    scores:         [N x 1] np.float32
                    image_size:     [1 x 2] np.int64
        """

        module_name = "FeatureDetectionSIFT"

        description = f"""
Detects existing keypoints(features) and descriptors in images using the feature detector 
SIFT. This Feature Detector is used when accurate and robust feature detection with
detailed description generation based algorithms are the priority. USE THIS MODULE in cases
where the environment is well-lit AND Highly-textured, and contains consistent lighting through the
set of images for a descriptive classical detector. If scene is Moderately or Mixed textured, EVEN if object
is highly textured, OPT to use SuperPoint instead. Only use in High-Textured scenes with Good Lighting,
When specified directly for SIFT or when classical based feature detection is called for with robust detection required
utilize the SIFT feature detector module. 

Initialization/Function Parameters: 
- cam_data: Data container to hold images and calibration data, read from the CameraDataManager.
- max_keypoints: Maximum number of Keypoints to detect per image from the feature detector
    - Default (int): 1024
- n_octave_layers: The number of layers in each octave. The number of octaves is computed automatically 
from the image resolution.
    - Default (int): 3 (NOTE: 3 is the value used in D. Lowe paper)
- contrast_threshold: The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions. 
The larger the threshold, the less features are produced by the detector. NOTE: The contrast threshold will be divided 
by nOctaveLayers when the filtering is applied.
    - Default (float): 0.04
- edge_threshold: The threshold used to filter out edge-like features. Note that the its meaning is different from the 
contrastThreshold, i.e. the larger the edgeThreshold, the more features are produced.
    - Default (int): 10
- sigma: The sigma of the Gaussian applied to the input image at the octave #0. If your image is captured with a weak 
camera with soft lenses, you might want to reduce the number. 
    - Default (float): 1.6

Module Output - Handled with SfMScene Object: 
    list[Points2D]:
        Points2D (Detected Features per Scene):
            points2D:       [N x 2] np.float32
            descriptors:    [N x 128] np.float32
            scores:         [N x 1] np.float32
            image_size:     [1 x 2] np.int64
"""
        
        example = f"""
Initialization of Module: 
from modules.baseclass import SfMScene
from modules.features import {module_name}

# Step 1: Read in Calibration/Image Data
reconstructed_scene = SfMScene(image_path = image_path, 
                            calibration_path = calibration_path)

# Step 2: Detect Features
reconstructed_scene.{module_name}(
    max_keypoints=9000,
    contrast_threshold=0.02,
    edge_threshold=20,
    )
"""
        
        # Set up Initialization
        super().__init__(cam_data=cam_data,
                         module_name=module_name,
                         description=description,
                         example=example)
        
        
        self.detector = cv2.SIFT_create(nfeatures = max_keypoints,
                                        nOctaveLayers = n_octave_layers,
                                        contrastThreshold = contrast_threshold,
                                        edgeThreshold =edge_threshold,
                                        sigma = sigma)


    def _detect_features(self) -> list[Points2D]:
        eps=1e-7

        for i in tqdm(range(len(self.image_list)), desc="Detecting Features"):

            img = np.asarray(self.image_list[i]) 
            im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 

            kp, des = self.detector.detectAndCompute(im_gray, None)
            pts = np.array([kp[i].pt for i in range(len(kp))], np.float32)
            des = np.float32(des)
            # apply the Hellinger kernel by first L1-normalizing and taking the
            # square-root
            des /= (des.sum(axis=1, keepdims=True) + eps)
            des = np.sqrt(des)

            scores = np.vstack(np.array([kp[i].response for i in range(len(kp))], np.float32))
            scale = np.array([[kp[i].size for i in range(len(kp))]], np.float32)
            ori = np.array([[kp[i].angle for i in range(len(kp))]], np.float32)
            image_size = np.array([im_gray.shape[1], im_gray.shape[0]])
            
            self.features.append(Points2D(points2D = pts, 
                                    descriptors = des,
                                    scores = scores, 
                                    image_size = image_size,
                                    reshape_scale=self.image_scale,
                                    scale=scale,
                                    orientation=ori))

        return self.features
    
class FeatureDetectionORB(FeatureClass):
    def __init__(self, #image_path:str | None,
                 cam_data: CameraData,
                 max_keypoints: int = 1024,
                 fast_threshold: int = 20,
                 edge_threshold: int = 31,
                 WTA_K: int = 2,
                 set_nms: bool = False,
                 set_nms_allowed_points: int = 3000,
                 set_nms_tolerance: float = 0.1):
        
        """
        Detect Features (Keypoints and Descriptors) using the ORB algorithm

        Input: Path to image list
        Output: 
            list[Points2D]:
                Points2D (Detected Features per Scene):
                    points2D: [N x 2] np.float32
                    descriptors: [N x 32] np.uint8
                    scores:         [N x 1] np.float32
                    image_size: [1 x 2] np.int64
        """

        module_name = "FeatureDetectionORB"

        description = f"""
Detects existing keypoints(features) and descriptors in images using the feature detector 
ORB. Detects oriented FAST keypoints and computes efficient binary descriptors. 
USE THIS DETECTOR in well-lit, highly textured environments with consistent image overlap and 
limited changes in scale or viewpoint. It is best suited to real-time visual odometry, SLAM, 
CPU-only systems, and embedded devices where speed is more important than maximum matching 
accuracy. Avoid it for severe illumination changes, weak texture, or large viewpoint differences.

Initialization/Function Parameters: 
- cam_data: Data container to hold images and calibration data, read from the CameraDataManager.
- max_keypoints: Maximum number of Keypoints to detect per image from the feature detector
    - Default (int): 1024
- edge_threshold: This is size of the border where the features are not detected. It should roughly match the patchSize parameter
  This prevents finding features too close to the image boundary where descriptors might not be computed correctly
    - Default (int): 31
- WTA_K: The number of points that produce each element of the oriented BRIEF descriptor.
    - Default (int): 2
- fast_threshold: This is the value to determine the pixel threshold for brightness, or dimness, that is used to estimate a point is a corner.
  A lower value will detect more potential corners, while a higher value will be more selective.
    - Default (int): 20
- set_nms: Utilize Non-Maximum Supression on detected feature points to create a uniform distribution of points and avoid clusters. 
  Useful in cases of highly textured regions and need a larger distance between points, but significantly increases time complexity.
    - Default (bool): False
- set_nms_allowed_points: number of points to search in algorithm. Ensure it is less that max_points to detect.
    - Defauilt (int): 3000
- set_nms_tolerance: adaptive nms tolerance value
    - Default (float): 0.1

Module Output - Handled with SfMScene Object: 
    list[Points2D]:
        Points2D (Detected Features per Scene):
            points2D: [N x 2] np.float32
            descriptors: [N x 32] np.uint8
            scores:         [N x 1] np.float32
            image_size: [1 x 2] np.int64
"""
        
        example = f"""
from modules.baseclass import SfMScene
from modules.features import {module_name}

Initialization of Module: 
# Step 1: Read in Calibration/Image Data
reconstructed_scene = SfMScene(image_path = image_path, 
                            calibration_path = calibration_path)

# Step 2: Detect Features
reconstructed_scene.FeatureDetectionORB(
    max_keypoints=9000,
    contrast_threshold=0.02,
    edge_threshold=20,
    )
"""        
        # Set up Initialization
        super().__init__(cam_data=cam_data,
                         module_name=module_name,
                         description=description,
                         example=example)


        err_msg = f"Value of the parameter of WTA_K is {WTA_K}. Ensure parameter value of 1, 2, 3, or 4. WTA_K can not be any other value."
        assert (WTA_K < 5 and WTA_K > 0), err_msg

        self.detector = cv2.ORB_create(nfeatures = max_keypoints,
                                       WTA_K = WTA_K,
                                       fastThreshold = fast_threshold,
                                       edgeThreshold = edge_threshold)

        self.nms = set_nms
        self.set_nms_allowed_points = set_nms_allowed_points
        self.set_nms_tolerance = set_nms_tolerance

    def _detect_features(self) -> list[Points2D]:
        
        for i in tqdm(range(len(self.image_list)), desc="Detecting Features"):
            img = np.asarray(self.image_list[i])
            im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # Convert Images to Gray

            kp, des = self.detector.detectAndCompute(im_gray, None)
            
            if self.nms: 
                sorted_indices = sorted(range(len(kp)), key=lambda x: kp[x].response, reverse=True)
                kp = sorted(kp, key=lambda x: x.response, reverse=True)
                des = des[sorted_indices]
                result_list = anms_ssc(kp, 
                                       tolerance=self.set_nms_tolerance,
                                       cols=im_gray.shape[1], 
                                       rows=im_gray.shape[0],
                                       num_ret_points=self.set_nms_allowed_points)
            
                pts = np.array([kp[i].pt for i in result_list], np.float32)
                scores = np.vstack(np.array([kp[i].response for i in result_list], np.float32))
                scale = np.array([[kp[i].size for i in result_list]], np.float32)
                ori = np.array([[kp[i].angle for i in result_list]], np.float32)
                des = des[result_list]
            else:
                pts = np.array([kp[i].pt for i in range(len(kp))], np.float32)
                scores = np.vstack(np.array([kp[i].response for i in range(len(kp))], np.float32))
                scale = np.array([[kp[i].size for i in range(len(kp))]], np.float32)
                ori = np.array([[kp[i].angle for i in range(len(kp))]], np.float32)

            image_size = np.array([im_gray.shape[1], im_gray.shape[0]])

            self.features.append(Points2D(points2D = pts, 
                                          descriptors = des,
                                          scores = scores,
                                          scale = scale,
                                          orientation = ori,
                                          image_size = image_size,
                                          reshape_scale=self.image_scale,
                                          binary_desc=True))
        return self.features

#### DEEP LEARNING MODELS #####
class FeatureDetectionALIKED(FeatureClass):
    def __init__(self, 
                 cam_data: CameraData,
                 max_keypoints: int = 1024,
                 det_thres: float = 0.005):
        """
            Detect Features (Keypoints and Descriptors) using the SuperPoint Deep Learning Model

            Assume Calibration is zero-based for now for proper image-reshaping

            Input: Path to image list
            Output: 
                list[Points2D]:
                    Points2D (Detected Features per Scene):
                        points2D:       [N x 2] np.float32
                        descriptors:    [N x 256] np.float32
                        scores:         [N x 1] np.float32
                        image_size:     [1 x 2] np.int64
            """

        module_name = "FeatureDetectionALIKED"

        description = f"""
    Detects existing keypoints(features) and descriptors in images using a Deep Learning Model Feature
    Detector denoted as ALIKED. Detects accurate subpixel keypoints and learned descriptors using a 
    lightweight neural architecture. USE THIS DETECTOR when scenes contain moderate or irregular texture 
    and strong localization accuracy is needed without the computational cost of a heavier learned detector. 
    It is suitable for real-time SfM, visual odometry, and localization in changing indoor or outdoor 
    environments, particularly when paired with LightGlue. Like other appearance-based detectors, performance 
    can degrade in extremely dark or nearly textureless regions.

    Initialization/Function Parameters: 
    - max_keypoints: Maximum number of Keypoints to detect per image from the feature detector
        - Default (int): 1024
    - det_thresh: Threshold for feature detection. Higher the number, more features are detected but
      less confident in point accuracy.

    Module Output - Handled with SfMScene Object: 
        list[Points2D]:
            Points2D (Detected Features per Scene):
                points2D: [N x 2] np.float32
                descriptors: [N x 256] np.float32
                scores:         [N x 1] np.float32
                image_size: [1 x 2] np.int64
    """

        example = f"""
    from modules.baseclass import SfMScene
    from modules.features import {module_name}

    Initialization of Module: 
    # Step 1: Read in Calibration/Image Data
    reconstructed_scene = SfMScene(image_path = image_path, 
                                calibration_path = calibration_path)

    # Step 2: Detect Features
    reconstructed_scene.{module_name}(
        max_keypoints=9000,
        det_thresh=0.005
        )
    """     

        # Set up Initialization
        super().__init__(cam_data=cam_data,
                        module_name=module_name,
                        description=description,
                        example=example)

        # Set Up Model
        self.device = torch.device(f"cuda:{self.cam_data.gpu_num}" if torch.cuda.is_available() else "cpu")
        aliked_detector = ALIKED(max_num_keypoints=max_keypoints, detection_threshold=det_thres)
        self.detector = aliked_detector.to(self.device).eval()

    def _detect_features(self) -> list[Points2D]:
        
        for i in tqdm(range(len(self.image_list)), desc="Detecting Features"): 
            
            img = np.asarray(self.image_list[i])
            img_torch = numpy_image_to_torch(img)
        
            # Detect Keypoints
            features = self.detector.extract(img_torch.to(self.device))
            
            keypoints = features['keypoints'].cpu().numpy().squeeze()
            desc = features['descriptors'].cpu().numpy().squeeze()
            scores = np.vstack(features['keypoint_scores'].cpu().numpy().squeeze())

            image_size = np.array([img_torch.shape[2], img_torch.shape[1]])
            
            self.features.append(Points2D(points2D = keypoints, 
                                        descriptors = desc,
                                        scores = scores, 
                                        image_size = image_size,
                                        reshape_scale=self.image_scale))
        
        # Output Metric Handled in set __Call__ functions

        return self.features

class FeatureDetectionSP(FeatureClass):
    def __init__(self, 
                 cam_data: CameraData,
                 max_keypoints: int = 1024):
        """
        Detect Features (Keypoints and Descriptors) using the SuperPoint Deep Learning Model

        Assume Calibration is zero-based for now for proper image-reshaping

        Input: Path to image list
        Output: 
            list[Points2D]:
                Points2D (Detected Features per Scene):
                    points2D:       [N x 2] np.float32
                    descriptors:    [N x 256] np.float32
                    scores:         [N x 1] np.float32
                    image_size:     [1 x 2] np.int64
        """

        module_name = "FeatureDetectionSP"

        description = f"""
Detects existing keypoints(features) and descriptors in images using a Deep Learning Model Feature
Detector denoted as SuperPoint.
Jointly detects learned keypoints and descriptors in a single neural-network pass. USE THIS DETECTOR 
for indoor or outdoor scenes with moderate texture, repeated patterns, illumination changes, or viewpoints 
where traditional corner detectors are unreliable. It is especially suitable for SfM and visual localization 
when paired with SuperGlue or LightGlue and GPU inference is available. Extremely dark, blurred, or 
domain-specific imagery may still require preprocessing or model adaptation.

Use in cases where SIFT and ORB may struggle in due to not well-lit settings, and cases where there is
diffuse lighting in the object and we need a feature detector more robust to these environments.
When specified directly to use the SuperPoint algorithm, mentioning to use a feature detector 
to handle view changes or material that lack texture in a given scene, or accurate dense features 
are necessary, use the SuperPoint detection Module.

Initialization/Function Parameters: 
- max_keypoints: Maximum number of Keypoints to detect per image from the feature detector
    - Default (int): 1024

Module Output - Handled with SfMScene Object: 
    list[Points2D]:
        Points2D (Detected Features per Scene):
            points2D: [N x 2] np.float32
            descriptors: [N x 256] np.float32
            scores:         [N x 1] np.float32
            image_size: [1 x 2] np.int64
"""

        example = f"""
from modules.baseclass import SfMScene
from modules.features import {module_name}

Initialization of Module: 
# Step 1: Read in Calibration/Image Data
reconstructed_scene = SfMScene(image_path = image_path, 
                            calibration_path = calibration_path)

# Step 2: Detect Features
reconstructed_scene.{module_name}(
    max_keypoints=9000,
    )
"""     

        # Set up Initialization
        super().__init__(cam_data=cam_data,
                         module_name=module_name,
                         description=description,
                         example=example)

        self.device = torch.device(f"cuda:{self.cam_data.gpu_num}" if torch.cuda.is_available() else "cpu") 

        self.detector = SuperPoint(max_num_keypoints=max_keypoints).eval().to(self.device) 

    def _detect_features(self) -> list[Points2D]:
        
        for i in tqdm(range(len(self.image_list)), desc="Detecting Features"): 
            
            img = np.asarray(self.image_list[i])
            img_torch = numpy_image_to_torch(img)
        
            # Detect Keypoints
            features = self.detector.extract(img_torch.to(self.device))
            
            keypoints = features['keypoints'].cpu().numpy().squeeze()
            desc = features['descriptors'].cpu().numpy().squeeze()
            scores = np.vstack(features['keypoint_scores'].cpu().numpy().squeeze())

            image_size = np.array([img_torch.shape[2], img_torch.shape[1]])
            
            self.features.append(Points2D(points2D = keypoints, 
                                        descriptors = desc,
                                        scores = scores, 
                                        image_size = image_size,
                                        reshape_scale=self.image_scale))
        
        # Output Metric Handled in set __Call__ functions

        return self.features