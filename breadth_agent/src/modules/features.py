import cv2
import math
import numpy as np
import glob
from tqdm import tqdm
from modules.DataTypes.datatype import Points2D, Calibration, CameraData
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
        #super().__init__(image_path)
        super().__init__(cam_data)
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
        self.module_name = "FeatureDetectionSIFT"
        self.description = f"""
Detects existing keypoints(features) and descriptors in images using the feature detector 
SIFT. This Feature Detector is used when accurate and robust feature detection with
detailed description generation based algorithms are the priority. When specified directly 
for SIFT or when classical based feature detection is called for with robust detection required
utilize the SIFT feature detector module. 

Initialization Parameters: 
- image_path (str): the image path in which the images are stored and to utilize for scene building
- max_keypoints: Maximum number of Keypoints to detect per image from the feature detector
    - Default (int): 1024
- n_octave_threshold: The number of layers in each octave. The number of octaves is computed automatically 
from the image resolution.
    - Default (int): 3 (NOTE: 3 is the value used in D. Lowe paper)
- contrast_threshold: The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions. 
The larger the threshold, the less features are produced by the detector. NOTE: The contrast threshold will be divided 
by nOctaveLayers when the filtering is applied.
    - Default (flaot): 0.04
- edgeThreshold: The threshold used to filter out edge-like features. Note that the its meaning is different from the 
contrastThreshold, i.e. the larger the edgeThreshold, the less features are filtered out (more features are retained).
    - Default (int): 10
- sigma: The sigma of the Gaussian applied to the input image at the octave #0. If your image is captured with a weak 
camera with soft lenses, you might want to reduce the number. 
    - Default (float): 1.6
- image_reshape: shape of image in the format of (int, int) = (Height, Width) to reshape current image to.
    - Default (int, int): None (No Reshape takes place by default)

Module Output: 
    list[Points2D]:
        Points2D (Detected Features per Scene):
            points2D:       [N x 2] np.float32
            descriptors:    [N x 128] np.float32
            scores:         [N x 1] np.float32
            image_size:     [1 x 2] np.int64
"""
        
        self.example = f"""
Initialization of Module: 
feature_detector = FeatureDetectionSIFT(image_path=image_path, max_keypoints = 2000) 

Function call of Module:  
features = feature_detector()
"""
        

        self.detector = cv2.SIFT_create(nfeatures = max_keypoints,
                                        nOctaveLayers = n_octave_layers,
                                        contrastThreshold = contrast_threshold,
                                        edgeThreshold =edge_threshold,
                                        sigma = sigma)

        # self.image_reshape, self.reshape_scale = self._det_img_shape(self.image_path[0],
        #                                                              image_reshape)
        # print(self.image_reshape)
        # print(self.reshape_scale)
        print("Number of Images", len(self.image_list))
        print("New Image Scale", self.image_scale)

        # Update Image Scale accordingly
        cam_data.update_calibration(self.image_scale)

        # self.image_reshape = None
        # self.reshape_scale = [1.0, 1.0]
        # print(self.reshape_scale)
        # if image_reshape is not None:
        #     img = self.image_path[0]
        #     h, w = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY).shape[:2]
        #     h_new, w_new = image_reshape
        #     self.reshape_scale = [w_new / w, h_new / h]
        #     self.image_rehsape = image_reshape
        # else:
        #     self.reshape_scale = [1.0, 1.0]
        #     self.image_rehsape = None

    def __call__(self) -> list[Points2D]:

        # New Version
        # if self.image_reshape is not None:
        #     self.features = self._detect_resize()
        # else: 
        #     self.features = self._detect_base()
        # return self.features

        features = []
        eps=1e-7

        #for i in tqdm(range(len(self.image_path))): # len(self.image_path)
        for i in tqdm(range(len(self.image_list)),
                      desc="Detecting Features"):

            img = self.image_list[i] #self.image_path[i]
            im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY)

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
            
            features.append(Points2D(points2D = pts, 
                                    descriptors = des,
                                    scores = scores, 
                                    image_size = image_size,
                                    reshape_scale=self.image_scale,
                                    scale=scale,
                                    orientation=ori))
            
        return features
    
    def _detect_resize(self) -> list[Points2D]:
        features = []
        eps=1e-7
        
        for i in tqdm(range(len(self.image_path))): # len(self.image_path)
        # img in self.image_path:
            img = self.image_path[i]
            im_gray = cv2.resize(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY),
                                 self.image_reshape, 
                                 interpolation=cv2.INTER_AREA)

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
            
            features.append(Points2D(points2D = pts, 
                                    descriptors = des,
                                    scores = scores, 
                                    image_size = image_size,
                                    reshape_scale=self.reshape_scale,
                                    scale=scale,
                                    orientation=ori))
            
        return features

    def _detect_base(self) -> list[Points2D]:
        features = []
        eps=1e-7

        #for i in tqdm(range(len(self.image_path))): # len(self.image_path)
        for i in range(len(self.image_list)):
        # img in self.image_path:
            img = self.image_list[i] #self.image_path[i]
            im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY)

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
            
            features.append(Points2D(points2D = pts, 
                                    descriptors = des,
                                    scores = scores, 
                                    image_size = image_size,
                                    reshape_scale=self.image_scale,
                                    scale=scale,
                                    orientation=ori))
            
        return features
    

class FeatureDetectionORB(FeatureClass):
    def __init__(self, #image_path:str | None,
                 cam_data: CameraData,
                 max_keypoints: int = 1024,
                 fast_threshold: int = 20,
                 edge_threshold: int = 31,
                 WTA_K: int = 2,
                 set_nms: bool = True):
        
        #super().__init__(image_path)
        super().__init__(cam_data)
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

        self.module_name = "FeatureDetectionORB"
#- image_path (str): the image path in which the images are stored and to utilize for scene building

        self.description = f"""
Detects existing keypoints(features) and descriptors in images using the feature detector 
ORB. This Feature Detector is used when efficiency is the priority as ORB utilizes faster feature detection
and description generation based algorithms. When specified directly for ORB or when quick/efficient 
feature detection is necessary, and called for, utilize the ORB feature detector module. 

Initialization Parameters: 
- image_list (Images): a Python class in which the images are stored and to utilize for scene building
- max_keypoints: Maximum number of Keypoints to detect per image from the feature detector
    - Default (int): 1024
- edge_threshold: This is size of the border where the features are not detected. It should roughly match the patchSize parameter
- WTA_K: The number of points that produce each element of the oriented BRIEF descriptor.
- fast_threshold: This is the value to determine the pixel threshold for brightness, or dimness, that is used to estimate a point is a corner.
- image_reshape: shape of image in the format of (int, int) = (Height, Width) to reshape current image to.
    - Default (int, int): None (No Reshape takes place by default)

Module Output: 
    list[Points2D]:
        Points2D (Detected Features per Scene):
            points2D: [N x 2] np.float32
            descriptors: [N x 32] np.uint8
            scores:         [N x 1] np.float32
            image_size: [1 x 2] np.int64
"""
        
        self.example = f"""
Initialization of Module: 
image_reader = ImageProcessor(image_path=image_path)
images = image_reader(calibration_data)

feature_detector = FeatureDetectionORB(image_list=images, max_keypoints = 2000) 

Function call of Module:  
features = feature_detector()

"""        
        err_msg = f"Value of the parameter of WTA_K is {WTA_K}. Ensure parameter value of 1, 2, 3, or 4. WTA_K can not be any other value."
        assert (WTA_K < 5 and WTA_K > 0), err_msg
        # print(self.image_path)

        self.detector = cv2.ORB_create(nfeatures = max_keypoints,
                                       WTA_K = WTA_K,
                                       fastThreshold = fast_threshold,
                                       edgeThreshold = edge_threshold)
        self.nms = set_nms

        # Update Calibration if Images are Resized
        cam_data.update_calibration(self.image_scale)

    def __call__(self) -> list[Points2D]:
        # for i in tqdm(range(len(self.image_path))): # len(self.image_path)
        # # img in self.image_path:
        #     img = self.image_path[i]
        #     im_gray = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY)
        for i in tqdm(range(len(self.image_list)), desc="Detecting Features"):
        # img in self.image_path:
            img = self.image_list[i] #self.image_path[i]
            im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            kp, des = self.detector.detectAndCompute(im_gray, None)
            
            if self.nms: 
                result_list = anms_ssc(kp, 
                                   cols=im_gray.shape[1], 
                                   rows=im_gray.shape[0],
                                   num_ret_points=3000)
            
                pts = np.array([kp[i].pt for i in result_list], np.float32)
                scores = np.vstack(np.array([kp[i].response for i in result_list], np.float32))
                des = des[result_list]
            else:
                pts = np.array([kp[i].pt for i in range(len(kp))], np.float32)
                scores = np.vstack(np.array([kp[i].response for i in range(len(kp))], np.float32))

            #des = des.astype(np.float32)
            image_size = np.array([im_gray.shape[1], im_gray.shape[0]])

            self.features.append(Points2D(points2D = pts, 
                                          descriptors = des,
                                          scores = scores,
                                          image_size = image_size,
                                          reshape_scale=self.image_scale,
                                          binary_desc=True))
        
        return self.features

            
class FeatureDetectionFAST(FeatureClass):
    def __init__(self, image_path:str | None,
                 image_reshape: tuple[int, int] | None = None):
        super().__init__(image_path)
        """
        Detect Features (Keypoints and Descriptors) using the FAST algorithm

        Assume Calibration is zero-based for now for proper image-reshaping

        Input: Path to image list
        Output: 
            list[Points2D]:
                Points2D (Detected Features per Scene):
                    points2D:       [N x 2] np.float32
                    descriptors:    [N x 32] np.uint8
                    scores:         [N x 1] np.float32
                    image_size:     [1 x 2] np.int64
        """

        
        self.module_name = "FeatureDetectionFAST"
        self.description = f"""
Detects existing keypoints(features) and descriptors in images using the FAST feature detector. 
For Descriptor generation, this module uses the BRIEF description extractor algorithm.
This module is used when feature detection needs to be very fast, and or real-time, 
for a wokring solution pipeline, or mentioning fast feature detection to simulate 
real-time use is directly stated.

Initialization Parameters: 
- image_path (str): the image path in which the images are stored and to utilize for scene building
- image_reshape: shape of image in the format of (int, int) = (Height, Width) to reshape current image to.
    - Default (int, int): None (No Reshape takes place by default)

Module Output: 
    list[Points2D]:
        Points2D (Detected Features per Scene):
            points2D:       [N x 2] np.float32
            descriptors:    [N x 32] np.uint8
            scores:         [N x 1] np.float32
            image_size:     [1 x 2] np.int64  
"""

        self.example = f"""
Initialization: 
feature_detector = FeatureDetectionFAST(image_path=image_path) # image_path is a string for path to saved images

Function call:  
features = feature_detector()
"""

        self.detector = cv2.FastFeatureDetector_create(threshold=20)
        self.brief_des = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    def __call__(self) -> list[Points2D]:
        for i in tqdm(range(12)): # len(self.image_path)
        # img in self.image_path:
            img = self.image_path[i]
            im_gray = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY)

            # Keypoints
            kp = self.detector.detect(im_gray, None)
            # Descriptors
            kp, des = self.brief_des.compute(im_gray, kp)
            
            pts = np.array([kp[i].pt for i in range(len(kp))], np.float32)
            scores = np.vstack(np.array([kp[i].response for i in range(len(kp))], np.float32))

            image_size = np.array([im_gray.shape[1], im_gray.shape[0]]) 

            self.features.append(Points2D(points2D = pts, 
                                          descriptors = des,
                                          scores = scores, 
                                          image_size = image_size,
                                          reshape_scale=[1.0, 1.0]))
        
        return self.features

#### DEEP LEARNING MODELS #####

class FeatureDetectionSP(FeatureClass):
    def __init__(self, 
                 cam_data: CameraData,
                 max_keypoints: int = 1024):
        super().__init__(cam_data)
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


        self.module_name = "FeatureDetectionSP"
        self.description = f"""
Detects existing keypoints(features) and descriptors in images using a Deep Learning Model Feature
Detector denoted as SuperPoint.
This Feature Detector is utilized in cases where diffuse lighting materials exist in scenes, images of 
environments that lack texture, or extreme view changes exist in scene video or images.
When specified directly to use the SuperPoint algorithm, mentioning to use a feature detector 
to handle view changes or material that lack texture in a given scene, or accurate dense features 
are necessary, use the SuperPoint detection Module.

Initialization Parameters: 
- image_path (str): the image path in which the images are stored and to utilize for scene building
- max_keypoints: Maximum number of Keypoints to detect per image from the feature detector
    - Default (int): 1024
- image_reshape: shape of image in the format of (int, int) = (Height, Width) to reshape current image to.
    - Default (int, int): None (No Reshape takes place by default)

Module Output: 
    list[Points2D]:
        Points2D (Detected Features per Scene):
            points2D: [N x 2] np.float32
            descriptors: [N x 256] np.float32
            scores:         [N x 1] np.float32
            image_size: [1 x 2] np.int64
"""

        self.example = f"""
Initialization: 
# Reshape Image
feature_detector = FeatureDetectionSP(image_path=image_path, max_keypoints=2000, image_reshape=(480, 640)) 

# Keep image size
feature_detector = FeatureDetectionSP(image_path=image_path, max_keypoints=2000) 

Function call:  
features = feature_detector()
"""     

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

        self.detector = SuperPoint(max_num_keypoints=max_keypoints).eval().to(self.device) 

    def __call__(self) -> list[Points2D]:
        for i in tqdm(range(len(self.image_list)),
                      desc="Detecting Features"): # len(self.image_path)
            
        # img in self.image_path:
            img = self.image_list[i]
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            #img_read, scale = load_image(img, self.image_reshape) # Need Image Shape
            img_torch = numpy_image_to_torch(img)
            # print(img_read.shape)
            # Keypoints
            features = self.detector.extract(img_torch.to(self.device))
            
            keypoints = features['keypoints'].cpu().numpy().squeeze()
            desc = features['descriptors'].cpu().numpy().squeeze()
            scores = np.vstack(features['keypoint_scores'].cpu().numpy().squeeze())

            image_size = np.array([img_torch.shape[2], img_torch.shape[1]])
            
            # print(keypoints.shape)
            # print(desc.shape)
            self.features.append(Points2D(points2D = keypoints, 
                                        descriptors = desc,
                                        scores = scores, 
                                        image_size = image_size,
                                        reshape_scale=self.image_scale))
        
        return self.features