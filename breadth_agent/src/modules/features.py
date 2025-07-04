import cv2
import numpy as np
import glob
from tqdm import tqdm
from .DataTypes.datatype import Points2D, Calibration
from baseclass import FeatureClass

# LISTED_TYPES = ['corner', 'sift', 'orb']
# MATCHERS = ['flann', 'bruteforce', 'bf']


class FeatureDetectionSIFT(FeatureClass):
    def __init__(self, image_path:str | None):
        self.module_name = "FeatureDetection"
        self.description = f"""
Detects existing keypoints(features) and descriptors in images using the feature detector 
SIFT. Default detector is SIFT. Unless specified, assume the feature detector that will be used is SIFT. 
The initialization parameter is the image path in which the images are stored.   
""" 
        # Provide examples of how to invoke the function calls -> Not Prompts to invoke it..
#         self.example = f"""
# 1. In this step, we will detect keypoints in sequential images using a feature detection algorithm (e.g., ORB, SIFT).
# 2. In this step, use the SIFT feature detector to detect features in sequential images
# 3. In this step, detect features in the left images of a stereo camera using the SIFT feature detector
# """
        self.example = f"""
Initialization: feature_detector = FeatureDetectionSIFT(image_path=image_path) # image_path is a string for path to saved images
Function call:  features = feature_detector()
"""
        super().__init__(image_path)
        print(self.image_path)

        self.detector = cv2.SIFT_create()

    def __call__(self) -> list[Points2D]:
        for i in tqdm(range(len(self.image_path))): # len(self.image_path)
        # img in self.image_path:
            img = self.image_path[i]
            im_gray = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY)

            kp, des = self.detector.detectAndCompute(im_gray, None)
            pts = np.array([kp[i].pt for i in range(len(kp))])

            self.features.append(Points2D(points2D = pts, descriptors=des))
        
        return self.features
    

class FeatureDetectionORB(FeatureClass):
    def __init__(self, image_path:str | None):
        self.module_name = "FeatureDetection"
        self.description = f"""
Detects existing keypoints(features) and descriptors in images using the feature detector 
ORB. When specified directly or efficient feature detection is necessary, utilize the ORB feature detector.
The initialization parameter is the image path in which the images are stored.        
"""
#         self.example = f"""
# 1. In this step, we will detect keypoints in sequential images using a feature detection algorithm ORB.
# 2. In this step, use the ORB feature detector to detect features in sequential images
# 3. In this step, we will detect keypoints in sequential images using a feature detection, but ensure this process is fast.
# 4. In this step, detect features in the left images of a stereo camera using the ORB feature detector
# """
        self.example = f"""
Initialization: feature_detector = FeatureDetectionORB(image_path=image_path) # image_path is a string for path to saved images
Function call:  features = feature_detector()
"""
        super().__init__(image_path)
        print(self.image_path)

        self.detector = cv2.ORB_create()

    def __call__(self) -> list[Points2D]:
        for i in tqdm(range(12)): # len(self.image_path)
        # img in self.image_path:
            img = self.image_path[i]
            im_gray = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY)

            kp, des = self.detector.detectAndCompute(im_gray, None)
            pts = np.array([kp[i].pt for i in range(len(kp))])

            self.features.append(Points2D(points2D = pts, descriptors=des))
        
        return self.features

            
class FeatureDetectionFAST(FeatureClass):
    def __init__(self, image_path:str | None):
        self.module_name = "FeatureDetection"
        self.description = f"""
Detects existing keypoints(features) and descriptors in images using the FAST feature detector. 
When specified directly or mentioning fast feature detection to simulate real-time use, 
utilize the FAST feature detector.
The initialization parameter is the image path in which the images are stored.   
"""
#         self.example = f"""
# 1. In this step, we will detect keypoints in sequential images using the FAST Feature Detector for corner points.
# 2. In this step, use the FAST feature detector to detect features in sequential images.
# 3. In this step, we will detect keypoints in sequential images using a feature detection, but ensure to use a very fast feature detector.
# 4. In this step, detect features in the left images of a stereo camera using the FAST feature detector.
# 5. In this step, we will use a feature detector to detect features for a process in a real-time setting. 
# """
        self.example = f"""
Initialization: feature_detector = FeatureDetectionFAST(image_path=image_path) # image_path is a string for path to saved images
Function call:  features = feature_detector()
"""
        super().__init__(image_path)
        print(self.image_path)

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
            
            pts = np.array([kp[i].pt for i in range(len(kp))])

            self.features.append(Points2D(points2D = pts, descriptors=des))
        
        return self.features

