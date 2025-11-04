from modules.utilities.utilities import CalibrationReader
from modules.features import FeatureDetectionSIFT, FeatureDetectionSP, FeatureDetectionORB, FeatureDetectionFAST
from modules.featurematching import (FeatureMatchFlannTracking, 
                                     FeatureMatchFlannPair,
                                     FeatureMatchBFPair,
                                     FeatureMatchLoftrPair,
                                     FeatureMatchLightGlueTracking, 
                                     FeatureMatchSuperGlueTracking, 
                                     FeatureMatchLightGluePair, 
                                     FeatureMatchSuperGluePair,
                                     FeatureMatchRoMAPair)
from modules.cameramanager import CameraDataManager
from modules.visualize import VisualizeScene
import glob
import cv2
import os
import numpy as np

# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\datasets\\Structure-from-Motion\\sfm_dataset"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\Structure-from-Motion\\calibration_new.npz"
# image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\scan6_low_lighting"
# calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\calibration_DTU.npz"

# Feature Module Initialization
# calibration_data = CalibrationReader(calibration_path).get_calibration()

CDM = CameraDataManager(image_path=image_path, calibrated=True, calibration_path=calibration_path)

cam_data = CDM.get_camera_data()

print(cam_data.get_K(0))
print(len(cam_data.image_list))

cam_data.update_calibration(cam_data.image_scale)
print(cam_data.get_K(0))