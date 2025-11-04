from modules.utilities.utilities import CalibrationReader
from modules.features import FeatureDetectionSIFT, FeatureDetectionSP, FeatureDetectionORB
from modules.featurematching import (FeatureMatchFlannPair, 
                                     FeatureMatchLoftrPair, 
                                     FeatureMatchLightGluePair, 
                                     FeatureMatchLightGlueTracking,
                                     FeatureMatchSuperGluePair,
                                     FeatureMatchRoMAPair)
from modules.camerapose import CamPoseEstimatorEssentialToPnP, CamPoseEstimatorVGGTModel
from modules.scenereconstruction import Sparse3DReconstructionVGGT
from modules.visualize import VisualizeScene
import glob
import cv2
# import matplotlib.pyplot as plt
# from matplotlib.patches import Circle

# import open3d as o3d
# import open3d.visualization.gui as gui
# import open3d.visualization.rendering as rendering
# import numpy as np
# import matplotlib.pyplot as plt


# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\datasets\\Structure-from-Motion\\sfm_dataset"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\Structure-from-Motion\\calibration.npz"
# image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\scan6_normal_lighting"
# calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\calibration_DTU.npz"

# Feature Module Initialization
calibration_data = CalibrationReader(calibration_path).get_calibration()

pose_estimator = CamPoseEstimatorVGGTModel(image_path=image_path, calibration=calibration_data) #CamPoseEstimatorEssentialToPnP(calibration=calibration_data, image_path=image_path, detector="sift")

scene_builder = Sparse3DReconstructionVGGT(calibration=calibration_data, image_path=image_path)
# Solution Pipeline

cam_poses = pose_estimator()#(matched_features) # (detected_features)

opt_scene = scene_builder(camera_poses=cam_poses)

print(opt_scene.points3D.points3D.shape)