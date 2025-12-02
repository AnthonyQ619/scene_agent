from modules.utilities.utilities import CalibrationReader
from modules.features import FeatureDetectionSIFT, FeatureDetectionSP, FeatureDetectionORB
from modules.featurematching import (FeatureMatchFlannPair, 
                                     FeatureMatchLoftrPair, 
                                     FeatureMatchBFPair,
                                     FeatureMatchLightGluePair, 
                                     FeatureMatchLightGlueTracking,
                                     FeatureMatchSuperGluePair,
                                     FeatureMatchRoMAPair)
from modules.camerapose import CamPoseEstimatorEssentialToPnP, CamPoseEstimatorVGGTModel
from modules.visualize import VisualizeScene
from modules.cameramanager import CameraDataManager
import glob
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import matplotlib.pyplot as plt


# Construct Modules with Initialized Arguments
# image_path = "C:\\Users\\Anthony\\Documents\\Projects\datasets\\Structure-from-Motion\\sfm_dataset"
# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\scan6_normal_lighting"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\calibration_DTU_new.npz"
# image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\scan6_normal_lighting"
# calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\calibration_DTU.npz"

# # Read Camera Data
# CDM = CameraDataManager(image_path=image_path,
#                         calibration_path=calibration_path)
#                         # target_resolution=(1024, 1024))

# cam_data = CDM.get_camera_data()
# STEP 1: Read in Camera Data
from modules.cameramanager import CameraDataManager

CDM = CameraDataManager(image_path=image_path,
                        calibration_path=calibration_path)

# Get Camera Data
camera_data = CDM.get_camera_data()

# STEP 2: Estimate Features per Image
from modules.features import FeatureDetectionORB
# Feature Module Initialization
feature_detector = FeatureDetectionORB(cam_data=camera_data, 
                                        max_keypoints=5000)

# Detect Features for all Images
features = feature_detector()

# STEP 3: Match Features Per Image 
# from modules.featurematching import FeatureMatchFlannPair
# # Pairwise Feature Matching Module Initialization
# feature_matcher = FeatureMatchFlannPair(detector='orb', 
#                                         cam_data=camera_data,
#                                         RANSAC_threshold=0.07,
#                                         lowes_thresh=0.75)

from modules.featurematching import FeatureMatchBFPair
# Pairwise Feature Matching Module Initialization
feature_matcher = FeatureMatchBFPair(detector='orb', 
                                     cam_data=camera_data,
                                     RANSAC_threshold=0.01,
                                     lowes_thresh=0.6)

# Detect Image Pair Correspondences for Pose Estimation
feature_pairs = feature_matcher(features=features)

# STEP 4: Estimate Camera Poses of Scene with Feature Pairs
from modules.camerapose import CamPoseEstimatorEssentialToPnP
# Camera Pose Module Initialization
pose_estimator = CamPoseEstimatorEssentialToPnP(cam_data=camera_data,
                                                reprojection_error=4.0,
                                                iteration_count=340,
                                                confidence=0.995)

# From estimated features, estimate the camera poses for all image frames
cam_poses = pose_estimator(feature_pairs=feature_pairs)
# # Feature Module Initialization
# feature_detector = FeatureDetectionSIFT(cam_data=cam_data,
#                                       max_keypoints=3000,
#                                       ) #FeatureDetectionORB(image_path=image_path, max_keypoints=3000)
# # feature_matcher = FeatureMatchBFPair(detector="sift", 
# #                                      cam_data=cam_data,
# #                                      cross_check=False,
# #                                      RANSAC_threshold=0.005)

# # feature_matcher = FeatureMatchRoMAPair(img_path=image_path, setting="indoor")
# pose_estimator = CamPoseEstimatorVGGTModel(cam_data=cam_data,
#                                            image_path=image_path) #CamPoseEstimatorEssentialToPnP(calibration=calibration_data, image_path=image_path, detector="sift")
# # pose_estimator = CamPoseEstimatorEssentialToPnP(cam_data=cam_data,
# #                                                 reprojection_error=4.0,
# #                                                 iteration_count=200,
# #                                                 confidence=0.995)

# # Solution Pipeline

# detected_features = feature_detector()
# # matched_features = feature_matcher(detected_features)
# # matched_features = feature_matcher()

# # print(matched_features.access_matching_pair(0)[0].shape)
# # print(len(detected_features))

# # Include Pairwise feature matching here

# cam_poses = pose_estimator() #(matched_features) # (detected_features)

# print(cam_poses.camera_pose)
# print(calibration_data.K_cams)

new_point_cloud = []
for i in range(len(cam_poses.camera_pose)):
    new_point_cloud.append(cam_poses.camera_pose[i][:,3:])

new_point_cloud = np.array(new_point_cloud).squeeze()
print(new_point_cloud.shape)
print(new_point_cloud)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(new_point_cloud)

gui.Application.instance.initialize()

window = gui.Application.instance.create_window("Mesh-Viewer", 1024, 750)

scene = gui.SceneWidget()
scene.scene = rendering.Open3DScene(window.renderer)

window.add_child(scene)

matGT = rendering.MaterialRecord()
matGT.shader = 'defaultUnlit'
matGT.point_size = 7.0
matGT.base_color = np.ndarray(shape=(4,1), buffer=np.array([0.0, 0.0, 1.0, 1.0]), dtype=float)

scene.scene.add_geometry("mesh_name2", pcd, matGT)
scene.scene.add_geometry("mesh_name3", o3d.geometry.TriangleMesh.create_coordinate_frame(), rendering.MaterialRecord())

bounds = pcd.get_axis_aligned_bounding_box()
scene.setup_camera(60, bounds, bounds.get_center())

gui.Application.instance.run()  # Run until user closes window