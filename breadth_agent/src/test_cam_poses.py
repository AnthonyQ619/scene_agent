from modules.utilities.utilities import CalibrationReader
from modules.features import FeatureDetectionSIFT
from modules.camerapose import CamPoseEstimatorEssentialToPnP
from modules.visualize import VisualizeScene
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
image_path = "C:\\Users\\Anthony\\Documents\\Projects\datasets\\Structure-from-Motion\\sfm_dataset"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\Structure-from-Motion\\calibration.npz"

# Feature Module Initialization
calibration_data = CalibrationReader(calibration_path).get_calibration()
feature_detector = FeatureDetectionSIFT(image_path=image_path)
pose_estimator = CamPoseEstimatorEssentialToPnP(calibration=calibration_data, image_path=image_path, detector="sift")


# Solution Pipeline

detected_features = feature_detector()
print(len(detected_features))

# Include Pairwise feature matching here

cam_poses = pose_estimator(detected_features)

new_point_cloud = []
for i in range(len(cam_poses.camera_pose)):
    new_point_cloud.append(cam_poses.camera_pose[i][:,3:])

new_point_cloud = np.array(new_point_cloud).squeeze()
print(new_point_cloud.shape)
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