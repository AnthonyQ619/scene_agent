from modules.features import FeatureDetectionSIFT, FeatureDetectionSP, FeatureDetectionORB, FeatureDetectionALIKED
from modules.featurematching import (FeatureMatchFlannTracking, 
                                     FeatureMatchBFTracking,
                                     FeatureMatchFlannPair,
                                     FeatureMatchBFPair,
                                     FeatureMatchLoftrPair,
                                     FeatureMatchLightGlueTracking, 
                                     FeatureMatchSuperGlueTracking, 
                                     FeatureMatchLightGluePair, 
                                     FeatureMatchSuperGluePair,
                                     FeatureMatchRoMAPair,)
                                     
from modules.featuretracking import FeatureTrackFromPairsUnionFind
from modules.optimization import BundleAdjustmentOptimizerLocal
from modules.camerapose import CamPoseEstimatorEssentialToPnP
from modules.scenereconstruction import Sparse3DReconstructionMono
from modules.visualize import VisualizeScene, visualize_camera_poses_plotly
from modules.baseclass import SfMScene

import glob
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import matplotlib.pyplot as plt


# Construct Modules with Initialized Arguments
image_path = "/home/anthonyq/datasets/DTU/scan15"
calibration_path = "/home/anthonyq/datasets/DTU/calibration_DTU_new.npz"
# image_path = "C:\\Users\\Anthony\\Documents\\Projects\datasets\\Structure-from-Motion\\sfm_dataset"
# calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\Structure-from-Motion\\calibration_new.npz"
# image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\ETH\\statue\\images\\dslr_images_undistorted"
# calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\ETH\\statue\\dslr_calibration_undistorted\\calibration_new.npz"

ID = "1"
log_dir = "/home/anthonyq/projects/scene_agent/breadth_agent/results/co3d/apple_110_13051_23361_vggt_random_10"
gpu_num = "5"

## NEW SFM PIPELINE
# Step 1: Read in Calibration/Image Data
reconstructed_scene = SfMScene(ID,
                                image_path = image_path, 
                                log_dir=log_dir,
                                gpu_num=gpu_num,
                                # max_images = 15,
                                calibration_path = calibration_path)

# Step 2: Detect Features
# reconstructed_scene.FeatureDetectionSIFT(
#     max_keypoints=10000,
#     contrast_threshold=0.009,
#     edge_threshold=20,
# )

## ALIKED
# reconstructed_scene.FeatureDetectionALIKED(max_keypoints=6000,
#                                             det_thres=0.2)
## SuperPoint
# reconstructed_scene.FeatureDetectionSP(max_keypoints=6000)

# # Step 3: Detect Feature Pairs
# reconstructed_scene.FeatureMatchBFPair(detector="sp",
#                                         k=2,
#                                         lowes_thresh=0.78,
#                                         RANSAC_homography=False,
#                                         RANSAC_threshold=2.0,
#                                         RANSAC_conf=0.999)

## LOFTR
# reconstructed_scene.FeatureMatchLoftrPair(setting="indoor",
#                                          RANSAC_homography=False,
#                                          RANSAC_threshold=1.5,
#                                          RANSAC_conf=0.999,
#                                          coarse_thr = 0.008,
#                                          border_rm = 0,
#                                          pseudo_merge_eps_px=1.10
#                                          )

## ROMA
reconstructed_scene.FeatureMatchRoMAPair(setting="indoor",
                                         RANSAC_homography=False,
                                         RANSAC_threshold=1.5,
                                         RANSAC_conf=0.999,
                                         pseudo_merge_eps_px=1.10,
                                         max_keypoints=5000
                                         )

# Step 4: Detect/Estimate Camera Poses
reconstructed_scene.CamPoseEstimatorEssentialToPnP(
    iteration_count=150,
    reprojection_error = 3.0,
    optimizer = ("BundleAdjustmentOptimizerLocal", {
        "max_num_iterations": 25,
        "robust_loss": True
    }),
)

visualize_camera_poses_plotly(reconstructed_scene.camera_poses)
# print(cam_poses.camera_pose)
# print(calibration_data.K_cams)

# new_point_cloud = []
# for i in range(len(cam_poses.camera_pose)):
#     new_point_cloud.append(cam_poses.camera_pose[i][:,3:])

# new_point_cloud = np.array(new_point_cloud).squeeze()
# print(new_point_cloud.shape)
# print(new_point_cloud)
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(new_point_cloud)

# gui.Application.instance.initialize()

# window = gui.Application.instance.create_window("Mesh-Viewer", 1024, 750)

# scene = gui.SceneWidget()
# scene.scene = rendering.Open3DScene(window.renderer)

# window.add_child(scene)

# matGT = rendering.MaterialRecord()
# matGT.shader = 'defaultUnlit'
# matGT.point_size = 7.0
# matGT.base_color = np.ndarray(shape=(4,1), buffer=np.array([0.0, 0.0, 1.0, 1.0]), dtype=float)

# scene.scene.add_geometry("mesh_name2", pcd, matGT)
# scene.scene.add_geometry("mesh_name3", o3d.geometry.TriangleMesh.create_coordinate_frame(), rendering.MaterialRecord())

# bounds = pcd.get_axis_aligned_bounding_box()
# scene.setup_camera(60, bounds, bounds.get_center())

# gui.Application.instance.run()  # Run until user closes window