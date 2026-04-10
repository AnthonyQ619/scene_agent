import pycolmap
import logging
import os
import pathlib
import shutil
import time
from typing import Dict, List, Type

import hydra
import omegaconf
import numpy as np
import cv2

import copy
import torch
import re

import theseus as th
import theseus.utils.examples as theg

# Hide Warnings
# os.environ["GLOG_log_dir"] = r"C:\Users\Anthony\Documents\Projects\scene_agent\breadth_agent\results"
# os.environ["GLOG_logtostderr"] = "0"
# os.environ["GLOG_alsologtostderr"] = "0"

# os.environ["GLOG_minloglevel"] = "0"        # keep INFO/WARNING in files
# os.environ["GLOG_stderrthreshold"] = "2"    # NOTHING below FATAL goes to terminal

# os.add_dll_directory(r"C:\\Users\\Anthony\\Desktop\\VCPKG\\vcpkg\\installed\\x64-windows\\bin")
# os.add_dll_directory(r"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4\\bin")
# os.add_dll_directory(r"C:\\Program Files\\NVIDIA cuDSS\\v0.7\\bin\\12")

import sys
print(sys.executable)

# import pycolmap

from modules.DataTypes.datatype import Scene, CameraData, PointsMatched, Points3D, CameraPose, Calibration, IncrementalSfMState
from modules.baseclass import OptimizationClass


class BundleAdjustmentOptimizerLocal(OptimizationClass):
    def __init__(
        self,
        cam_data: CameraData,
        window_size: int = 8,
        min_track_len: int = 3,
        refine_focal_length: bool = False,
        refine_principal_point: bool = False,
        refine_extra_params: bool = False,
        max_num_iterations: int = 50,
        use_gpu: bool = True,
        gpu_index: int = 0,
        robust_loss: bool = True,
    ):
        super().__init__(cam_data=cam_data,
                         refine_focal_length=refine_focal_length,
                         refine_principal_point=refine_principal_point,
                         refine_extra_params=refine_extra_params,
                         max_num_iterations=max_num_iterations,
                         use_gpu=use_gpu,
                         gpu_index=gpu_index,
                         robust_loss=robust_loss)
        
        # Set up window size and min_track_length 
        self.window_size = window_size
        self.min_track_len = min_track_len

        self.module_name = "BundleAdjustmentOptimizerLocal"
        self.description = f"""
Local Optimization tool using the bundle adjustment optimization algorithm to optimize the reconstructed sparse 
scene using the 3D estimated points, 2D feature tracks for each 3D estimated point, and estimated camera poses of the 
monocular camera scene to optimize the reprojection error loss of each estimated 3D point WITH the PURPOSE to CORRECT DRIFT
in the estimate camera pose of the scene. USE THIS MODULE in cases where SIFT, ORB, and SuperPoint features will be less accurate
either through usage, or environmental factors where initial feature detection will be geometrically incorrect despite outlier rejection.
The output of the module is the optimized poses within the local estimated frame to correct drift in pose esimtation. 
The algorithm optimizes the 3D points, camera poses, and intrinsic parameters of the calibrated camera (If permitted to). 

Initialization Parameters:
- cam_data: Data container to hold images and calibration data, read from the CameraDataManager.
    - default (CameraData): MUST BE INCLUDED in initialization for usage
- refine_focal_length: Whether to refine the focal length parameter group. 
    - default (bool): False
- refine_principal_point: Whether to refine the principal point parameter group. (bool, )
    - default (bool): False
- refine_extra_params: Whether to refine the extra parameter group.
    - default (bool): False
- max_num_iterations: maximum number of iterations to run the Levenberg-Marquardt algorithm for bundle adjustment
    - default (int): 50
- use_gpu: Whether to use Ceres CUDA linear algebra library, if available. 
    - default (bool): True
- gpu_index: Which GPU to use for solving the problem.
    - default (int): 0 
- robust_loss: Determins whether to use one of the loss function types
    Loss function types: Trivial (non-robust, robust = False) and Cauchy (robust, robust = True) loss
    - default (bool): True
- window_size: the number of total frames (N) used in the local bundle adjustment optimizer (Current frame and N - 1 previous frames)
    - default (int): 8
- min_track_len: minimum length of features tracked across images to consider the 3D point to be used for pose optimization
    - default (int): 3

Function Calls:
- Function: Module call (Python __call__ function)
   - Default: NOT used for this module.
"""

        self.example = f"""
Initialization: 
from modules.optimization import BundleAdjustmentOptimizerLocal

# Build Optimizer
optimizer = BundleAdjustmentOptimizerLocal(max_num_iterations=20,
                                           window_size=5)

NO FUNCTION CALL:     

USAGE (Alongside of CameraPose Estimation Module):
# Initiialized Module
from modules.camerapose import CamPoseEstimatorEssentialToPnP

# Camera Pose Module Initialization (APPLY OPTIMIZER IN INITIALIZATION)
pose_estimator = CamPoseEstimatorEssentialToPnP(cam_data=camera_data,
                                                reprojection_error=3.0,
                                                iteration_count=300,
                                                confidence=0.995,
                                                optimizer=optimizer)

# From estimated features, estimate the camera poses for all image frames (With Local Bundle Adjustment Activated)
cam_poses = pose_estimator(feature_pairs=feature_pairs)

"""
        

    def optimize(self, 
                 state: IncrementalSfMState, 
                 new_image_id: int):

        recon, window = self._build_reconstruction(state, new_image_id)

        # --- BA config: include all registered images ---
        config = pycolmap.BundleAdjustmentConfig()
        for image_id in recon.reg_image_ids():  # registered images
            config.add_image(image_id)
        # --- BA config: Fix the first camera for stability
        config.set_constant_rig_from_world_pose(int(window[0]))


        # --- BA options ---
        ba_opts = pycolmap.BundleAdjustmentOptions()
        ba_opts.refine_focal_length = self.refine_focal_length
        ba_opts.refine_principal_point = self.refine_principal_point
        ba_opts.refine_extra_params = self.refine_extra_params

        # GPU knobs (only used when supported in your build)
        if self.use_gpu:
            ba_opts.use_gpu = bool(self.use_gpu)
            ba_opts.gpu_index = str(self.gpu_index)

        # Ceres solver options (requires PyCeres installed in your environment)
        # was ba_opts.solver_options.max_num_iterations = int(self.max_num_iterations) 
        ba_opts.ceres.solver_options.max_num_iterations = int(self.max_num_iterations) 

        # Optional robust loss
        if self.robust_loss:
            # Was: ba_opts.loss_function_type = pycolmap.LossFunctionType.CAUCHY
            ba_opts.ceres.loss_function_type = pycolmap.LossFunctionType.CAUCHY

        bundle_adjuster = pycolmap.create_default_bundle_adjuster(ba_opts, config, recon)
        _ = bundle_adjuster.solve()

        # --- Export optimized results back into your Scene ---
        # Write refined poses back into state
        for img_id in window:
            img = recon.image(int(img_id))
            T = img.cam_from_world()
            R = T.rotation.matrix()
            t = np.asarray(T.translation).reshape(3, 1)
            state.poses[img_id] = np.hstack([R, t])

        return state

    def _build_reconstruction(self, 
                              state: IncrementalSfMState, 
                              new_image_id: int):
        recon = pycolmap.Reconstruction()

        # ---- 1) Add camera (single-camera monocular case) ----
        # You must map your cam_data -> COLMAP camera model + params.
        # Common choices:
        #   PINHOLE:   [fx, fy, cx, cy]
        #   OPENCV:    [fx, fy, cx, cy, k1, k2, p1, p2]
        #   FULL_OPENCV: [fx, fy, cx, cy, k1,k2,p1,p2,k3,k4,k5,k6]

        # Get active Window Length for BA
        last = new_image_id
        first = max(0, last - self.window_size + 1)
        window = list(range(first, last + 1))

        K = state.K  # (3,3)
        fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])

        if getattr(self.cam_data, "dist", None) is None:
            model = pycolmap.CameraModelId.PINHOLE
            params = np.array([fx, fy, cx, cy], dtype=np.float64)
        else:
            # assume OpenCV 4-dist for monocular
            d = self.cam_data.get_distortion().ravel().astype(np.float64)
            model = pycolmap.CameraModelId.OPENCV
            params = np.array([fx, fy, cx, cy, d[0], d[1], d[2], d[3]], dtype=np.float64)

        camera_id = 1
        cam = pycolmap.Camera(
            camera_id=camera_id,
            model=model,
            width=state.width,
            height=state.height,
            params=params,
        )
        recon.add_camera(cam)

        # Build Camera Rig 
        rig = pycolmap.Rig()
        rig.rig_id = camera_id 
        sensor_t = pycolmap.sensor_t()
        sensor_t.type = pycolmap.SensorType.CAMERA
        sensor_t.id = cam.camera_id
        rig.add_ref_sensor(sensor_t)
        recon.add_rig(rig)

        # # ---- 2) Build per-image keypoints and (track_id, frame)->point2D_idx ----
        # # points.data_matrix rows: [track_id, frame_num, x, y]
        # obs = scene.observations #points.data_matrix
        # track_ids = obs[:, 1].astype(np.int64)
        # frame_nums = obs[:, 0].astype(np.int64)
        # xys = obs[:, 2:4].astype(np.float64)

        # num_frames = len(scene.cam_poses)

        # # group observations by frame
        # frame_to_rows = [[] for _ in range(num_frames)]
        # for row_i, f in enumerate(frame_nums):
        #     if 0 <= f < num_frames:
        #         frame_to_rows[f].append(row_i)

        # maps (track_id, frame) -> point2D_idx in that image's keypoint list
        keypoint_index = {}

        # Add frames/images
        for img_id in window:
            # Create frame + set pose from your initial estimate
            frame = pycolmap.Frame(frame_id = img_id,
                                   rig_id = rig.rig_id)
            # frame.frame_id = camera_id
            # frame.rig_id = rig.rig_id

            # print("Proposed Pose", scene.cam_poses[f])
            # Get Camera Rotation
            cam_from_world = pycolmap.Rigid3d(state.poses[img_id])#camera_poses.camera_pose[f])  # 3x4
            # print("CFW", cam_from_world)
            # frame.set_cam_from_world(camera_id, cam_from_world)
            # recon.add_frame(frame)
            # recon.register_frame(frame_id)

            # image keypoints: keypoints (Nx2)
            kps = np.asarray(state.keypoints[img_id], dtype=np.float64)

            # Create Point2D list
            pts2d = pycolmap.Point2DList()
            for xy in kps.astype(np.float64):
                # Point2D(xy=[x,y]) optionally with point3D_id
                pts2d.append(pycolmap.Point2D(xy))

            img = pycolmap.Image(
                name=f"{img_id:06d}.png",
                points2D=pts2d,
                camera_id=camera_id,
                image_id=img_id,
                frame_id=frame.frame_id
            )
            frame.add_data_id(img.data_id)
            recon.add_frame(frame)
            # recon.register_frame(frame_id)
            recon.frame(frame.frame_id).set_cam_from_world(camera_id=camera_id, 
                                                           cam_from_world=cam_from_world)
            recon.register_frame(frame.frame_id)
            recon.add_image(img)

        # ---- 3) Add 3D points with tracks (observations) ----
        # Using your state.points3D keyed by track_id, and state.tracks for observations
        for track_id, obs in state.tracks.items():
            # keep only observations that are in the window
            # obs_w = [(im, kp) for (im, kp) in obs if im in window]
            # if len(obs_w) < self.min_track_len:
            #     continue
            # if track_id not in state.points3D:
            #     continue
            # print(obs_w)
            # track_elems = [pycolmap.TrackElement(int(im), int(kp)) for im, kp in obs_w]
            # track = pycolmap.Track(track_elems)
            # xyz = np.asarray(state.points3D[track_id], dtype=np.float64).reshape(3, 1)
            # recon.add_point3D(xyz, track)

            # Keep only observations in the current window
            obs_w = [(int(im), int(kp)) for (im, kp) in obs if im in window]
            if len(obs_w) < self.min_track_len:
                continue
            if track_id not in state.points3D:
                continue

            # Check whether any observation is already assigned to a point3D
            existing_pids = set()
            for im, kp in obs_w:
                pid = recon.image(im).points2D[kp].point3D_id
                if pid != pycolmap.INVALID_POINT3D_ID:
                    existing_pids.add(pid)

            # ---- Case 1: conflicting assignments → skip track ----
            if len(existing_pids) > 1:
                # Multiple different point3D IDs → inconsistent track
                continue

            # ---- Case 2: exactly one existing point3D → extend it ----
            if len(existing_pids) == 1:
                pid = existing_pids.pop()
                p3d = recon.point3D(pid)

                for im, kp in obs_w:
                    img = recon.image(im)
                    if img.points2D[kp].point3D_id == pycolmap.INVALID_POINT3D_ID:
                        recon.add_observation(pid, pycolmap.TrackElement(im, kp))

                continue  # do NOT create a new point3D

            # ---- Case 3: no existing assignments → create a new point3D ----
            track_elems = [
                pycolmap.TrackElement(im, kp)
                for im, kp in obs_w
                if recon.image(im).points2D[kp].point3D_id == pycolmap.INVALID_POINT3D_ID
            ]

            if len(track_elems) < self.min_track_len:
                continue

            xyz = np.asarray(state.points3D[track_id], dtype=np.float64).reshape(3, 1)
            track = pycolmap.Track(track_elems)
            recon.add_point3D(xyz, track)

        # If no points, skip
        if len(recon.points3D) == 0:
            return

        return recon, window

class BundleAdjustmentOptimizerGlobal(OptimizationClass):
    def __init__(
        self,
        cam_data: CameraData,
        refine_focal_length: bool = False,
        refine_principal_point: bool = False,
        refine_extra_params: bool = False,
        max_num_iterations: int = 50,
        use_gpu: bool = True,
        gpu_index: int = 0,
        robust_loss: bool = True,
    ):
        super().__init__(cam_data=cam_data,
                         refine_focal_length=refine_focal_length,
                         refine_principal_point=refine_principal_point,
                         refine_extra_params=refine_extra_params,
                         max_num_iterations=max_num_iterations,
                         use_gpu=use_gpu,
                         gpu_index=gpu_index,
                         robust_loss=robust_loss)

        self.module_name = "BundleAdjustmentOptimizerGlobal"
        self.description = f"""
Global Optimization tool using the bundle adjustment optimization algorithm to optimize the reconstructed sparse 
scene using the 3D estimated points, 2D feature tracks for each 3D estimated point, and estimated camera poses of the 
monocular camera scene to optimize the reprojection error loss of each estimated 3D point. The output is a newly 
optimize sparse 3D reconstructed scene. The algorithm optimizes the 3D points, camera poses, and intrinsic parameters 
of the calibrated camera (If permitted to). ALWAYS USE THIS MODULE AT THE END OF A SPARSE RECONSTRUCTION PIPELINE, AND 
PRIOR TO ESTIMATING DENSE RECONSTRUCTION!

Initialization Parameters:
- cam_data: Data container to hold images and calibration data, read from the CameraDataManager.
    - default (CameraData): MUST BE INCLUDED in initialization for usage
- refine_focal_length: Whether to refine the focal length parameter group. 
    - default (bool): False
- refine_principal_point: Whether to refine the principal point parameter group. (bool, )
    - default (bool): False
- refine_extra_params: Whether to refine the extra parameter group.
    - default (bool): False
- max_num_iterations: maximum number of iterations to run the Levenberg-Marquardt algorithm for bundle adjustment
    - default (int): 50
- use_gpu: Whether to use Ceres CUDA linear algebra library, if available. 
    - default (bool): True
- gpu_index: Which GPU to use for solving the problem.
    - default (int): 0 
- robust_loss: Determins whether to use one of the loss function types
    Loss function types: Trivial (non-robust, robust = False) and Cauchy (robust, robust = True) loss
    - default (bool): True


Function Calls:
- Function: Module call (Python __call__ function)
    - Parameters:
        - scene: Data type that stores the camera pose and 3D point information of the reconstructed scene.
            - default (Scene): Stores information of the reconstructed scene, typically sparse when the function is called
                - Important items stored:
                    - points3D: Points3D              # Set of 3D points
                    - cam_poses: list[np.ndarray]     # Should be formatted as a 3x4 matrix
                    - observations: np.ndarray        # Mx4 matrices for each point observation where M=num_of_observations, and each row = [frame, 3d_point_ind, pix_x, pix_y]
                    - sparse: bool                    # Used to determine if current scene is sparse or dense (Sparse=True)
        - Output: scene (Optimized)
"""

        self.example = f"""
Initialization: 
from modules.optimization import BundleAdjustmentOptimizerGlobal

# Build Optimizer
optimizer = BundleAdjustmentOptimizerGlobal(max_num_iterations=40,
                                            cam_data=camera_data)

Function Call:                                      
# Run Optimizer
optimal_scene = optimizer(sparse_scene)
"""

        # Define the workspace for Sparse Reconstruction
        self.directory_path = "C:\\Users\\Anthony\\Documents\\Projects\\scene_agent\\breadth_agent\\results\\workspace\\sparse"
        if os.path.exists(self.directory_path):
            # Delete the directory and all its contents
            shutil.rmtree(self.directory_path)

        # Recreate an empty directory
        os.makedirs(self.directory_path)

    def optimize(self, 
                 scene: Scene):
                #  points: PointsMatched, 
                #  camera_poses: CameraPose, 
                #  cam_data: CameraData):
        """
        scene.points3D.points3D: (N,3) float
        camera_poses.camera_pose[i]: (3,4) cam_from_world for frame i
        points.data_matrix: (M,4) [track_id, frame_num, x, y]  (pixels)
        """
        # if not points.multi_view:
        #     raise ValueError("BA requires multi-view tracks (points.multi_view=True).")

        recon, trackid_to_point3Did = self._build_reconstruction(scene)

        # --- BA config: include all registered images ---
        config = pycolmap.BundleAdjustmentConfig()
        for image_id in recon.reg_image_ids():  # registered images
            config.add_image(image_id)
        # --- BA config: Fix the first camera for stability
        config.set_constant_rig_from_world_pose(recon.reg_image_ids()[0])


        # --- BA options ---
        ba_opts = pycolmap.BundleAdjustmentOptions()
        ba_opts.refine_focal_length = self.refine_focal_length
        ba_opts.refine_principal_point = self.refine_principal_point
        ba_opts.refine_extra_params = self.refine_extra_params

        # GPU knobs (only used when supported in your build)
        # ba_opts.use_gpu = bool(self.use_gpu)
        # ba_opts.gpu_index = str(self.gpu_index)

        # # Ceres solver options (requires PyCeres installed in your environment)
        # ba_opts.solver_options.max_num_iterations = int(self.max_num_iterations)

        # # Optional robust loss
        # if self.robust_loss:
        #     ba_opts.loss_function_type = pycolmap.LossFunctionType.CAUCHY

        # GPU knobs (only used when supported in your build)
        if self.use_gpu:
            ba_opts.use_gpu = bool(self.use_gpu)
            ba_opts.gpu_index = str(self.gpu_index)

        # Ceres solver options (requires PyCeres installed in your environment)
        ba_opts.ceres.solver_options.max_num_iterations = int(self.max_num_iterations) 

        # Optional robust loss
        if self.robust_loss:
            ba_opts.ceres.loss_function_type = pycolmap.LossFunctionType.CAUCHY

        bundle_adjuster = pycolmap.create_default_bundle_adjuster(ba_opts, config, recon)
        summary = bundle_adjuster.solve()

        # --- Export optimized results back into your Scene ---
        self._write_back_to_scene(scene, recon, trackid_to_point3Did)

        # Write reconstructed scene to workspace (Sparse Scene!)
        recon.write(self.directory_path)

        return scene #, summary

    def _build_reconstruction(self, 
                              scene: Scene):
        recon = pycolmap.Reconstruction()

        # ---- 1) Add camera (single-camera monocular case) ----
        # You must map your cam_data -> COLMAP camera model + params.
        # Common choices:
        #   PINHOLE:   [fx, fy, cx, cy]
        #   OPENCV:    [fx, fy, cx, cy, k1, k2, p1, p2]
        #   FULL_OPENCV: [fx, fy, cx, cy, k1,k2,p1,p2,k3,k4,k5,k6]

        # Set up Camera for reconstruction class
        if self.multi_cam:
            camera_map = {}
            frame_to_camera = {}
            next_camera_id = 1

            for f in range(len(scene.cam_poses)):

                K = self.K[f]
                dist = self.dist[f]

                fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])

                if dist is None:
                    key = ("PINHOLE", fx, fy, cx, cy)
                else:
                    d = tuple(dist.ravel())
                    key = ("OPENCV", fx, fy, cx, cy) + d

                if key not in camera_map:

                    if dist is None:
                        model = pycolmap.CameraModelId.PINHOLE
                        params = np.array([fx, fy, cx, cy], np.float64)
                    else:
                        # model = pycolmap.CameraModelId.OPENCV
                        # params = np.array([fx, fy, cx, cy, *d], np.float64)
                        d = np.asarray(dist, dtype=np.float64).ravel()

                        if len(d) == 4:
                            # k1, k2, p1, p2
                            model = pycolmap.CameraModelId.OPENCV
                            params = np.array([fx, fy, cx, cy, d[0], d[1], d[2], d[3]], dtype=np.float64)
                        elif len(d) == 5:
                            # OpenCV often gives k1, k2, p1, p2, k3
                            # drop k3 and use OPENCV
                            model = pycolmap.CameraModelId.OPENCV
                            params = np.array([fx, fy, cx, cy, d[0], d[1], d[2], d[3]], dtype=np.float64)
                        elif len(d) == 8:
                            # k1, k2, p1, p2, k3, k4, k5, k6
                            model = pycolmap.CameraModelId.FULL_OPENCV
                            params = np.array([fx, fy, cx, cy, *d], dtype=np.float64)

                    cam = pycolmap.Camera(
                        camera_id=next_camera_id,
                        model=model,
                        width=self.W,
                        height=self.H,
                        params=params,
                    )

                    recon.add_camera(cam)
                    camera_map[key] = next_camera_id
                    next_camera_id += 1

                frame_to_camera[f] = camera_map[key]

            # Create Camera rig for single camera
            rig = pycolmap.Rig()
            rig.rig_id = 1

            # for cam_id in camera_map.values():
            #     sensor = pycolmap.sensor_t()
            #     sensor.type = pycolmap.SensorType.CAMERA
            #     sensor.id = cam_id
            #     rig.add_sensor(sensor)
            cam_ids = list(camera_map.values())

            for i, cam_id in enumerate(cam_ids):
                sensor = pycolmap.sensor_t()
                sensor.type = pycolmap.SensorType.CAMERA
                sensor.id = cam_id

                if i == 0:
                    # Make first camera the reference sensor
                    rig.add_ref_sensor(sensor)

                    # # Optional but explicit: reference sensor has identity transform
                    # rig.set_sensor_from_rig(sensor, pycolmap.Rigid3d())
                else:
                    # Additional sensors need an explicit pose wrt rig
                    rig.add_sensor(sensor, pycolmap.Rigid3d())

            # choose a reference sensor (first camera)
            # ref = pycolmap.sensor_t()
            # ref.type = pycolmap.SensorType.CAMERA
            # ref.id = list(camera_map.values())[0]
            # rig.add_ref_sensor(ref)

            recon.add_rig(rig)
        else:
            fx, fy, cx, cy = float(self.K[0,0]), float(self.K[1,1]), float(self.K[0,2]), float(self.K[1,2])

            if getattr(self.cam_data, "dist", None) is None:
                model = pycolmap.CameraModelId.PINHOLE
                params = np.array([fx, fy, cx, cy], dtype=np.float64)
            else:
                # assume OpenCV 4-dist for monocular
                d = self.dist.ravel().astype(np.float64)
                model = pycolmap.CameraModelId.OPENCV
                params = np.array([fx, fy, cx, cy, d[0], d[1], d[2], d[3]], dtype=np.float64)
            
            # Create Single Camera if using a single Monocular camera
            camera_id = 1
            cam = pycolmap.Camera(
                camera_id=camera_id,
                model=model,
                width=self.W,
                height=self.H,
                params=params,
            )
            recon.add_camera(cam)

            # Create Camera rig for single camera
            rig = pycolmap.Rig()
            rig.rig_id = camera_id 
            sensor_t = pycolmap.sensor_t()
            sensor_t.type = pycolmap.SensorType.CAMERA
            sensor_t.id = cam.camera_id
            rig.add_ref_sensor(sensor_t)
            recon.add_rig(rig)

        # ---- 2) Build per-image keypoints and (track_id, frame)->point2D_idx ----
        # points.data_matrix rows: [track_id, frame_num, x, y]
        obs = scene.observations #points.data_matrix
        track_ids = obs[:, 1].astype(np.int64)
        frame_nums = obs[:, 0].astype(np.int64)
        xys = obs[:, 2:4].astype(np.float64)

        num_frames = len(scene.cam_poses)

        # group observations by frame
        frame_to_rows = [[] for _ in range(num_frames)]
        for row_i, f in enumerate(frame_nums):
            if 0 <= f < num_frames:
                frame_to_rows[f].append(row_i)

        # maps (track_id, frame) -> point2D_idx in that image's keypoint list
        keypoint_index = {}

        for f in range(num_frames):
            image_id = f + 1
            frame_id = f + 1
            if self.multi_cam:
                camera_id = frame_to_camera[f]
            else:
                camera_id = 1

            # Create frame + set pose from your initial estimate
            frame = pycolmap.Frame(frame_id = frame_id,
                                   rig_id = rig.rig_id)

            # Get Camera Rotation
            cam_from_world = pycolmap.Rigid3d(scene.cam_poses[f])#camera_poses.camera_pose[f])  # 3x4

            rows = frame_to_rows[f]
            if len(rows) == 0:
                # still add an image (pose exists), but no keypoints
                img = pycolmap.Image(
                    name=f"{f:06d}.png",
                    points2D=pycolmap.Point2DList(),
                    camera_id=camera_id,
                    image_id=image_id,
                    frame_id=frame_id
                )
                frame.add_data_id(img.data_id)
                recon.add_frame(frame)
                # recon.register_frame(frame_id)
                recon.frame(frame.frame_id).set_cam_from_world(camera_id=camera_id, 
                                                            cam_from_world=cam_from_world)
                recon.register_frame(frame_id)
                recon.add_image(img)
                continue

            kps = xys[rows]  # (m,2)
            # store idx mapping
            for local_idx, row_i in enumerate(rows):
                tid = int(track_ids[row_i])
                keypoint_index[(tid, f)] = local_idx

            # Create Point2D list
            pts2d = pycolmap.Point2DList()
            for xy in kps.astype(np.float64):
                # Point2D(xy=[x,y]) optionally with point3D_id
                pts2d.append(pycolmap.Point2D(xy))

            img = pycolmap.Image(
                name=f"{f:06d}.png",
                points2D=pts2d,
                camera_id=camera_id,
                image_id=image_id,
                frame_id=frame_id
            )
            frame.add_data_id(img.data_id)
            recon.add_frame(frame)
            # recon.register_frame(frame_id)
            recon.frame(frame.frame_id).set_cam_from_world(camera_id=camera_id, 
                                                           cam_from_world=cam_from_world)
            recon.register_frame(frame_id)
            recon.add_image(img)

        # ---- 3) Add 3D points with tracks (observations) ----
        # We assume your scene.points3D.points3D aligns with track_id 0..N-1 (or you have a mapping).
        # If you filtered points during triangulation, pass/keep an explicit mapping instead.
        xyzs = scene.points3D.points3D.astype(np.float64)

        trackid_to_point3Did = {}

        # Build a quick map: track_id -> list of (frame, xy idx)
        # from tracked observations
        from collections import defaultdict
        tid_to_frames = defaultdict(list)
        for row_i in range(obs.shape[0]):
            tid = int(track_ids[row_i])
            f = int(frame_nums[row_i])
            if (tid, f) in keypoint_index:
                tid_to_frames[tid].append(f)

        for tid in range(xyzs.shape[0]):
            if tid not in tid_to_frames:
                continue

            elements = []
            for f in tid_to_frames[tid]:
                image_id = f + 1
                p2d_idx = keypoint_index[(tid, f)]
                elements.append(pycolmap.TrackElement(image_id=image_id, point2D_idx=int(p2d_idx)))

            # minimum 2 observations to be a valid track
            if len(elements) < 2:
                continue

            track = pycolmap.Track(elements=elements)
            point3D_id = recon.add_point3D(xyzs[tid], track)  # returns new ID
            trackid_to_point3Did[tid] = point3D_id

        return recon, trackid_to_point3Did


    def _write_back_to_scene(self, 
                            scene: Scene,
                            recon,
                            trackid_to_point3Did):
        # Update camera poses: pull optimized cam_from_world from each image
        num_frames = len(scene.cam_poses)

        for f in range(num_frames):
            image_id = f + 1
            img = recon.image(image_id)
            if not img.has_pose:
                continue
            scene.cam_poses[f] = img.cam_from_world().matrix()

        # Update 3D points
        # If you need to keep ordering by track_id, write into that index.
        xyzs = scene.points3D.points3D
        for tid, p3did in trackid_to_point3Did.items():
            p3d = recon.point3D(p3did)
            xyzs[tid] = p3d.xyz  # (3,)

        
class BundleAdjustmentOptimizerLeastSquares(OptimizationClass):
    def __init__(self, 
                 cam_data: CameraData,
                 max_iterations: int = 10, 
                 step_size: float = 0.1, 
                 learning_rate: float = 0.1,
                 num_epochs: int = 20,
                 optimizer_cls: str = "LevenbergMarquardt"
                 ):
        super().__init__(cam_data = cam_data)
        
        self.optimizer_choices = ["LevenbergMarquardt", "GaussNewton"]

        assert (optimizer_cls in self.optimizer_choices), "Must Choose Optimizer from supported classes: LevenbergMarquardt or GaussNewton"

        self.module_name = "BundleAdjustmentOptimizerLeastSquares"
        self.description = f"""
Global Optimization tool using the non-linear least square optimization algorithm to optimize the reconstructed sparse 
scene using the 3D estimated points, 2D feature tracks for each 3D estimated point, and estimated camera poses of the 
monocular camera scene to optimize the reprojection error loss of each estimated 3D point. This tool applies the algorithm
Bundle Adjustment to globally optimize the scene. The output is a newly optimize sparse 3D reconstructed scene. The algorithm
optimizes the 3D points, camera poses, and distortion parameters of the calibrated camera. This algorithm keeps the focal length
of the original calibration by normalizing the detected 2D feature points prior to optimization.

Initialization Parameters:
- calibration: Data type that stores the camera's calibration data initialized from the calibration 
reader module
    - default (Calibration): None - Include in initialization for usage
- scene: Data type that stores the camera pose and 3D point information of the reconstructed scene.
    - default (Scene): None - Include in initialization for usage

Function Calls:
- Function: prep_optimizer
    - Parameters:
        - max_iterations: Maximum number of iterations to run the bundle adjustment algorithm (larger number 
        guarantees convergence -- between 40 and 50 iterations)
            - Default (int) = 10,
        - ratio_known_cameras: Used if ground truth cameras are known prior, which can be used to optimize the reconstructed scen
        in training cases. Input should be percentage of known cameras (input < 1.0)
            - Default (float) = 0.0
        - optimizer_cls: Optimzer class utilized in the bundle adjustment algorithm. Levenberg-Marquardt is default due to 
        utilizing the non-linear least square optimization algorithm. Use Gauss Newton if initial data is less noisy and 
        can use a more aggressive optimizer, but it is less robust to outliers. 
            - Default (str) = "LevenbergMarquardt"
            - Options (str) ] ["LevenbergMarquardt", "GaussNewton"]

- Function: Module call (Python __call__ function)
    - Parameters:
        -ba_file: Path to BAL file generated from SceneReconstruction module that stores the data from the reconstructed
        scene for bundle adjustment application.
            - Default (str): Not Applicable. Needs input to run.
"""

        self.example = f"""
Initialization: 
optimizer = BundleAdjustmentOptimizerLeastSquares(scene=sparse_scene, 
                                                  cam_data=camera_data)
optimizer.prep_optimizer(ratio_known_cameras=0.0, 
                         max_iterations=30, 
                         num_epochs=1, 
                         step_size=0.1,
                         optimizer_cls="GaussNewton")

Function Call: 
optimal_scene = optimizer(bal_path)
"""
        # Logger
        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger(__name__)

        # Setup Path for Saving Results
        result_dir = "results/scene_data"
        file_path = os.path.realpath(__file__).split('\\')
        file_path[0] = file_path[0] + "\\"
        home_dir = pathlib.Path(os.path.join(*file_path[:7])) # Sets Path to Breadth_agent
        self.results_path = home_dir / result_dir

        # Get Device
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        config_settings = {"inner_optim": {
                                "optimizer_cls": optimizer_cls,
                                "max_iters": max_iterations,
                                "step_size": step_size,
                                "backward_mode": "implicit",
                                "verbose": True,
                                "track_err_history": True,
                                "keep_step_size": True,
                                "regularize": True,
                                "ratio_known_cameras": 0.0,
                                "reg_w": 1e-4},
                            "outer_optim": {
                                "lr": learning_rate,
                                "num_epochs": num_epochs},
                            "hydra": {
                                    "run":{
                                        "dir": "examples/outputs"}}}
        
        self.cfg = omegaconf.OmegaConf.create(config_settings)

        # End of Initialization

    ################################## HELPER FUNCTIONS ##################################
    # Copyright (c) Meta Platforms, Inc. and affiliates.
    #
    # This source code is licensed under the MIT license found in the
    # LICENSE file in the root directory facebookresearch/theseus
    # Credit belongs to Facebook Research Team with Theseus library Examples:Bundle Adjustment
    
    def print_histogram(self,
        ba: theg.BundleAdjustmentDataset, var_dict: Dict[str, torch.Tensor], msg: str
    ):
        # print(ba.observations)
        self.log.info(msg)
        histogram = theg.ba_histogram(
            cameras=[
                theg.Camera(
                    th.SE3(tensor=var_dict[c.pose.name]),
                    c.focal_length,
                    c.calib_k1,
                    c.calib_k2,
                )
                for c in ba.cameras
            ],
            points=[th.Point3(tensor=var_dict[pt.name]) for pt in ba.points],
            observations=ba.observations,
        )
        for line in histogram.split("\n"):
            self.log.info(line)


    # loads (the only) batch
    def get_batch(self,
                ba: theg.BundleAdjustmentDataset,
                orig_poses: Dict[str, torch.Tensor],
                orig_points: Dict[str, torch.Tensor],
                ) -> Dict[str, torch.Tensor]:
        
        retv = {}
        for cam in ba.cameras:
            retv[cam.pose.name] = orig_poses[cam.pose.name].clone()
        for pt in ba.points:
            retv[pt.name] = orig_points[pt.name].clone()
        return retv

    def save_epoch(self,
                results_path: pathlib.Path,
                epoch: int,
                log_loss_radius: th.Vector,
                theseus_outputs: Dict[str, torch.Tensor],
                info: th.optimizer.OptimizerInfo,
                loss_value: float,
                total_time: float):
        
        def _clone(t_):
            return t_.detach().cpu().clone()

        results = {
            "log_loss_radius": _clone(log_loss_radius.tensor),
            "theseus_outputs": dict((s, _clone(t)) for s, t in theseus_outputs.items()),
            "err_history": info.err_history,  # type: ignore
            "loss": loss_value,
            "total_time": total_time,
        }
        torch.save(results, results_path / f"results_epoch{epoch}.pt")
    # End
    ################################## HELPER FUNCTIONS ##################################

    def __call__(self,
                 scene: Scene) -> Scene:

        # Setup Theseus BA dataset -- Read BAL dataset
        ba = copy.deepcopy(scene.bal_data.dataset) #theg.BundleAdjustmentDataset.load_from_file(ba_file) # Look into switching this

        # hyper parameters (ie outer loop's parameters) -> Not needed
        log_loss_radius = th.Vector(1, name="log_loss_radius", dtype=torch.float64)

        # Set up objective
        objective = th.Objective(dtype=torch.float64)

        pose_check = set()
        weight = th.ScaleCostWeight(torch.tensor(1.0).to(dtype=ba.cameras[0].pose.dtype, device=self.device))
        for obs in ba.observations:
            cam = ba.cameras[obs.camera_index]
            cost_function = th.eb.Reprojection(
                camera_pose=cam.pose,
                world_point=ba.points[obs.point_index],
                focal_length=cam.focal_length,
                calib_k1=cam.calib_k1,
                calib_k2=cam.calib_k2,
                image_feature_point=obs.image_feature_point,
                weight=weight,
            )
            robust_cost_function = th.RobustCostFunction(
                cost_function,
                th.HuberLoss,
                log_loss_radius,
                name=f"robust_{cost_function.name}",
            )
            objective.add(robust_cost_function)
            #objective.add
        dtype = objective.dtype

        # Add regularization
        if self.cfg.inner_optim.regularize:
            zero_point3 = th.Point3(dtype=dtype, name="zero_point")
            identity_se3 = th.SE3(dtype=dtype, name="zero_se3")
            w = np.sqrt(self.cfg.inner_optim.reg_w)
            damping_weight = th.ScaleCostWeight(w * torch.ones(1, dtype=dtype, device=self.device))
            for name, var in objective.optim_vars.items():
                target: th.Manifold
                if isinstance(var, th.SE3):
                    target = identity_se3
                elif isinstance(var, th.Point3):
                    target = zero_point3
                else:
                    assert False
                objective.add(
                    th.Difference(var, target, damping_weight, name=f"reg_{name}")
                )

        # Create optimizer
        optimizer_cls: Type[th.NonlinearLeastSquares] = getattr(
            th, self.cfg.inner_optim.optimizer_cls
        )
        # optimizer_cls: Type[th.LevenbergMarquardt] = getattr(
        #     th, self.cfg.inner_optim.optimizer_cls
        # )
        
        optimizer = optimizer_cls(
            objective,
            max_iterations=self.cfg.inner_optim.max_iters,
            step_size=self.cfg.inner_optim.step_size,
        )

        # Set up Theseus layer
        theseus_optim = th.TheseusLayer(optimizer)
        theseus_optim = theseus_optim.to(device=self.device)

        # copy the poses/pts to feed them to each outer iteration
        orig_poses = {cam.pose.name: cam.pose.tensor.clone().to(self.device) for cam in ba.cameras}
        orig_points = {pt.name: pt.tensor.clone() for pt in ba.points}
        print("CAMERAS", pose_check)
        # print(orig_poses)

        # Outer optimization loop
        loss_radius_tensor = torch.nn.Parameter(torch.tensor([3.0], dtype=torch.float64, device=self.device))
        model_optimizer = torch.optim.Adam([loss_radius_tensor], lr=self.cfg.outer_optim.lr)

        num_epochs = self.cfg.outer_optim.num_epochs

        theseus_inputs = self.get_batch(ba, orig_poses, orig_points)
        theseus_inputs["log_loss_radius"] = loss_radius_tensor.unsqueeze(1).clone()

        # with torch.no_grad():
        #     #camera_loss_ref = self.camera_loss(ba, camera_pose_vars).item()
        #     reproj_loss = self.reprojection_loss(ba, theseus_inputs)
        # # self.log.info(f"CAMERA LOSS (no learning):  {camera_loss_ref: .3f}")
        # reproj_loss = objective.error_metric() 
        # reproj_loss = loss.sum()
        # self.log.info(f"Reprojection Loss (no learning):  {reproj_loss: .3f}")

        self.print_histogram(ba, theseus_inputs, "Input histogram:")

        for epoch in range(num_epochs):
            self.log.info(f" ******************* EPOCH {epoch} ******************* ")
            start_time = time.time_ns()
            model_optimizer.zero_grad()
            theseus_inputs["log_loss_radius"] = loss_radius_tensor.unsqueeze(1).clone()

            theseus_outputs, info = theseus_optim.forward(
                input_tensors=theseus_inputs,
                optimizer_kwargs={
                    "verbose": self.cfg.inner_optim.verbose,
                    "track_err_history": self.cfg.inner_optim.track_err_history,
                    "backward_mode": self.cfg.inner_optim.backward_mode,
                    "__keep_final_step_size__": self.cfg.inner_optim.keep_step_size,
                },
            )

            # print(theseus_outputs)
            # objective.update(theseus_outputs)
            loss = objective.error_metric(input_tensors=theseus_outputs) #(self.camera_loss(ba, camera_pose_vars) - camera_loss_ref) / camera_loss_ref
            loss = loss.sum()
            # loss = self.reprojection_loss(ba, theseus_outputs=theseus_outputs)
            # print(loss)
            # print(loss.device)
            # loss.backward()
            # print("AFTER BACK PROP")
            # model_optimizer.step()
            loss_value = torch.sum(loss.cpu().detach()).item()
            end_time = time.time_ns()
            
            # Calibration Check -> Focal Length should be 1, distortion params should be 0
            for c in ba.cameras:
                print("Focal Length", c.focal_length,
                      "Distortion 1:", c.calib_k1,
                      "Distortion 2:", c.calib_k2)
            # self.print_histogram(ba, theseus_outputs, "Output histogram:")
            self.log.info(
                f"Epoch: {epoch} Loss: {loss_value} "
                f"Kernel Radius: exp({loss_radius_tensor.data.item()})="
                f"{torch.exp(loss_radius_tensor.data).item()}"
            )
            self.log.info(f"Epoch took {(end_time - start_time) / 1e9: .3f} seconds")

            self.save_epoch(
                self.results_path,
                epoch,
                log_loss_radius,
                theseus_outputs,
                info,
                loss_value,
                end_time - start_time,
            )

        poses = []
        # poses_track = set()
        points_3d = Points3D()

        # Rebuild Scene Datatype
        # for obs in ba.observations:
        #     cam = ba.cameras[obs.camera_index]

        #     cam_pose_tensor = theseus_outputs[cam.pose.name]  # shape (6,) or (batch,6)
        #     if cam.pose.name not in poses_track:
        #         print("Here")
        #         pose = cam_pose_tensor.cpu().detach().numpy()
        #         poses.append(pose)
        #         poses_track.add(cam.pose.name)
              
        #     world_point = theseus_outputs[ba.points[obs.point_index].name].cpu().detach().numpy()  # shape (3,) or (batch,3)
        #     points_3d.update_points(world_point)
        print(theseus_outputs.keys())
        for key in theseus_outputs.keys():
            if 'Cam' in key:
                pose = theseus_outputs[key].cpu().detach().numpy()
                poses.append(pose)
            if 'Pt' in key:
                # print(obs.point_index)
                # print("KEY", int(re.findall(r'\d+',key)[0]))
                world_point = theseus_outputs[ba.points[int(re.findall(r'\d+',key)[0])].name].cpu().detach().numpy()  # shape (3,) or (batch,3)
                points_3d.update_points(world_point)#.astype(float))
        
        # print(theseus_outputs)
        # print(theseus_outputs.keys())
        # print(len(poses))
        print("Number of 3D Points", points_3d.points3D.shape)
        new_scene = Scene(points3D = points_3d,
                          cam_poses = poses,
                          representation = "point cloud")
        
        # print(points_3d.points3D)
        return new_scene
    
    # def _create_result_path(self, result_dir: str) -> pathlib.Path:
    #     file_path = os.path.realpath(__file__).split('\\')
    #     file_path[0] = file_path[0] + "\\"
    #     home_dir = pathlib.Path(os.path.join(*file_path[:7])) # Sets Path to Breadth_agent
    #     result_path = home_dir / result_dir

    #     return result_path