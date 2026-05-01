import pycolmap
import logging
import os
import pathlib
import shutil
import time
from typing import Dict, List, Type
from pathlib import Path

# import hydra
import omegaconf
import numpy as np
import cv2

import copy
import torch
import re

# import theseus as th
# import theseus.utils.examples as theg

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
from modules.baseclass import OptimizationClass,  module_metric

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
        # use_gpu: bool = True,
        # gpu_index: int = 0,
        robust_loss: bool = True,
    ):

        module_name = "BundleAdjustmentOptimizerLocal"
        description = f"""
Local Optimization tool using the bundle adjustment optimization algorithm to optimize the reconstructed sparse 
scene using the 3D estimated points, 2D feature tracks for each 3D estimated point, and estimated camera poses of the 
monocular camera scene to optimize the reprojection error loss of each estimated 3D point WITH the PURPOSE to CORRECT DRIFT
in the estimated camera pose of the scene. 

USE THIS MODULE in cases where SIFT, ORB, and SuperPoint features will be less accurate either through usage, or environmental 
factors where initial feature detection will be geometrically incorrect despite outlier rejection.

The output of the module is the optimized poses within the local estimated frame to correct drift in pose esimtation. 
The algorithm optimizes the 3D points, camera poses, and intrinsic parameters of the calibrated camera (If permitted to) for 
a given window of data and estimated camera poses.

Initialization/Function Parameters:
- refine_focal_length: Whether to refine the focal length parameter group. 
    - default (bool): False
- refine_principal_point: Whether to refine the principal point parameter group. (bool, )
    - default (bool): False
- refine_extra_params: Whether to refine the extra parameter group.
    - default (bool): False
- max_num_iterations: maximum number of iterations to run the Levenberg-Marquardt algorithm for bundle adjustment
    - default (int): 50
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
# To use GPU Context now removed.
# - use_gpu: Whether to use Ceres CUDA linear algebra library, if available. 
#     - default (bool): True
# - gpu_index: Which GPU to use for solving the problem.
#     - default (int): 0 

        example = f"""
Initialization modules
from modules.baseclass import SfMScene
from modules.features import ....
from modules.featurematching import ....
from modules.optimization import {module_name}
from modules.camerapose import ...


# Start SfM Pipeline 
# Step 1: Read in Calibration/Image Data
reconstructed_scene = SfMScene(image_path = image_path, 
                            calibration_path = calibration_path)

# Step 2: Detect Features must be completed prior!
# Step 3: Feature Matching Pairs module must be completed prior!

# Step 4: Detect Camera Poses and apply Bundle Adjustment (Local Optimizer)
reconstructed_scene.CamPoseEstimatorEssentialToPnP(
    iteration_count=150,
    reprojection_error = 3.0,
    optimizer = ("BundleAdjustmentOptimizerLocal", {{
        "max_num_iterations": 25,
        "robust_loss": True,
        "use_gpu": False
    }}),
)
"""     
        super().__init__(cam_data=cam_data,
                         module_name=module_name,
                         description=description,
                         example=example)
        
        # Set up Internal Parameters/Settigns for Bundle Adjustment
        self.refine_focal_length = refine_focal_length
        self.refine_principal_point = refine_principal_point
        self.refine_extra_params = refine_extra_params
        self.max_num_iterations = max_num_iterations
        # self.use_gpu = use_gpu
        # self.gpu_index = gpu_index
        self.robust_loss = robust_loss

        # Set up window size and min_track_length 
        self.window_size = window_size
        self.min_track_len = min_track_len

        # Do not use metrics for this module
        self.use_base_metrics = False
        self.use_no_metrics = True
    
    def _optimize_scene(self, 
                        state: IncrementalSfMState, 
                        new_image_id: int) -> IncrementalSfMState:
        # return super()._optimize_scene(current_scene) 
        # --- Step 1 - Build Reconstruction ---
        recon, window = self._build_reconstruction(state, new_image_id)
        # --- Step 2 - Set BA Solver Options and Config settings --- 
        ba_opts, config = self._build_adjuster(recon, window)

        # --- Step 3: Build and Run Solver with Options/Config/Reconstruction ---
        # bundle_adjuster = pycolmap.create_default_bundle_adjuster(ba_opts, config, recon)
        # summary = bundle_adjuster.solve()
        new_state = self._solve_poses(state, ba_opts, config, recon, window)

        return new_state
    
    def _solve_poses(self, state: IncrementalSfMState, 
                     ba_opts, config, recon, window) -> IncrementalSfMState:
        # Solve and Optimize given Scene
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

    def _build_adjuster(self, recon, window):
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
        # if self.use_gpu:
        #     ba_opts.use_gpu = bool(self.use_gpu)
        #     ba_opts.gpu_index = str(self.gpu_index)

        # Ceres solver options (requires PyCeres installed in your environment)
        # was ba_opts.solver_options.max_num_iterations = int(self.max_num_iterations) 
        ba_opts.ceres.solver_options.max_num_iterations = int(self.max_num_iterations) 

        # Optional robust loss
        if self.robust_loss:
            # Was: ba_opts.loss_function_type = pycolmap.LossFunctionType.CAUCHY
            ba_opts.ceres.loss_function_type = pycolmap.LossFunctionType.CAUCHY

        return ba_opts, config
    
    # def optimize(self, 
    #              state: IncrementalSfMState, 
    #              new_image_id: int):

    #     recon, window = self._build_reconstruction(state, new_image_id)

    #     # --- BA config: include all registered images ---
    #     config = pycolmap.BundleAdjustmentConfig()
    #     for image_id in recon.reg_image_ids():  # registered images
    #         config.add_image(image_id)
    #     # --- BA config: Fix the first camera for stability
    #     config.set_constant_rig_from_world_pose(int(window[0]))


    #     # --- BA options ---
    #     ba_opts = pycolmap.BundleAdjustmentOptions()
    #     ba_opts.refine_focal_length = self.refine_focal_length
    #     ba_opts.refine_principal_point = self.refine_principal_point
    #     ba_opts.refine_extra_params = self.refine_extra_params

    #     # GPU knobs (only used when supported in your build)
    #     if self.use_gpu:
    #         ba_opts.use_gpu = bool(self.use_gpu)
    #         ba_opts.gpu_index = str(self.gpu_index)

    #     # Ceres solver options (requires PyCeres installed in your environment)
    #     # was ba_opts.solver_options.max_num_iterations = int(self.max_num_iterations) 
    #     ba_opts.ceres.solver_options.max_num_iterations = int(self.max_num_iterations) 

    #     # Optional robust loss
    #     if self.robust_loss:
    #         # Was: ba_opts.loss_function_type = pycolmap.LossFunctionType.CAUCHY
    #         ba_opts.ceres.loss_function_type = pycolmap.LossFunctionType.CAUCHY

    #     bundle_adjuster = pycolmap.create_default_bundle_adjuster(ba_opts, config, recon)
    #     _ = bundle_adjuster.solve()

    #     # --- Export optimized results back into your Scene ---
    #     # Write refined poses back into state
    #     for img_id in window:
    #         img = recon.image(int(img_id))
    #         T = img.cam_from_world()
    #         R = T.rotation.matrix()
    #         t = np.asarray(T.translation).reshape(3, 1)
    #         state.poses[img_id] = np.hstack([R, t])

    #     return state

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
        output_dir: str | None = None,
        refine_focal_length: bool = False,
        refine_principal_point: bool = False,
        refine_extra_params: bool = False,
        max_num_iterations: int = 50,
        # use_gpu: bool = True,
        # gpu_index: int = 0,
        robust_loss: bool = True,
    ):
        # super().__init__(cam_data=cam_data,
        #                  refine_focal_length=refine_focal_length,
        #                  refine_principal_point=refine_principal_point,
        #                  refine_extra_params=refine_extra_params,
        #                  max_num_iterations=max_num_iterations,
        #                  use_gpu=use_gpu,
        #                  gpu_index=gpu_index,
        #                  robust_loss=robust_loss)

        module_name = "BundleAdjustmentOptimizerGlobal"
        description = f"""
Global Optimization tool using the bundle adjustment optimization algorithm to optimize the reconstructed sparse 
scene using the 3D estimated points, 2D feature tracks for each 3D estimated point, and estimated camera poses of the 
monocular camera scene to optimize the reprojection error loss of each estimated 3D point. The output is a newly 
optimize sparse 3D reconstructed scene. The algorithm optimizes the 3D points, camera poses, and intrinsic parameters 
of the calibrated camera (If permitted to). ALWAYS USE THIS MODULE AT THE END OF A SPARSE RECONSTRUCTION PIPELINE, AND 
PRIOR TO ESTIMATING DENSE RECONSTRUCTION!

Initialization/Function Parameters:
- refine_focal_length: Whether to refine the focal length parameter group. 
    - default (bool): False
- refine_principal_point: Whether to refine the principal point parameter group. (bool, )
    - default (bool): False
- refine_extra_params: Whether to refine the extra parameter group.
    - default (bool): False
- max_num_iterations: maximum number of iterations to run the Levenberg-Marquardt algorithm for bundle adjustment
    - default (int): 50
- robust_loss: Determins whether to use one of the loss function types
    Loss function types: Trivial (non-robust, robust = False) and Cauchy (robust, robust = True) loss
    - default (bool): True

Function Calls - HANDLED INTERNALLY, DO NOT USE IF SFMCORE IN USE:
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

        example = f"""
Initialization:
from modules.features import ...
from modules.featurematching import ... (Pair Module), ... (Tracking Module)
from modules.camerapose import ...
from modules.scenereconstruction import ...
from modules.optimization import {module_name}
from modules.baseclass import SfMScene

Function Use:
# Step 1: Read in Calibration/Image Data
reconstructed_scene = SfMScene(image_path = image_path, 
                               calibration_path = calibration_path)

# Step 2: Detect Features Prior to Step 3 (Data filled in SfMScene)

# Step 3: Detect Feature Pairwise Matches Prior to Step 4 (Data filled in SfMScene)

# Step 4: Detect Cam Poses Prior to Step 5 Using a Pose Modules

# Step 5: Detect Feature Tracks Prior to Step 6 (Data filled in SfMScene)

# Step 6: Estimate Sparse Reconstruction using Prior to Global Optimization (Step 7)

# Step 7: Run Global Optimization on Sparsely Reconstructed Scene 
reconstructed_scene.BundleAdjustmentOptimizerGlobal(
    max_num_iterations=200
)
"""
        super().__init__(cam_data=cam_data,
                         module_name=module_name,
                         description=description,
                         example=example)
        
        # Set up Internal Parameters/Settigns for Bundle Adjustment
        self.refine_focal_length = refine_focal_length
        self.refine_principal_point = refine_principal_point
        self.refine_extra_params = refine_extra_params
        self.max_num_iterations = max_num_iterations
        # self.use_gpu = use_gpu
        # self.gpu_index = gpu_index
        self.robust_loss = robust_loss

        # Define workspace location
        self.output_dir = output_dir

        # Define the workspace for Sparse Reconstruction
        self.dir_path = Path(__file__).resolve().parents[2]
        self.directory_path = str(self.dir_path / "results" / "workspace" / "sparse") #C:\\Users\\Anthony\\Documents\\Projects\\scene_agent\\breadth_agent\\results\\workspace\\sparse"
        if os.path.exists(self.directory_path):
            # Delete the directory and all its contents
            shutil.rmtree(self.directory_path)

        # Recreate an empty directory
        os.makedirs(self.directory_path)

    def _optimize_scene(self, current_scene: Scene) -> Scene:
        # return super()._optimize_scene(current_scene) 
        # --- Step 1 - Build Reconstruction ---
        recon, trackid_to_point3Did = self._build_reconstruction(current_scene)
        # --- Step 2 - Set BA Solver Options and Config settings --- 
        ba_opts, config = self._build_adjuster(recon)

        # --- Step 3: Build and Run Solver with Options/Config/Reconstruction ---
        # bundle_adjuster = pycolmap.create_default_bundle_adjuster(ba_opts, config, recon)
        # summary = bundle_adjuster.solve()
        # Get initial Cost
        recon.update_point_3d_errors()
        self.initial_mean_error = recon.compute_mean_reprojection_error()
        # Run Optimizer
        summary = self._solve(ba_opts, config, recon)

        print("SUMMARY", summary)
        self.summary = summary
        # --- Step 4: Export optimized results back into the Scene ---
        self._write_back_to_scene(current_scene, recon, trackid_to_point3Did)

        # Write reconstructed scene to workspace (Sparse Scene Currently)
        recon.write(self.directory_path)
        recon.export_PLY(str(self.dir_path / "results" / "workspace" / "sparse.ply"))

        # Get final Metric (reprojection errors)
        recon.update_point_3d_errors()
        # Mean reprojection error in pixels (final cost)
        self.mean_error = recon.compute_mean_reprojection_error()

        #Record Camera Poses
        self._store_extrinsics_information(recon)

        return current_scene #, summary
    
    def _solve(self, ba_opts, config, recon):
        bundle_adjuster = pycolmap.create_default_bundle_adjuster(ba_opts, config, recon)
        return bundle_adjuster.solve()

    def _build_adjuster(self, recon):
        # --- BA config: include all registered images ---
        config = pycolmap.BundleAdjustmentConfig()
        for image_id in recon.reg_image_ids():  # registered images
            config.add_image(image_id)
        # --- BA config: Fix the first camera for stability
        reg_ids = recon.reg_image_ids()
        if len(reg_ids) > 0:
            # config.set_constant_rig_from_world_pose(recon.reg_image_ids()[0])
            config.set_constant_rig_from_world_pose(reg_ids[0])


        # --- BA options ---
        ba_opts = pycolmap.BundleAdjustmentOptions()
        ba_opts.refine_focal_length = self.refine_focal_length
        ba_opts.refine_principal_point = self.refine_principal_point
        ba_opts.refine_extra_params = self.refine_extra_params

        # GPU knobs (only used when supported in your build)
        # if self.use_gpu:
        #     ba_opts.use_gpu = bool(self.use_gpu)
        #     ba_opts.gpu_index = str(self.gpu_index)

        # Ceres solver options (requires PyCeres installed in your environment)
        ba_opts.ceres.solver_options.max_num_iterations = int(self.max_num_iterations) 

        # Optional robust loss
        if self.robust_loss:
            ba_opts.ceres.loss_function_type = pycolmap.LossFunctionType.CAUCHY
        
        return ba_opts, config

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

    def _store_extrinsics_information(self, recon) -> None:
        out_path = os.path.join(self.log_dir, f"cam_poses_log.npz")
        image_records = []

        for image_id, image in recon.images.items():
            if not image.has_pose:
                continue

            camera = recon.cameras[image.camera_id]

            R_wc, t_wc = self.get_image_pose_world_to_cam(image)
            C_w = -R_wc.T @ t_wc

            K, cam_params = self.camera_to_K_and_dist(camera)

            image_records.append(
                {
                    "image_id": int(image_id),
                    "image_name": image.name,
                    "camera_id": int(image.camera_id),
                    "width": int(camera.width),
                    "height": int(camera.height),
                    "camera_model": str(camera.model),
                    "camera_params": cam_params,
                    "K": K,
                    "R_world_to_cam": R_wc,
                    "t_world_to_cam": t_wc,
                    "camera_center_world": C_w,
                }
            )

        # Sort by image name for deterministic ordering.
        image_records = sorted(image_records, key=lambda x: x["image_name"])

        image_ids = np.array([r["image_id"] for r in image_records], dtype=np.int64)
        image_names = np.array([r["image_name"] for r in image_records])
        camera_ids = np.array([r["camera_id"] for r in image_records], dtype=np.int64)

        widths = np.array([r["width"] for r in image_records], dtype=np.int64)
        heights = np.array([r["height"] for r in image_records], dtype=np.int64)
        camera_models = np.array([r["camera_model"] for r in image_records])

        K = np.stack([r["K"] for r in image_records], axis=0)
        R_world_to_cam = np.stack([r["R_world_to_cam"] for r in image_records], axis=0)
        t_world_to_cam = np.stack([r["t_world_to_cam"] for r in image_records], axis=0)
        camera_center_world = np.stack([r["camera_center_world"] for r in image_records], axis=0)

        # Camera params can be different lengths depending on model.
        # Store as object array.
        camera_params = np.array([r["camera_params"] for r in image_records], dtype=object)

        save_dict = {
            "image_ids": image_ids,
            "image_names": image_names,
            "camera_ids": camera_ids,
            "widths": widths,
            "heights": heights,
            "camera_models": camera_models,
            "camera_params": camera_params,
            "K": K,
            "R_world_to_cam": R_world_to_cam,
            "t_world_to_cam": t_world_to_cam,
            "camera_center_world": camera_center_world,
        }

        np.savez_compressed(out_path, **save_dict)

    def _write_back_to_scene(self, 
                            scene: Scene,
                            recon,
                            trackid_to_point3Did) -> None:
        
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

    @module_metric
    def _metric_ba_results(self) -> dict:
        return {"Convergence": str(self.summary.termination_type.name),
                "Initial Cost": float(self.summary.ceres_summary.initial_cost),
                "Final Cost": float(self.summary.ceres_summary.final_cost),
                "Initial Reprojection Error": float(self.initial_mean_error), 
                "Final Reprojection Error": float(self.mean_error)}
    

    # Helper File to extract camera pose information
    def get_image_pose_world_to_cam(self, image):
        """
        Returns R_world_to_cam, t_world_to_cam from a pycolmap Image.

        pycolmap versions differ slightly, so this tries the common APIs.
        """
        # Newer pycolmap versions
        if hasattr(image, "cam_from_world"):
            cam_from_world = image.cam_from_world

            # Rigid3d-like object
            if hasattr(cam_from_world, "rotation") and hasattr(cam_from_world, "translation"):
                rot = cam_from_world.rotation
                t = np.asarray(cam_from_world.translation, dtype=np.float64)

                if hasattr(rot, "matrix"):
                    R = np.asarray(rot.matrix(), dtype=np.float64)
                elif hasattr(rot, "to_matrix"):
                    R = np.asarray(rot.to_matrix(), dtype=np.float64)
                else:
                    R = np.asarray(rot, dtype=np.float64)

                return R, t

            # Sometimes transform matrix may be exposed
            if hasattr(cam_from_world, "matrix"):
                T = np.asarray(cam_from_world.matrix(), dtype=np.float64)
                return T[:3, :3], T[:3, 3]

        # Older pycolmap-style API
        if hasattr(image, "qvec") and hasattr(image, "tvec"):
            qvec = np.asarray(image.qvec, dtype=np.float64)
            t = np.asarray(image.tvec, dtype=np.float64)
            R = self.qvec_to_rotmat(qvec)
            return R, t

        raise RuntimeError(f"Could not extract pose for image {image.name}")


    def qvec_to_rotmat(self, qvec):
        """
        COLMAP quaternion convention: q = [qw, qx, qy, qz].
        """
        qvec = np.asarray(qvec, dtype=np.float64)
        qw, qx, qy, qz = qvec

        return np.array([
            [
                1 - 2 * qy ** 2 - 2 * qz ** 2,
                2 * qx * qy - 2 * qz * qw,
                2 * qz * qx + 2 * qy * qw,
            ],
            [
                2 * qx * qy + 2 * qz * qw,
                1 - 2 * qx ** 2 - 2 * qz ** 2,
                2 * qy * qz - 2 * qx * qw,
            ],
            [
                2 * qz * qx - 2 * qy * qw,
                2 * qy * qz + 2 * qx * qw,
                1 - 2 * qx ** 2 - 2 * qy ** 2,
            ],
        ], dtype=np.float64)


    def camera_to_K_and_dist(self, camera):
        """
        Returns a 3x3 calibration matrix K and raw COLMAP camera params.

        camera.params is model-dependent. We save both:
        - K for easy evaluation
        - raw params/model for exact reconstruction reference
        """
        if hasattr(camera, "calibration_matrix"):
            K = np.asarray(camera.calibration_matrix(), dtype=np.float64)
        else:
            # Fallback for common COLMAP models.
            params = np.asarray(camera.params, dtype=np.float64)
            model = str(camera.model)

            if "SIMPLE" in model:
                f, cx, cy = params[:3]
                fx, fy = f, f
            else:
                fx, fy, cx, cy = params[:4]

            K = np.array(
                [
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )

        params = np.asarray(camera.params, dtype=np.float64)

        return K, params

    # def optimize(self, 
    #              scene: Scene):
    #             #  points: PointsMatched, 
    #             #  camera_poses: CameraPose, 
    #             #  cam_data: CameraData):
    #     """
    #     scene.points3D.points3D: (N,3) float
    #     camera_poses.camera_pose[i]: (3,4) cam_from_world for frame i
    #     points.data_matrix: (M,4) [track_id, frame_num, x, y]  (pixels)
    #     """
    #     # if not points.multi_view:
    #     #     raise ValueError("BA requires multi-view tracks (points.multi_view=True).")

    #     recon, trackid_to_point3Did = self._build_reconstruction(scene)

    #     # --- BA config: include all registered images ---
    #     config = pycolmap.BundleAdjustmentConfig()
    #     for image_id in recon.reg_image_ids():  # registered images
    #         config.add_image(image_id)
    #     # --- BA config: Fix the first camera for stability
    #     config.set_constant_rig_from_world_pose(recon.reg_image_ids()[0])


    #     # --- BA options ---
    #     ba_opts = pycolmap.BundleAdjustmentOptions()
    #     ba_opts.refine_focal_length = self.refine_focal_length
    #     ba_opts.refine_principal_point = self.refine_principal_point
    #     ba_opts.refine_extra_params = self.refine_extra_params

    #     # GPU knobs (only used when supported in your build)
    #     # ba_opts.use_gpu = bool(self.use_gpu)
    #     # ba_opts.gpu_index = str(self.gpu_index)

    #     # # Ceres solver options (requires PyCeres installed in your environment)
    #     # ba_opts.solver_options.max_num_iterations = int(self.max_num_iterations)

    #     # # Optional robust loss
    #     # if self.robust_loss:
    #     #     ba_opts.loss_function_type = pycolmap.LossFunctionType.CAUCHY

    #     # GPU knobs (only used when supported in your build)
    #     if self.use_gpu:
    #         ba_opts.use_gpu = bool(self.use_gpu)
    #         ba_opts.gpu_index = str(self.gpu_index)

    #     # Ceres solver options (requires PyCeres installed in your environment)
    #     ba_opts.ceres.solver_options.max_num_iterations = int(self.max_num_iterations) 

    #     # Optional robust loss
    #     if self.robust_loss:
    #         ba_opts.ceres.loss_function_type = pycolmap.LossFunctionType.CAUCHY

    #     bundle_adjuster = pycolmap.create_default_bundle_adjuster(ba_opts, config, recon)
    #     summary = bundle_adjuster.solve()

    #     # --- Export optimized results back into your Scene ---
    #     self._write_back_to_scene(scene, recon, trackid_to_point3Did)

    #     # Write reconstructed scene to workspace (Sparse Scene!)
    #     recon.write(self.directory_path)

    #     return scene #, summary
    