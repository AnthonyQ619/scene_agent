from __future__ import annotations
from PIL import Image
import numpy as np
import torch
import cv2
import os

import inspect
from abc import ABC
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Optional, Tuple, Union, Dict, Any

@dataclass
class CameraData:
    # --- Image Data ---
    image_names: List[str]
    image_list: List[Image.Image]        
    image_shape_old: Tuple[int, int]    # (width, height)
    image_shape_new: Tuple[int, int]    # (width, height)
    image_scale: Tuple[float, float]    # (width, height)

    # --- Calibration ---
    intrinsics: List[np.ndarray]  = field(default_factory=[np.zeros((3,3))])   # Camera Intrinsics (3x3 K Matrix) 
    distortions: List[np.ndarray] = field(default_factory=[np.zeros((1, 5))])     # Camera Distortions (OpenCV convention of 1x5 [k1, k2, p1, p2, k3])

    stereo: bool = False
    multi_cam: bool = False
    extrinsic: Optional[np.ndarray] = None    # Rotation | Translation of Stereo Camera

    # Logging Info
    metric_file_path: str = ""
    logging_dir: str = ""
    script_id: str = ""
    gpu_num: str = ""

    def update_K(self, cam_idx: int, img_scale: Tuple[float, float]):
        # Assume Monocular Camera for now with OpenCV calibration convention (Wide belief)
        # Meaning: Skew Parameter will be zero in these cases.
        # height_scale, width_scale = img_scale[:]
        assert (cam_idx >= 0 and cam_idx < len(self.intrinsics)), "Assumed to be more cameras. Intrinsics likely not loaded properly. Use CameraDataManager to load calibration or estimate through VGGTPoseEsimtation."
        width_scale, height_scale = img_scale[:]

        self.intrinsics[cam_idx][0,0] = width_scale * self.intrinsics[cam_idx][0,0]   # fx x width_scale = fx'
        self.intrinsics[cam_idx][1,1] = height_scale * self.intrinsics[cam_idx][1,1]  # fy x height_scale = fy'
        self.intrinsics[cam_idx][0,2] = width_scale * self.intrinsics[cam_idx][0,2]   # cx x width_scale = cx'
        self.intrinsics[cam_idx][1,2] = height_scale * self.intrinsics[cam_idx][1,2]  # cy x height_scale = cy'

    def update_calibration(self, img_scale: Tuple[float, float]):
        if self.intrinsics is not None:
            for cam_idx in range(len(self.intrinsics)):
                width_scale, height_scale = img_scale[:]

                self.intrinsics[cam_idx][0,0] = width_scale * self.intrinsics[cam_idx][0,0]   # fx x width_scale = fx'
                self.intrinsics[cam_idx][1,1] = height_scale * self.intrinsics[cam_idx][1,1]  # fy x height_scale = fy'
                self.intrinsics[cam_idx][0,2] = width_scale * self.intrinsics[cam_idx][0,2]   # cx x width_scale = cx'
                self.intrinsics[cam_idx][1,2] = height_scale * self.intrinsics[cam_idx][1,2]  # cy x height_scale = cy'

    def apply_new_calibration(self, 
                              intrinsics: List[np.ndarray],
                              distortion: List[np.ndarray] | None = None):
        if len(intrinsics) > 1:
            if distortion is None:
                self.intrinsics = intrinsics
                self.distortions = [np.zeros(1, 5)]*len(intrinsics) # Ensure distortion Param exists with equivalent list size
            else:
                self.intrinsics = intrinsics
                self.distortions = distortion
            self.multi_cam = True
        else:
            self.intrinsics = intrinsics[0]
            self.distortions = distortion[0]

    def get_K(self):
        #assert(self.intrinsics is not None), "Calibration Data is not properly loaded. Ensure necessary steps are taken to generate calibration through VGGT tools or calibration is properly read with CameraDataManager."
        if self.intrinsics is None:
            return None
        elif self.stereo:
            return self.intrinsics[0], self.intrinsics[1]
        elif self.multi_cam:
            return self.intrinsics
        else:
            return self.intrinsics[0]
    
    def get_distortion(self):
        #assert(self.intrinsics is not None), "Calibration Data is not properly loaded. Ensure necessary steps are taken to generate calibration through VGGT tools or calibration is properly read with CameraDataManager."
        if self.intrinsics is None:
            return None
        elif self.stereo:
            return self.distortions[0], self.distortions[1]
        elif self.multi_cam:
            return self.distortions
        else:
            return self.distortions[0]


@dataclass
class Calibration:
    K1: np.ndarray # Camera intrinsics (Single)
    K2: np.ndarray # 2nd Camera Intrinsics (Stereo)
    K_cams: list[np.ndarray] # Set of camera intrinics for multi camera approach (Mono cameras)
    distort: np.ndarray # Camera 1 Distortion (Single)
    distort2: np.ndarray # Camera 2 Distortion (Stereo)
    cam_dists: list[np.ndarray] # set of camera distortion params for multi camera approach (Mono cameras)
    R12: np.ndarray # Rotation of Stereo Camera
    T12: np.ndarray # Translation of Stereo Camera (Baseline)
    stereo: bool
    multi_cam: bool # Determine if the Mono camera setup takes into account multiple cameras

    def __init__(self, 
                 K1 = np.eye(3),  
                 R: np.ndarray | None = None, 
                 T: np.ndarray | None = None, 
                 stereo: bool = False, 
                 K2: np.ndarray | None = None, 
                 dist: np.ndarray | None = np.zeros((1,5)),
                 dist2: np.ndarray | None = np.zeros((1,5))):
        self.multi_cam = False
        self.K1 = K1 # For single camera mono setup
        self.distort = dist
        self.stereo = stereo

        self.K2 = K2 # For stereo camera
        self.distort2 = dist2
        self.R12 = R # Rotation from camera 1 to camera 2
        self.T12 = T # Baseline between camera 1 and camera 2
        if self.stereo:
            self.distort2 = None
        
    def setup_multi_cam(self, cam_ints: list[np.ndarray], cam_dists: list[np.ndarray]) -> None:
        # Set up old Mono cams back to default if changed
        self.K1 = np.eye(3)
        self.distort = np.zeros((1,5))

        # Update Mono Camera settings to multi_cam setup
        self.K_cams = cam_ints  
        self.cam_dists = cam_dists
        self.multi_cam = True

    def update_cal_img_shape(self, img_scale: list[float]):
        # Assume Monocular Camera for now with OpenCV calibration convention (Wide belief)
        # Meaning: Skew Parameter will be zero in these cases.
        # height_scale, width_scale = img_scale[:]
        width_scale, height_scale = img_scale[:]

        self.K1[0,0] = width_scale * self.K1[0,0]   # fx x width_scale = fx'
        self.K1[1,1] = height_scale * self.K1[1,1]  # fy x height_scale = fy'
        self.K1[0,2] = width_scale * self.K1[0,2]   # cx x width_scale = cx'
        self.K1[1,2] = height_scale * self.K1[1,2]  # cy x height_scale = cy'



    def get_intrinsics(self):
        if self.stereo:
            return self.K1, self.K2
        
        return self.K1
    
    def get_extrinsics(self):
        assert(self.stereo)
        
        return self.R12, self.T12