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
class CameraPose:
    camera_pose: list[np.ndarray]   # Camera pose for each corresponding frame. Each pose is 3x4 (R, T)
    rotations: list[np.ndarray]     # Rotation matrices for each corresponding frame (Derived from camera_pose)
    translations: list[np.ndarray]  # Translation matrices for each corresponding frame (Derived from camera_pose)

    def __init__(self, cam_poses: list[np.ndarray] | None = [], 
                 rot: list[np.ndarray] | None = [], trans: list[np.ndarray] | None = []):
        self.camera_pose = cam_poses
        self.rotations = rot
        self.translations = trans

    def set_rotations(self) -> None:
        if self.camera_pose is None:
            message = 'Camera Poses have not been set. Please initate camera poses using the module "CameraPoseEstimator"'
            raise Exception(message)
        
        for i in range(len(self.camera_pose)):  
            rot = self.camera_pose[i][:,:3]
            self.rotations.append(rot)
    
    def set_translation(self) -> None:
        if self.camera_pose is None:
            message = 'Camera Poses have not been set. Please initate camera poses using the module "CameraPoseEstimator"'
            raise Exception(message)
        
        for i in range(len(self.camera_pose)):  
            trans = self.camera_pose[i][:,3:]
            self.translations.append(trans)
    
    def get_translations_np(self) -> np.ndarray: # Returns an Nx3 Array, where each row is the translation vector of the pose
        if self.camera_pose is None:
            message = 'Camera Poses have not been set. Please initate camera poses using the module "CameraPoseEstimator"'
            raise Exception(message)
        
        temp_trans = []
        for i in range(len(self.camera_pose)):  
            trans = self.camera_pose[i][:,3:]
            temp_trans.append(trans.T[0])

        return np.array(temp_trans)

    def set_rot_2_angle_axis(self) -> None:
        if self.camera_pose is None:
            message = 'Camera Poses have not been set. Please initate camera poses using the module "CameraPoseEstimator"'
            raise Exception(message)
        if len(self.rotations) > 0:
            self.rotations = []
        
        for i in range(len(self.camera_pose)):  
            rot = self.camera_pose[i][:,:3]
            rotation_vector, _ = cv2.Rodrigues(rot)
            self.rotations.append(rotation_vector)

    # Returns an Nx3 array, where each row is the rodrigues rotation vector for the estimated poses.
    def get_rot_2_angle_axis(self) -> np.ndarray: #list[np.ndarray]:
        if self.camera_pose is None:
            message = 'Camera Poses have not been set. Please initate camera poses using the module "CameraPoseEstimator"'
            raise Exception(message)
        
        rot_vecs = []
        for i in range(len(self.camera_pose)):  
            rot = self.camera_pose[i][:,:3].astype(np.float32)
            rotation_vector, _ = cv2.Rodrigues(rot)
            rot_vecs.append(rotation_vector.T[0])
        
        rot_vecs_np = np.array(rot_vecs)
        return rot_vecs_np

    def set_rot_2_quaternion(self) -> None:
        if self.camera_pose is None:
            message = 'Camera Poses have not been set. Please initate camera poses using the module "CameraPoseEstimator"'
            raise Exception(message)
        if len(self.rotations) > 0:
            self.rotations = []

        def quaternion(rotation: np.ndarray) -> np.ndarray:
            trace = rotation.trace()

            if trace > 0.0:
                s = np.sqrt(trace + 1.0)
                v1 = s*0.5
                s = 0.5 / s
                quat = np.array([v1,
                                rotation[2,1] - rotation[1,2]*s,
                                rotation[0,2] - rotation[2,0]*s,
                                rotation[1,0] - rotation[0,1]*s])
                return quat
            else:
                if rotation[0,0] < rotation[1,1]:
                    if rotation[1,1] < rotation[2,2]:
                        i = 2
                    else:
                        i = 1
                else:
                    if rotation[0,0] < rotation[2,2]:
                        i = 2
                    else:
                        i = 0
                j = (i + 1) % 3
                k = (i + 2) % 3

                quat = np.zeros((4,1))
                s = np.sqrt(rotation[i,i] - rotation[j,j] - rotation[k,k] + 1.0)

                quat[i] = s * 0.5
                s = 0.5 / s
                quat[3] = rotation[k,j] - rotation[j,k]*s
                quat[j] = rotation[j,i] + rotation[i,j]*s
                quat[k] = rotation[k,i] + rotation[i,k]*s

                return quat
            
        for i in range(len(self.camera_pose)):  
            rot = self.camera_pose[i][:,:3]
            quat = quaternion(rot)
            self.rotations.append(quat)
