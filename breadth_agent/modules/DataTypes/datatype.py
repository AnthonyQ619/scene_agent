import numpy as np
from dataclasses import dataclass

@dataclass
class Points2D:
    points2D: np.ndarray
    descriptors: np.ndarray

    def update_2D_points_index(self, indices: list[int]) -> None:
        self.points2D = self.points2D[indices]
        self.descriptors = self.descriptors[indices]

    def update_2D_points_values(self, points2D) -> None:
        self.points2D = points2D
        self.descriptors = None


@dataclass
class Points3D:
    points3D = np.ndarray
    cam_poses = np.ndarray # Scene Corresponding to each point value 

    def update_3D_points(self, points3D) -> None:
        self.Points3D = points3D

@dataclass
class Calibration:
    K: np.ndarray # Camera intrinsics 
    K2: np.ndarray # 2nd Camera Intrinsics
    distort: np.ndarray # Camera 1 Distortion
    distort2: np.ndarray # Camera 2 Distortion
    R12: np.ndarray # Rotation of Stereo Camera
    T12: np.ndarray # Translation of Stereo Camera (Baseline)
    camera_poses: np.ndarray #Nx12 [R1 R2 ... R9 T1 T2 T3] -> N = # of scenes
    stereo: bool

    def __init__(self, K = np.eye(3), cam_poses = None, 
                 R = None, T = None, stereo = False, K2 = None):
        self.K = K # If Stereo, this will be a lsit of calibration vals
        self.stereo = stereo

        if self.stereo:
            self.K2 = K2
            self.R12 = R # Rotation from camera 1 to camera 2
            self.T12 = T # Baseline between camera 1 and camera 2
        
        self.extrinsics = cam_poses

    def get_intrinsics(self):
        if self.stereo:
            return self.K, self.K2
        
        return self.K
    
    def get_extrinsics(self):
        assert(self.stereo)
        
        return self.R12, self.T12
    
    def set_cam_poses(self, cam_poses):
        self.extrinsics = cam_poses
    
    def get_cam_pose(self, index):
        assert self.extrinsics is not None

        return self.extrinsics[index]
    
    def get_cam_poses(self):
        assert self.extrinsics is not None

        return self.extrinsics


