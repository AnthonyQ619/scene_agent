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
from .pointDT import Points2D, Points3D
from .featmatchDT import PointsMatched
from .cameraposeDT import CameraPose

Obs = Tuple[int, int]  # (image_id, kp_idx)

@dataclass
class IncrementalSfMState:
    K: np.ndarray
    dist: Optional[np.ndarray]
    width: int
    height: int

    # poses in cam-from-world 3x4
    poses: List[np.ndarray] = field(default_factory=list)

    # For BA-compatible states:
    #     keypoints[image_id][kp_idx] -> xy
    #
    # For obs-ID pose states:
    #     this may be empty or unused.
    keypoints: Dict[int, np.ndarray] = field(default_factory=dict)

    # Pose-estimation state:
    #     track_id -> list[(image_id, obs_id)]
    #
    # BA-compatible state:
    #     track_id -> list[(image_id, kp_idx)]
    tracks: Dict[int, List[Obs]] = field(default_factory=dict)

    # structure: track_id -> point3D xyz, shape (3,)
    points3D: Dict[int, np.ndarray] = field(default_factory=dict)

@dataclass
class Scene:
    points3D: Points3D              # Set of 3D points
    cam_poses: list[np.ndarray]     # Should be formatted as a 3x4 matrix
    observations: np.ndarray        # Mx4 matrices for each point observation where M=num_of_observations, and each row = [frame, 3d_point_ind, pix_x, pix_y]
    representation: str             # Represnetation of the scene (Future use cases here)
    # bal_data: BundleAdjustmentData  # Data stored in the BAL format, and write file to reconstructed scene (REFACTOR TO REMOVE)
    depth_maps: list[np.ndarray]    # Depth Maps per frame, formated as HeightxWidth of image shape
    sparse: bool                    # Used to determine if current scene is sparse or dense (Sparse=True)
    recon: object | None            # Pycolmap.Reconstruction type if provided (from VGGT no Feature Detection is the current tool where this exists!)

    def __init__(self, points3D: Points3D | None = Points3D(), 
                 cam_poses: list[np.ndarray] = [], 
                 observations: np.ndarray | None = None, 
                 representation: str = "point cloud",
                 sparse: bool = True,
                 bal_data : BundleAdjustmentData | None = None,
                 depth_maps: np.ndarray | None = None,
                 recon: object | None = None):
        self.SceneRepresentation = ["point cloud", "mesh", 'NeRF']

        
        self.points3D = points3D
        self.cam_poses = cam_poses
        self.observations = observations
        self.representation = representation
        self.depth_maps = depth_maps
        self.sparse = sparse
        self.recon = recon

        assert(self.representation in self.SceneRepresentation)

        if bal_data is not None:
            self._write_BAL_file(bal_data=bal_data)
            self.bal_data = bal_data

    def update_cam_pose(self, cam_pose: np.ndarray) -> None:
        self.cam_poses.append(cam_pose)

    def update_3d_points(self, points: Points3D) -> None:
        self.points3D.update_points(points=points.points3D)

    def _write_BAL_file(self, bal_data: BundleAdjustmentData) -> None:
        # File is fixed with name and location placement
        
        file_name = "BAL_Scene_Data"

@dataclass
class SceneState:
    cam_data: CameraData

    features: list[Points2D] | None = None
    feature_pairs: PointsMatched | None = None
    tracked_features: PointsMatched | None = None
    camera_poses: CameraPose | None = None

    sparse_scene: Scene | None = None
    dense_scene: Scene | None = None
    optimized_scene: Scene | IncrementalSfMState | None = None

    last_output: Any = None
    history: list[dict[str, Any]] = field(default_factory=list)

