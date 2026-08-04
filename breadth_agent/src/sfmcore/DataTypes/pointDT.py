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
class Points2D:
    points2D: np.ndarray        # Nx2 [np.float32] (Mono or Left image)
    descriptors: np.ndarray     # NxM [np.float32] (32, 128, or 256 Depending on Detector)
    scores: np.ndarray          # Nx1 [np.float32] 
    orientation: np.ndarray     # 1xN [np.float32] orientation of the feature detected (SIFT)
    scale: np.ndarray           # 1xN [np.float32] scale of the feature detected
    binary_desc: bool           # Determine whether the descriptor from features are binary or float based.
    image_size: np.ndarray      # 1x2 [np.int64] (Simply Image Shape: (W, H))
    reshape_scale: list[float]  # 1x2 [float] (Simply Image reshape scalee: (W, H))

    def __init__(self, 
                 points2D: np.ndarray,
                 descriptors: np.ndarray,
                 scores: np.ndarray,
                 image_size: np.ndarray,
                 reshape_scale: list[float],
                 scale: np.ndarray | None = None,
                 orientation: np.ndarray | None = None,
                 binary_desc: bool = False):
        self.points2D = points2D
        self.descriptors = descriptors
        self.scores = scores
        self.image_size = image_size
        self.reshape_scale = reshape_scale
        self.binary_desc = binary_desc

        # For Detectors that include scale and orientation (SIFT)
        self.scale = scale
        self.orientation = orientation

    def update_2D_points_index(self, indices: list[int]) -> None:
        self.points2D = self.points2D[indices]
        self.descriptors = self.descriptors[indices]
        self.scores = self.scores[indices]

    def update_2D_points_values(self, points2D) -> None:
        self.points2D = points2D
        self.descriptors = None

    def splice_2D_points(self, indices: list[int]) -> dict[str: np.ndarray]:
        points2D = self.points2D[indices]
        descriptors = self.descriptors[indices]
        scores = self.scores[indices]

        return {"points2D" : points2D, "descriptors": descriptors, 'scores': scores, 'image_size': self.image_size, 'reshape_scale': self.reshape_scale}

    def set_inliers(self, mask: np.ndarray) -> dict[str: np.ndarray]:
        points2D = self.points2D[mask.ravel() == 1]
        if self.descriptors is not None:
            descriptors = self.descriptors[mask.ravel() == 1]
        else:
            descriptors = None
        if self.scores is not None:
            scores = self.scores[mask.ravel() == 1]
        else:
            scores = None

        return {"points2D" : points2D, "descriptors": descriptors, "scores": scores, 'image_size': self.image_size, 'reshape_scale': self.reshape_scale}

@dataclass
class Points3D:
    points3D: np.ndarray    # Point position in 3D space [x, y, z] : Nx3
    color: np.ndarray       # Point Color [r, g, b] : Nx3
    
    def __init__(self,  points: list[np.ndarray] | None = None, #np.array([[0.0, 0.0, 0.0]]), 
                        color: list[np.ndarray] | None = None): #np.array([0, 0, 0])):
        if points is None:
            self.points3D = None
        else:
            self.points3D = np.array(points)

        self.color = None #np.array(color)

    def update_points(self, 
                      points: list[np.ndarray], 
                      color: list[np.ndarray] | None = None) -> None:
        if self.points3D is None:
            if isinstance(points, list):
                self.points3D = np.array(points)
            elif isinstance(points, np.ndarray):
                self.points3D = points
            if color is not None:
                self.color = np.array(color)
        else:
            new_points = np.array(points)
            self.points3D = np.vstack((self.points3D, new_points))
            if color is not None:
                self.color = np.vstack((self.color,np.array(color)))
    
    def set_all_points(self, 
                       points: np.ndarray, 
                       color: np.ndarray | None = None) -> None:
        self.points3D = points
        self.color = color
