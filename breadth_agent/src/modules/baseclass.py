'''
Base Class designs for each module to standardize the class design
for each tool/module.

This is to reduce the possiblility of the Agent to hallucinate code
'''

import numpy as np
import cv2
from DataTypes.datatype import Scene, Calibration, Points2D, PointsMatched, CameraPose
import glob

class GeometryClass():
    def __init__(self, format: str):
        self.module_name = "..."
        self.description = "..."
        self.example = "..."

        self.format = format

    def __call__(self):
        pass

class SceneEstimation():
    def __init__(self, calibration: Calibration, image_path: str):
        self.module_name = "..."
        self.description = "..."
        self.example = "..."

        self.stereo = calibration.stereo
        self.K1 = calibration.K1
        self.dist1 = calibration.distort
        if self.stereo:
            self.K2 = calibration.K2
            self.dist2 = calibration.distort2
            self.R12 = calibration.R12
            self.T12 = calibration.T12
        
        self.image_path = sorted(glob.glob(image_path + "\\*"))

    def __call__(self, tracked_features: PointsMatched, cam_poses: CameraPose) -> Scene:
        scene = Scene()

        return scene


class CameraPoseEstimatorClass():
    def __init__(self, calibration: Calibration, image_path:str):
        self.module_name = "..."
        self.description = "..."
        self.example = "..."

        self.stereo = calibration.stereo
        self.K1 = calibration.K1
        self.dist1 = calibration.distort
        if self.stereo:
            self.K2 = calibration.K2
            self.dist2 = calibration.distort2
        
        self.image_path = sorted(glob.glob(image_path + "\\*"))

        
    def __call__(self, features: list[Points2D] | None = None) -> CameraPose:
        poses = CameraPose()

        return poses

class FeatureClass():
    def __init__(self, image_path:str):
        self.module_name = "..."
        self.description = "..."
        self.example = "..."
        self.features = []

        self.image_path = sorted(glob.glob(image_path + "\\*"))

    def __call__(self) -> list[Points2D]:
        return self.features

class FeatureMatching():
    def __init__(self, detector:str):
        self.module_name = "..."
        self.description = "..."
        self.example = "..."
        self.detector = detector

        self.DETECTORS = ["sift", "orb", "fast"]

    def __call__(self) -> PointsMatched:
        # Points Matched for Tracking -> data = N x [track_id, frame_num, x, y]
        # Points Matched for Pairwise Matching -> 
        matched_points = PointsMatched() 


        return matched_points

class OptimizationClass():
    def __init__(self, calibration: Calibration, format: str):
        self.module_name = "..."
        self.description = "..."
        self.example = "..."

        if calibration is not None:
            self.K = calibration.K1

        self.OPTIONS = ['essential', 'fundamental', 'homography', 'projective']
        self.FORMATS = ['full', 'partial', 'pair']

        self.format = format
        
    def __call__(self):
        pass

class VisualizeClass():
    def __init__(self, path:str):
        self.module_name = "..."
        self.description = "..."
        self.example = "..."

        self.path = path

    def __call__(self, data: Scene | np.ndarray, store:bool = False) -> None:
        pass