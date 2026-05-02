from __future__ import annotations
'''
Base Class designs for each module to standardize the class design
for each tool/module.

This is to reduce the possiblility of the Agent to hallucinate code
'''
from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial import cKDTree
import cv2
from modules.cameramanager import CameraDataManager
from modules.DataTypes.datatype import (Scene, 
                                CameraData, 
                                Calibration, 
                                Points2D, 
                                PointsMatched, 
                                CameraPose, 
                                Points3D,
                                BundleAdjustmentData,
                                IncrementalSfMState,
                                SceneState)
import glob
from collections.abc import Callable
import torch
import open3d as o3d

import inspect
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Tuple

from pathlib import Path
import copy
import random
import json
from tqdm import tqdm

############################################# HELPER CLASSES #############################################

class PipelineModule(ABC):
    """
    Auto-register every concrete subclass by its public name.
    """
    REGISTRY: dict[str, type["PipelineModule"]] = {}

    exposed_name: str | None = None
    output_key: str | None = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if inspect.isabstract(cls):
            return

        public_name = cls.exposed_name or cls.__name__

        if public_name in PipelineModule.REGISTRY:
            raise ValueError(f"Duplicate pipeline module name: {public_name}")

        PipelineModule.REGISTRY[public_name] = cls

    def run_from_state(self, state: "SceneState") -> Any:
        raise NotImplementedError
    
# Convert points to normalized iamge coordinates!
class Normalization():
    def __init__(self, 
                 K: np.ndarray | None = None,
                 dist: np.ndarray | None = None,
                 multi_cam: bool = False):
        if K is None:
            self.K = None
            self.dist = None
            self.calibration = False
        else:
            if multi_cam:
                self.K_cams = K
                self.dists = dist
                self.calibration = True
                self.multi_cam = multi_cam
            else:
                self.K = K
                self.dist = dist
                self.multi_cam = False
                self.calibration = True

    def __call__(self, pts: Points2D, frame_id:int) -> np.ndarray:
        if self.calibration:
            return self._calibrated(pts, frame_id)
        else:
            return self._uncalibrated(pts)
        
    def _calibrated(self, pts: Points2D, frame_id: int) -> np.ndarray:
        # print(pts.points2D.shape)
        # print(self.dist1.shape)
        # print(self.K1.shape)
        if self.multi_cam:
            pts_norm = []
            # for i in range(cams.shape[0]): 
            #     cam = int(cams[i])
            K = self.K_cams[frame_id]
            dist = self.dists[frame_id]
            # pt = pts[i, :]
            pts_norm = cv2.undistortPoints(pts.points2D.T, K, dist)[:, 0, :]
        else:
            pts_norm = cv2.undistortPoints(pts.points2D.T, self.K, self.dist)[:, 0, :]

        return pts_norm
    

    def _uncalibrated(self, pts:Points2D) -> Points2D:
        pass 

class FeatureTracker():
    def __init__(self, 
                 matcher_parser: Callable[[Points2D, Points2D], tuple],
                 normalization: Normalization,
                 RANSAC_threshold: float,
                 RANSAC_conf: float,
                 homography: bool = False):

        # Establish the data structures 
        self.track_map = {}
        self.next_track_id = 0
        self.observations = []

        # Set up Outlier Check Parameters
        self.ransac_threshold = RANSAC_threshold
        self.ransac_conf = RANSAC_conf
        self.homography = homography

        # Set the Matcher
        self.matcher = matcher_parser

        # Set 2D Point Normalization
        self.normalize = normalization
    
    def tracking_points(self, frame_id: float, pts1: Points2D, pts2: Points2D) -> None:        
        for i in range(pts1.points2D.shape[0]):
            pt1_id = pts1.points2D[i, :].tobytes()
            pt2_id = pts2.points2D[i, :].tobytes()

            key1 = (frame_id, pt1_id)
            key2 = (frame_id + 1, pt2_id)

            if key2 in self.track_map:
                continue

            if key1 in self.track_map:
                track_id = self.track_map[key1]
            else:
                track_id = self.next_track_id
                self.next_track_id += 1
                pt1 = pts1.points2D[i, :]
                self.observations.append([float(track_id), frame_id, pt1[0], pt1[1]])
                self.track_map[key1] = track_id
        
            pt2 = pts2.points2D[i, :] 
            self.observations.append([float(track_id), frame_id + 1, pt2[0], pt2[1]])
            self.track_map[key2] = track_id

    def outlier_reject(self, pts1: Points2D, pts2: Points2D, frame_id: int) -> tuple[Points2D, Points2D]:
        pts1_norm = self.normalize(pts1, frame_id)
        pts2_norm = self.normalize(pts2, frame_id+1)
        pts1_t = pts1.points2D
        pts2_t = pts2.points2D
        if self.homography:
            F, mask = cv2.findHomography(pts1_t, pts2_t, #pts1_norm, pts2_norm, 
                                         cv2.USAC_MAGSAC, 
                                         ransacReprojThreshold=self.ransac_threshold,
                                         maxIters=10000,
                                         confidence=self.ransac_conf)
        else:
            F, mask = cv2.findFundamentalMat(pts1_t, pts2_t, #pts1_norm, pts2_norm, 
                                             cv2.USAC_MAGSAC, 
                                             ransacReprojThreshold=self.ransac_threshold,
                                             maxIters=10000,
                                             confidence=self.ransac_conf)

        # Could update points2D to inlier points with Mask
        inlier_pts1 = Points2D(**pts1.set_inliers(mask))
        inlier_pts2 = Points2D(**pts2.set_inliers(mask))
    
        return inlier_pts1, inlier_pts2
    
    def match_full(self, features: list[Points2D]) -> PointsMatched:
        img_size = features[0].image_size
        img_scale = features[0].reshape_scale
        tracked_features = PointsMatched(image_size=img_size, 
                                         multi_view=True,
                                         image_scale=img_scale)
        
        outlier_count = []
        matching_pair_ct = []

        for scene in tqdm(range(0, len(features) - 1), desc="Tracking Points"):
            pt1 = features[scene]
            pt2 = features[scene + 1]

            # matches, idx1, idx2 = self.matcher_parser(pt1, pt2) # Match and Lowe's Ratio Test
            idx1, idx2 = self.matcher(pt1, pt2)

            new_pt1 = Points2D(**pt1.splice_2D_points(idx1))
            new_pt2 = Points2D(**pt2.splice_2D_points(idx2))

            # Outlier Rejection Here
            # matches_inlier, inlier_pts1, inlier_pts2 = self.outlier_reject(matches, new_pt1, new_pt2)
            inlier_pts1, inlier_pts2 = self.outlier_reject(new_pt1, new_pt2, scene)
            # inlier_pts1, inlier_pts2 = self.ep_check(inlier_pts1, inlier_pts2)

            # Collect Metric Information
            matching_pair_ct.append(inlier_pts1.points2D.shape[0])
            # outlier_count.append(self._z_score(inlier_pts1.points2D, inlier_pts2.points2D, sigma_th=3))

            # Feature Tracking algorithm here
            self.tracking_points(scene, inlier_pts1, inlier_pts2) #, matches_inlier)

            # matched_points.append([new_pt1, new_pt2])
        
        # # Output Metric Information
        # counts_np = np.array(matching_pair_ct)
        # mean_ct = float(counts_np.mean())
        # min_ct = int(counts_np.min())
        # max_ct = int(counts_np.max())
        # avg_outlier = float(np.mean(np.array(outlier_count)))

        tracked_features.set_matched_matrix(self.observations)
        tracked_features.track_map = self.track_map
        tracked_features.point_count = self.next_track_id - 1
        
        return tracked_features
    
    # def _calculate_avg_track_length(self, data_mat: np.ndarray, total_points: int):

    #     track_ids = data_mat[:, 0].astype(int)

    #     unique_ids, counts = np.unique(track_ids, return_counts=True)

    #     # Track Length Average, Median, and Maximum
    #     max_track = counts.max()
    #     median_track_length = np.median(counts)
    #     avg_track = counts.mean()

    #     # Survival Curve Metric (Num of Tracks lasting N frames)
    #     survive_ge_3  = np.mean(counts >= 3) # Multi-View Suppert rate 3
    #     survive_ge_5  = np.mean(counts >= 5) # Multi-View Suppert rate 5
    #     survive_ge_10 = np.mean(counts >= 10) # Multi-View Suppert rate 10
       
    #     # Lower fragmentation and higher observations-per-track are usually better.
    #     fragmentation = len(counts) / np.sum(counts) # tracks per observation
    #     obs_per_track = np.sum(counts) / len(counts) # ovservation per track


    #     return {"Avg. track length": avg_track, 
    #             "Max. track length": max_track, 
    #             "Median track length": median_track_length, 
    #             "Survival Tracks of 3": survive_ge_3, 
    #             "Survival Tracks of 5": survive_ge_5, 
    #             "Survival Tracks of 10": survive_ge_10,
    #             "Fragmentation": fragmentation,
    #             "Obs. per Track": obs_per_track}

class TriangulationCheck():
    def __init__(self, 
                 K_mat: np.ndarray,
                 dist: np.ndarray): #min_angle: float = 1.0):
        # Calibration Set up
        self.K = K_mat
        self.dist = dist

        # # Minimum Angle Necessary 
        # self.min_angle = min_angle

    # views = [cam, x, y]:Nx3, camera_poses = [R, t]:4x4
    def __call__(self, views: np.ndarray, 
                 cam_poses: list[np.ndarray],
                 minimum_angle: float = 1.0) -> tuple[bool, float]:
        
        # Setup
        track_len = views.shape[0]
        max_angle = 0.0
        min_angle = 180.0

        for i in range(track_len):
            for j in range(i + 1, track_len):
                cam1, pt1 = views[i, 0], views[i, 1:]
                cam2, pt2 = views[j, 0], views[j, 1:]
                cam1 = int(cam1)
                cam2 = int(cam2)

                pt_vec1 = self.copmute_bearing_vec(pt1)
                pt_vec2 = self.copmute_bearing_vec(pt2)

                R1 = cam_poses[cam1][:, :3]
                R2 = cam_poses[cam2][:, :3]

                pt_vec1_R = R1 @ pt_vec1
                pt_vec2_R = R2 @ pt_vec2

                angle = self.angle_from_pts(pt1_vec=pt_vec1_R, pt2_vec=pt_vec2_R)

                # if angle > max_angle:
                #     max_angle = angle
                if angle <= min_angle:
                    min_angle = angle


        # return max_angle >= self.min_angle, max_angle
        return min_angle >= minimum_angle, min_angle

    def copmute_bearing_vec(self, pt: np.ndarray):
        pt_norm = cv2.undistortPoints(pt, cameraMatrix=self.K, distCoeffs=self.dist)[:,0,:]

        x = np.array([[pt_norm[0,0]], [pt_norm[0,1]], [1.0]])

        x_cam = np.linalg.inv(self.K)@(x)
        x_cam = x_cam / np.linalg.norm(x_cam)
        return x_cam
    
    def angle_from_pts(self, pt1_vec: np.ndarray, pt2_vec: np.ndarray):
        angle = np.dot(pt1_vec[:, 0], pt2_vec[:, 0])
        angle = np.clip(angle, -1.0, 1.0)

        return np.degrees(np.arccos(angle))

class KeypointRefinement():
    def __init__(self,
                 detector: str = "splg"):
        SUPPORTED_DETECTORS = ["splg", "spsg", "dedode", "aliked", "xfeat"]
        self.det = detector.lower()
        if self.det not in SUPPORTED_DETECTORS:
            message = 'Error: detector is not supported for keypoint refinement. Use one of ' + str(SUPPORTED_DETECTORS) + ' instead to use this Feature Matcher.'
            raise Exception(message)
        
        self.keypt2subpx = torch.hub.load('KimSinjeong/keypt2subpx', 'Keypt2Subpx', pretrained=True, detector=self.det)

    def get_refiner(self):
        return self.keypt2subpx
    
##########################################################################################################

# Define Decorator for Metric Functions
def module_metric(func):
    """Mark a method as a metric provider."""
    func._is_metric_provider = True
    return func

class SparseSceneEstimation(PipelineModule, ABC):
    use_base_metrics = True
    _registered_metric_methods: tuple[str, ...] = ()
    output_key = "sparse_scene"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        metric_names = []

        # inherit metrics from parent classes
        for base in reversed(cls.__mro__[1:]):
            metric_names.extend(getattr(base, "_registered_metric_methods", ()))

        # add metrics declared directly in this subclass
        for name, value in cls.__dict__.items():
            if callable(value) and getattr(value, "_is_metric_provider", False):
                metric_names.append(name)

        # remove duplicates, preserve order
        cls._registered_metric_methods = tuple(dict.fromkeys(metric_names))

    def run_from_state(self, state: SceneState) -> Scene:
        tracked = state.tracked_features or state.feature_pairs
        if tracked is None:
            raise RuntimeError(
                "SparseSceneEstimation requires tracked_features or feature_pairs."
            )
        if state.camera_poses is None:
            raise RuntimeError("SparseSceneEstimation requires camera_poses.")
        
        try:
            return self(tracked, state.camera_poses)
        except Exception as e:
            min_obs = getattr(self, "min_observations", None)

            raise RuntimeError(
                "[SparseReconstruction Error]\n"
                "Sparse reconstruction failed because there were not enough valid feature "
                "tracks to triangulate a reliable 3D point cloud.\n\n"
                "Likely causes:\n"
                "- Too few tracks satisfied the minimum observation requirement.\n"
                "- Most feature tracks were only visible in 2 views, which is not enough "
                "for stable multi-view reconstruction.\n"
                "- The feature detector and matcher combination did not produce long, "
                "consistent tracks across the image sequence.\n"
                "- Outlier rejection or matching may have removed too many correspondences "
                "before reconstruction.\n\n"
                "Action needed:\n"
                "- Improve the feature detector and matcher combination so that the majority "
                "of tracks are observed in at least 3 views.\n"
                "- Check the feature tracking metrics, especially average track length, "
                "median track length, and the number of tracks with length >= 3.\n"
                "- If most tracks are shorter than 3 views, switch to a stronger detector, "
                "increase detected keypoints, loosen matching/outlier rejection thresholds, "
                "or improve frame overlap.\n"
                "- Sparse reconstruction should only run successfully when enough 3+ tracks"
                "exist to build stable 3D points.\n\n"
                f"Minimum observations required per 3D point (if already at 3, consider improving detector/tracker combo): {min_obs}\n"
                f"Original error: {type(e).__name__}: {e}"
            ) from e

    
    def __init__(self, cam_data: CameraData,
                 module_name: str,
                 description: str,
                 example: str):

        # Define Module Name, Description, etc. 
        # Under modules.
        self.module_name = module_name
        self.description = description
        self.example = example

        # Setting up Calibration Data
        self.cam_data = cam_data
        self.image_list = copy.copy(cam_data.image_list)
        self.K_mat = cam_data.get_K()
        self.dist = cam_data.get_distortion()
        self.stereo = cam_data.stereo
        self.multi_cam = cam_data.multi_cam

        # Setup Minimum Angle Check Function
        self.angle_check = TriangulationCheck(self.K_mat, self.dist)
        
        #self.image_path = sorted(glob.glob(image_path + "\\*"))[:10]

    def __call__(self, tracked_features: PointsMatched, cam_poses: CameraPose) -> Scene:
        return self.build_reconstruction(tracked_features, cam_poses)
    
    @abstractmethod
    def build_reconstruction(self, 
                             tracked_features: PointsMatched,
                             cam_poses: CameraPose) -> Scene:
        """Implement Algorithm to reconstruct scene here."""
        raise NotImplementedError

    # Point Maps estimation must have this function follow with tracked features
    # Point Maps must be in the shape of 
    def match_tracks_to_point_maps(self, #Keep here
                                   tracked_features: PointsMatched,
                                   point_maps: np.ndarray,
                                   conf_maps: np.ndarray,
                                   minimum_observation: int,
                                   img_width: int,
                                   num_cameras: int,
                                   camera_poses: CameraPose,
                                   ) -> Scene:
       
        # points_3d = []
        points_3d = Points3D()
        w_scale, h_scale = tracked_features.image_scale[:]

        # # BAL File for Optimization Module
        # num_observations = 0
        # num_cameras = len(camera_poses.camera_pose)
        # observations = []
        scale = conf_maps.shape[-1] / img_width
        observations = []
        num_observations = 0
        point_index = 0

        observations_pix = []
        print(scale)
        print(img_width)
        print(point_maps[0].shape)
        for i in tqdm(range(tracked_features.point_count)):
                # views = [cam, x, y]:Nx3, camera_poses = [R, t]:4x4
                views = tracked_features.access_point3D(i)

                if views.shape[0] < minimum_observation:
                    track_len = views.shape[0]

                    # for j in range(track_len):
                    # Take the first view, and build the 3D points around that View
                    frame, point2d = views[0, 0], views[0, 1:]
                    frame = int(frame)
                    x, y = round(point2d[0]*scale), round(point2d[1]*scale) # Determine whether to scale these points or not...

                    # BAL Data Construction
                    point_ind = np.array([point_index for _ in range(views.shape[0])]).reshape((views.shape[0],1))
                    # norm_pts = self._normalize_points_for_BAL(views)#views[:, 1:])
                    # observation = np.hstack((np.vstack(views[:,0]), point_ind, norm_pts))#views[:,1:]))
                    observation_pix = np.hstack((np.vstack(views[:,0]), point_ind, views[:, 1:]))
                    observations_pix.append(observation_pix)
                    # observations.append(observation)
                    # num_observations += views.shape[0] # Number of observations

                    # get 3D point
                    pred_point_3d = point_maps[frame][y, x]

                    points_3d.update_points(pred_point_3d)

                    # Update Point Index
                    point_index += 1

        scene = Scene(points3D = points_3d,
                      cam_poses = camera_poses.camera_pose,
                      observations= np.vstack(observations_pix),
                      representation = "point cloud",
                    #   bal_data=ba_data,
                      sparse=True)
        return scene
    
    # Helper Function to Normalize for 2D Points for 3D estimation if Features are used
    def _normalize_points(self, view: np.ndarray): #KEEP but rename to normalize points
        cams, pts = view[:, 0], view[:, 1:]
        # Normalize Points without undistorting points for Bundle Adjustment Optimization
        if self.multi_cam:
            pts_norm = []
            for i in range(cams.shape[0]): 
                cam = int(cams[i])
                K = self.K_mat[cam]
                dist = self.dist[cam]
                pt = pts[i, :]
                pt_norm = cv2.undistortPoints(pt.T, K, dist)[:, 0, :][0]#np.zeros((1,5)))[:, 0, :][0]
                pts_norm.append(pt_norm)
            pts_norm = np.array(pts_norm)
        else:
            pts_norm = cv2.undistortPoints(pts.T, self.K_mat, np.zeros((1,5)))[:, 0, :]

        return pts_norm
    
    # Metric Function for reprojection error calculation
    def _reprojection_error(self, 
                            point: np.ndarray, 
                            views: np.ndarray, 
                            cam_poses: list[np.ndarray]) -> float:
        # X_h = np.append(point, 1)
        errors = []
        # print("POINT SHAPE:", X_h.shape)
        if self.multi_cam:
            for i in range(views.shape[0]):
                cam, pt = views[i, 0], views[i, 1:]
                cam = int(cam)
                K = self.K_mat[cam]
                dist = self.dist[cam]
                rvec = cam_poses[cam][:, :3]
                tvec = cam_poses[cam][:, 3]
                # Pmat = np.eye() @ cam_poses[cam] 
                # dist = self.dist[cam]

                # xUnd = cv2.undistortPoints(pt, K, dist)

                # x_proj_h = Pmat @ X_h
                # x_proj = x_proj_h[:2] / x_proj_h[2]
                # errors.append(np.linalg.norm(x_proj - xUnd))

                imgpt_proj, _ = cv2.projectPoints(objectPoints=point.reshape((1,3)),
                                                  rvec=rvec,
                                                  tvec=tvec,
                                                  cameraMatrix=K,
                                                  distCoeffs=dist)#cv2.projectPoints(point.reshape((1,3)), rvec, tvec, self.K_mat, self.dist)
                imgpt_proj = imgpt_proj.ravel()
                pixel_error = np.linalg.norm(imgpt_proj - pt)

                errors.append(pixel_error)

            return np.mean(errors)
        else:
            for i in range(views.shape[0]):
                cam, pt = views[i, 0], views[i, 1:]
                cam = int(cam)
                # Pmat = np.eye(3) @ cam_poses[cam]
                # xUnd = cv2.undistortPoints(pt, self.K_mat, self.dist)
                rvec = cam_poses[cam][:, :3]
                tvec = cam_poses[cam][:, 3]
                # print(rvec)
                # x_proj_h = Pmat @ X_h
                # x_proj = x_proj_h[:2] / x_proj_h[2]
                # errors.append(np.linalg.norm(x_proj - xUnd))
                imgpt_proj, _ = cv2.projectPoints(objectPoints=point.reshape((1,3)),
                                                  rvec=rvec,
                                                  tvec=tvec,
                                                  cameraMatrix=self.K_mat,
                                                  distCoeffs=self.dist)#cv2.projectPoints(point.reshape((1,3)), rvec, tvec, self.K_mat, self.dist)
                imgpt_proj = imgpt_proj.ravel()
                pixel_error = np.linalg.norm(imgpt_proj - pt)
                # print("Point SHAPE", imgpt_proj.shape)
                # print("Point Value", imgpt_proj)
                # print("Original Point:", pt)
                errors.append(pixel_error)

            return np.mean(errors)

class DenseSceneEstimation(PipelineModule, ABC):
    use_base_metrics = True
    _registered_metric_methods: tuple[str, ...] = ()
    output_key = "dense_scene"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        metric_names = []

        # inherit metrics from parent classes
        for base in reversed(cls.__mro__[1:]):
            metric_names.extend(getattr(base, "_registered_metric_methods", ()))

        # add metrics declared directly in this subclass
        for name, value in cls.__dict__.items():
            if callable(value) and getattr(value, "_is_metric_provider", False):
                metric_names.append(name)

        # remove duplicates, preserve order
        cls._registered_metric_methods = tuple(dict.fromkeys(metric_names))

    def run_from_state(self, state: SceneState) -> Scene:
        return self(
            sparse_scene=state.sparse_scene,
            camera_poses=state.camera_poses,
        )
    
    def __init__(self, cam_data: CameraData,
                 module_name: str,
                 description: str,
                 example: str):

        # Define Module Name, Description, etc. 
        # Under modules.
        self.module_name = module_name
        self.description = description
        self.example = example

        # Setting up Calibration Data
        self.cam_data = cam_data
        self.image_list = copy.copy(cam_data.image_list)
        self.K_mat = cam_data.get_K()
        self.dist = cam_data.get_distortion()
        self.stereo = cam_data.stereo
        self.multi_cam = cam_data.multi_cam


    def __call__(self, sparse_scene: Scene | None = None,
                 camera_poses: CameraPose | None = None) -> Scene:

        return self.build_reconstruction(sparse_scene = sparse_scene, 
                                         cam_poses = camera_poses)
    @abstractmethod
    def build_reconstruction(self, 
                             sparse_scene: Scene | None = None,
                             cam_poses: CameraPose | None = None) -> Scene:
        """Implement Algorithm to reconstruct scene here."""
        raise NotImplementedError
    
    # For Point Map Reconstruction Models, collect all Points!
    def collect_PM_points(self,
                          point_maps:np.ndarray,
                          conf_maps: np.ndarray =None, 
                          conf_thresh: float =0.5):
        all_points = []

        for i, pm in enumerate(point_maps):
            H, W, _ = pm.shape

            if conf_maps is not None:
                mask = conf_maps[i] > conf_thresh
            else:
                # assume invalid points are NaN
                mask = np.isfinite(pm).all(axis=-1)

            pts = pm[mask]          # (N, 3)
            all_points.append(pts)

        return np.concatenate(all_points, axis=0)
    
    # Down sample voxels for better reconstruction of Point Map based Models
    def voxel_downsample(self,
                         points: np.ndarray, 
                         voxel_size: float =0.01):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        pcd = pcd.voxel_down_sample(voxel_size)
        return np.asarray(pcd.points)
    
    # Placebo Function to call at run-time for Optimizer/Debugger -> Replace at code execution
    def build_dense(self, 
                    sparse_scene: Scene | None = None,
                    camera_poses: CameraPose | None = None) -> Scene:
        if sparse_scene is not None:
            assert isinstance(sparse_scene, Scene), "Incorrect Parameterization. Ensure sparse_scene parameter is a reconstructed Scene typing from SceneReconstruction Module."
        if camera_poses is not None:
            assert isinstance(camera_poses, CameraPose), "Incorrect Parameterization. Ensure camera_poses parameter is a CameraPose type from the Pose Estimation Module."
        return sparse_scene

class CameraPoseEstimatorClass(PipelineModule, ABC):
    use_base_metrics = True
    _registered_metric_methods: tuple[str, ...] = ()
    output_key = "camera_poses"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        metric_names = []

        # inherit metrics from parent classes
        for base in reversed(cls.__mro__[1:]):
            metric_names.extend(getattr(base, "_registered_metric_methods", ()))

        # add metrics declared directly in this subclass
        for name, value in cls.__dict__.items():
            if callable(value) and getattr(value, "_is_metric_provider", False):
                metric_names.append(name)

        # remove duplicates, preserve order
        cls._registered_metric_methods = tuple(dict.fromkeys(metric_names))

    def run_from_state(self, state: SceneState) -> CameraPose:
        correspondences = state.feature_pairs #state.tracked_features or state.feature_pairs
        # return self(correspondences)
        try: 
            return self(correspondences)
        except Exception as e:
            raise RuntimeError(
                "[CameraPoseEstimation Error]\n"
                "Camera pose estimation failed because there were not enough reliable "
                "pairwise feature correspondences to estimate camera motion.\n\n"
                "Likely cause:\n"
                "- The pairwise matcher did not produce enough valid feature matches.\n"
                "- Too few features were detected in one or more image pairs.\n"
                "- Too many matches were removed during outlier rejection.\n"
                "- The detector and matcher combination is not strong enough for this image set.\n\n"
                "Action needed:\n"
                "- Change the feature detector or feature matcher.\n"
                "- Use a detector that produces more reliable keypoints across image pairs.\n"
                "- Use a matcher that produces more consistent pairwise correspondences.\n"
                "- Increase the number of detected keypoints if the detector supports it.\n"
                "- Check pairwise matching metrics before camera pose estimation.\n"
                "- Camera pose estimation should only run when enough valid pairwise matches "
                "exist to estimate relative pose between images.\n\n"
                f"Original error: {type(e).__name__}: {e}"
            ) from e
    
    def __init__(self, 
                 cam_data: CameraData,
                 module_name: str,
                 description: str,
                 example: str):
        
        # Define Module Name, Description, etc. per Sub-Module
        self.module_name = module_name
        self.description = description
        self.example = example

        # Shared Camera data
        self.cam_data = cam_data
        self.image_list = copy.copy(cam_data.image_list)
        self.K_mat = cam_data.get_K()
        self.dist = cam_data.get_distortion()

        # Output container
        self.camera_poses: CameraPose = CameraPose()

        # Metric state
        self._residual_means: list[float] = []
        self._residual_medians: list[float] = []
        
    def __call__(self, 
                 feature_pairs: PointsMatched | None = None) -> CameraPose:
        # poses = CameraPose() # Empty Data Condainer 

        # poses = self._estimate_camera_poses(camera_poses=poses,
        #                                     feature_pairs=feature_pairs)
        self._estimate_camera_poses(feature_pairs=feature_pairs)
        self.calculate_metrics()
        
        return self.camera_poses
    
    @abstractmethod
    def _estimate_camera_poses(self, feature_pairs: PointsMatched | None) -> None:
        # Input Custom Pose Estimation Algorithm in this Function
        raise NotImplementedError

    def calculate_metrics(self) -> None:
        event_msg = self._collect_metrics()
        print(json.dumps(event_msg), flush=True)

        with open(self.cam_data.metric_file_path, "a", encoding="utf-8") as file:
            file.write("============================================================\n")
            file.write("=====================CameraPose Metrics=====================\n")
            json.dump(event_msg, file, indent=4)
            file.write("\n============================================================\n")

    def _collect_metrics(self) -> dict:
        metrics = {}

        if self.use_base_metrics:
            metrics.update(self._base_metric_calculation())

        for method_name in self._registered_metric_methods:
            method = getattr(self, method_name)
            result = method()

            if result is None:
                continue
            if not isinstance(result, dict):
                raise TypeError(
                    f"Metric method '{method_name}' must return dict | None, "
                    f"got {type(result).__name__}"
                )

            overlap = set(metrics).intersection(result)
            if overlap:
                raise ValueError(
                    f"Duplicate metric keys from '{method_name}': {sorted(overlap)}"
                )

            metrics.update(result)

        return metrics

    def _base_metric_calculation(self) -> dict:
        num_poses = len(self.camera_poses.camera_pose)

        if len(self._residual_means) == 0:
            return {
                "Number of Camera Poses": int(num_poses),
                "Average Reprojection Error per Frame": None,
                "Average Median Reprojection Error per Frame": None,
            }

        residual_error = float(np.mean(np.asarray(self._residual_means)))
        residual_median = float(np.median(np.asarray(self._residual_medians)))

        return {
            "Number of Camera Poses": int(num_poses),
            "Average Reprojection Error per Frame": residual_error,
            "Average Median Reprojection Error per Frame": residual_median,
        }


    def _record_residual_metric(
        self,
        object_points: np.ndarray | None,
        image_points: np.ndarray | None,
        pose: np.ndarray | None,
    ) -> None:
        if object_points is None or image_points is None or pose is None:
            return

        if len(object_points) == 0 or len(image_points) == 0:
            return

        mean_error, median_error = self._metric_calculation_residuals(
            object_points=object_points,
            image_points=image_points,
            pose=pose,
        )
        self._residual_means.append(float(mean_error))
        self._residual_medians.append(float(median_error))

    def _metric_calculation_residuals(self, 
                                      object_points: np.ndarray, 
                                      image_points: np.ndarray,
                                      pose: np.ndarray):
        R = pose[:, :3]
        T = pose[:, 3:]
        # proj, _ = cv2.projectPoints(object_points, R, T, self.K_mat, self.dist)
        rvec, _ = cv2.Rodrigues(R)
        proj, _ = cv2.projectPoints(object_points, rvec, T, self.K_mat, self.dist)

        residual_error = np.linalg.norm(image_points - proj.reshape(-1,2), axis=1)
        error = float(np.mean(residual_error))
        median_error = float(np.median(residual_error))

        return error, median_error

class FeatureClass(PipelineModule, ABC):
    output_key = "features"

    def run_from_state(self, state: SceneState) -> list[Points2D]:
        # return self()
        try:
            return self()
        except Exception as e:
            raise RuntimeError(
                "[FeatureDetection Error]\n"
                "Feature detection failed. This is usually caused by one of the following:\n"
                "1. The image path is incorrect, empty, or does not contain readable images.\n"
                "2. The feature detector module name was specified incorrectly.\n"
                "3. The selected detector is not supported or was not registered correctly.\n\n"
                "Suggested fixes:\n"
                "- Verify that the image directory exists and contains valid image files.\n"
                "- Check that the feature detector name matches one of the supported module names.\n"
                "- Use the correct registered feature detection module, such as SIFT, SuperPoint, or another implemented detector.\n\n"
                f"Original error: {type(e).__name__}: {e}"
            ) from e
    
    def __init__(self, 
                 cam_data: CameraData,
                 module_name: str,
                 description: str,
                 example: str):

        # Define Module Name, Description, etc. 
        # Under modules.
        self.module_name = module_name
        self.description = description
        self.example = example
        
        # Define Camera Data variables to use. 
        self.cam_data = cam_data
        self.image_list = cam_data.image_list
        self.image_scale = cam_data.image_scale
        self.image_shape = cam_data.image_list[0].shape[:2]
        
        # Personal Variables of Feature Module
        self.features: list[Points2D] = []


    def __call__(self) -> list[Points2D]:

        # Detect Features from Designated Detector
        self.features = self._detect_features()

        # Output Metric
        self.calculate_metrics()

        return self.features
    
    @abstractmethod
    def _detect_features(self) -> list[Points2D]:
        # Write Code Here to Fill Feature Module per Detector Implemented
        raise NotImplementedError

    def calculate_metrics(self) -> None:
        # Output Metric
        mean_ct, min_count, max_count = self._metric_calculation()
        event_msg_feats = {"Average": mean_ct, "Minimum": min_count, "Maximum": max_count}
        print(json.dumps(event_msg_feats), flush=True)
        mean_ct, min_count, max_count = self._spatial_dist_calc()
        event_msg_coverage = {"Average": mean_ct, "Minimum": min_count, "Maximum": max_count}
        print(json.dumps(event_msg_coverage), flush=True)

        # Write to file
        with open(self.cam_data.metric_file_path, "a", encoding="utf-8") as file:
            file.write("============================================================\n")
            file.write("======================Features Metrics======================\n")
            json.dump({"Features Detected per Frame": event_msg_feats, "Feature Spatial Distribution per Frame": event_msg_coverage}, file, indent=4)
            file.write("\n============================================================\n")
        # with open('data.json', 'w', encoding='utf-8') as f:
        #     json.dump({"Features Detected per Frame": event_msg_feats, "Feature Spatial Distribution per Frame": event_msg_coverage}, f, indent = 4)

    def _spatial_dist_calc(self):
        set_of_coverages = []
        for pts_2D in self.features:
            pts = pts_2D.points2D

            if pts.shape[0] < 2:
                set_of_coverages.append(0)
                continue

            # print("DUPE CHECK: ", len(pts) - len(np.unique(pts, axis=0)))
            # Remove duplicate points
            pts_non_dupe = np.unique(pts, axis=0)

            tree = cKDTree(pts_non_dupe)

            # k=2 because first neighbor is itself
            dists, _ = tree.query(pts_non_dupe, k=2)

            nearest = dists[:,1]

            # Clamp points to avoid 0 distance error.
            nearest = np.maximum(nearest, 1e-6)

            coverage = nearest.shape[0] / np.sum(1.0 / nearest)

            set_of_coverages.append(coverage)

        coverages_np = np.array(set_of_coverages)


        mean_ct = coverages_np.mean()
        min_count = coverages_np.min()
        max_count = coverages_np.max()

        return float(mean_ct), float(min_count), float(max_count)

    def _metric_calculation(self):
        set_of_pt_counts = []

        for pts_2D in self.features:
            num_pts = pts_2D.points2D.shape[0]
            set_of_pt_counts.append(num_pts)
        
        counts_np = np.array(set_of_pt_counts)

        mean_ct = counts_np.mean()
        min_count = counts_np.min()
        max_count = counts_np.max()

        return float(mean_ct), int(min_count), int(max_count)

class FeatureMatching(PipelineModule, ABC):
    output_key = "feature_pairs"

    def run_from_state(self, state: SceneState) -> PointsMatched:
        if state.features is None:
            raise RuntimeError("FeatureMatching requires features. Run a FeatureClass module first.")
        try:
            return self(state.features)
        except Exception as e:
            threshold = self.ransac_threshold #getattr(self, "RANSAC_threshold", None)

            if threshold is not None and threshold >= 3.0:
                fix_hint = (
                    "The outlier rejection threshold is already at or above 3.0, so the "
                    "failure is likely caused by too few reliable feature points being detected. "
                    "Switch to a stronger/more dense feature detector or increase the number of "
                    "detected keypoints before feature matching."
                )
            else:
                fix_hint = (
                    "The outlier rejection threshold may be too strict. Try loosening the "
                    "RANSAC/outlier rejection threshold. If the threshold reaches 3.0 and the "
                    "module still fails, switch feature detectors because not enough reliable "
                    "points were detected."
                )

            raise RuntimeError(
                "[FeatureMatching Error]\n"
                "Feature matching failed.\n\n"
                "Likely causes:\n"
                "- Not enough feature points were detected to produce enough matches.\n"
                "- Not enough matches survived outlier rejection.\n"
                "- The outlier rejection threshold is too strict.\n"
                "- The selected feature detector may not be producing enough reliable keypoints.\n\n"
                "Suggested fixes:\n"
                f"- {fix_hint}\n"
                "- Improve parameterization to increase the number of detected keypoints if the detector supports it.\n"
                "- Use a more robust detector/descriptor combination for the image set (If SuperPoint is used, think about Detector Free approach).\n"
                "- Final try, swap to a detector free approach, such as using VGGT directly for Pose/Reconstruction.\n\n"
                f"Current outlier rejection threshold: {threshold}\n"
                f"Original error: {type(e).__name__}: {e}"
            ) from e
    
    def __init__(self, 
                 cam_data:CameraData,
                 module_name: str,
                 description: str,
                 example: str,
                 RANSAC_threshold: float,
                 RANSAC_conf: float,
                 RANSAC_homography: bool = False):
        
        # Define Module Name, Description, etc. 
        # Under modules.
        self.module_name = module_name
        self.description = description
        self.example = example

        # Set up Calibration and Image Data
        self.cam_data = cam_data
        self.K = cam_data.get_K()
        self.dist = cam_data.get_distortion()

        # Setup Outlier Rejection
        self.normalize = Normalization(K=self.K, 
                                       dist=self.dist,
                                       multi_cam=self.cam_data.multi_cam)
        self.homography = RANSAC_homography
        self.ransac_threshold = RANSAC_threshold
        self.ransac_conf = RANSAC_conf
    
    def __call__(self, features: list[Points2D]) -> PointsMatched:
        
        matched_points = self.find_correspondences(features) 

        self.calculate_metrics(matched_points) 

        return matched_points
    
    @abstractmethod
    def find_correspondences(self, features: list[Points2D]) -> PointsMatched:
        """Override for custom matching algorithm in here"""
        raise NotImplementedError
        # matched_points = PointsMatched() 

        # return matched_points
    
    def outlier_reject(self, 
                       pts1: Points2D, 
                       pts2: Points2D, 
                       idx1: list,
                       idx2: list,
                       frame_id: int) -> tuple[Points2D, 
                                               Points2D,
                                               np.ndarray,
                                               np.ndarray,
                                               np.ndarray]: # Move to Base Class
        
        # pts1_norm = self.normalize(pts1, frame_id)
        # pts2_norm = self.normalize(pts2, frame_id+1)

        pts1_t = pts1.points2D
        pts2_t = pts2.points2D
        # print("SHAPE", pts2_t.shape)
        if self.homography:
            # F, mask = cv2.findFundamentalMat(pts1_norm, pts2_norm, cv2.FM_RANSAC, 
            #                                  ransacReprojThreshold=self.ransac_threshold, 
            #                                  confidence=self.ransac_conf)
            F, mask = cv2.findHomography(pts1_t, pts2_t, #pts1_norm, pts2_norm, 
                                         cv2.USAC_MAGSAC, 
                                         ransacReprojThreshold=self.ransac_threshold,
                                         maxIters=10000,
                                         confidence=self.ransac_conf)
        else:
            # F, mask = cv2.findFundamentalMat(pts1_norm, pts2_norm, cv2.FM_LMEDS)
            F, mask = cv2.findFundamentalMat(pts1_t, pts2_t, #pts1_norm, pts2_norm, 
                                             cv2.USAC_MAGSAC, 
                                             ransacReprojThreshold=self.ransac_threshold,
                                             maxIters=10000,
                                             confidence=self.ransac_conf)

        # Could update points2D to inlier points with Mask
        inlier_pts1 = Points2D(**pts1.set_inliers(mask))
        inlier_pts2 = Points2D(**pts2.set_inliers(mask))
        idx1_inliers = np.array(idx1)[mask.ravel() == 1]
        idx2_inliers = np.array(idx2)[mask.ravel() == 1]

        return inlier_pts1, inlier_pts2, idx1_inliers, idx2_inliers, F

    def calculate_metrics(self, matching_points: PointsMatched, sigma_th: int = 3):
        # outlier_count = self._z_score(matched_points=matching_points.pairwise_matches, sigma_th=sigma_th)
        repeatability = self._calc_repeatability(matching_points=matching_points, epsilon=3.5) # Since we don't have ground truth, we assume 1px of noise.
        mean_ct, inlier_yield_avg, inlier_yield_median = self._calculate_proxy_matching_score(matched_points=matching_points.pairwise_matches, features = matching_points.img_features)
        gric_score_F, gric_score_H = self.evaluate_models(matching_points=matching_points)

        event_msg = {"Average Corresponding Features": mean_ct, "Average Inlier Yield per Frame": inlier_yield_avg,
                     "Median Inlier Yield per Frame": inlier_yield_median,
                     "Average Repeatability per Image Pair": repeatability, "Gric Score - Fundamental": gric_score_F, 
                     "Gric Score - Homography": gric_score_H}
        print(json.dumps(event_msg), flush=True)

        # Write to file
        with open(self.cam_data.metric_file_path, "a", encoding="utf-8") as file:
            file.write("============================================================\n")
            file.write("==================Feature Matching Metrics==================\n")
            json.dump({"Corresponding Feature Metrics per Image Pair": event_msg}, file, indent=4)
            file.write("\n============================================================\n")
        
        # with open('data.json', 'w', encoding='utf-8') as f:
        #     json.dump({"Corresponding Feature Metrics per Image Pair": event_msg}, f, indent = 4)

        # return mean_ct, inlier_yield, repeatability, gric_score_F, gric_score_H
    
    def _calculate_proxy_matching_score(self, matched_points: list[np.ndarray], features: list[np.ndarray]):
        set_of_pt_counts = np.zeros((len(matched_points), 1))
        inlier_yields = np.zeros((len(features), 1))

        for i in range(len(matched_points)):
            matching_points = matched_points[i]
            num_pts = matching_points.shape[0]
            set_of_pt_counts[i] = num_pts # Get total correspondences

            # Get inlier Yield
            feats1 = features[i]
            feats2 = features[i + 1]
            inlier_yields[i] = num_pts/min(feats1.shape[0], feats2.shape[0])
            # inlier_yields[i + 1] = num_pts/feats2.shape[0]
        
        # counts_np = np.array(set_of_pt_counts)

        mean_ct = set_of_pt_counts.mean()
        inlier_yield_avg = inlier_yields.mean()
        inlier_yield_median = np.median(inlier_yields)
        # min_count = counts_np.min()
        # max_count = counts_np.max()

        return float(mean_ct), float(inlier_yield_avg), float(inlier_yield_median) #int(min_count), int(max_count)

    def _warp_points(self, src: np.ndarray, mat: np.ndarray, img_shape: list):
        KA_prime = cv2.perspectiveTransform(src, mat)
        KA_prime = KA_prime.reshape(-1, 2)
        W, H = img_shape[:]
        # Filter Warped Points
        KA_prime = KA_prime[KA_prime[:, 0] >= 0]
        KA_prime = KA_prime[KA_prime[:, 1] >= 0]
        KA_prime = KA_prime[KA_prime[:, 0] < W]
        KA_prime = KA_prime[KA_prime[:, 1] < H]

        return KA_prime
           

    def _calc_repeatability(self, matching_points: PointsMatched, epsilon: float = 3.0):
        matched_points = matching_points.pairwise_matches
        features = matching_points.img_features
        W, H = matching_points.image_size[:]

        repeatabilities = []
        for pair_idx in range(len(matched_points)):
            pt_set = matched_points[pair_idx]

            if pt_set.shape[0] < 4:
                repeatabilities.append(0.0)
                continue

            pts_A = pt_set[:, :2].astype(np.float32)
            pts_B = pt_set[:, 2:].astype(np.float32)

            H_mat, inlier_mask = cv2.findHomography(
                pts_A,
                pts_B,
                cv2.RANSAC,
                5.0
            )

            if H_mat is None:
                repeatabilities.append(0.0)
                continue

            try:
                H_inv = np.linalg.inv(H_mat)
            except np.linalg.LinAlgError:
                repeatabilities.append(0.0)
                continue

            KA = features[pair_idx].astype(np.float32)
            KB = features[pair_idx + 1].astype(np.float32)

            if len(KA) == 0 or len(KB) == 0:
                repeatabilities.append(0.0)
                continue

            # A features projected into B
            KA_to_B = self._warp_points(
                KA.reshape(-1, 1, 2),
                H_mat,
                [W, H]
            )

            # B features projected into A, only for visibility count
            KB_to_A = self._warp_points(
                KB.reshape(-1, 1, 2),
                H_inv,
                [W, H]
            )

            if len(KA_to_B) == 0 or len(KB_to_A) == 0:
                repeatabilities.append(0.0)
                continue

            # Compare warped A features against actual B features
            tree = cKDTree(KB)

            dists, _ = tree.query(KA_to_B, k=1)

            repeated = np.sum(dists <= epsilon)

            denom = min(len(KA_to_B), len(KB_to_A))

            repeatability = repeated / denom if denom > 0 else 0.0
            repeatabilities.append(float(repeatability))
        # for pair in range(len(matched_points)):
        #     pt_set = matched_points[pair]
        #     # print(pt_set)
        #     pt_set_A = pt_set[:, :2]
        #     pt_set_B = pt_set[:, 2:]

        #     H_mat, _ = cv2.findHomography(pt_set_A, pt_set_B, cv2.RANSAC, 5.0)

        #     KA = features[pair]
        #     KB = features[pair + 1]

        #     # print("BEFORE", KB.shape)
        #     KA_prime = self._warp_points(KA.reshape(-1, 1, 2), H_mat, [W, H])
        #     KB_prime = self._warp_points(KB.reshape(-1, 1, 2), np.linalg.inv(H_mat),[W, H])
        #     KB = self._warp_points(KB_prime.reshape(-1, 1, 2), H_mat, [W, H])
        #     # print("AFTER", KB.shape)
        #     tree = cKDTree(KB)

        #     # k=2 because first neighbor is itself
        #     dists, _ = tree.query(KA_prime, k=1)

        #     nearest = dists[:,1]

        #     repeated = nearest[nearest <= epsilon].shape[0]
        #     # print(repeated)
        #     repeatability = repeated / min(len(KA_prime), len(KB))

        #     repeatabilities.append(repeatability)


        return float(np.array(repeatabilities).mean())

    def fundamental_error(self, pts1, pts2, F):
        """
        Sampson error for fundamental matrix residuals
        """
        pts1_h = np.hstack([pts1, np.ones((pts1.shape[0],1))])
        pts2_h = np.hstack([pts2, np.ones((pts2.shape[0],1))])

        Fx1 = F @ pts1_h.T
        Ftx2 = F.T @ pts2_h.T

        denom = Fx1[0]**2 + Fx1[1]**2 + Ftx2[0]**2 + Ftx2[1]**2
        err = np.sum(pts2_h.T * (F @ pts1_h.T), axis=0)

        return (err**2) / denom


    def homography_error(self, pts1, pts2, H):
        """
        symmetric transfer error for homography
        """
        pts1_h = np.hstack([pts1, np.ones((pts1.shape[0],1))])
        pts2_h = np.hstack([pts2, np.ones((pts2.shape[0],1))])

        proj = (H @ pts1_h.T).T
        proj = proj[:,:2] / proj[:,2:]

        err = np.linalg.norm(proj - pts2, axis=1)**2
        return err


    def compute_gric(self, errors, sigma, model_dim, param_dim, r: int = 4, lambda3: int = 2):
        """
        GRIC computation
        Fundamental: model_dim = 3, param_dim = 7
        Homography: model_dim = 2, param_dim = 8
        """
        n = len(errors)

        lambda1 = np.log(r)
        lambda2 = np.log(r*n)
        
        # robust truncation
        errors = np.minimum(errors / (sigma**2), lambda3*(r - model_dim))

        gric = np.sum(errors) + lambda1 * model_dim * n + lambda2 * param_dim
        return gric


    def evaluate_models(self, matching_points: PointsMatched, sigma: float = 1.0):
        """
        Compare homography vs fundamental matrix using GRIC
        """

        matched_points = matching_points.pairwise_matches
        features = matching_points.img_features
        W, H = matching_points.image_size[:]

        gric_scores = {'fundamental':[], 'homography':[]}
        for pair in range(len(matched_points)):
            pt_set = matched_points[pair]
            pts1 = pt_set[:, :2]
            pts2 = pt_set[:, 2:]

            # estimate models
            H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)
            F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

            # compute residuals
            h_errors = self.homography_error(pts1, pts2, H)
            f_errors = self.fundamental_error(pts1, pts2, F)

            # model parameters
            # homography: 8 parameters, dimension=2
            # fundamental: 7 parameters, dimension=3
            gric_H = self.compute_gric(h_errors, sigma, model_dim=2, param_dim=8)
            gric_F = self.compute_gric(f_errors, sigma, model_dim=3, param_dim=7)

            gric_scores['fundamental'].append(gric_F)
            gric_scores['homography'].append(gric_H)

        return np.array(gric_scores['fundamental']).mean(), np.array(gric_scores['homography']).mean()
        # if gric_H < gric_F:
        #     model = "homography"
        # else:
        #     model = "fundamental"

        # return {
        #     "GRIC_H": gric_H,
        #     "GRIC_F": gric_F,
        #     "best_model": model
        # }

class FeatureTracking(PipelineModule, ABC):
    output_key = "tracked_features"

    def run_from_state(self, state: SceneState) -> PointsMatched:
        if state.features is None:
            raise RuntimeError("FeatureTracking requires features. Run a FeatureClass module first.")
        # return self(state.features)
        try:
            return self(state.features)
        except Exception as e:
            # threshold = getattr(self, "RANSAC_threshold", None)
            threshold = self.RANSAC_threshold 

            if threshold is not None and threshold >= 3.0:
                fix_hint = (
                    "The outlier rejection threshold is already at or above 3.0, so the "
                    "failure is likely caused by too few reliable feature points being detected. "
                    "Switch to a stronger/more dense feature detector or increase the number of "
                    "detected keypoints before feature matching."
                )
            else:
                fix_hint = (
                    "The outlier rejection threshold may be too strict. Try loosening the "
                    "RANSAC/outlier rejection threshold. If the threshold reaches 3.0 and the "
                    "module still fails, switch feature detectors because not enough reliable "
                    "points were detected."
                )

            raise RuntimeError(
                "[FeatureTracking Error]\n"
                "Feature tracking failed.\n\n"
                "Likely causes:\n"
                "- Not enough feature points were detected to produce valid matches/tracks.\n"
                "- Not enough matches survived outlier rejection.\n"
                "- The outlier rejection threshold is too strict.\n"
                "- The selected feature detector may not be producing enough reliable keypoints.\n\n"
                "Suggested fixes:\n"
                f"- {fix_hint}\n"
                "- Increase the maximum number of detected keypoints if the detector supports it.\n"
                "- Use a more robust feature matcher module for the image set.\n"
                "- Use a more robust detector/descriptor combination for the image set.\n"
                "- Check that image overlap is sufficient between matched frames.\n"
                "- Final try, swap to a detector free approach, such as using VGGT directly for Pose/Reconstruction.\n\n"
                f"Current outlier rejection threshold: {threshold}\n"
                f"Original error: {type(e).__name__}: {e}"
            ) from e
    
    def __init__(self, detector:str, 
                 cam_data:CameraData,
                 module_name: str,
                 description: str,
                 example: str,
                 RANSAC_threshold: float,
                 RANSAC_conf: float,
                 RANSAC_homography: bool = False):
        
        # Define Module Name, Description, etc. per Sub-Module
        self.module_name = module_name
        self.description = description
        self.example = example
        self.RANSAC_threshold = RANSAC_threshold
        self.detector = detector
        self.det_free = False

        self.cam_data = cam_data
        self.K = cam_data.get_K()
        self.dist = cam_data.get_distortion()

        self.DETECTORS = ["sift", "superpoint", "orb"]

        if self.detector not in self.DETECTORS:
            self.det_free = True

        # Set normalization function
        normalization = Normalization(K=self.K,
                                           dist=self.dist,
                                           multi_cam=cam_data.multi_cam)
        
        # Fixed Algorithm to Track Features
        self.feature_tracker = FeatureTracker(matcher_parser = self.matcher_parser, 
                                              normalization = normalization,
                                              RANSAC_threshold = RANSAC_threshold,
                                              RANSAC_conf = RANSAC_conf,
                                              homography = RANSAC_homography)

    def __call__(self, features: list[Points2D]) -> PointsMatched: # Fixed to the Module
        
        # Points Matched for Tracking -> data = N x [track_id, frame_num, x, y]

        matched_points = self.feature_tracker.match_full(features)

        self.calculate_metrics(data_mat=matched_points.data_matrix, total_points=matched_points.point_count)

        return matched_points
    
    @abstractmethod
    def matcher_parser(self, pt1: Points2D, pt2: Points2D) -> tuple[list, list]:
        # To Fill per Module Basis
        # return [], [] 
        raise NotImplementedError
    
    # Metric Function
    def calculate_metrics(self, data_mat: np.ndarray, total_points: int):

        track_ids = data_mat[:, 0].astype(int)

        unique_ids, counts = np.unique(track_ids, return_counts=True)

        # Track Length Average, Median, and Maximum
        max_track = counts.max()
        median_track_length = np.median(counts)
        avg_track = counts.mean()

        # Survival Curve Metric (Num of Tracks lasting N frames)
        survive_ge_3  = np.mean(counts >= 3) # Multi-View Suppert rate 3
        survive_ge_5  = np.mean(counts >= 5) # Multi-View Suppert rate 5
        survive_ge_10 = np.mean(counts >= 10) # Multi-View Suppert rate 10
       
        # Lower fragmentation and higher observations-per-track are usually better.
        fragmentation = len(counts) / np.sum(counts) # tracks per observation
        obs_per_track = np.sum(counts) / len(counts) # observation per track


        event_msg = {"Avg. track length": float(avg_track), 
                     "Max. track length": float(max_track), 
                     "Median track length": float(median_track_length), 
                     "Survival Tracks of 3": float(survive_ge_3), 
                     "Survival Tracks of 5": float(survive_ge_5), 
                     "Survival Tracks of 10": float(survive_ge_10),
                     "Fragmentation": float(fragmentation),
                     "Obs. per Track": float(obs_per_track)}

        print("HERE")
        print(json.dumps(event_msg), flush=True)

        # Write to file
        # Write to file
        with open(self.cam_data.metric_file_path, "a", encoding="utf-8") as file:
            file.write("============================================================\n")
            file.write("==================Feature Tracking Metrics==================\n")
            json.dump({"Feature Track Metrics for Survivability and Stability": event_msg}, file, indent=4)
            file.write("\n============================================================\n")
        # with open('data.json', 'w', encoding='utf-8') as f:
        #     json.dump({"Feature Track Metrics for Survivability and Stability": event_msg}, f, indent = 4)

class OptimizationClass(PipelineModule, ABC):
    use_base_metrics = True
    use_no_metrics = False
    _registered_metric_methods: tuple[str, ...] = ()
    output_key = "optimized_scene"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        metric_names = []

        # inherit metrics from parent classes
        for base in reversed(cls.__mro__[1:]):
            metric_names.extend(getattr(base, "_registered_metric_methods", ()))

        # add metrics declared directly in this subclass
        for name, value in cls.__dict__.items():
            if callable(value) and getattr(value, "_is_metric_provider", False):
                metric_names.append(name)

        # remove duplicates, preserve order
        cls._registered_metric_methods = tuple(dict.fromkeys(metric_names))

    def run_from_state(self, state: SceneState, **kwargs) -> Scene | IncrementalSfMState:
        current_scene = state.dense_scene or state.sparse_scene
        if current_scene is None:
            raise RuntimeError("Optimization requires a sparse or dense scene.")
        # return self(current_scene, **kwargs)
        try: 
            return self(current_scene, **kwargs)
        except Exception as e:
            points3d = getattr(state.sparse_scene, "points3D", None)
            num_points3d = points3d.points3D.shape[0]

            raise RuntimeError(
                "[Optimization Error]\n"
                "Bundle adjustment / optimization failed because the reconstruction contains "
                "too few valid 3D points to optimize reliably.\n\n"
                "Likely cause:\n"
                "- The sparse reconstruction produced a very low number of 3D points.\n"
                "- If fewer than 10 3D points were reconstructed, the optimization problem is "
                "likely too under-constrained to solve reliably.\n"
                "- The detector, matcher, tracker, camera pose, or sparse reconstruction stages "
                "did not produce enough valid geometry before optimization.\n\n"
                "Action needed:\n"
                "- Improve the detector and tracker/matcher combination so more stable 3D points "
                "are created before optimization.\n"
                "- Check feature tracking metrics, especially track length and number of tracks "
                "observed in at least 3 views.\n"
                "- Check sparse reconstruction metrics, especially Number Points3D.\n"
                "- If Number Points3D is below 10, do not tune the optimizer first. Instead, "
                "revisit the parameterization of previous modules.\n"
                "- Consider increasing detected keypoints, changing the detector, changing the "
                "matcher, loosening outlier rejection, improving track generation, or adjusting "
                "the sparse reconstruction minimum observation settings.\n\n"
                f"Current Number Points3D: {num_points3d}\n"
                f"Original error: {type(e).__name__}: {e}"
            ) from e
    
    def __init__(self, 
                 cam_data: CameraData,
                 module_name: str,
                 description: str,
                 example: str,
                #  refine_focal_length: bool = False,
                #  refine_principal_point: bool = False,
                #  refine_extra_params: bool = False,
                #  max_num_iterations: int = 50,
                #  use_gpu: bool = True,
                #  gpu_index: int = 0,
                #  robust_loss: bool = True,
                 ):
        
        # Define Module Name, Description, etc. per Sub-Module
        self.module_name = module_name
        self.description = description
        self.example = example

        # Set up Camera Data/Image Resolution
        self.K = cam_data.get_K()
        self.dist = cam_data.get_distortion()
        self.cam_data = cam_data
        self.H, self.W = cam_data.image_list[0].shape[:2] 
        self.multi_cam = cam_data.multi_cam

        # Set up Bundle Adjustment Params
        # self.refine_focal_length = refine_focal_length
        # self.refine_principal_point = refine_principal_point
        # self.refine_extra_params = refine_extra_params
        # self.max_num_iterations = max_num_iterations
        # self.use_gpu = use_gpu
        # self.gpu_index = gpu_index
        # self.robust_loss = robust_loss

        # self.optimizer = ["BA"]
        # self.dataset = scene.bal_data.dataset
        
        # self.OPTIONS = ['essential', 'fundamental', 'homography', 'projective']
        # self.FORMATS = ['full', 'partial', 'pair']

        # self.format = format
        
    def __call__(self, current_scene: Scene, **kwargs) -> Scene:
        """Fixed Function Call specifically for global bundle adjustment pipelines"""
        optimized_scene = self._optimize_scene(current_scene, **kwargs)
        self.calculate_metrics(optimized_scene)
        return optimized_scene
    

    @abstractmethod
    def _optimize_scene(self, current_scene: Scene | IncrementalSfMState) -> Scene | IncrementalSfMState:
        """Write optimizer-specific implementation here."""
        raise NotImplementedError

    def calculate_metrics(self, current_scene: Scene) -> None:
        if self.use_no_metrics:
            return
        
        event_msg = self._collect_metrics(current_scene)
        print(json.dumps(event_msg), flush=True)

        with open(self.cam_data.metric_file_path, "a", encoding="utf-8") as file:
            file.write("============================================================\n")
            file.write("====================Optimization Metrics====================\n")
            json.dump({"Scene Metrics from Optimization": event_msg}, file, indent=4)
            file.write("\n============================================================\n")

    def _collect_metrics(self, current_scene: Scene) -> dict:
        metrics = {}

        if self.use_base_metrics:
            metrics.update(self._base_metric_calculation(current_scene))

        for method_name in self._registered_metric_methods:
            method = getattr(self, method_name)
            result = method()

            if result is None:
                continue
            if not isinstance(result, dict):
                raise TypeError(
                    f"Metric method '{method_name}' must return dict | None, "
                    f"got {type(result).__name__}"
                )

            overlap = set(metrics).intersection(result)
            if overlap:
                raise ValueError(
                    f"Duplicate metric keys from '{method_name}': {sorted(overlap)}"
                )

            metrics.update(result)

        return metrics

    def _base_metric_calculation(self) -> dict:
        return 

    # def calculate_metrics(self, current_scene: Scene | IncrementalSfMState) -> None:
    #     metrics = self._metric_calculation(current_scene)
    #     print(json.dumps(metrics), flush=True)

    #     with open(self.cam_data.metric_file_path, "a", encoding="utf-8") as file:
    #         file.write("============================================================\n")
    #         file.write("====================Optimization Metrics====================\n")
    #         json.dump({"Scene Metrics from Optimization": metrics}, file, indent=4)
    #         file.write("\n============================================================\n")

    def _base_metric_calculation(self, current_scene: Scene | IncrementalSfMState) -> dict:
        metrics = {"Optimizer": self.module_name}

        if hasattr(current_scene, "points3D") and current_scene.points3D is not None:
            pts3d = getattr(current_scene.points3D, "points3D", None)
            if pts3d is not None:
                metrics["Num Points3D"] = int(len(pts3d))

        if hasattr(current_scene, "camera_poses") and current_scene.camera_poses is not None:
            poses = getattr(current_scene.cam_poses, "camera_pose", None)
            if poses is not None:
                metrics["Num Camera Poses"] = int(len(poses))

        return metrics

    # def optimize(self, current_scene: Scene | IncrementalSfMState):
    #     """Edit Function Call for optimizer library and BA style (Global or Local)"""
    #     pass
    
    # def _build_reconstruction(self, 
    #                           current_scene: Scene | IncrementalSfMState):
    #     """Edit Function Call specifically to set up state or optimization solver"""
    #     pass
 
class VisualizeClass():
    def __init__(self, path:str):
        self.module_name = "..."
        self.description = "..."
        self.example = "..."

        self.path = path

    def __call__(self, data: Scene | np.ndarray, store:bool = False) -> None:
        pass


############################################ Orchestrator ############################################
class SfMScene:
    def __init__(
        self,
        id,
        log_dir,
        image_path: str | None = None,
        calibration_path: str | None = None,
        cam_data: CameraData | None = None,
        max_images: int | None = None,
        target_resolution: Tuple[int, int] | None = None,
    ):
        self.id = id
        self.log_dir = log_dir
        #Colmap Workspace
        colmap_dir = log_dir + f"/{id}/workspace"
        if cam_data is None:
            # if image_path is None or calibration_path is None:
                # raise ValueError(
                    # "Provide either cam_data or both image_path and calibration_path."
                # )
            if max_images is None:
                CDM = CameraDataManager(image_path=image_path,
                            calibration_path=calibration_path,
                            target_resolution=target_resolution,
                            colmap_workspace=colmap_dir)
            else:
                CDM = CameraDataManager(image_path=image_path,
                            max_images=max_images,
                            calibration_path=calibration_path,
                            target_resolution=target_resolution,
                            colmap_workspace=colmap_dir)

            # Get Camera Data
            cam_data = CDM.get_camera_data()
            
            # parent_metric_path = Path(__file__).resolve().parents[2]
            # metric_file_path = str(parent_metric_path / "results" / f"metrics_results_{id}.txt")
            metric_file_path = self.log_dir + f"/metrics_results_{id}.txt"
            # metric_file_path = "/work/tmp/metric_" + str(self.id) + ".txt"
            # Create file or erase contents of existing one
            with open(metric_file_path, "w") as file:
                pass
            # cam_data = CameraData(
            #     image_path=image_path,
            #     calibration_path=calibration_path,
            # )

        self.cam_data = cam_data
        self.cam_data.metric_file_path = metric_file_path 
        self.cam_data.logging_dir = self.log_dir
        self.cam_data.script_id = id
        self.state = SceneState(cam_data=cam_data)

    def __getattr__(self, name: str):
        module_cls = PipelineModule.REGISTRY.get(name)
        if module_cls is None:
            raise AttributeError(f"{type(self).__name__!s} has no module '{name}'")

        def runner(**kwargs):
            if "optimizer" in kwargs:
                kwargs["optimizer"] = self._build_dependency(kwargs["optimizer"])
                
            module = module_cls(cam_data=self.cam_data, **kwargs)
            result = module.run_from_state(self.state)

            output_key = module.output_key
            if output_key is None:
                raise RuntimeError(f"{module_cls.__name__} is missing output_key")

            setattr(self.state, output_key, result)
            self.state.last_output = result
            self.state.history.append(
                {
                    "module": name,
                    "params": kwargs,
                    "output_key": output_key,
                }
            )
            return self  # enables chaining

        return runner

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(PipelineModule.REGISTRY.keys()))
    
    # Helper Function to pass Function Instances in Params of Sub-Modules
    def _build_dependency(self, dep):
        if dep is None:
            return None

        # already-instantiated object
        if not isinstance(dep, tuple):
            return dep

        module_name, kwargs = dep
        module_cls = PipelineModule.REGISTRY[module_name]

        return module_cls(cam_data=self.cam_data, **kwargs)
    
    @property
    def features(self):
        return self.state.features

    @property
    def feature_pairs(self):
        return self.state.feature_pairs

    @property
    def tracked_features(self):
        return self.state.tracked_features

    @property
    def camera_poses(self):
        return self.state.camera_poses

    @property
    def sparse_scene(self):
        return self.state.sparse_scene

    @property
    def dense_scene(self):
        return self.state.dense_scene

    @property
    def optimized_scene(self):
        return self.state.optimized_scene
######################################################################################################