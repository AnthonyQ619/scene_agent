'''
Base Class designs for each module to standardize the class design
for each tool/module.

This is to reduce the possiblility of the Agent to hallucinate code
'''

import numpy as np
from scipy.spatial import cKDTree
import cv2
from modules.DataTypes.datatype import (Scene, 
                                CameraData, 
                                Calibration, 
                                Points2D, 
                                PointsMatched, 
                                CameraPose, 
                                Points3D,
                                BundleAdjustmentData,
                                IncrementalSfMState)
import glob
from collections.abc import Callable
import torch
import open3d as o3d

import copy
import random
import json
from tqdm import tqdm

############################################# HELPER CLASSES #############################################

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
                 K: np.ndarray,
                 dist: np.ndarray,
                 RANSAC_threshold: float,
                 RANSAC_conf: float,
                 cam_data: CameraData):

        # Establish the data structures 
        self.track_map = {}
        self.next_track_id = 0
        self.observations = []

        # Set up Outlier Check Parameters
        self.ransac_threshold = RANSAC_threshold
        self.ransac_conf = RANSAC_conf

        # Set the Matcher
        self.matcher = matcher_parser

        # self.ep_check = EpipoleChecker(pxl_min=25)
        self.normalization = Normalization(K=K,
                                           dist=dist,
                                           multi_cam=cam_data.multi_cam)
    
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
        pts_norm1 = self.normalization(pts1, frame_id)
        pts_norm2 = self.normalization(pts2, frame_id + 1)

        # _, mask = cv2.findFundamentalMat(pts1.points2D, pts2.points2D, cv2.FM_LMEDS)
        # _, mask = cv2.findFundamentalMat(pts_norm1, pts_norm2, cv2.FM_LMEDS)
        F, mask = cv2.findFundamentalMat(pts_norm1, pts_norm2, cv2.FM_RANSAC, 
                                             ransacReprojThreshold=self.ransac_threshold, 
                                             confidence=self.ransac_conf)

        # Could update points2D to inlier points with Mask
        inlier_pts1 = Points2D(**pts1.set_inliers(mask))
        inlier_pts2 = Points2D(**pts2.set_inliers(mask))
        # print(pts_norm1.shape)
        # print(inlier_pts1.points2D.shape)
        # # print(matches)
        # matches_np = np.array(matches)

        # # print(inlier_pts1.points2D.shape)
        # matches_inlier = matches_np[mask.ravel()==1].tolist()
        # # print(len(matches_inlier))
        # return matches_inlier, inlier_pts1, inlier_pts2
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
            outlier_count.append(self._z_score(inlier_pts1.points2D, inlier_pts2.points2D, sigma_th=3))

            # Feature Tracking algorithm here
            self.tracking_points(scene, inlier_pts1, inlier_pts2) #, matches_inlier)

            # matched_points.append([new_pt1, new_pt2])
        
        # Output Metric Information
        counts_np = np.array(matching_pair_ct)
        mean_ct = float(counts_np.mean())
        min_ct = int(counts_np.min())
        max_ct = int(counts_np.max())
        avg_outlier = float(np.mean(np.array(outlier_count)))

        tracked_features.set_matched_matrix(self.observations)
        tracked_features.track_map = self.track_map
        tracked_features.point_count = self.next_track_id - 1

        avg_track, max_track = self._calculate_avg_track_length(data_mat=tracked_features.data_matrix, total_points=tracked_features.point_count)
        event_msg = {"avg track length": avg_track, "max track length": max_track, "avg_outlier": avg_outlier, "avg_feats": mean_ct, "min_feats": min_ct, "max_feats": max_ct}
        print(json.dumps(event_msg), flush=True)
        
        return tracked_features
    
    def _calculate_avg_track_length(self, data_mat: np.ndarray, total_points: int):
        # sum_of_tracks = 0
        # max_track = 0
        # for val in range(total_points):
        #     sum_of_tracks += data_mat[data_mat[:, 0] == val].shape[0]
        #     if data_mat[data_mat[:, 0] == val].shape[0] > max_track:
        #         max_track = data_mat[data_mat[:, 0] == val].shape[0] 
        # print("MAX TRACK LENGTH:", max_track)

        track_ids = data_mat[:, 0].astype(int)

        unique_ids, counts = np.unique(track_ids, return_counts=True)

        max_track = counts.max()
        avg_track = counts.mean()
        
        # return sum_of_tracks / total_points
        return float(avg_track), float(max_track)

    def _z_score(self, pts1: np.ndarray, pts2: np.ndarray, sigma_th: int) -> np.ndarray:
        pixel_diff = pts1 - pts2

        pixel_dist = np.linalg.norm(pixel_diff, axis=1)

        mu = np.mean(pixel_dist)
        sigma = np.std(pixel_dist)
        z = (pixel_dist - mu) / (sigma + 1e-12)
        out_count = np.sum(np.abs(z) > sigma_th)

        return out_count

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



class SparseSceneEstimation():
    def __init__(self, cam_data: CameraData):
        self.module_name = "..."
        self.description = "..."
        self.example = "..."

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
    
    def build_reconstruction(self, 
                             tracked_features: PointsMatched,
                             cam_poses: CameraPose) -> Scene:
        """Implement Algorithm to reconstruct scene here."""
        pass

    # Point Maps estimation must have this function follow with tracked features
    # Point Maps must be in the shape of 
    def match_tracks_to_point_maps(self,
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
                    norm_pts = self._normalize_points_for_BAL(views)#views[:, 1:])
                    observation = np.hstack((np.vstack(views[:,0]), point_ind, norm_pts))#views[:,1:]))
                    observation_pix = np.hstack((np.vstack(views[:,0]), point_ind, views[:, 1:]))
                    observations_pix.append(observation_pix)
                    observations.append(observation)
                    num_observations += views.shape[0] # Number of observations

                    # get 3D point
                    pred_point_3d = point_maps[frame][y, x]

                    points_3d.update_points(pred_point_3d)

                    # Update Point Index
                    point_index += 1
       
        # Build BAL data
        if self.multi_cam:
            ba_data = BundleAdjustmentData(num_cameras=num_cameras,
                                            num_points=points_3d.points3D.shape[0],
                                            num_observations=num_observations,
                                            observations=observations,
                                            cameras=camera_poses,
                                            points=points_3d.points3D,
                                            dist=self.dist,
                                            mono=False)
        else:
            ba_data = BundleAdjustmentData(num_cameras=num_cameras,
                                            num_points=points_3d.points3D.shape[0],
                                            num_observations=num_observations,
                                            observations=observations,
                                            cameras=camera_poses,
                                            points=points_3d.points3D,
                                            dist=[self.dist],
                                            mono=True)


        scene = Scene(points3D = points_3d,
                      cam_poses = camera_poses.camera_pose,
                      observations= np.vstack(observations_pix),
                      representation = "point cloud",
                      bal_data=ba_data,
                      sparse=True)
        return scene
    
    # Normalize for BAL data optimization
    def _normalize_points_for_BAL(self, view: np.ndarray): #pts1: np.ndarray):
        cams, pts = view[:, 0], view[:, 1:]
        #cam = int(cam)
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
    
    def _reprojection_error(self, 
                            point: np.ndarray, 
                            views: np.ndarray, 
                            cam_poses: list[np.ndarray]) -> float:
        X_h = np.append(point, 1)
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

class DenseSceneEstimation():
    def __init__(self, cam_data: CameraData):
        self.module_name = "..."
        self.description = "..."
        self.example = "..."

        # Setting up Calibration Data
        self.cam_data = cam_data
        self.image_list = copy.copy(cam_data.image_list)
        self.K_mat = cam_data.get_K()
        self.dist = cam_data.get_distortion()
        self.stereo = cam_data.stereo
        self.multi_cam = cam_data.multi_cam

        # Setup Minimum Angle Check Function
        # self.angle_check = TriangulationCheck(self.K_mat, self.dist)
        
        #self.image_path = sorted(glob.glob(image_path + "\\*"))[:10]

    def __call__(self, sparse_scene: Scene | None = None,
                 camera_poses: CameraPose | None = None) -> Scene:

        return self.build_reconstruction(sparse_scene = sparse_scene, 
                                         cam_poses = camera_poses)
    
    def build_reconstruction(self, 
                             sparse_scene: Scene | None = None,
                             cam_poses: CameraPose | None = None) -> Scene:
        """Implement Algorithm to reconstruct scene here."""
        pass
    
    # For Point Map Reconstruction Models
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
    
    # Down sample voxels for better reconstruction
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

class CameraPoseEstimatorClass():
    def __init__(self, cam_data: CameraData):
        self.module_name = "..."
        self.description = "..."
        self.example = "..."

        self.cam_data = cam_data
        self.image_list = copy.copy(cam_data.image_list)
        self.K_mat = cam_data.get_K()
        self.dist = cam_data.get_distortion()

    # def _setup_calibration(self, image_scale: list[float] | None = None):
    #     if image_scale is not None:
    #         self.calibration.update_cal_img_shape(image_scale)

    #     # print(image_scale)

    #     if isinstance(self.calibration, Calibration):
    #         self.stereo = self.calibration.stereo
    #         self.K1 = self.calibration.K1
    #         self.dist1 = self.calibration.distort
    #         if self.stereo:
    #             self.K2 = self.calibration.K2
    #             self.dist2 = self.calibration.distort2
    #     else:
    #         self.stereo = False
    #         self.K1 = None
    #         self.dist1 = None
        
    def __call__(self, 
                 feature_pairs: PointsMatched | None = None) -> CameraPose:
        poses = CameraPose() # Empty Data Condainer 

        poses = self._estimate_camera_poses(camera_poses=poses,
                                            feature_pairs=feature_pairs)

        return poses
    
    def _estimate_camera_poses(self, 
                               camera_poses: CameraPose,
                               feature_pairs: PointsMatched) -> CameraPose:
        # Input Custom Pose Estimation Algorithm in this Function
        
        return camera_poses

    def _metric_calculation_residuals(self, 
                                      object_points: np.ndarray, 
                                      image_points: np.ndarray,
                                      pose: np.ndarray):
        R = pose[:, :3]
        T = pose[:, 3:]
        proj, _ = cv2.projectPoints(object_points, R, T, self.K_mat, self.dist)

        residual_error = np.linalg.norm(image_points - proj.reshape(-1,2), axis=1)
        error = np.mean(residual_error)
        median_error = np.median(residual_error)

        return error, median_error

class FeatureClass():
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
    
    def _detect_features(self) -> list[Points2D]:
        # Write Code Here to Fill Feature Module per Detector Implemented

        return self.features

    def calculate_metrics(self) -> None:


        # Output Metric
        mean_ct, min_count, max_count = self._metric_calculation()
        event_msg = {"avg": mean_ct, "min": min_count, "max": max_count}
        print(json.dumps(event_msg), flush=True)
        mean_ct, min_count, max_count = self._spatial_dist_calc()
        event_msg = {"avg Coverage": mean_ct, "min Coverage": min_count, "max Coverage": max_count}
        print(json.dumps(event_msg), flush=True)

        # Write to file
        with open('data.json', 'w', encoding='utf-8') as f:
            json.dump(data, f)

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

class FeatureMatching():
    def __init__(self, 
                 detector:str, 
                 cam_data:CameraData,
                 RANSAC_threshold: float,
                 RANSAC: bool,
                 RANSAC_conf: float):
        self.module_name = "..."
        self.description = "..."
        self.example = "..."

        self.detector = detector
        self.det_free = False

        self.cam_data = cam_data
        self.K = cam_data.get_K()
        self.dist = cam_data.get_distortion()

        self.DETECTORS = ["sift", "superpoint", "orb"]

        if self.detector not in self.DETECTORS:
            self.det_free = True

        # Setup Outlier Rejection
        self.normalize = Normalization(K=self.K, 
                                       dist=self.dist,
                                       multi_cam=self.cam_data.multi_cam)
        self.ransac = RANSAC
        self.ransac_threshold = RANSAC_threshold
        self.ransac_conf = RANSAC_conf
    
    def __call__(self, features: list[Points2D]) -> PointsMatched:
        
        matched_points = self.match_full(features) 

        return matched_points
    
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
        
        pts1_norm = self.normalize(pts1, frame_id)
        pts2_norm = self.normalize(pts2, frame_id+1)

        if self.ransac:
            # F, mask = cv2.findFundamentalMat(pts1_norm, pts2_norm, cv2.FM_RANSAC, 
            #                                  ransacReprojThreshold=self.ransac_threshold, 
            #                                  confidence=self.ransac_conf)
            F, mask = cv2.findHomography(pts1_norm, pts2_norm, cv2.RANSAC, 
                                         ransacReprojThreshold=self.ransac_threshold)
        else:
            F, mask = cv2.findFundamentalMat(pts1_norm, pts2_norm, cv2.FM_LMEDS)

        # Could update points2D to inlier points with Mask
        inlier_pts1 = Points2D(**pts1.set_inliers(mask))
        inlier_pts2 = Points2D(**pts2.set_inliers(mask))
        idx1_inliers = np.array(idx1)[mask.ravel() == 1]
        idx2_inliers = np.array(idx2)[mask.ravel() == 1]

        return inlier_pts1, inlier_pts2, idx1_inliers, idx2_inliers, F
    
    def match_full(self, features: list[Points2D]) -> PointsMatched:
        """Override for custom matching algorithm in here"""
        
        matched_points = PointsMatched() 

        return matched_points

    def _metric_calculation(self, matching_points: PointsMatched, sigma_th: int = 3):
        # outlier_count = self._z_score(matched_points=matching_points.pairwise_matches, sigma_th=sigma_th)
        repeatability = self._calc_repeatability(matching_points=matching_points, epsilon=3.5) # Since we don't have ground truth, we assume 1px of noise.
        mean_ct, inlier_yield = self._matching_feat_counts(matched_points=matching_points.pairwise_matches, features = matching_points.img_features)
        gric_score_F, gric_score_H = self.evaluate_models(matching_points=matching_points)

        return mean_ct, inlier_yield, repeatability, gric_score_F, gric_score_H
    
    def _matching_feat_counts(self, matched_points: list[np.ndarray], features: list[np.ndarray]):
        set_of_pt_counts = np.zeros((len(matched_points), 1))
        inlier_yields = np.zeros((len(features), 1))

        for i in range(len(matched_points)):
            matching_points = matched_points[i]
            num_pts = matching_points.shape[0]
            set_of_pt_counts[i] = num_pts # Get total correspondences

            # Get inlier Yield
            feats1 = features[i]
            feats2 = features[i + 1]
            inlier_yields[i] = num_pts/feats1.shape[0]
            inlier_yields[i + 1] = num_pts/feats2.shape[0]
        
        # counts_np = np.array(set_of_pt_counts)

        mean_ct = set_of_pt_counts.mean()
        inlier_yield_avg = inlier_yields.mean()
        # min_count = counts_np.min()
        # max_count = counts_np.max()

        return float(mean_ct), float(inlier_yield_avg) #int(min_count), int(max_count)

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
        for pair in range(len(matched_points)):
            pt_set = matched_points[pair]
            pt_set_A = pt_set[:, :2]
            pt_set_B = pt_set[:, 2:]

            H_mat, _ = cv2.findHomography(pt_set_A, pt_set_B, cv2.RANSAC, 5.0)

            KA = features[pair]
            KB = features[pair + 1]

            print("BEFORE", KB.shape)
            KA_prime = self._warp_points(KA.reshape(-1, 1, 2), H_mat, [W, H])
            KB_prime = self._warp_points(KB.reshape(-1, 1, 2), np.linalg.inv(H_mat),[W, H])
            KB = self._warp_points(KB_prime.reshape(-1, 1, 2), H_mat, [W, H])
            print("AFTER", KB.shape)
            tree = cKDTree(KB)

            # k=2 because first neighbor is itself
            dists, _ = tree.query(KA_prime, k=2)

            nearest = dists[:,1]

            repeated = nearest[nearest <= epsilon].shape[0]
            print(repeated)
            repeatability = repeated / min(len(KA_prime), len(KB))

            repeatabilities.append(repeatability)


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

class FeatureTracking():
    def __init__(self, detector:str, 
                 cam_data:CameraData,
                 RANSAC_threshold: float,
                 RANSAC_conf: float):
        self.module_name = "..." # To Fill per Module Basis
        self.description = "..." # To Fill per Module Basis
        self.example = "..."     # To Fill per Module Basis

        self.detector = detector
        self.det_free = False

        self.cam_data = cam_data
        self.K = cam_data.get_K()
        self.dist = cam_data.get_distortion()

        self.DETECTORS = ["sift", "superpoint", "orb"]

        if self.detector not in self.DETECTORS:
            self.det_free = True

        # Fixed Algorithm to Track Features
        self.feature_tracker = FeatureTracker(self.matcher_parser, 
                                              K=self.K,
                                              dist=self.dist,
                                              RANSAC_threshold=RANSAC_threshold,
                                              RANSAC_conf=RANSAC_conf,
                                              cam_data = self.cam_data)

    def __call__(self, features: list[Points2D]) -> PointsMatched: # Fixed to the Module
        
        # Points Matched for Tracking -> data = N x [track_id, frame_num, x, y]

        matched_points = self.feature_tracker.match_full(features)

        return matched_points
    
    def matcher_parser(self, pt1: Points2D, pt2: Points2D) -> tuple[list, list]:
        # To Fill per Module Basis
        return [], [] 

class OptimizationClass():
    def __init__(self, 
                 cam_data: CameraData,
                 refine_focal_length: bool = False,
                 refine_principal_point: bool = False,
                 refine_extra_params: bool = False,
                 max_num_iterations: int = 50,
                 use_gpu: bool = True,
                 gpu_index: int = 0,
                 robust_loss: bool = True,
                 ):
        self.module_name = "..."
        self.description = "..."
        self.example = "..."

        # Set up Camera Data/Image Resolution
        self.K = cam_data.get_K()
        self.dist = cam_data.get_distortion()
        self.cam_data = cam_data
        self.H, self.W = cam_data.image_list[0].shape[:2] 
        self.multi_cam = cam_data.multi_cam

        # Set up Bundle Adjustment Params
        self.refine_focal_length = refine_focal_length
        self.refine_principal_point = refine_principal_point
        self.refine_extra_params = refine_extra_params
        self.max_num_iterations = max_num_iterations
        self.use_gpu = use_gpu
        self.gpu_index = gpu_index
        self.robust_loss = robust_loss

        # self.optimizer = ["BA"]
        # self.dataset = scene.bal_data.dataset
        
        # self.OPTIONS = ['essential', 'fundamental', 'homography', 'projective']
        # self.FORMATS = ['full', 'partial', 'pair']

        # self.format = format
        
    def __call__(self, current_scene: Scene) -> Scene:
        """Fixed Function Call specifically for global bundle adjustment pipelines"""
        return self.optimize(current_scene)

    def optimize(self, current_scene: Scene | IncrementalSfMState):
        """Edit Function Call for optimizer library and BA style (Global or Local)"""
        pass
    
    def _build_reconstruction(self, 
                              current_scene: Scene | IncrementalSfMState):
        """Edit Function Call specifically to set up state or optimization solver"""
        pass
 
class VisualizeClass():
    def __init__(self, path:str):
        self.module_name = "..."
        self.description = "..."
        self.example = "..."

        self.path = path

    def __call__(self, data: Scene | np.ndarray, store:bool = False) -> None:
        pass