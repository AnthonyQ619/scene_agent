'''
Base Class designs for each module to standardize the class design
for each tool/module.

This is to reduce the possiblility of the Agent to hallucinate code
'''

import numpy as np
import cv2
from modules.DataTypes.datatype import (Scene, 
                                CameraData, 
                                Calibration, 
                                Points2D, 
                                PointsMatched, 
                                CameraPose, 
                                Points3D,
                                BundleAdjustmentData)
import glob
from collections.abc import Callable

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

        # print(pts_norm1.shape)
        # print(pts_norm2.shape)
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

        event_msg = {"avg_outlier": avg_outlier, "avg_feats": mean_ct, "min_feats": min_ct, "max_feats": max_ct}
        print(json.dumps(event_msg), flush=True)

        tracked_features.set_matched_matrix(self.observations)
        tracked_features.track_map = self.track_map
        tracked_features.point_count = self.next_track_id - 1
        
        return tracked_features
    

    def _z_score(self, pts1: np.ndarray, pts2: np.ndarray, sigma_th: int) -> np.ndarray:
        pixel_diff = pts1 - pts2

        pixel_dist = np.linalg.norm(pixel_diff, axis=1)

        mu = np.mean(pixel_dist)
        sigma = np.std(pixel_dist)
        z = (pixel_dist - mu) / (sigma + 1e-12)
        out_count = np.sum(np.abs(z) > sigma_th)

        return out_count


class TriangulationCheck:
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
##########################################################################################################


class GeometryClass():
    def __init__(self, format: str):
        self.module_name = "..."
        self.description = "..."
        self.example = "..."

        self.format = format

    def __call__(self):
        pass

class ImageProcessorClass():
    def __init__(self, format: str):
        self.module_name = "..."
        self.description = "..."
        self.example = "..."

        self.format = format

    def __call__(self):
        pass

class SceneEstimation():
    def __init__(self, cam_data: CameraData):
        self.module_name = "..."
        self.description = "..."
        self.example = "..."

        # Setting up Calibration Data
        self.cam_data = cam_data
        self.image_list = copy.copy(cam_data.image_list)
        self.K_mat = cam_data.get_K(0)
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
                      cam_poses = camera_poses,
                      representation = "point cloud",
                      bal_data=ba_data)
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
                # dist = self.dist[cam]
                pt = pts[i, :]
                pt_norm = cv2.undistortPoints(pt.T, K, np.zeros((1,5)))[:, 0, :][0]
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
    
class CameraPoseEstimatorClass():
    def __init__(self, cam_data: CameraData):
        self.module_name = "..."
        self.description = "..."
        self.example = "..."

        self.cam_data = cam_data
        self.image_list = copy.copy(cam_data.image_list)
        self.K_mat = cam_data.get_K(0)
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

class FeatureClass():
    def __init__(self, cam_data: CameraData):
                 #image_path:str):
        self.module_name = "..."
        self.description = "..."
        self.example = "..."
        self.features: list[Points2D] = []

        self.image_list = cam_data.image_list
        self.image_scale = cam_data.image_scale
        self.image_shape = cam_data.image_list[0].shape[:2]


    def __call__(self) -> list[Points2D]:
        return self.features
    
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
        self.K = cam_data.get_K(0)
        self.dist = cam_data.get_distortion()

        self.DETECTORS = ["sift", "superpoint", "orb", "fast"]

        if self.detector not in self.DETECTORS:
            self.det_free = True

        # Setup Outlier Rejection
        self.normalize = Normalization(K=self.K, 
                                       dist=self.dist,
                                       multi_cam=self.cam_data.multi_cam)
        self.ransac = RANSAC
        self.ransac_threshold = RANSAC_threshold
        self.ransac_conf = RANSAC_conf

    # def __call__(self) -> PointsMatched:
    #     # Points Matched for Tracking -> data = N x [track_id, frame_num, x, y]
    #     # Points Matched for Pairwise Matching -> 
    #     matched_points = PointsMatched() 


    #     return matched_points
    
    def __call__(self, features: list[Points2D]) -> PointsMatched:
        
        matched_points = self.match_full(features) 

        return matched_points
    
    def outlier_reject(self, pts1: Points2D, pts2: Points2D, frame_id: int) -> tuple[Points2D, Points2D]: # Move to Base Class
        
        pts1_norm = self.normalize(pts1, frame_id)
        pts2_norm = self.normalize(pts2, frame_id+1)

        # F, mask = cv2.findFundamentalMat(pts1.points2D, pts2.points2D, cv2.FM_LMEDS)
        # F, mask = cv2.findFundamentalMat(pts1_norm, pts2_norm, cv2.FM_LMEDS)
        if self.ransac:
            F, mask = cv2.findFundamentalMat(pts1_norm, pts2_norm, cv2.FM_RANSAC, 
                                             ransacReprojThreshold=self.ransac_threshold, 
                                             confidence=self.ransac_conf)
        else:
            F, mask = cv2.findFundamentalMat(pts1_norm, pts2_norm, cv2.FM_LMEDS)

        # Could update points2D to inlier points with Mask
        inlier_pts1 = Points2D(**pts1.set_inliers(mask))
        inlier_pts2 = Points2D(**pts2.set_inliers(mask))

        return inlier_pts1, inlier_pts2, F
    
    def match_full(self, features: list[Points2D]) -> PointsMatched:
        """Override for custom matching algorithm in here"""
        
        matched_points = PointsMatched() 

        return matched_points

    def _metric_calculation(self, matching_points: PointsMatched, sigma_th: int = 3):
        outlier_count = self._z_score(matched_points=matching_points.pairwise_matches, sigma_th=sigma_th)
        mean_ct, min_count, max_count = self._matching_feat_counts(matched_points=matching_points.pairwise_matches)
        
        return float(np.mean(outlier_count)), mean_ct, min_count, max_count
    
    def _matching_feat_counts(self, matched_points: list[np.ndarray]):
        set_of_pt_counts = []

        for i in range(len(matched_points)):
            matching_points = matched_points[i]
            num_pts = matching_points.shape[0]
            set_of_pt_counts.append(num_pts)
        
        counts_np = np.array(set_of_pt_counts)

        mean_ct = counts_np.mean()
        min_count = counts_np.min()
        max_count = counts_np.max()

        return float(mean_ct), int(min_count), int(max_count)

    def _z_score(self, matched_points: list[np.ndarray], sigma_th: int) -> np.ndarray:
        outlier_counts = []

        for i in range(len(matched_points)):
            matching_points = matched_points[i]

            pixel_diff = matching_points[:, :2] - matching_points[:, 2:]

            pixel_dist = np.linalg.norm(pixel_diff, axis=1)

            mu = np.mean(pixel_dist)
            sigma = np.std(pixel_dist)
            z = (pixel_dist - mu) / (sigma + 1e-12)
            out_count = np.sum(np.abs(z) > sigma_th)

            outlier_counts.append(int(out_count))
        
        return np.array(outlier_counts)

class FeatureTracking():
    def __init__(self, detector:str, 
                 cam_data:CameraData,
                 RANSAC_threshold: float,
                 RANSAC_conf: float):
        self.module_name = "..."
        self.description = "..."
        self.example = "..."

        self.detector = detector
        self.det_free = False

        self.cam_data = cam_data
        self.K = cam_data.get_K(0)
        self.dist = cam_data.get_distortion()

        self.DETECTORS = ["sift", "superpoint", "orb", "fast"]

        if self.detector not in self.DETECTORS:
            self.det_free = True

        self.feature_tracker = FeatureTracker(self.matcher_parser, 
                                              K=self.K,
                                              dist=self.dist,
                                              RANSAC_threshold=RANSAC_threshold,
                                              RANSAC_conf=RANSAC_conf,
                                              cam_data = self.cam_data)

    def __call__(self, features: list[Points2D]) -> PointsMatched:
        
        # Points Matched for Tracking -> data = N x [track_id, frame_num, x, y]

        matched_points = self.feature_tracker.match_full(features)

        return matched_points
    
    def matcher_parser(self, pt1: Points2D, pt2: Points2D) -> tuple[list, list]:
        return [], []

class OptimizationClass():
    def __init__(self, cam_data: CameraData):
        self.module_name = "..."
        self.description = "..."
        self.example = "..."

        self.cal = cam_data.get_K(0)
        self.dist = cam_data.get_distortion()
        # self.optimizer = ["BA"]
        # self.dataset = scene.bal_data.dataset
        
        # self.OPTIONS = ['essential', 'fundamental', 'homography', 'projective']
        # self.FORMATS = ['full', 'partial', 'pair']

        # self.format = format
        
    def __call__(self):
        pass
    
    def _prep_optimizer(self) -> None:
        # Write Data here? Or have BALData class write data then..., and keep scene disjointed from BAL.
        pass
    
class VisualizeClass():
    def __init__(self, path:str):
        self.module_name = "..."
        self.description = "..."
        self.example = "..."

        self.path = path

    def __call__(self, data: Scene | np.ndarray, store:bool = False) -> None:
        pass