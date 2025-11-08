import cv2
import numpy as np
from modules.baseclass import SceneEstimation
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms as TF
import os
from modules.models.sfm_models.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from modules.models.sfm_models.vggt.utils.geometry import unproject_depth_map_to_point_map
from modules.models.sfm_models.vggt.models.vggt import VGGT
from modules.models.sfm_models.vggt.utils.load_fn import load_and_preprocess_images

from modules.DataTypes.datatype import (Points2D, 
                                        CameraData, 
                                        Points3D, 
                                        CameraPose, 
                                        Scene, 
                                        PointsMatched,
                                        BundleAdjustmentData)

############################################## HELPER CLASS ##############################################

# class TriangulationCheck:
#     def __init__(self, calibration: Calibration, min_angle: float = 1.0):
#         # Calibration Set up
#         self.K1 = calibration.K1
#         self.dist1 = calibration.distort

#         # Minimum Angle Necessary 
#         self.min_angle = min_angle

#     # views = [cam, x, y]:Nx3, camera_poses = [R, t]:4x4
#     def __call__(self, views: np.ndarray, cam_poses: list[np.ndarray]) -> tuple[bool, float]:
#         track_len = views.shape[0]
#         max_angle = 0.0
#         min_angle = 180.0

#         for i in range(track_len):
#             for j in range(i + 1, track_len):
#                 cam1, pt1 = views[i, 0], views[i, 1:]
#                 cam2, pt2 = views[j, 0], views[j, 1:]
#                 cam1 = int(cam1)
#                 cam2 = int(cam2)

#                 pt_vec1 = self.copmute_bearing_vec(pt1)
#                 pt_vec2 = self.copmute_bearing_vec(pt2)

#                 R1 = cam_poses[cam1][:, :3]
#                 R2 = cam_poses[cam2][:, :3]

#                 pt_vec1_R = R1 @ pt_vec1
#                 pt_vec2_R = R2 @ pt_vec2

#                 angle = self.angle_from_pts(pt1_vec=pt_vec1_R, pt2_vec=pt_vec2_R)

#                 # if angle > max_angle:
#                 #     max_angle = angle
#                 if angle <= min_angle:
#                     min_angle = angle


#         # return max_angle >= self.min_angle, max_angle
#         return min_angle >= self.min_angle, min_angle

#     def copmute_bearing_vec(self, pt: np.ndarray):
#         pt_norm = cv2.undistortPoints(pt, cameraMatrix=self.K1, distCoeffs=self.dist1)[:,0,:]

#         x = np.array([[pt_norm[0,0]], [pt_norm[0,1]], [1.0]])

#         x_cam = np.linalg.inv(self.K1)@(x)
#         x_cam = x_cam / np.linalg.norm(x_cam)
#         return x_cam
    
#     def angle_from_pts(self, pt1_vec: np.ndarray, pt2_vec: np.ndarray):
#         angle = np.dot(pt1_vec[:, 0], pt2_vec[:, 0])
#         angle = np.clip(angle, -1.0, 1.0)

#         return np.degrees(np.arccos(angle))


##########################################################################################################

##########################################################################################################
############################################### ML MODULES ###############################################

class Sparse3DReconstructionVGGT(SceneEstimation):
    def __init__(self,
                 cam_data: CameraData,
                 min_observe: int = 3):
        super().__init__(cam_data = cam_data)

        self.module_name = "Sparse3DReconstructionVGGT"
        self.description = f"""
Sparsely, and densely, reconstructs a 3D scene utilizing pre-processed information of camera poses and
images of the scene. Camera Poses are estimated prior to thie module through the camera pose estimation 
module, specifically from VGGT pose estimation. Features do NOT need to be tracked or matched between frames.
This module can reconstruct sparse 3D scenes specifically using a monocular camera. 
This module can reconstruct sparse 3D scenes either through single view or multi-view scenes.
This is determined by the how many images exist in the scene and how many poses were estimated from the previous
module using the VGGT pose estimation tool specifically.
Use this module when specified for sparse/dense reconstruction and the scene doesn't allow for many features to be detected
from given feature detectors. Utilize this module in conjuction with the VGGT pose estimation module in these cases
where feature detection is low. This  module is for reconstructing the scene using the deep learning approach. 
Computation time should not matter when invoking this tool.

Initialization Parameters:
- image_path (str): the image path where the images are stored to utilize for scene building.
- calibration (Calibration): Data type that stores the camera's calibration data initialized from the calibration 
reader module

Function Call Parameters:
- camera_poses (CameraPose): Estimated camera poses for the given scene. Poses are estimated prior to this function call, 
specifically from the CameraPoseEstimation modules. This scene reconstruction is called in conjuction with the VGGT 
pose estimation module.
"""
        self.example = f"""
Initialization:
image_path = ...
calibration_path = ...
calibration_data = CalibrationReader(calibration_path).get_calibration()

# Camera Pose Module Initialization
pose_estimator = CamPoseEstimatorVGGTModel(image_path=image_path, 
                                            calibration=calibration_data)

# Scene Reconstruction Module Initialization
sparse_reconstruction = Sparse3DReconstructionVGGT(calibration=calibration_data, 
                                                   image_path=image_path)

Function Use:
# From estimated features, estimate the camera poses for all image frames
cam_poses = pose_estimator()

# Estimate sparse 3D scene from tracked features and camera poses
sparse_scene = sparse_reconstruction(camera_poses=cam_poses)
"""

        # Initialize Model
        WEIGHT_MODULE = str(os.path.dirname(__file__)) + "\\models\\sfm_models\\vggt\\weights\\model.pt"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        self.model = VGGT().to(self.device)
        self.model.load_state_dict(torch.load(WEIGHT_MODULE, weights_only=True))

        self.height, self.width = self.image_list[0].shape[:2]
        # Load Images in correct format for VGGT inference
        to_tensor = TF.ToTensor()
        tensor_img_list = []
        for ind in range(len(self.image_list)):
            tensor_img_list.append(to_tensor(self.image_list[ind]))
        self.images = torch.stack(tensor_img_list).to(self.device) 

        self.minimum_observation = min_observe


    def build_reconstruction(self, 
                             tracked_features: PointsMatched, 
                             cam_poses: CameraPose) -> Scene:

        ext_torch = torch.from_numpy(np.array(cam_poses.camera_pose)).to(self.device)
        int_torch = torch.from_numpy(np.array(self.K_mat)).to(self.device)

        # VGGT Fixed Resolution to 518 for Inference
        images = F.interpolate(self.images, size=(518, 518), mode="bilinear", align_corners=False)
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=self.dtype):
                images = images[None]  # add batch dimension
                aggregated_tokens_list, ps_idx = self.model.aggregator(images)

            # Predict Depth Maps
            depth_map, depth_conf = self.model.depth_head(aggregated_tokens_list, self.images, ps_idx)

            point_map = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                                                                ext_torch, 
                                                                int_torch)
        
        num_cameras = len(cam_poses.camera_pose)

        # Here we use the ext, int, depth_map, and point_map (points3D) to initialize the sparse reconstruction with tracked feature points
        scene = self.match_tracks_to_point_maps(tracked_features=tracked_features,
                                                point_maps = point_map,
                                                conf_maps = depth_conf,
                                                minimum_observation = self.minimum_observation,
                                                img_width = self.width,
                                                num_cameras = num_cameras)

        return scene
        # points3D = Points3D(points = point_map.reshape(-1, 3))

        # return Scene(points3D=points3D, 
        #              cam_poses=cam_poses.camera_pose, 
        #              representation="point cloud")
    

###########################################################################################################
############################################ CLASSICAL MODULES ############################################

class Sparse3DReconstructionStereo(SceneEstimation):
    def __init__(self,
                 cam_data: CameraData):
        super().__init__(cam_data=cam_data)

        self.module_name = "Sparse3DReconstructionMono"
        self.description = f"""
Sparsely reconstructs a 3D scene utilizing pre-processed information of camera poses and
detected features tracked across the scene. Camera Poses are estimated prior to thie module
through the camera pose estimation module. Features matched, or tracked are estimated 
prior to this module through the feature matcher module. 
This module can reconstruct sparse 3D scenes specifically using a stereo camera. This is 
determined by the data used and the parameter 'view' on module function call. 
This module can reconstruct sparse 3D scenes either through multi-view or two-view triangulation
with a stereo camera. This is determined by the method used to find matching features. If features 
are detected from pairwise matching, use the "two" method for the 'method' parameter. If features 
are tracked across multiple frames, use the "multi" method for the 'method' parameter.
Use this module when specified for sparse reconstruction and calibration data is provided when
the data used is from a stereo camera specifically. This module is for reconstructing the scene 
using the direct mathematical approach.
"""
        self.example = f"""
Initialization:
image_path = ...
calibration_path = ...
calibration_data = CalibrationReader(calibration_path).get_calibration()

sparse_reconstruction = Sparse3DReconstructionStereo(calibration=calibration_data, image_path=image_path)

Function Use (multi-view):
feature_tracker = FeatureMatchLightGlueTracking(detector="superpoint")

cam_poses = pose_estimator(features=features) # To Estimate Camera Poses from detected features

tracked_features = feature_tracker(features=features) # To track features across multiple images 

# Estimate 3D scene using multi-view due to tracking features from multiple images in previous step
sparse_scene = sparse_reconstruction(tracked_features, cam_poses, view="multi") 

Function Use (two-view):
feature_matcher = FeatureMatchLoftrPair(img_path=image_path)

cam_poses = pose_estimator(features=features) # To Estimate Camera Poses from detected features

matched_features = feature_matcher(features=features) # To track features across multiple images 

# Estimate 3D scene using multi-view due to tracking features from multiple images in previous step
sparse_scene = sparse_reconstruction(tracked_features, cam_poses, view="two") 
"""
        self.VIEWS = ["multi", "two"]

    def __call__(self, points: PointsMatched, camera_poses: CameraPose, view: str | None = "multi") -> Scene:
        if view not in self.VIEWS:
            message = 'Error: setting is not supported. Use one of ' + str(self.VIEWS) + ' instead to use this Reconstruction Module.'
            raise Exception(message)
        
        # points_3d = []
        points_3d = Points3D()

        if view == self.VIEWS[0]: # Multi-view
            if not points.multi_view:
                message = 'Error: features are not tracked. Use the setting ' + str(self.VIEW[1]) + ' instead to use this Reconstruction Module for pairwise feature matching.'
                raise Exception(message)
            
            for i in tqdm(range(points.point_count)):
                views = points.access_point3D(i)

                point = self.triangulate_nView_points_Mono(views, camera_poses.camera_pose)

                points_3d.update_points(point)

            scene = Scene(points3D = points_3d,cam_poses = camera_poses, representation = "point cloud") 
            return scene
        elif view == self.VIEWS[1]: # Two-View
            if points.multi_view:
                message = 'Error: features are tracked. Use the setting ' + str(self.VIEW[0]) + ' instead to use this Reconstruction Module for feature tracking.'
                raise Exception(message)
            
            points_3d = Points3D()

            for i in tqdm(range(len(points.pairwise_matches))):
                pts1, pts2 = points.access_matching_pair(i) # Left and Right Image Features
                pose = camera_poses.camera_pose[i]         # Left Camera

                points3d = self.triangulate_points_stereo(pts1, pts2, pose)

                points_3d.update_points(points3d)

            scene = Scene(points3D = points_3d,cam_poses = camera_poses, representation = "point cloud") 
            return scene
    
    # Triangulation of points (Stereo Camera) - 2View (Purely Stereo Camera)
    def triangulate_points_stereo(self, pts1: np.ndarray, pts2: np.ndarray, camera_pose: np.ndarray) -> np.ndarray:
        Rot_R = self.R12 @ camera_pose[:, :3] 
        Trans_R = self.R12 @ camera_pose[:, 3:] + self.T12
        stereo_pose = np.hstack((Rot_R, Trans_R))

        if self.dist1 is not None:
            pt1 = cv2.undistortPoints(pts1, self.K1, self.dist1)
            pt2 = cv2.undistortPoints(pts2, self.K2, self.dist2)
            P1mtx = np.eye(3) @ camera_pose
            P2mtx = np.eye(3) @ stereo_pose
        else:
            pt1 = pts1.T
            pt2 = pts2.T
            P1mtx = self.K1 @ camera_pose
            P2mtx = self.K2 @ stereo_pose

        X = cv2.triangulatePoints(P1mtx, P2mtx, pt1, pt2)
        X = (X[:-1]/X[-1]).T[0]

        return X 
    
    # def triangulate_nView_points_Stereo(self, views: np.ndarray, cam_poses: list[np.ndarray]) -> np.ndarray:

    #     # total_cameras = len(self.scene_point_2d_map[pt_index])
    #     total_cameras = len(views.shape[0])
    #     A = np.zeros((4*total_cameras, 4))

    #     Rot_R = self.R12 @ camera_pose[:, :3] 
    #     Trans_R = self.R12 @ camera_pose[:, 3:] + self.T12
    #     stereo_pose = np.hstack((Rot_R, Trans_R))

    #     if self.dist1 is not None:
    #         pt1 = cv2.undistortPoints(pts1.points2D, self.K1, self.dist1)
    #         pt2 = cv2.undistortPoints(pts2.points2D, self.K2, self.dist2)
    #         P1mtx = np.eye(3) @ camera_pose
    #         P2mtx = np.eye(3) @ stereo_pose
    #     else: 
    #         pt1 = pts1.points2D
    #         pt2 = pts2.points2D
    #         P1mtx = self.K1 @ camera_pose
    #         P2mtx = self.K2 @ stereo_pose

    #     # Read Hartley and Zisserman to see if we need the normalization factor??
    #     if self.dist1 is None:
    #         for i in range(views.shape[0]):
    #             cam, pt = views[i, 0], views[i, 1:]
    #             Pmat = self.K1 @ cam_poses[cam]

    #             row1 = pt[0]*Pmat[2, :] - Pmat[0, :]
    #             row2 = pt[1]*Pmat[2, :] - Pmat[1, :]

    #             A[2*i, :] = row1
    #             A[2*i + 1, :] = row2
    #     else: 
    #         for i in range(views.shape[0]):
    #             cam, pt = views[i, 0], views[i, 1:]
    #             Pmat = np.eye(3) @ cam_poses[cam]
    #             xUnd = cv2.undistortPoints(pt, self.K1, self.dist1)

    #             row1 = xUnd[0, 0, 0]*Pmat[2, :] - Pmat[0, :]
    #             row2 = xUnd[0, 0, 1]*Pmat[2, :] - Pmat[1, :]

    #             A[2*i, :] = row1
    #             A[2*i + 1, :] = row2

    #     U, S, V = np.linalg.svd(A)
    #     X = V[-1, :]
    #     X = (X[:-1]/X[-1]).T[0]

    #     return X


    # def triangulate_nView_points_stereo(self, views: np.ndarray, cam_poses):

    #     # total_cameras = len(self.scene_point_2d_map[pt_index])
    #     total_cameras = len(self.scene_2d_pts[pt_index])
    #     A = np.zeros((4*total_cameras, 4))

    #     index = 0

    #     # Read Hartley and Zisserman to see if we need the normalization factor??
    #     for cam, pt in self.scene_2d_pts[pt_index].items():

    #         PmatLeft = np.eye(3) @ self.camera_poses[cam][0]
    #         PmatRight = np.eye(3) @ self.camera_poses[cam][1]

    #         xU1 = cv2.undistortPoints(np.hstack(pt[0]), self.cam_left, self.dist_left)
    #         xU2 = cv2.undistortPoints(np.hstack(pt[1]), self.cam_right, self.dist_right)

    #         row1 = xU1[0, 0, 0]*PmatLeft[2, :] - PmatLeft[0, :]
    #         row2 = xU1[0, 0, 1]*PmatLeft[2, :] - PmatLeft[1, :]
    #         row3 = xU2[0, 0, 0]*PmatRight[2, :] - PmatRight[0, :]
    #         row4 = xU2[0, 0, 1]*PmatRight[2, :] - PmatRight[1, :]

    #         A[4*index, :] = row1
    #         A[4*index + 1, :] = row2
    #         A[4*index + 2, :] = row3
    #         A[4*index + 3, :] = row4

    #     index += 1
    #     U, S, V = np.linalg.svd(A)
    #     X = V[-1, :]
    #     X = (X[:-1]/X[-1]).T

    #     return X
    

class Sparse3DReconstructionMono(SceneEstimation):
    def __init__(self, cam_data: CameraData, 
                 view: str = "multi",
                 reproj_error: float = 3.0,
                 min_observe: int = 3,
                 min_angle: float = 1.0):
        
        super().__init__(cam_data=cam_data)

        self.module_name = "Sparse3DReconstructionMono"
        self.description = f"""
Sparsely reconstructs a 3D scene utilizing pre-processed information of camera poses and
detected features tracked across the scene. Camera Poses are estimated prior to thie module
through the camera pose estimation module. Features matched, or tracked are estimated 
prior to this module through the feature matcher module. 
This module can reconstruct sparse 3D scenes specifically using a monocular camera. This is 
determined by the data used and the parameter 'view' on module function call. 
This module can reconstruct sparse 3D scenes either through multi-view or two-view triangulation.
This is determined by the method used to find matching features. If features are detected from 
pairwise matching, use the "two" method for the 'method' parameter. If features are tracked
across multiple frames, use the "multi" method for the 'method' parameter.
Use this module when specified for sparse reconstruction and calibration data is provided,
with the camera being used is a monocular cmaera. This  module is for reconstructing the 
scene using the direct mathematical approach.

Initialization Parameters:
- image_path (str): The image path where the images are stored to utilize for scene building.
- calibration (Calibration): Data type that stores the camera's calibration data initialized from the calibration 
reader module
- min_observe: The minimum number of observations (number of tracked feature points) needed to conduct a 3D 
point estimation. Note: this must be greater than 2
    - Default (int): 3 
- min_angle: The minimum angle required between bearing rays from paired 2D feature point to accept a 3D point 
estimation from the set of corresponding 2D feature points. Used for the Triangulation Angle Test.
    - Default (float): 1.0 (Typically 1.0 - 3.0 [Number represents angle degree])

Function Call Parameters:
- camera_poses (CameraPose): Estimated camera poses for the given scene. Poses are estimated prior to this function call, 
specifically from the CameraPoseEstimation modules. 
"""
        self.example = f"""
Initialization:
image_path = ...
calibration_path = ...
calibration_data = CalibrationReader(calibration_path).get_calibration()

sparse_reconstruction = Sparse3DReconstructionMono(calibration=calibration_data, image_path=image_path)

Function Use (multi-view):
feature_tracker = FeatureMatchLightGlueTracking(detector="superpoint")

cam_poses = pose_estimator(features=features) # To Estimate Camera Poses from detected features

tracked_features = feature_tracker(features=features) # To track features across multiple images 

# Estimate 3D scene using multi-view due to tracking features from multiple images in previous step
sparse_scene = sparse_reconstruction(tracked_features, cam_poses, view="multi") 

Function Use (two-view):
feature_matcher = FeatureMatchLoftrPair(img_path=image_path)

cam_poses = pose_estimator(features=features) # To Estimate Camera Poses from detected features

matched_features = feature_matcher(features=features) # To track features across multiple images 

# Estimate 3D scene using multi-view due to tracking features from multiple images in previous step
sparse_scene = sparse_reconstruction(tracked_features, cam_poses, view="two") 
"""
        self.VIEWS = ["multi", "two"]
        self.view = view
        self.minimum_observation = min_observe # N-view functionality only
        self.min_angle = min_angle
        self.reproj_error_min = reproj_error
        # self.angle_check = TriangulationCheck(calibration=calibration, min_angle=min_angle)

    # tracked_features: PointsMatched, cam_poses: CameraPose
    def build_reconstruction(self, 
                 points: PointsMatched, 
                 camera_poses: CameraPose) -> Scene:

        if self.view not in self.VIEWS:
            message = 'Error: setting is not supported. Use one of ' + str(self.VIEWS) + ' instead to use this Reconstruction Module.'
            raise Exception(message)
        
        # points_3d = []
        points_3d = Points3D()

        # BAL File for Optimization Module
        num_observations = 0 
        num_cameras = len(camera_poses.camera_pose)
        observations = []

        if self.view == self.VIEWS[0]: # Multi-view
            if not points.multi_view:
                message = 'Error: features are not tracked. Use the setting ' + str(self.VIEWS[1]) + ' instead to use this Reconstruction Module for pairwise feature matching.'
                raise Exception(message)
            
            point_index = 0
            for i in tqdm(range(points.point_count)):
                views = points.access_point3D(i)

                if views.shape[0] < self.minimum_observation: # Below minimum observation for accurate 3D triangulation
                    continue 
                # Check triangulation angle of points
                min_angle, max_angle = self.angle_check(views, 
                                                        camera_poses.camera_pose,
                                                        minimum_angle=self.min_angle)
                if not min_angle:
                    continue

                # # BAL Data Construction
                # point_ind = np.array([point_index for _ in range(views.shape[0])]).reshape((views.shape[0],1))
                # norm_pts = self._normalize_points_for_BAL(views)#views[:, 1:])
                # observation = np.hstack((np.vstack(views[:,0]), point_ind, norm_pts))#views[:,1:]))
                # observations.append(observation)
                # num_observations += views.shape[0] # Number of observations

                # Estimate 3D point
                point = self.triangulate_nView_points_Mono(views, camera_poses.camera_pose)
                
                reproj_error = self._reprojection_error(point, views, camera_poses.camera_pose)
                if reproj_error <= self.reproj_error_min:
                    # BAL Data Construction
                    point_ind = np.array([point_index for _ in range(views.shape[0])]).reshape((views.shape[0],1))
                    norm_pts = self._normalize_points_for_BAL(views)#views[:, 1:])
                    observation = np.hstack((np.vstack(views[:,0]), point_ind, norm_pts))#views[:,1:]))
                    observations.append(observation)
                    num_observations += views.shape[0] # Number of observations

                    # Keep 3D point here
                    points_3d.update_points(point)

                    point_index += 1 # Successfully Estimated Point

            # Build BAL data
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
        elif self.view == self.VIEWS[1]: # Two-View
            if points.multi_view:
                message = 'Error: features are tracked. Use the setting ' + str(self.VIEWS[0]) + ' instead to use this Reconstruction Module for feature tracking.'
                raise Exception(message)
            
            points_3d = Points3D()

            for i in tqdm(range(len(points.pairwise_matches))):
                pts1, pts2 = points.access_matching_pair(i) # frame_i and frame_i+1
                pose1 = camera_poses.camera_pose[i]         # frame_i
                pose2 = camera_poses.camera_pose[i + 1]     # frame_i+1

                points3d = self.triangulate_points_mono(pts1, pts2, [pose1, pose2])

                points_3d.update_points(points3d)

            scene = Scene(points3D = points_3d,cam_poses = camera_poses, representation = "point cloud") 
            return scene
    
    # def min_angle_test(self):
    #     pass

    # Triangulation of points (Monocular Camera) - 2View
    def triangulate_points_mono(self, pts1: np.ndarray, pts2: np.ndarray, camera_pose: list[np.ndarray]) -> np.ndarray:
        if self.dist1 is not None:
            pt1 = cv2.undistortPoints(pts1, self.K_mat, self.dist)
            pt2 = cv2.undistortPoints(pts2, self.K_mat, self.dist)
            
            P1mtx = np.eye(3) @ camera_pose[0]
            P2mtx = np.eye(3) @ camera_pose[1]
        else:
            pt1, pt2 = pts1.T, pts2.T

            P1mtx = self.K_mat @ camera_pose[0]
            P2mtx = self.K_mat @ camera_pose[1]

        X = cv2.triangulatePoints(P1mtx, P2mtx, pt1, pt2)
        X = (X[:-1]/X[-1]).T 

        # pts3D = Points3D(points = X)
        # return pts3D

        return X    

    def triangulate_nView_points_Mono(self, views: np.ndarray, cam_poses: list[np.ndarray]) -> np.ndarray:

        # total_cameras = len(self.scene_point_2d_map[pt_index])
        total_cameras = views.shape[0]
        A = np.zeros((2*total_cameras, 4))

        # Read Hartley and Zisserman to see if we need the normalization factor??
        # if self.dist is None: # Keep Points in Pixel Coordinates
        #     for i in range(views.shape[0]):
        #         cam, pt = views[i, 0], views[i, 1:]
        #         cam = int(cam)
        #         Pmat = self.K1 @ cam_poses[cam]

        #         row1 = pt[0]*Pmat[2, :] - Pmat[0, :]
        #         row2 = pt[1]*Pmat[2, :] - Pmat[1, :]

        #         A[2*i, :] = row1
        #         A[2*i + 1, :] = row2
        # else: 
        if self.multi_cam:
            for i in range(views.shape[0]):
                cam, pt = views[i, 0], views[i, 1:]
                cam = int(cam)
                Pmat = np.eye(3) @ cam_poses[cam]
                K = self.K_mat[cam]
                dist = self.dist[cam]

                xUnd = cv2.undistortPoints(pt, K, dist) # Undistort and Normalize Points

                row1 = xUnd[0, 0, 0]*Pmat[2, :] - Pmat[0, :]
                row2 = xUnd[0, 0, 1]*Pmat[2, :] - Pmat[1, :]

                A[2*i, :] = row1
                A[2*i + 1, :] = row2
        else:
            for i in range(views.shape[0]):
                cam, pt = views[i, 0], views[i, 1:]
                cam = int(cam)
                Pmat = np.eye(3) @ cam_poses[cam]
                xUnd = cv2.undistortPoints(pt, self.K_mat, self.dist) # Undistort and Normalize Points

                row1 = xUnd[0, 0, 0]*Pmat[2, :] - Pmat[0, :]
                row2 = xUnd[0, 0, 1]*Pmat[2, :] - Pmat[1, :]

                A[2*i, :] = row1
                A[2*i + 1, :] = row2

        U, S, V = np.linalg.svd(A)
        X = V[-1, :]

        X = (X[:-1]/X[-1]).T

        return X