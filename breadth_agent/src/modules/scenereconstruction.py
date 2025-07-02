import cv2
import numpy as np
from baseclass import SceneEstimation
from tqdm import tqdm
from .DataTypes.datatype import Points2D, Calibration, Points3D, CameraPose, Scene, PointsMatched



class Sparse3DReconstruction(SceneEstimation):
    def __init__(self, calibration: Calibration, image_path: str):
        super().__init__(calibration=calibration, image_path=image_path)

        self.module_name = "Sparse3DReconstruction"
        self.description = f"""
Sparsely reconstructs a 3D scene utilizing pre-processed information of camera poses and
detected features tracked across the scene. Camera Poses are estimated prior to thie module
through camera pose estimation algorithms. Features tracked are estimated prior to this module
through feature tracking algorothms. 
This module can reconstruct sparse 3D scenes using a monocular or stereo camera. This is 
determined by the data used and the parameter 'camera_type' on the function call. 
This module can reconstruct sparse 3D scenes either through multi-view or two-view triangulation.
This is determined by the method used to find matching features. If features are detected from 
pairwise matching, use the "two-view" method for the 'view' parameter. If features are tracked
across multiple frames, use the "multi-view" method for the 'view' parameter.
Use this module when specified for sparse reconstruction and calibration data is provided. This 
module is for reconstructing the scene using the direct mathematical approach.
"""
        self.example = f"""
Initialization:
image_path = ...
calibration_path = ...
calibration_data = CalibrationReader(calibration_path).get_calibration()

sparse_reconstruction = Sparse3DReconstruction(calibration=calibration_data, image_path=image_path)


Function Use:
cam_poses = pose_estimator(features=features) # To Estimate Camera Poses from detected features

tracked_features = feature_tracker(features=features) # To track features across multiple images 

# Estimate 3D scene using multi-view due to tracking features from multiple images in previous step
sparse_scene = sparse_reconstruction(tracked_features, cam_poses, view="multi") 
"""
        self.VIEWS = ["multi", "two"]
        
    # tracked_features: PointsMatched, cam_poses: CameraPose
    def __call__(self, points: PointsMatched, camera_poses: CameraPose, view: str | None = "multi") -> Scene:
        # if format.lower() not in self.FORMATS:
        #     message = 'Error: no such option exist. Use on of ' + str(self.FORMATS)
        #     raise Exception(message)
        # if self.cam_setting.lower() not in self.CAM_SETTINGS:
        #     message = 'Error: no such option exist. Use on of ' + str(self.CAM_SETTINGS)
        #     raise Exception(message)

        # points_3d = []
        points_3d = Points3D()

        if view == self.VIEWS[0]: # Multi-view
            for i in tqdm(range(points.point_count)):
                views = points.access_point3D(i)

                point = self.triangulate_nView_points_Mono(views, camera_poses.camera_pose)

                points_3d.update_points(point)

            scene = Scene(points3D = points_3d,cam_poses = camera_poses, representation = "point cloud") 
            return scene
        elif view == self.VIEWS[1]:
            pass
        # scene = Scene(cam_poses= camera_poses, representation="point cloud")
        # if format.lower() == self.FORMATS[0]:
        #     if self.cam_setting.lower() in self.CAM_SETTINGS[0]: # Mono Cam Setting
        #         for i in range(len(points)):
        #             pts1 = points[i][0]
        #             pts2 = points[i][1]
        #             cam_pose1 = camera_poses[i]
        #             cam_pose2 = camera_poses[i + 1]

        #             pts = self.triangulate_points_mono(pts1, pts2, [cam_pose1, cam_pose2])
        #             print(pts.points3D.shape)
        #             # points_3d.append(pts)
        #             scene.update_3d_points(pts)
        #         # Construct 3D scene given points
        #         #scene = Scene(points3D = Points3D(points=points_3d), cam_poses= camera_poses, representation="point cloud")

        #         return scene
        #     elif self.cam_setting.lower() in self.CAM_SETTINGS[1]: # Stereo Cam Setting
        #         for i in range(len(points)):
        #             pts1 = points[i][0].points2D # Right Camera
        #             pts2 = points[i][1].points2D # Left Camera
        #             cam_pose1 = camera_poses[i] # Left camera pose

        #             pts = self.triangulate_points_stereo(pts1, pts2, [cam_pose1, cam_pose2])

        #             # points_3d.append(pts)

        #         # scene = Scene(points3D = Points3D(points=points_3d), cam_poses= camera_poses, representation="point cloud")
        #         scene.update_3d_points(pts)
        #         return scene
        # elif format.lower() == self.FORMATS[1]: # TODO: Think about removing this if-else completely and the format option. Need an input/output for all functions, and this doesn't seem good as a solution for single 3D point estimation
        #     if self.cam_setting.lower() in self.CAM_SETTINGS[0]:
        #         pts1 = points[0][0].points2D
        #         pts2 = points[0][1].points2D
        #         cam_pose1 = camera_poses[0]
        #         cam_pose2 = camera_poses[1]

        #         pts = self.triangulate_points_mono(pts1, pts2, [cam_pose1, cam_pose2])

        #         # points_3d.append(pts)

        #         #scene = Scene(points3D = Points3D(points=points_3d), cam_poses= camera_poses, representation="point cloud")
        #         scene.update_3d_points(pts)
        #         return scene
        #     elif self.cam_setting.lower() in self.CAM_SETTINGS[1]:
        #         pts1 = points[0][0].points2D # Right Camera
        #         pts2 = points[0][1].points2D # Left Camera
        #         cam_pose1 = camera_poses[0] # Left camera pose

        #         pts = self.triangulate_points_stereo(pts1, pts2, [cam_pose1, cam_pose2])

        #         # points_3d.append(pts)

        #         #scene = Scene(points3D = Points3D(points=points_3d), cam_poses= camera_poses, representation="point cloud")
        #         scene.update_3d_points(pts)
        #         return scene
    
    # Triangulation of points (Monocular Camera) - 2View
    def triangulate_points_mono(self, pts1: Points2D, pts2: Points2D, camera_pose: list[np.ndarray]) -> Points3D:
        if self.dist1 is not None:
            pt1 = cv2.undistortPoints(pts1.points2D, self.K1, self.dist1)
            pt2 = cv2.undistortPoints(pts2.points2D, self.K1, self.dist1)
            
            P1mtx = np.eye(3) @ camera_pose[0]
            P2mtx = np.eye(3) @ camera_pose[1]
        else:
            pt1, pt2 = pts1.points2D.T, pts2.points2D.T

            P1mtx = self.K1 @ camera_pose[0]
            P2mtx = self.K2 @ camera_pose[1]

        X = cv2.triangulatePoints(P1mtx, P2mtx, pt1, pt2)
        X = (X[:-1]/X[-1]).T 

        pts3D = Points3D(points = X)
        return pts3D

    # Triangulation of points (Stereo Camera) - 2View
    def triangulate_points_stereo(self, pts1: Points2D, pts2: Points2D, camera_pose: np.ndarray) -> np.ndarray:
        Rot_R = self.R12 @ camera_pose[:, :3] 
        Trans_R = self.R12 @ camera_pose[:, 3:] + self.T12
        stereo_pose = np.hstack((Rot_R, Trans_R))

        if self.dist1 is not None:
            pt1 = cv2.undistortPoints(pts1.points2D, self.K1, self.dist1)
            pt2 = cv2.undistortPoints(pts2.points2D, self.K2, self.dist2)
            P1mtx = np.eye(3) @ camera_pose
            P2mtx = np.eye(3) @ stereo_pose
        else:
            pt1 = pts1.points2D
            pt2 = pts2.points2D
            P1mtx = self.K1 @ camera_pose
            P2mtx = self.K2 @ stereo_pose

        X = cv2.triangulatePoints(P1mtx, P2mtx, pt1, pt2)
        X = (X[:-1]/X[-1]).T[0]

        # pts3D = Points3D(points3D = X)
        return X # pts3D

    def triangulate_nView_points_Mono(self, views: np.ndarray, cam_poses: list[np.ndarray]) -> np.ndarray:

        # total_cameras = len(self.scene_point_2d_map[pt_index])
        total_cameras = views.shape[0]
        A = np.zeros((2*total_cameras, 4))

        # Read Hartley and Zisserman to see if we need the normalization factor??
        if self.dist1 is None:
            for i in range(views.shape[0]):
                cam, pt = views[i, 0], views[i, 1:]
                cam = int(cam)
                Pmat = self.K1 @ cam_poses[cam]

                row1 = pt[0]*Pmat[2, :] - Pmat[0, :]
                row2 = pt[1]*Pmat[2, :] - Pmat[1, :]

                A[2*i, :] = row1
                A[2*i + 1, :] = row2
        else: 
            for i in range(views.shape[0]):
                cam, pt = views[i, 0], views[i, 1:]
                Pmat = np.eye(3) @ cam_poses[cam]
                xUnd = cv2.undistortPoints(pt, self.K1, self.dist1)

                row1 = xUnd[0, 0, 0]*Pmat[2, :] - Pmat[0, :]
                row2 = xUnd[0, 0, 1]*Pmat[2, :] - Pmat[1, :]

                A[2*i, :] = row1
                A[2*i + 1, :] = row2

        U, S, V = np.linalg.svd(A)
        X = V[-1, :]
        X = (X[:-1]/X[-1]).T

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