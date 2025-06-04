import cv2
import numpy as np
from DataTypes.datatype import Points2D, Points3D, Calibration

class GeometryProcessing:
    def __init__(self, calibration: Calibration):
        self.K = calibration.K
        self.OPTIONS = ['essential', 'fundamental', 'homography', 'projective']
        self.FORMATS = ['full', 'partial', 'pair']
    
    def __call__(self, option: str, format: str, pts: list[list[Points2D]]) -> list[np.ndarray]:
        if option.lower() not in self.OPTIONS:
            message = 'Error: no such option exist. Use on of ' + str(self.OPTIONS)
            raise Exception(message)
        if format.lower not in self.FORMATS:
            message = 'Error: no such option exist. Use on of ' + str(self.FORMATS)
            raise Exception(message)

        if format.lower() == self.FORMATS[0]:
            matrices = []

            if option.lower() == self.OPTIONS[0]:
                for scene in pts:
                    pts1, pts2 = scene[0], scene[1]
                    M = self.find_essential(pts1, pts2)
                    matrices.append(M)
                
            elif option.lower() == self.OPTIONS[1]:
                for scene in pts:
                    pts1, pts2 = scene[0], scene[1]
                    M = self.find_fundamental(pts1, pts2)
                    matrices.append(M)

            elif option.lower() in self.OPTIONS[2:]:
                for scene in pts:
                    pts1, pts2 = scene[0], scene[1]
                    M = self.find_projective(pts1, pts2)
                    matrices.append(M)
            
            return matrices
        elif format.lower() in self.FOMRATS[1:]:
            pts1, pts2 = pts[0][0], pts[0][1]
            if option.lower() == self.OPTIONS[0]:
                M = self.find_essential(pts1, pts2)
            if option.lower() == self.OPTIONS[1]:
                M = self.find_fundamental(pts1, pts2)
            if option.lower() in self.OPTIONS[2:]:
                M = self.find_projective(pts1, pts2)

            return [M]

    def find_fundamental(self, pts1: Points2D, pts2: Points2D) -> np.ndarray:
        F, mask = cv2.findFundamentalMat(pts1.points2D, pts2.points2D, cv2.FM_LMEDS)

        # Could update points2D to inlier points with Mask

        return F
    
    def find_essential(self, pts1: Points2D, pts2: Points2D) -> np.ndarray:
        E = cv2.findEssentialMat(pts1.points2D, pts2.points2D, self.K)

        return E
    
    def find_projective(self, pts1: Points2D, pts2: Points2D) -> np.ndarray:
        MIN_MATCH_COUNT = 10

        if pts1.points2D.shape[0] > MIN_MATCH_COUNT:
            H, mask = cv2.findHomography(pts1.points2D, pts2.points2D, cv2.RANSAC, 5.0)
            # Could update points2D to inlier points with Mask

            return H
        else:
            message = "Error: Not enough points for computation. Requires: " + str(MIN_MATCH_COUNT)
            raise Exception(message)



class CameraPoseEstimator:
    def __init__(self, calibration: Calibration, format = 'full'):
        self.OPTIONS = ['essential', 'pnp']
        self.FORMAT = ['full', 'partial']
        self.format = format

        if format.lower() not in self.FORMAT:
            message = 'Error: no such option exist. Use on of ' + str(self.FORMAT)
            raise Exception(message)

        self.K = calibration.K
        self.cam_distortion = calibration.distort
        self.stereo = calibration.stereo

    def __call__(self, option:str, points:list[list[Points2D, Points2D]], 
                 points3D: list[Points3D] | None, e_mat: np.ndarray | None) -> list[tuple[np.ndarray, np.ndarray]]:
        if option.lower() not in self.OPTIONS:
            message ='Error: no such option exist. Use on of ' + str(self.OPTIONS)
            raise Exception(message)
        
        cam_poses = [(np.eye(3), np.zeros((1,3)))]

        if self.format == self.FORMAT[0]:
            if option.lower() == self.OPTIONS[0]:
                for pts in points:
                    R, T = self.find_pose_essential(e_mat, pts)

                    cam_poses.append((R, T))
            elif option.lower() == self.OPTIONS[1]:
                message = 'Error: no such option exist. Use on of ' + str(self.OPTIONS)
                raise Exception(message)
        elif self.format == self.FORMAT[1]:
            if option.lower() == self.OPTIONS[0]:
                pts = points[0]
                R, T = self.find_pose_essential(e_mat, pts)

                return [(R, T)]
            elif option.lower() == self.OPTIONS[1]:
                message = 'Error: no such option exist. Use on of ' + str(self.OPTIONS)
                raise Exception(message)

    def find_pose_essential(self, E: np.ndarray, pts: list[Points2D]):
        pts1 = pts[0].points2D
        pts2 = pts[1].points2D

        R, T = cv2.recoverPose(E = E, points1= pts1, points2= pts2, cameraMatrix=self.K)

        return R, T
    
    def find_pose_3D(self, points3D: Points3D, points2D: Points2D):
        pts3D = points3D.points3D 
        pts2D = points2D.points2D 
        R, T = cv2.solvePnP(pts3D, pts2D, self.K, self.cam_distortion)

        return R, T
        

class ImageRectifier:
    def __init__(self, calibration: Calibration | None, format: str):
        self.FORMAT = ['full', 'partial']
        self.format = format

        if calibration is not None:
            self.K = calibration.K
            self.K2 = calibration.K2
            self.cam_distortion = calibration.distort
            self.cam_distortion2 = calibration.distort2
            self.R = calibration.R12
            self.T = calibration.T12
            self.stereo = calibration.stereo   

        if not self.stereo:
            message = 'Error: Camera settings must set Stereo to TRUE'
            raise Exception(message)

    def __call__(self, calibrated: bool, image_shape: tuple[int,int], points: list[list[Points2D, Points2D]] | None, F_mats: list[np.ndarray] | None) -> list[list[np.ndarray]]:
        if calibrated:
            R1, R2, P1, P2, _ = cv2.stereoRectify(self.K, self.cam_distortion, self.K2,
                                                  self.cam_distortion2, image_shape, self.R, self.T)
            
            return [[R1, R2], [P1, P2]]
        else:
            if self.format.lower() == self.FORMAT[0]:
                rectify_mats = []
                for i in range(len(points)):
                    pts = points[i]
                    pts1, pts2 = pts[0].points2D, pts[1].points2D
                    F = F_mats[i]
                    H1, H2 = cv2.stereoRectifyUncalibrated(pts1, pts2, F)
                    rectify_mats.append([H1, H2])
            elif self.format.lower() == self.FORMAT[1]:
                pts1, pts2 = points[0][0].points2D, points[0][1].points2D
                F = F_mats[0]
                H1, H2 = cv2.stereoRectifyUncalibrated(pts1, pts2, F)
                return[[H1, H2]]

class GeometryComposition:
    def __init__(self):
        pass

class GeometrySceneEstimate:
    def __init__(self, calibration: Calibration):
        self.K = calibration.K
        self.cam_distortion = calibration.distort
        if calibration.stereo:
            self.K2 = calibration.K2
            self.cam_distortion2 = calibration.distort2
            self.R12 = calibration.R12
            self.T12 = calibration.T12
            #self.extrinsics = np.hstack((calibration.R12, calibration.T12))

        self.CAM_SETTINGS = ['mono', 'stereo']
        self.FORMATS = ['full', 'partial']

    def __call__(self, cam_setting: str, points: list[list[Points2D, Points2D]], camera_poses: list, format = 'full') -> Points3D:
        if format.lower() not in self.FORMATS:
            message = 'Error: no such option exist. Use on of ' + str(self.FORMATS)
            raise Exception(message)
        if cam_setting.lower() not in self.CAM_SETTINGS:
            message = 'Error: no such option exist. Use on of ' + str(self.CAM_SETTINGS)
            raise Exception(message)

        if format.lower() == self.FORMATS[0]:
            if cam_setting.lower() in self.CAM_SETTINGS[0]: # Mono Cam Setting
                
            elif cam_setting.lower() in self.CAM_SETTINGS[1]: # Stereo Cam Setting
        elif format.lower() == self.FORMATS[1]:
            if cam_setting.lower() in self.CAM_SETTINGS[0]:
            elif cam_setting.lower() in self.CAM_SETTINGS[1]:

    
    # Triangulation of points (Monocular Camera)
    def triangulate_points_mono(self, pts1: Points2D, pts2: Points2D, camera_pose: list) -> Points3D:
        xU1 = cv2.undistortPoints(pts1.points2D, self.K, self.cam_distortion)
        xU2 = cv2.undistortPoints(pts2.points2D, self.K2, self.cam_distortion2)
        
        P1mtx = np.eye(3) @ camera_pose[0]
        P2mtx = np.eye(3) @ camera_pose[1]

        X = cv2.triangulatePoints(P1mtx, P2mtx, xU1, xU2)
        X = (X[:-1]/X[-1]).T 

        pts3D = Points3D(points3D = X)
        return pts3D

    # Triangulation of points (Stereo Camera)
    def triangulate_points_stereo(self, pts1: Points2D, pts2: Points2D, camera_pose) -> Points3D:
        xU1 = cv2.undistortPoints(pts1.points2D, self.K, self.cam_distortion)
        xU2 = cv2.undistortPoints(pts2.points2D, self.K2, self.cam_distortion2)
        
        Rot_R = self.R12 @ camera_pose[:, :3]
        Trans_R = self.R12 @ camera_pose[:, 3:] + self.T12
        stereo_pose = np.hstack((Rot_R, Trans_R))
        P1mtx = np.eye(3) @ camera_pose
        P2mtx = np.eye(3) @ stereo_pose

        X = cv2.triangulatePoints(P1mtx, P2mtx, xU1, xU2)
        X = (X[:-1]/X[-1]).T

        pts3D = Points3D(points3D = X)
        return pts3D

    def triangulate_nView_points(self, pt_index):

        # total_cameras = len(self.scene_point_2d_map[pt_index])
        total_cameras = len(self.scene_2d_pts[pt_index])
        A = np.zeros((4*total_cameras, 4))

        index = 0

        # Read Hartley and Zisserman to see if we need the normalization factor??
        for cam, pt in self.scene_2d_pts[pt_index].items():

            PmatLeft = np.eye(3) @ self.camera_poses[cam][0]
            PmatRight = np.eye(3) @ self.camera_poses[cam][1]

            xU1 = cv2.undistortPoints(np.hstack(pt[0]), self.cam_left, self.dist_left)
            xU2 = cv2.undistortPoints(np.hstack(pt[1]), self.cam_right, self.dist_right)

            row1 = xU1[0, 0, 0]*PmatLeft[2, :] - PmatLeft[0, :]
            row2 = xU1[0, 0, 1]*PmatLeft[2, :] - PmatLeft[1, :]
            row3 = xU2[0, 0, 0]*PmatRight[2, :] - PmatRight[0, :]
            row4 = xU2[0, 0, 1]*PmatRight[2, :] - PmatRight[1, :]

            A[4*index, :] = row1
            A[4*index + 1, :] = row2
            A[4*index + 2, :] = row3
            A[4*index + 3, :] = row4

        index += 1
        U, S, V = np.linalg.svd(A)
        X = V[-1, :]
        X = (X[:-1]/X[-1]).T

        return X