import cv2
import numpy as np
import glob
from tqdm import tqdm
from .DataTypes.datatype import Points2D, Points3D, Calibration, CameraPose, Scene 

class GeometryProcessing:
    def __init__(self, calibration: Calibration | None = None):
        if calibration is not None:
            self.K = calibration.K1
        self.OPTIONS = ['essential', 'fundamental', 'homography', 'projective']
        self.FORMATS = ['full', 'partial', 'pair']
    
    def __call__(self, option: str, format: str, pts: list[list[Points2D]]) -> list[np.ndarray]:
        if option.lower() not in self.OPTIONS:
            message = 'Error: no such option exist. Use on of ' + str(self.OPTIONS)
            raise Exception(message)
        if format.lower() not in self.FORMATS:
            message = 'Error: no such format exist. Use on of ' + str(self.FORMATS)
            raise Exception(message)

        if format.lower() == self.FORMATS[0]:
            matrices = []

            if option.lower() == self.OPTIONS[0]:
                for i in tqdm(range(len(pts))):
                    scene = pts[i]
                    pts1, pts2 = scene[0], scene[1]
                    M = self.find_essential(pts1, pts2)
                    matrices.append(M)
                
            elif option.lower() == self.OPTIONS[1]:
                for i in tqdm(range(len(pts))):
                    scene = pts[i]
                    pts1, pts2 = scene[0], scene[1]
                    M = self.find_fundamental(pts1, pts2)
                    matrices.append(M)

            elif option.lower() in self.OPTIONS[2:]:
                for i in tqdm(range(len(pts))):
                    scene = pts[i]
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
        E, mask = cv2.findEssentialMat(pts1.points2D, pts2.points2D, self.K)

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

        self.K = calibration.K1
        self.cam_distortion = calibration.distort
        self.stereo = calibration.stereo

        self.camera_poses = CameraPose()

    def __call__(self, option:str, points:list[list[Points2D, Points2D]], 
                 points3D: Points3D | None = None, e_mat: list[np.ndarray] | None = None) -> list[np.ndarray]: #CameraPose:
        if option.lower() not in self.OPTIONS:
            message ='Error: no such option exist. Use on of ' + str(self.OPTIONS)
            raise Exception(message)
        
        cam_poses = [np.hstack((np.eye(3), np.zeros((3,1))))]

        if self.format == self.FORMAT[0]:
            if option.lower() == self.OPTIONS[0]:
                for i in tqdm(range(len(points))):
                    pts = points[i]
                    E = e_mat[i]
                    R, T = self.find_pose_essential(E, pts)

                    cam_poses.append(np.hstack((R, T)))
                
                #self.camera_poses.camera_pose = cam_poses
                return cam_poses #self.camera_poses
            elif option.lower() == self.OPTIONS[1]:
                message = 'Error: no such option exist. Use on of ' + str(self.OPTIONS)
                raise Exception(message)
        elif self.format == self.FORMAT[1]:
            if option.lower() == self.OPTIONS[0]:
                pts = points[0]
                R, T = self.find_pose_essential(e_mat, pts)

                self.camera_poses.camera_pose = [np.hstack((R, T))]
                return [np.hstack((R, T))] #self.camera_poses
            elif option.lower() == self.OPTIONS[1]:
                message = 'Error: no such option exist. Use on of ' + str(self.OPTIONS)
                raise Exception(message)

    def find_pose_essential(self, E: np.ndarray, pts: list[Points2D]):
        pts1 = pts[0].points2D
        pts2 = pts[1].points2D

        print(pts1.shape, pts1.dtype)
        print(pts2.shape, pts2.dtype)
        print(E.shape, E.dtype)
        print(self.K.shape, self.K.dtype)

        _, R, T, _ = cv2.recoverPose(points1 = pts1, points2=pts2, cameraMatrix=self.K, E = E)
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
            self.K = calibration.K1
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

class HomographyApplication:
    def __init__(self, image_path: str, output_path: str, format = 'full'):
        self.images = sorted(glob.glob(image_path + "\\*"))
        self.output_path = output_path
        self.FORMAT = ['full', 'partial']
        self.format = format.lower()

        if format.lower() not in self.FORMAT:
            message = 'Error: no such option exist. Use on of ' + str(self.FORMAT)
            raise Exception(message)
    

    def __call__(self, h_matrices: list[np.ndarray]) -> np.ndarray:
        if self.format == self.FORMAT[0]:
            img = cv2.imread(self.images[0])

            output_image = self.define_new_image(h_matrices, img)

            output_image = self.warp_image_all(h_matrices, output_image)

            return output_image
        elif self.format == self.FORMAT[1]:
            input_img = cv2.imread(self.images[0])
            dst_img = cv2.imread(self.images[1])

            output_image = self.warp_image_single(h_matrices, input_img, dst_img)

            return output_image
        else:
            if format.lower() not in self.FORMAT:
                message = 'Error: no such option exist. Use on of ' + str(self.FORMAT)
                raise Exception(message)
            
        
    def define_new_image(self, h_matrices:list[np.ndarray], image:np.ndarray):
        h, w, _ = image.shape()

        cornerPointsofFrame = np.array([[0,w,w,0],
                                        [0,0,h,h]],dtype=np.float32).T.reshape(-1,1,2)

        for i in range(len(h_matrices)):
            som = cv2.perspectiveTransform(cornerPointsofFrame, h_matrices[i])
            som1 = som[:,0,:].T
            for i in range(som1.shape[1]):
                x, y, = som1[:, i]
                if x_min > x: x_min = x

                if x > x_max: x_max = x

                if y_min > y: y_min = y

                if y > y_max: y_max = y
        
        newImg = np.zeros((int(np.ceil(y_max - y_min)), int(np.ceil(x_max - x_min))), dtype = np.uint8)

        return newImg
    
    def warp_image_all(self, h_matrices:list[np.ndarray], output_image: np.ndarray):

        image = cv2.imread(self.images[0])
        H = h_matrices[0]
        prev_img = cv2.warpPerspective(image, H, output_image.shape[:2], flags=cv2.INTER_LINEAR, borderValue = 0)
        for i in range(1, len(self.images)):
            image = cv2.imread(self.images[i])
            H = h_matrices[i]
            curr_img = cv2.warpPerspective(image, H, output_image.shape[:2], flags=cv2.INTER_LINEAR, borderValue = 0)

            prev_img = self.blend_image(prev_img, curr_img)
        
        return prev_img

        
    def warp_image_single(self, h_matrices:list[np.ndarray], input_image:np.ndarray, output_image: np.ndarray):
        H = h_matrices[0]
        output_mask = cv2.warpPerspective(input_image, H, output_image.shape[:2], flags=cv2.INTER_LINEAR, borderValue = 0)
        # TODO: Blend single images together here... (Would likley not need H_composed_matrices, so figure out what is really passsed into here)

        output = self.blend_image(output_mask, output_image)

        return output

    def blend_image(self, foreground: np.ndarray, background: np.ndarray) -> np.ndarray:
        # Set alpha (transparency) value (0.0 = fully transparent, 1.0 = fully opaque)
        alpha = 0.0

        # Blend images
        blended_image = cv2.addWeighted(background, 1 - alpha, foreground, alpha, 0)

        return blended_image
    

class GeometryComposition:
    def __init__(self, image_path: str):
        self.images = sorted(glob.glob(image_path + "\\*"))
        self.OPTIONS = ['pose', 'projective', 'homography']

        if len(self.images) < 1:
            message = 'Error: no such path exist. Use correct path for images'
            raise Exception(message)
        
    
    def __call__(self, option: str, matrices:list[np.ndarray] | None) -> list[np.ndarray]:
        new_composed_matrices = []

        if option.lower() == self.OPTIONS[0]:

            new_composed_matrices.append(matrices[0])
            for i in range(1, len(matrices)):
                new_pose = self.compose_cam_pose_matrices(new_composed_matrices[i-1], matrices[i])

                new_composed_matrices.append(new_pose)

            return new_composed_matrices
        
        elif option.lower() in self.OPTIONS[1:]:
            return self.compose_homog_matrices(matrices)
        else:
            message ='Error: no such option exist. Use on of ' + str(self.OPTIONS)
            raise Exception(message)
        

    def compose_cam_pose_matrices(self, mat1: np.ndarray, mat2:np.ndarray) -> np.ndarray:
        mat1 = np.vstack((mat1, [[0,0,0,1]]))
        mat2 = np.vstack((mat2, [[0,0,0,1]]))

        r_mat = (mat1 @ mat2)[:3, :]

        return r_mat
    
    def compose_homog_matrices(self, h_matrices: list[np.ndarray]) -> list[np.ndarray]:
        img = cv2.imread(self.images[0])
        h, w, _ = img.shape

        cornerPointsofFrame = np.array([[0,w,w,0],
                                        [0,0,h,h]],dtype=np.float32).T.reshape(-1,1,2)

        for i in range(len(h_matrices)):
            som = cv2.perspectiveTransform(cornerPointsofFrame, h_matrices[i])
            som1 = som[:,0,:].T
            for i in range(som1.shape[1]):
                x, y, = som1[:, i]
                if x_min > x: x_min = x

                if x > x_max: x_max = x

                if y_min > y: y_min = y

                if y > y_max: y_max = y

        w1 = int(abs(x_min))
        h1 = int(abs(y_min))
        cornerPointsofMosiac1 = np.array([[w1,w1+w,w1+w,w1],
                                        [h1,h1,h1+h,h1+h]],dtype=np.float32).T.reshape(-1,1,2)

        H_mosaic = cv2.getPerspectiveTransform(cornerPointsofFrame,cornerPointsofMosiac1)

        mosaic_H_mats = []
        for i in range(len(h_matrices)):
            newHM = H_mosaic @ h_matrices[i]
            mosaic_H_mats.append(newHM)

        return mosaic_H_mats
    
class GeometrySceneEstimate:
    def __init__(self, calibration: Calibration):
        self.CAM_SETTINGS = ['mono', 'stereo']

        # Load Calibration Data
        self.K = calibration.K1
        self.cam_distortion = calibration.distort
        if calibration.stereo:
            self.K2 = calibration.K2
            self.cam_distortion2 = calibration.distort2
            self.R12 = calibration.R12
            self.T12 = calibration.T12
            self.cam_setting = "stereo"
            #self.extrinsics = np.hstack((calibration.R12, calibration.T12))
        else:
            self.cam_setting = "mono"

        self.FORMATS = ['full', 'partial']

    def __call__(self, points: list[list[Points2D]], camera_poses: list[np.ndarray], format:str = 'full') -> Scene:
        if format.lower() not in self.FORMATS:
            message = 'Error: no such option exist. Use on of ' + str(self.FORMATS)
            raise Exception(message)
        if self.cam_setting.lower() not in self.CAM_SETTINGS:
            message = 'Error: no such option exist. Use on of ' + str(self.CAM_SETTINGS)
            raise Exception(message)

        # points_3d = []
        scene = Scene(cam_poses= camera_poses, representation="point cloud")
        if format.lower() == self.FORMATS[0]:
            if self.cam_setting.lower() in self.CAM_SETTINGS[0]: # Mono Cam Setting
                for i in range(len(points)):
                    pts1 = points[i][0]
                    pts2 = points[i][1]
                    cam_pose1 = camera_poses[i]
                    cam_pose2 = camera_poses[i + 1]

                    pts = self.triangulate_points_mono(pts1, pts2, [cam_pose1, cam_pose2])
                    print(pts.points3D.shape)
                    # points_3d.append(pts)
                    scene.update_3d_points(pts)
                # Construct 3D scene given points
                #scene = Scene(points3D = Points3D(points=points_3d), cam_poses= camera_poses, representation="point cloud")

                return scene
            elif self.cam_setting.lower() in self.CAM_SETTINGS[1]: # Stereo Cam Setting
                for i in range(len(points)):
                    pts1 = points[i][0].points2D # Right Camera
                    pts2 = points[i][1].points2D # Left Camera
                    cam_pose1 = camera_poses[i] # Left camera pose

                    pts = self.triangulate_points_stereo(pts1, pts2, [cam_pose1, cam_pose2])

                    # points_3d.append(pts)

                # scene = Scene(points3D = Points3D(points=points_3d), cam_poses= camera_poses, representation="point cloud")
                scene.update_3d_points(pts)
                return scene
        elif format.lower() == self.FORMATS[1]: # TODO: Think about removing this if-else completely and the format option. Need an input/output for all functions, and this doesn't seem good as a solution for single 3D point estimation
            if self.cam_setting.lower() in self.CAM_SETTINGS[0]:
                pts1 = points[0][0].points2D
                pts2 = points[0][1].points2D
                cam_pose1 = camera_poses[0]
                cam_pose2 = camera_poses[1]

                pts = self.triangulate_points_mono(pts1, pts2, [cam_pose1, cam_pose2])

                # points_3d.append(pts)

                #scene = Scene(points3D = Points3D(points=points_3d), cam_poses= camera_poses, representation="point cloud")
                scene.update_3d_points(pts)
                return scene
            elif self.cam_setting.lower() in self.CAM_SETTINGS[1]:
                pts1 = points[0][0].points2D # Right Camera
                pts2 = points[0][1].points2D # Left Camera
                cam_pose1 = camera_poses[0] # Left camera pose

                pts = self.triangulate_points_stereo(pts1, pts2, [cam_pose1, cam_pose2])

                # points_3d.append(pts)

                #scene = Scene(points3D = Points3D(points=points_3d), cam_poses= camera_poses, representation="point cloud")
                scene.update_3d_points(pts)
                return scene
    
    # Triangulation of points (Monocular Camera)
    def triangulate_points_mono(self, pts1: Points2D, pts2: Points2D, camera_pose: list) -> Points3D:
        xU1 = cv2.undistortPoints(pts1.points2D, self.K, self.cam_distortion)
        xU2 = cv2.undistortPoints(pts2.points2D, self.K, self.cam_distortion)
        
        P1mtx = np.eye(3) @ camera_pose[0]
        P2mtx = np.eye(3) @ camera_pose[1]

        X = cv2.triangulatePoints(P1mtx, P2mtx, xU1, xU2)
        X = (X[:-1]/X[-1]).T 

        pts3D = Points3D(points = X)
        return pts3D

    # Triangulation of points (Stereo Camera)
    def triangulate_points_stereo(self, pts1: Points2D, pts2: Points2D, camera_pose) -> np.ndarray:
        xU1 = cv2.undistortPoints(pts1.points2D, self.K, self.cam_distortion)
        xU2 = cv2.undistortPoints(pts2.points2D, self.K2, self.cam_distortion2)
        
        Rot_R = self.R12 @ camera_pose[:, :3]
        Trans_R = self.R12 @ camera_pose[:, 3:] + self.T12
        stereo_pose = np.hstack((Rot_R, Trans_R))
        P1mtx = np.eye(3) @ camera_pose
        P2mtx = np.eye(3) @ stereo_pose

        X = cv2.triangulatePoints(P1mtx, P2mtx, xU1, xU2)
        X = (X[:-1]/X[-1]).T

        # pts3D = Points3D(points3D = X)
        return X # pts3D

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