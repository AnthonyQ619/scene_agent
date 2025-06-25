import numpy as np
import cv2
from dataclasses import dataclass

@dataclass
class Points2D:
    points2D: np.ndarray # 2xN
    descriptors: np.ndarray # MxN

    def update_2D_points_index(self, indices: list[int]) -> None:
        self.points2D = self.points2D[indices]
        self.descriptors = self.descriptors[indices]

    def update_2D_points_values(self, points2D) -> None:
        self.points2D = points2D
        self.descriptors = None

    def splice_2D_points(self, indices: list[int]) -> dict[str: np.ndarray]:
        points2D = self.points2D[indices]
        descriptors = self.descriptors[indices]

        return {"points2D" : points2D, "descriptors": descriptors}

    def set_inliers(self, mask: np.ndarray) -> dict[str: np.ndarray]:
        points2D = self.points2D[mask.ravel() == 1]
        descriptors = self.descriptors[mask.ravel() == 1]

        return {"points2D" : points2D, "descriptors": descriptors}

@dataclass
class PointsMatched:
    data_matrix: np.ndarray # Data Structure to store corresponding points. In the form of Nx4 -> [track_id, frame_num, x, y]
    track_map: dict         # Used to aid in the feature matching process.
    point_count: int        # Based on track_id max count -> tells us how many 3D points exist
    
    def __init__(self,  data_matrix: np.ndarray | None = None, 
                        track_map: dict = {},
                        point_count: int = 0):
        self.data_matrix = data_matrix
        self.track_map = track_map
        self.point_count = point_count

    def set_matched_matrix(self, data: list[list]) -> None:
        self.data_matrix = np.array(data)

    def access_point3D(self, track_id: int) -> np.ndarray: # 3D point returns a Nx3 Matrix of (Cam, x, y)
        indicies = np.where(self.data_matrix[:, 0] == track_id)[0]
        return self.data_matrix[indicies, 1:] 


@dataclass
class Points3D:
    points3D: np.ndarray    # Point position in 3D space [x, y, z] : Nx3
    color: np.ndarray       # Point Color [r, g, b] : Nx3
    
    def __init__(self,  points: list[np.ndarray] | None = np.array([0.0, 0.0, 0.0]), 
                        color: list[np.ndarray] | None = np.array([0, 0, 0])):
        self.points3D = np.array(points)
        self.color = np.array(color)

    def update_points(self, points: list[np.ndarray], color: list[np.ndarray] | None = None) -> None:
        if self.points3D.shape[0] == 1:
            self.points3D = np.array(points)
            if color is not None:
                self.color = np.array(color)
        else:
            new_points = np.array(points)
            self.points3D = np.vstack((self.points3D, new_points))
            if color is not None:
                self.color = np.vstack((self.color,np.array(color)))


@dataclass
class CameraPose:
    camera_pose: list[np.ndarray]   # Camera pose for each corresponding frame
    rotations: list[np.ndarray]     # Rotation matrices for each corresponding frame (Derived from camera_pose)
    translations: list[np.ndarray]  # Translation matrices for each corresponding frame (Derived from camera_pose)

    def __init__(self, cam_poses: list[np.ndarray] | None = None, 
                 rot: list[np.ndarray] | None = [], trans: list[np.ndarray] | None = []):
        self.camera_pose = cam_poses
        self.rotations = rot
        self.translations = trans

    def set_rotations(self) -> None:
        if self.camera_pose is None:
            message = 'Camera Poses have not been set. Please initate camera poses using the module "CameraPoseEstimator"'
            raise Exception(message)
        
        for i in range(self.camera_pose):  
            rot = self.camera_pose[i][:,:3]
            self.rotations.append(rot)
    
    def set_translation(self) -> None:
        if self.camera_pose is None:
            message = 'Camera Poses have not been set. Please initate camera poses using the module "CameraPoseEstimator"'
            raise Exception(message)
        
        for i in range(self.camera_pose):  
            trans = self.camera_pose[i][:,3:]
            self.translations.append(trans)

    def set_rot_2_angle_axis(self) -> None:
        if self.camera_pose is None:
            message = 'Camera Poses have not been set. Please initate camera poses using the module "CameraPoseEstimator"'
            raise Exception(message)
        if len(self.rotations) > 0:
            self.rotations = []
        
        for i in range(self.camera_pose):  
            rot = self.camera_pose[i][:,:3]
            rotation_vector, _ = cv2.Rodrigues(rot)
            self.rotations.append(rotation_vector)

    def set_rot_2_quaternion(self) -> None:
        if self.camera_pose is None:
            message = 'Camera Poses have not been set. Please initate camera poses using the module "CameraPoseEstimator"'
            raise Exception(message)
        if len(self.rotations) > 0:
            self.rotations = []

        def quaternion(rotation: np.ndarray) -> np.ndarray:
            trace = rotation.trace()

            if trace > 0.0:
                s = np.sqrt(trace + 1.0)
                v1 = s*0.5
                s = 0.5 / s
                quat = np.array([v1,
                                rotation[2,1] - rotation[1,2]*s,
                                rotation[0,2] - rotation[2,0]*s,
                                rotation[1,0] - rotation[0,1]*s])
                return quat
            else:
                if rotation[0,0] < rotation[1,1]:
                    if rotation[1,1] < rotation[2,2]:
                        i = 2
                    else:
                        i = 1
                else:
                    if rotation[0,0] < rotation[2,2]:
                        i = 2
                    else:
                        i = 0
                j = (i + 1) % 3
                k = (i + 2) % 3

                quat = np.zeros((4,1))
                s = np.sqrt(rotation[i,i] - rotation[j,j] - rotation[k,k] + 1.0)

                quat[i] = s * 0.5
                s = 0.5 / s
                quat[3] = rotation[k,j] - rotation[j,k]*s
                quat[j] = rotation[j,i] + rotation[i,j]*s
                quat[k] = rotation[k,i] + rotation[i,k]*s

                return quat
            
        for i in range(self.camera_pose):  
            rot = self.camera_pose[i][:,:3]
            quat = quaternion(rot)
            self.rotations.append(quat)

@dataclass
class Scene:
    points3D: Points3D        # Set of 3D points
    cam_poses: list[np.ndarray] # Should be formatted as a 3x4 matrix
    point_to_pose: np.ndarray   # List of corresponding camera poses to 3D points
    representation: str         # Represnetation of the scene (Future use cases here)

    def __init__(self, points3D: Points3D | None = Points3D(), cam_poses: list[np.ndarray] = [], 
                 point_to_pose: np.ndarray | None = None, representation: str = "point cloud"):
        self.SceneRepresentation = ["point cloud", "mesh", 'NeRF']

        
        self.points3D = points3D
        self.cam_poses = cam_poses
        self.point_to_pose = point_to_pose
        self.representation = representation

        assert(self.representation in self.SceneRepresentation)

    def update_cam_pose(self, cam_pose: np.ndarray) -> None:
        self.cam_poses.append(cam_pose)

    def update_3d_points(self, points: Points3D) -> None:
        self.points3D.update_points(points=points.points3D)

@dataclass
class Calibration:
    K1: np.ndarray # Camera intrinsics 
    K2: np.ndarray # 2nd Camera Intrinsics
    distort: np.ndarray # Camera 1 Distortion
    distort2: np.ndarray # Camera 2 Distortion
    R12: np.ndarray # Rotation of Stereo Camera
    T12: np.ndarray # Translation of Stereo Camera (Baseline)
    stereo: bool

    def __init__(self, K1 = np.eye(3),  R: np.ndarray | None = None, T: np.ndarray | None = None, 
                 stereo: bool = False, K2: np.ndarray | None = None, dist: np.ndarray = np.zeros((1, 5)),
                 dist2: np.ndarray = np.zeros((1, 5))):
        self.K1 = K1 # If Stereo, this will be a lsit of calibration vals
        self.distort = dist
        self.stereo = stereo

        if self.stereo:
            self.K2 = K2
            self.distort2 = dist2
            self.R12 = R # Rotation from camera 1 to camera 2
            self.T12 = T # Baseline between camera 1 and camera 2

    def get_intrinsics(self):
        if self.stereo:
            return self.K1, self.K2
        
        return self.K1
    
    def get_extrinsics(self):
        assert(self.stereo)
        
        return self.R12, self.T12
    


