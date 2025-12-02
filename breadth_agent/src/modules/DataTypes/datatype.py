import numpy as np
import torch
import cv2
import os
import theseus as th
import theseus.utils.examples as theg
from theseus.utils.examples.bundle_adjustment.data import Camera, Observation

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union


@dataclass
class CameraData:
    # --- Image Data ---
    image_list: List[np.ndarray]        
    image_shape_old: Tuple[int, int]    # (width, height)
    image_scale: Tuple[float, float]    # (width, height)

    # --- Calibration ---
    intrinsics: List[np.ndarray]  = field(default_factory=[np.zeros((3,3))])   # Camera Intrinsics (3x3 K Matrix) 
    distortions: List[np.ndarray] = field(default_factory=[np.zeros((1, 5))])     # Camera Distortions (OpenCV convention of 1x5 [k1, k2, p1, p2, k3])

    stereo: bool = False
    multi_cam: bool = False
    extrinsic: Optional[np.ndarray] = None    # Rotation | Translation of Stereo Camera


    def update_K(self, cam_idx: int, img_scale: Tuple[float, float]):
        # Assume Monocular Camera for now with OpenCV calibration convention (Wide belief)
        # Meaning: Skew Parameter will be zero in these cases.
        # height_scale, width_scale = img_scale[:]
        assert (cam_idx >= 0 and cam_idx < len(self.intrinsics)), "Assumed to be more cameras. Intrinsics likely not loaded properly. Use CameraDataManager to load calibration or estimate through VGGTPoseEsimtation."
        width_scale, height_scale = img_scale[:]

        self.intrinsics[cam_idx][0,0] = width_scale * self.intrinsics[cam_idx][0,0]   # fx x width_scale = fx'
        self.intrinsics[cam_idx][1,1] = height_scale * self.intrinsics[cam_idx][1,1]  # fy x height_scale = fy'
        self.intrinsics[cam_idx][0,2] = width_scale * self.intrinsics[cam_idx][0,2]   # cx x width_scale = cx'
        self.intrinsics[cam_idx][1,2] = height_scale * self.intrinsics[cam_idx][1,2]  # cy x height_scale = cy'

    def update_calibration(self, img_scale: Tuple[float, float]):
        if self.intrinsics is not None:
            for cam_idx in range(len(self.intrinsics)):
                width_scale, height_scale = img_scale[:]

                self.intrinsics[cam_idx][0,0] = width_scale * self.intrinsics[cam_idx][0,0]   # fx x width_scale = fx'
                self.intrinsics[cam_idx][1,1] = height_scale * self.intrinsics[cam_idx][1,1]  # fy x height_scale = fy'
                self.intrinsics[cam_idx][0,2] = width_scale * self.intrinsics[cam_idx][0,2]   # cx x width_scale = cx'
                self.intrinsics[cam_idx][1,2] = height_scale * self.intrinsics[cam_idx][1,2]  # cy x height_scale = cy'

    def apply_new_calibration(self, 
                              intrinsics: List[np.ndarray],
                              distortion: List[np.ndarray] | None = None):
        if len(intrinsics) > 1:
            if distortion is None:
                self.intrinsics = intrinsics
                self.distortions = [np.zeros(1, 5)]*len(intrinsics) # Ensure distortion Param exists with equivalent list size
            else:
                self.intrinsics = intrinsics
                self.distortions = distortion
            self.multi_cam = True
        else:
            self.intrinsics = intrinsics[0]
            self.distortions = distortion[0]

    def get_K(self, cam_idx: int):
        #assert(self.intrinsics is not None), "Calibration Data is not properly loaded. Ensure necessary steps are taken to generate calibration through VGGT tools or calibration is properly read with CameraDataManager."
        if self.intrinsics is None:
            return None
        elif self.stereo:
            return self.intrinsics[0], self.intrinsics[1]
        elif self.multi_cam:
            return self.intrinsics
        else:
            return self.intrinsics[0]
    
    def get_distortion(self):
        #assert(self.intrinsics is not None), "Calibration Data is not properly loaded. Ensure necessary steps are taken to generate calibration through VGGT tools or calibration is properly read with CameraDataManager."
        if self.intrinsics is None:
            return None
        elif self.stereo:
            return self.distortions[0], self.distortions[1]
        elif self.multi_cam:
            return self.distortions
        else:
            return self.distortions[0]
    
@dataclass
class Points2D:
    points2D: np.ndarray        # Nx2 [np.float32] (Mono or Left image)
    # points2D_stereo: np.ndarray  # Nx2 [np.float32] (Right Image if Stereo=True)
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
        descriptors = self.descriptors[mask.ravel() == 1]
        scores = self.scores[mask.ravel() == 1]

        return {"points2D" : points2D, "descriptors": descriptors, "scores": scores, 'image_size': self.image_size, 'reshape_scale': self.reshape_scale}

@dataclass
class PointsMatched:
    data_matrix: np.ndarray             # Data Structure to store corresponding points. In the form of Nx4 -> [track_id, frame_num, x, y]
    pairwise_matches: list[np.ndarray]  # Data Structure to store Pairwise feature matches. Form of Nx4 -> [x1, y1, x2, y2]
    track_map: dict                     # Used to aid in the feature matching process.
    point_count: int                    # Based on track_id max count -> tells us how many 3D points exist
    multi_view: bool                    # Determine if Pairwise/Feature Matching
    image_size: np.ndarray              # 1x2 [np.int64] (Simply Image Shape: (W, H))
    image_scale: list[float]            # [W_scale, H_scale] if image is resized
    stereo_cam: bool                    # Deterine if the camera utilized is a stereo camera for feature matching/tracking

    def __init__(self,  data_matrix: np.ndarray | None = None, 
                        pairwise_matches: list[np.ndarray] | None = None,
                        multi_view: bool = False,
                        track_map: dict = {},
                        point_count: int = 0,
                        image_size: np.ndarray | None = None,
                        image_scale: list[float] = [1.0, 1.0]):
        self.data_matrix = data_matrix
        self.pairwise_matches = pairwise_matches
        self.track_map = track_map
        self.point_count = point_count
        self.image_size = image_size
        self.image_scale = image_scale

        self.multi_view = multi_view

    # Feature Tracking Functions (Multi-View)
    def set_matched_matrix(self, data: list[list]) -> None:
        self.data_matrix = np.array(data)
        self.multi_view = True

    def access_point3D(self, track_id: int) -> np.ndarray: # 3D point returns a Nx3 Matrix of (Cam, x, y)
        indicies = np.where(self.data_matrix[:, 0] == track_id)[0]
        return self.data_matrix[indicies, 1:] 

    # Feature Matching (Two-View)
    def set_matching_pair(self, data:np.ndarray) -> None:
        assert (data.shape[1] == 4), "Not enough data stored in column. Each row must contain: [x1, y1, x2, y2] of matching feature pair."
        self.pairwise_matches.append(data)
        self.multi_view = False

    def access_matching_pair(self, pair_index: int) -> tuple[np.ndarray, np.ndarray]:
        data = self.pairwise_matches[pair_index]

        pts1 = data[:, :2]
        pts2 = data[:, 2:]

        return pts1, pts2

@dataclass
class Points3D:
    points3D: np.ndarray    # Point position in 3D space [x, y, z] : Nx3
    color: np.ndarray       # Point Color [r, g, b] : Nx3
    
    def __init__(self,  points: list[np.ndarray] | None = None, #np.array([[0.0, 0.0, 0.0]]), 
                        color: list[np.ndarray] | None = np.array([0, 0, 0])):
        if points is None:
            self.points3D = None
        else:
            self.points3D = np.array(points)

        self.color = np.array(color)

    def update_points(self, points: list[np.ndarray], color: list[np.ndarray] | None = None) -> None:
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

@dataclass
class CameraPose:
    camera_pose: list[np.ndarray]   # Camera pose for each corresponding frame. Each pose is 3x4 (R, T)
    rotations: list[np.ndarray]     # Rotation matrices for each corresponding frame (Derived from camera_pose)
    translations: list[np.ndarray]  # Translation matrices for each corresponding frame (Derived from camera_pose)

    def __init__(self, cam_poses: list[np.ndarray] | None = [], 
                 rot: list[np.ndarray] | None = [], trans: list[np.ndarray] | None = []):
        self.camera_pose = cam_poses
        self.rotations = rot
        self.translations = trans

    def set_rotations(self) -> None:
        if self.camera_pose is None:
            message = 'Camera Poses have not been set. Please initate camera poses using the module "CameraPoseEstimator"'
            raise Exception(message)
        
        for i in range(len(self.camera_pose)):  
            rot = self.camera_pose[i][:,:3]
            self.rotations.append(rot)
    
    def set_translation(self) -> None:
        if self.camera_pose is None:
            message = 'Camera Poses have not been set. Please initate camera poses using the module "CameraPoseEstimator"'
            raise Exception(message)
        
        for i in range(len(self.camera_pose)):  
            trans = self.camera_pose[i][:,3:]
            self.translations.append(trans)
    
    def get_translations_np(self) -> np.ndarray: # Returns an Nx3 Array, where each row is the translation vector of the pose
        if self.camera_pose is None:
            message = 'Camera Poses have not been set. Please initate camera poses using the module "CameraPoseEstimator"'
            raise Exception(message)
        
        temp_trans = []
        for i in range(len(self.camera_pose)):  
            trans = self.camera_pose[i][:,3:]
            temp_trans.append(trans.T[0])

        return np.array(temp_trans)

    def set_rot_2_angle_axis(self) -> None:
        if self.camera_pose is None:
            message = 'Camera Poses have not been set. Please initate camera poses using the module "CameraPoseEstimator"'
            raise Exception(message)
        if len(self.rotations) > 0:
            self.rotations = []
        
        for i in range(len(self.camera_pose)):  
            rot = self.camera_pose[i][:,:3]
            rotation_vector, _ = cv2.Rodrigues(rot)
            self.rotations.append(rotation_vector)

    # Returns an Nx3 array, where each row is the rodrigues rotation vector for the estimated poses.
    def get_rot_2_angle_axis(self) -> np.ndarray: #list[np.ndarray]:
        if self.camera_pose is None:
            message = 'Camera Poses have not been set. Please initate camera poses using the module "CameraPoseEstimator"'
            raise Exception(message)
        
        rot_vecs = []
        for i in range(len(self.camera_pose)):  
            rot = self.camera_pose[i][:,:3].astype(np.float32)
            rotation_vector, _ = cv2.Rodrigues(rot)
            rot_vecs.append(rotation_vector.T[0])
        
        rot_vecs_np = np.array(rot_vecs)
        return rot_vecs_np

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
            
        for i in range(len(self.camera_pose)):  
            rot = self.camera_pose[i][:,:3]
            quat = quaternion(rot)
            self.rotations.append(quat)

@dataclass
class BundleAdjustmentData:
    num_cameras: int
    num_points: int
    num_observations: int
    camera_int: list[np.ndarray]    # Camera Intrinsics, for each cam_i in list contains calibration matrix
    observations: np.ndarray        # Mx4 matrices for each point observation where M=num_of_observations, and each row = [frame, 3d_point_ind, norm_x, norm_y]
    cameras: list[np.ndarray]       # List of cameras, with each row containing R(rodriguez), T, f, K1, K2 (k1 and k2 radial distortion)
    points: np.ndarray              # Nx3 matrix containing the X, Y, Z coordinates of points 
    dataset: theg.BundleAdjustmentDataset 

    def __init__(self, 
                 num_cameras: int, 
                 num_points: int,
                 num_observations: int,
                 observations: list[np.ndarray],
                 cameras: CameraPose,
                 points: np.ndarray,
                 dist: list[np.ndarray],
                 mono: bool = True):
        self.num_cameras = num_cameras
        self.num_points = num_points
        self.num_observations = num_observations
        self.mono = mono

        self.observations = self._setup_observations(observations, num_observations)
        self.cam_data = self._setup_cameras(cameras, dist)
        self.points = points

        self._create_BADataset() # Write BAL file down
    
    def _create_BADataset(self) -> None:
        # Fixed File Name
        # file_path = os.path.realpath(__file__).split('\\')
        # file_path[0] = file_path[0] + "\\"
        # chosen_dir = "scene_data"
        # file_name = "bal_data.txt"
        # # Path to File
        # file_path = os.path.join(*file_path[:7], "results", chosen_dir, file_name)
        # print(self.points.shape)

        # for p in self.points[1, :]:
        #             print(f"{p}")
        # for p in self.points[-1, :]:
        #             print(f"{p}")

        print("Current Dataset Features")
        print("Number of Observations:", self.observations.shape[0])
        print("Number of Observations:", self.num_observations)
        print("Number of Cameras in the Scene:", self.num_cameras)
        print("Number of Cameras in the Scene:", self.cam_data.shape[0])
        print("Number of 3D Points in the Scene:", self.num_points)
        observations = []
        cameras = []
        points = []

        # with open(file_path, "w+") as bal_file:
            # print(f"{self.num_cameras} {self.num_points} {self.num_observations}", file=bal_file)
        # Track Cameras in Scene
        # camera_tracker = set()

        # Write Observations
        for i in range(self.observations.shape[0]):
            ci, pi, x, y = self.observations[i, :] # ci = camera index, pi = point index (3D point), x,y = feature point
            # print(f"{int(ci)} {int(pi)} {x} {y}", file=bal_file)
            # For Theseus Bundle Adjustment Dataset
            feat = th.Point2(
                tensor=torch.tensor(
                    [float(x), float(y)], dtype=torch.float64
                ).unsqueeze(0),
                name=f"Feat{i}",
            )
            observations.append(
                Observation(
                    camera_index=int(ci),
                    point_index=int(pi),
                    image_feature_point=feat,
                )
            )

            # # Track Cameras in use
            # camera_tracker.add(int(ci))

        # Write Camera Params
        for cam in range(self.cam_data.shape[0]):
            params = []
            # if cam in camera_tracker:
            for p in self.cam_data[cam, :]:
                # print(f"{p}", file=bal_file)
                # For Theseus Bundle Adjustment Dataset
                params.append(float(p))
            cameras.append(Camera.from_params(params, name=f"Cam{cam}"))
        # print("NUMBER OF CAMERAS", len(cameras))
        # Write Point Data (3D points)
        for pt in range(self.points.shape[0]):
            params = []
            for p in self.points[pt, :]:
                # print(f"{p}", file=bal_file)
                # For Theseus Bundle Adjustment Dataset
                params.append(float(p))

            points.append(
                th.Point3(
                    tensor=torch.tensor(params, dtype=torch.float64).unsqueeze(0),
                    name=f"Pt{pt}",
                )
            )
                
            
        # Create Theseus variable for Bundle Adjustment Dataset
        self.dataset =  theg.BundleAdjustmentDataset(
                            cameras=cameras,
                            points=points,
                            observations=observations,
                            gt_cameras=None,
                            gt_points=None,
                        )

    # def _create_BADataset(self) -> theg.BundleAdjustmentDataset:
    #     pass

    def _setup_cameras(self, cameras: CameraPose, cam_dist: list[np.ndarray]) -> np.ndarray:
        self.rotations = cameras.get_rot_2_angle_axis() # Ensure this is an Nx3 Matrix
        self.translations = cameras.get_translations_np()       # Ensure this is an Nx3 Matrix

        # We know points are normalized, so set F (Focal Length) = 1
        f = 1

        if self.mono:
            dist = cam_dist[0]
            cam_data = np.array([[f, dist[0, 0], dist[0, 1]]]*self.translations.shape[0])
        else: # Assume Multi-view for now...
            dist = cam_dist[0]
            cam_data = np.array([[f, dist[0, 0], dist[0, 1]]]*self.translations.shape[0])


        # N x 9 matrix, where each row is the R, T, f, k1, k2 (where k1 and k2 are the dist params)
        camera_data = np.hstack((self.rotations, self.translations, cam_data)) 

        return camera_data
    
    def _setup_observations(self, obs: list[np.ndarray], num_of_obs: int) -> np.ndarray:
        observation = np.zeros((num_of_obs, 4))
        start_index = 0
        end_index = 0

        for i in range(0, len(obs)):
            end_index = start_index + obs[i].shape[0]
            observation[start_index:end_index,:] = obs[i]

            start_index = end_index

        return observation

@dataclass
class Scene:
    points3D: Points3D        # Set of 3D points
    cam_poses: list[np.ndarray] # Should be formatted as a 3x4 matrix
    point_to_pose: np.ndarray   # List of corresponding camera poses to 3D points
    representation: str         # Represnetation of the scene (Future use cases here)
    bal_data: BundleAdjustmentData # Data stored in the BAL format, and write file to reconstructed scene

    def __init__(self, points3D: Points3D | None = Points3D(), 
                 cam_poses: list[np.ndarray] = [], 
                 point_to_pose: np.ndarray | None = None, 
                 representation: str = "point cloud",
                 bal_data : BundleAdjustmentData | None = None):
        self.SceneRepresentation = ["point cloud", "mesh", 'NeRF']

        
        self.points3D = points3D
        self.cam_poses = cam_poses
        self.point_to_pose = point_to_pose
        self.representation = representation

        assert(self.representation in self.SceneRepresentation)

        if bal_data is not None:
            self._write_BAL_file(bal_data=bal_data)
            self.bal_data = bal_data

    def update_cam_pose(self, cam_pose: np.ndarray) -> None:
        self.cam_poses.append(cam_pose)

    def update_3d_points(self, points: Points3D) -> None:
        self.points3D.update_points(points=points.points3D)

    def _write_BAL_file(self, bal_data: BundleAdjustmentData) -> None:
        # File is fixed with name and location placement
        
        file_name = "BAL_Scene_Data"

@dataclass
class Calibration:
    K1: np.ndarray # Camera intrinsics (Single)
    K2: np.ndarray # 2nd Camera Intrinsics (Stereo)
    K_cams: list[np.ndarray] # Set of camera intrinics for multi camera approach (Mono cameras)
    distort: np.ndarray # Camera 1 Distortion (Single)
    distort2: np.ndarray # Camera 2 Distortion (Stereo)
    cam_dists: list[np.ndarray] # set of camera distortion params for multi camera approach (Mono cameras)
    R12: np.ndarray # Rotation of Stereo Camera
    T12: np.ndarray # Translation of Stereo Camera (Baseline)
    stereo: bool
    multi_cam: bool # Determine if the Mono camera setup takes into account multiple cameras

    def __init__(self, 
                 K1 = np.eye(3),  
                 R: np.ndarray | None = None, 
                 T: np.ndarray | None = None, 
                 stereo: bool = False, 
                 K2: np.ndarray | None = None, 
                 dist: np.ndarray | None = np.zeros((1,5)),
                 dist2: np.ndarray | None = np.zeros((1,5))):
        self.multi_cam = False
        self.K1 = K1 # For single camera mono setup
        self.distort = dist
        self.stereo = stereo

        self.K2 = K2 # For stereo camera
        self.distort2 = dist2
        self.R12 = R # Rotation from camera 1 to camera 2
        self.T12 = T # Baseline between camera 1 and camera 2
        if self.stereo:
            self.distort2 = None
        
    def setup_multi_cam(self, cam_ints: list[np.ndarray], cam_dists: list[np.ndarray]) -> None:
        # Set up old Mono cams back to default if changed
        self.K1 = np.eye(3)
        self.distort = np.zeros((1,5))

        # Update Mono Camera settings to multi_cam setup
        self.K_cams = cam_ints  
        self.cam_dists = cam_dists
        self.multi_cam = True

    def update_cal_img_shape(self, img_scale: list[float]):
        # Assume Monocular Camera for now with OpenCV calibration convention (Wide belief)
        # Meaning: Skew Parameter will be zero in these cases.
        # height_scale, width_scale = img_scale[:]
        width_scale, height_scale = img_scale[:]

        self.K1[0,0] = width_scale * self.K1[0,0]   # fx x width_scale = fx'
        self.K1[1,1] = height_scale * self.K1[1,1]  # fy x height_scale = fy'
        self.K1[0,2] = width_scale * self.K1[0,2]   # cx x width_scale = cx'
        self.K1[1,2] = height_scale * self.K1[1,2]  # cy x height_scale = cy'



    def get_intrinsics(self):
        if self.stereo:
            return self.K1, self.K2
        
        return self.K1
    
    def get_extrinsics(self):
        assert(self.stereo)
        
        return self.R12, self.T12
    


