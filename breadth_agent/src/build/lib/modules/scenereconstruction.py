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
from mapanything.models import MapAnything
from mapanything.utils.geometry import closed_form_pose_inverse, depthmap_to_world_frame
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT

from modules.DataTypes.datatype import (Points2D, 
                                        CameraData, 
                                        Points3D, 
                                        CameraPose, 
                                        Scene, 
                                        PointsMatched,
                                        BundleAdjustmentData)


##########################################################################################################
############################################### ML MODULES ###############################################

class Sparse3DReconstructionMapAnything(SceneEstimation):
    def __init__(self,
                 cam_data: CameraData,
                 min_observe: int = 3,
                 update_intrinsics = False):
        
        super().__init__(cam_data = cam_data)

        dtype = (
        torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = MapAnything.from_pretrained("facebook/map-anything").to(self.device)
        self.model.eval()        

        data_norm_type = self.model.encoder.data_norm_type
        print(data_norm_type)
        if data_norm_type is None:
            # No normalization, just convert to tensor
            img_transform = TF.ToTensor()
        elif data_norm_type in IMAGE_NORMALIZATION_DICT.keys():
            # Use the specified normalization
            img_norm = IMAGE_NORMALIZATION_DICT[data_norm_type]
            img_transform = TF.Compose(
                [TF.ToTensor(), TF.Normalize(mean=img_norm.mean, std=img_norm.std)]
            )

        self.height, self.width = self.image_list[0].shape[:2]
        self.minimum_observation = min_observe
        self.update_intrinsics = update_intrinsics

        print(self.height)
        print(self.width)
        tensor_img_list = []
        for ind in range(len(self.image_list)):
            tensor_img_list.append(img_transform(self.image_list[ind]))
        self.images = torch.stack(tensor_img_list).to(self.device) 


    def build_reconstruction(self, 
                             tracked_features: PointsMatched, 
                             cam_poses: CameraPose | None = None) -> Scene:
        torch.cuda.empty_cache() #Empty GPU cache

        # VGGT Fixed Resolution to 518 for Inference
        images = F.interpolate(self.images, size=(518, 518), mode="bilinear", align_corners=False)


        if cam_poses is None and self.K_mat is None:
            views = []
            for view_idx in range(images.shape[0]):
                view = {
                    "img": images[view_idx][None],  # Add batch dimension
                    "data_norm_type": [self.model.encoder.data_norm_type],
                }
                views.append(view)
        elif cam_poses is None:
            print("Old INTRINSICS", self.K_mat)
            int_torch = torch.from_numpy(np.array(self.K_mat.astype(np.float32))).to(self.device)
            int_torch[:2, :] *= (518/self.width)
            print("INTRISNICS SHAPE:", int_torch.shape)
            views = []
            for view_idx in range(images.shape[0]):
                print("IMAGE SHAPE:", images[view_idx][None].shape)
                view = {
                    "img": images[view_idx][None],  # Add batch dimension
                    "intrinsics": int_torch[None],
                    "data_norm_type": [self.model.encoder.data_norm_type],
                }
                views.append(view)
        else:
            print("USING CAL AND POSES")
            int_torch = torch.from_numpy(np.array(self.K_mat.astype(np.float32))).to(self.device)
            int_torch[:2, :] *= (518/self.width)
            camera_poses = cam_poses.camera_pose
            views = []
            for view_idx in range(images.shape[0]):
                world2cam_matrix = np.eye(4)
                world2cam_matrix[:3, :3] = camera_poses[view_idx][:3, :3]
                world2cam_matrix[:3, 3] = camera_poses[view_idx][:3, 3]

                pose_matrix = closed_form_pose_inverse(world2cam_matrix[None, :, :])[0]
                view = {
                    "img": images[view_idx][None],  # Add batch dimension
                    "intrinsics": int_torch[None],
                    "camera_poses": torch.from_numpy(pose_matrix.astype(np.float32)).to(self.device)[None],
                    "data_norm_type": [self.model.encoder.data_norm_type],
                    "is_metric_scale": torch.tensor([False]),  # COLMAP data is non-metric
                }
                views.append(view)

        predictions = self.model.infer(
            views, memory_efficient_inference=False
        )
        
        # Process predictions
        (
            all_extrinsics,
            all_intrinsics,
            all_depth_maps,
            all_depth_confs,
            all_pts3d,
        ) = (
            [],
            [],
            [],
            [],
            [],
        )

        for pred in predictions:
            # Compute 3D points from depth, intrinsics, and camera pose
            depthmap_torch = pred["depth_z"][0].squeeze(-1)  # (H, W)
            intrinsics_torch = pred["intrinsics"][0]  # (3, 3)
            camera_pose_torch = pred["camera_poses"][0]  # (4, 4)
            pts3d, valid_mask = depthmap_to_world_frame(
                depthmap_torch, intrinsics_torch, camera_pose_torch
            )

            # Extract mask from predictions and combine with valid depth mask
            mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
            mask = mask & valid_mask.cpu().numpy()  # Combine with valid depth mask

            # Convert tensors to numpy arrays
            extrinsic = (
                closed_form_pose_inverse(pred["camera_poses"])[0].cpu().numpy()
            )  # c2w -> w2c
            intrinsic = intrinsics_torch.cpu().numpy()
            depth_map = depthmap_torch.cpu().numpy()
            depth_conf = pred["conf"][0].cpu().numpy()
            pts3d = pts3d.cpu().numpy()


            # Collect results
            print("OLD VERSION EXT", extrinsic)
            all_extrinsics.append(extrinsic[:3, :])
            all_intrinsics.append(intrinsic)
            all_depth_maps.append(depth_map)
            all_depth_confs.append(depth_conf)
            all_pts3d.append(pts3d)

        print("Previous Intrinsics", int_torch)
        print("INTRINSICS", all_intrinsics[:2])
        # print("PREVIOUS EXTRINISICS:", cam_poses.camera_pose[:2])
        print("CURRENT EXT:", all_extrinsics[:2])
        # Stack results into arrays
        # all_extrinsics = np.stack(all_extrinsics)
        # all_intrinsics = np.stack(all_intrinsics)
        all_depth_maps = np.stack(all_depth_maps)
        all_depth_confs = np.stack(all_depth_confs)
        all_pts3d = np.stack(all_pts3d)

        # Update Camera Poses
        if cam_poses is None:
            rotations = []
            translations = []
            num_cameras = len(all_extrinsics) #.shape[0]
            # all_extrinsics[0] = np.hstack((np.eye(3), np.zeros((3, 1))))
            for i in range(num_cameras):
                rotations.append(all_extrinsics[i][:, :3])
                translations.append(all_extrinsics[i][:, 3:])
            cam_poses = CameraPose(cam_poses=all_extrinsics,
                                   rot=rotations,
                                   trans=translations)
        else:
            num_cameras = len(cam_poses.camera_pose)

        if self.update_intrinsics:
            self.cam_data.apply_new_calibration(intrinsics=all_intrinsics)
            self.K_mat = self.cam_data.get_K(0)
            self.dist = self.cam_data.get_distortion()
            self.multi_cam = self.cam_data.multi_cam
        # Here we use the ext, int, depth_map, and point_map (points3D) to initialize the sparse reconstruction with tracked feature points
        # print("DISTORTION", self.dist)
        scene = self.match_tracks_to_point_maps(tracked_features=tracked_features,
                                                point_maps = all_pts3d,
                                                conf_maps = all_depth_confs,
                                                minimum_observation = self.minimum_observation,
                                                img_width = self.width,
                                                num_cameras = num_cameras,
                                                camera_poses=cam_poses)
        
        return scene

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
from classical feature detectors (SIFT or ORB). Utilize this module in conjuction with the VGGT pose estimation module in these cases
where feature detection is low. This module is for reconstructing the scene using the deep learning approach. 
Computation time should not matter when invoking this tool, but keep in mind of system constraints such as GPU memory.

Initialization Parameters:
- cam_data: Data container to hold images and calibration data, read from the CameraDataManager.
- min_observe: The minimum number of observations (number of tracked feature points) needed to conduct a 3D 
point estimation. Note: this must be greater than 2
    - Default (int): 3 

Function Call Parameters:
- camera_poses (CameraPose): Estimated camera poses for the given scene. Poses are estimated prior to this function call, 
specifically from the CameraPoseEstimation modules. 
- tracked_features (PointsMatched): Feature points tracked across multiple frames to allow Multi-View 3D point estimation. Feature Tracks are 
estimated from the FeatureTracking modules.

Module Input:
    PointsMatched (Matched Features across image pairs)
        pairwise_matches: list[np.ndarray]  [N x 4] -> [x1, y1, x2, y2]. Data Structure to store Pairwise feature matches.
        multi_view: bool                    Determine if Pairwise/Multi-View Feature Matching (Should be False for Pairwise in this function)
        image_size: np.ndarray              [1 x 2] [np.int64] (Simply Image Shape: (W, H))
        image_scale: list[float]            [W_scale, H_scale] if image is resized
    
    CameraPose:
        camera_pose: list[np.ndarray]   [3 x 4] (np.float) Camera pose for each corresponding frame. Each pose is 3x4 (R, T)
        rotations: list[np.ndarray]     [3 x 3] (np.float) Rotation matrices for each corresponding frame (Derived from camera_pose)
        translations: list[np.ndarray]  [3 x 1] (np.float) Translation matrices for each corresponding frame (Derived from camera_pose)

Module Output:
    Scene:
        points3D: Points3D 
            Points3D
                - points3D: np.ndarray      [N x 3] Point position in 3D space [x, y, z]
                - color: np.ndarray         [N x 3] Point Color [r, g, b]               
        cam_poses: list[np.ndarray]         [3 x 4] (np.float) Camera pose for each corresponding frame. Each pose is 3x4 (R, T)
        point_to_pose: np.ndarray           [N x 2] List of corresponding camera poses to 3D points [cam_frame, point_index]
        bal_data: BundleAdjustmentData      Data stored in the BAL format, and write file to reconstructed scene
            BundleAdjustmentData
                - num_cameras: int                Number of cameras in the scene (Not number of observerations)
                - num_points: int                 Number of 3D Points
                - num_observations: int           Number of observations.
                - camera_int: list[np.ndarray]    [3 x 3] Camera Intrinsics, for each cam_i in list contains calibration matrix
                - observations: np.ndarray        [M x 4] matrices for each point observation where M=num_of_observations, and each row = [frame, 3d_point_ind, norm_x, norm_y]
                - cameras: list[np.ndarray]       [1 x 9] per element | List of cameras, with each row containing R(rodriguez), T, f, K1, K2 (k1 and k2 radial distortion)
                - points: np.ndarray              [N x 3] matrix containing the X, Y, Z coordinates of points. N = Number of total points.

"""
        self.example = f"""
Initialization:
# Camera Pose Module Initialization
pose_estimator = CamPoseEstimatorVGGTModel(cam_data = camera_data)

# Scene Reconstruction Module Initialization
sparse_reconstruction = Sparse3DReconstructionVGGT(cam_data = camera_data)

# Feature Tracker Module Iniitalization
feature_tracker = FeatureMatchLightGlueTracking(detector="superpoint")

Function Use:
# From estimated features, estimate the camera poses for all image frames
cam_poses = pose_estimator()

 # To track features across multiple images 
tracked_features = feature_tracker(features=features)

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
        torch.cuda.empty_cache() #Empty GPU cache

        ext_torch = torch.from_numpy(np.array(cam_poses.camera_pose)).to(self.device)
        int_torch = torch.from_numpy(np.array(self.K_mat)).to(self.device)
        int_torch[:, :2, :] *= (518/self.width) # Bring back to fixed VGGT Resolution

        # VGGT Fixed Resolution to 518 for Inference
        images = F.interpolate(self.images, size=(518, 518), mode="bilinear", align_corners=False)
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=self.dtype):
                images = images[None]  # add batch dimension
                aggregated_tokens_list, ps_idx = self.model.aggregator(images)

            # Predict Depth Maps
            depth_map, depth_conf = self.model.depth_head(aggregated_tokens_list, images, ps_idx)

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
                                                num_cameras = num_cameras,
                                                camera_poses=cam_poses)
        
        torch.cuda.empty_cache() #Empty GPU cache

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
This module can reconstruct sparse 3D scenes specifically using a monocular camera as primary sensor. 
This is determined by the data used and the parameter 'view' on module function call. 
This module can reconstruct sparse 3D scenes either through multi-view or two-view triangulation.
This is determined by the method used to find matching features.
Use this module when specified for sparse reconstruction and calibration data is provided,
with the camera being used is a monocular camera, and when enough features are detected in the scene. 
This can apply for scenes with high textured with good lighting, but also scenes that do not apply if 
the prerequisite for enough features detected are met. The module is for reconstructing the 
scene using the direct mathematical (Classical) approach.

Initialization Parameters:
- cam_data: Data container to hold images and calibration data, read from the CameraDataManager.
- view: Method used to trace feature points across frames (Two-View [Image Pairs] or Multi-View [Tracking])
    - Default (str): multi
    - Options: ["multi", "two"]
- min_observe: The minimum number of observations (number of tracked feature points) needed to conduct a 3D 
point estimation. Note: this must be greater than 2
    - Default (int): 3 
- min_angle: The minimum angle required between bearing rays from paired 2D feature point to accept a 3D point 
estimation from the set of corresponding 2D feature points. Used for the Triangulation Angle Test.
    - Default (float): 1.0 (Typically 1.0 - 3.0 [Number represents angle degree])
- reproj_error: Maximum reprojection error accepted for a potential 3D point estimation to keep in a point cloud. Error is measured in pixel coordinates.
    - Default (float): 3.0

Function Call Parameters:
- cam_poses (CameraPose): Estimated camera poses for the given scene. Poses are estimated prior to this function call, 
specifically from the CameraPoseEstimation modules. 
- tracked_points (PointsMatched): Feature points tracked across multiple frames to allow Multi-View 3D point estimation. Feature Tracks are 
estimated from the FeatureTracking modules.

Module Input:
    PointsMatched (Matched Features across image pairs)
        pairwise_matches: list[np.ndarray]  [N x 4] -> [x1, y1, x2, y2]. Data Structure to store Pairwise feature matches.
        multi_view: bool                    Determine if Pairwise/Multi-View Feature Matching (Should be False for Pairwise in this function)
        image_size: np.ndarray              [1 x 2] [np.int64] (Simply Image Shape: (W, H))
        image_scale: list[float]            [W_scale, H_scale] if image is resized
    
    CameraPose:
        camera_pose: list[np.ndarray]   [3 x 4] (np.float) Camera pose for each corresponding frame. Each pose is 3x4 (R, T)
        rotations: list[np.ndarray]     [3 x 3] (np.float) Rotation matrices for each corresponding frame (Derived from camera_pose)
        translations: list[np.ndarray]  [3 x 1] (np.float) Translation matrices for each corresponding frame (Derived from camera_pose)

Module Output:
    Scene:
        points3D: Points3D 
            Points3D
                - points3D: np.ndarray      [N x 3] Point position in 3D space [x, y, z]
                - color: np.ndarray         [N x 3] Point Color [r, g, b]               
        cam_poses: list[np.ndarray]         [3 x 4] (np.float) Camera pose for each corresponding frame. Each pose is 3x4 (R, T)
        point_to_pose: np.ndarray           [N x 2] List of corresponding camera poses to 3D points [cam_frame, point_index]
        bal_data: BundleAdjustmentData      Data stored in the BAL format, and write file to reconstructed scene
            BundleAdjustmentData
                - num_cameras: int                Number of cameras in the scene (Not number of observerations)
                - num_points: int                 Number of 3D Points
                - num_observations: int           Number of observations.
                - camera_int: list[np.ndarray]    [3 x 3] Camera Intrinsics, for each cam_i in list contains calibration matrix
                - observations: np.ndarray        [M x 4] matrices for each point observation where M=num_of_observations, and each row = [frame, 3d_point_ind, norm_x, norm_y]
                - cameras: list[np.ndarray]       [1 x 9] per element | List of cameras, with each row containing R(rodriguez), T, f, K1, K2 (k1 and k2 radial distortion)
                - points: np.ndarray              [N x 3] matrix containing the X, Y, Z coordinates of points. N = Number of total points.
"""
        self.example = f"""
Initialization:
sparse_reconstruction = Sparse3DReconstructionMono(cam_data = camera_data, reproj_error = 2.0, min_observe = 3, min_angle = 1.5)

Function Use (multi-view):
feature_tracker = FeatureMatchLightGlueTracking(detector="superpoint")

cam_poses = pose_estimator(features=features) # To Estimate Camera Poses from detected features

tracked_features = feature_tracker(features=features) # To track features across multiple images 

# Estimate 3D scene using multi-view due to tracking features from multiple images in previous step
sparse_scene = sparse_reconstruction(tracked_features, cam_poses, view="multi") 

Function Use (two-view):
feature_matcher = FeatureMatchLightGluePair(cam_data = camera_data)

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