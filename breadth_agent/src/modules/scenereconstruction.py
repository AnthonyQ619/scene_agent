import cv2
import numpy as np
from modules.baseclass import SparseSceneEstimation, DenseSceneEstimation
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms as TF
import os
import shutil
import struct
import open3d as o3d
import glob
from modules.models.sfm_models.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from modules.models.sfm_models.vggt.utils.geometry import unproject_depth_map_to_point_map
from modules.models.sfm_models.vggt.models.vggt import VGGT
from modules.models.sfm_models.vggt.utils.load_fn import load_and_preprocess_images
from mapanything.models import MapAnything
from mapanything.utils.geometry import closed_form_pose_inverse, depthmap_to_world_frame
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT
from pathlib import Path

from modules.DataTypes.datatype import (Points2D, 
                                        CameraData, 
                                        Points3D, 
                                        CameraPose, 
                                        Scene, 
                                        PointsMatched,
                                        BundleAdjustmentData)



# Import Pycolmap
# os.add_dll_directory(r"C:\\Users\\Anthony\\Desktop\\VCPKG\\vcpkg\\installed\\x64-windows\\bin")
# os.add_dll_directory(r"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4\\bin")
# os.add_dll_directory(r"C:\\Program Files\\NVIDIA cuDSS\\v0.7\\bin\\12")
import pycolmap

##########################################################################################################
############################################### ML MODULES ###############################################

class Sparse3DReconstructionMapAnything(SparseSceneEstimation):
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
            self.K_mat = self.cam_data.get_K()
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

class Sparse3DReconstructionVGGT(SparseSceneEstimation):
    def __init__(self,
                 cam_data: CameraData,
                 min_observe: int = 3):
        
        module_name = "Sparse3DReconstructionVGGT"
        description = f"""
Sparsely reconstructs a 3D scene utilizing pre-processed information of camera poses and
images of the scene. Camera Poses are estimated prior to thie module through the camera pose estimation 
module, specifically from VGGT pose estimation. Features do need to be tracked to build a sparse reconstruction 
from the estimated point maps of VGGT.
This module can reconstruct sparse 3D scenes specifically using a monocular camera. 
This module can reconstruct sparse 3D scenes either through single view or multi-view scenes.
This is determined by the how many images exist in the scene and how many poses were estimated from the previous
module using the VGGT pose estimation tool specifically.
Use this module when specified for ONLY SPARSE reconstruction and the scene doesn't allow for many features to be detected
from classical feature detectors (SIFT or ORB). Utilize this module in conjuction with the VGGT pose estimation module in these cases
where feature detection is low. This module is for reconstructing the scene using the deep learning approach. 
Computation time should not matter when invoking this tool, but keep in mind of system constraints such as GPU memory.

Initialization/Function Parameters:
- min_observe: The minimum number of observations (number of tracked feature points) needed to conduct a 3D 
point estimation. Note: this must be greater than 2
    - Default (int): 3 

Function Call Parameters - Handled Internally from SfMScene in the common API Workflow:
- cam_poses (CameraPose): Estimated camera poses for the given scene. Poses are estimated prior to this function call, 
specifically from the CameraPoseEstimation modules. 
- tracked_features (PointsMatched): Feature points tracked across multiple frames to allow Multi-View 3D point estimation. Feature Tracks are 
estimated from the FeatureTracking modules.

Module Input - Handled Internally from SfMScene in the common API Workflow:
    PointsMatched (Matched Features across image pairs)
        pairwise_matches: list[np.ndarray]  [N x 4] -> [x1, y1, x2, y2]. Data Structure to store Pairwise feature matches.
        multi_view: bool                    Determine if Pairwise/Multi-View Feature Matching (Should be False for Pairwise in this function)
        image_size: np.ndarray              [1 x 2] [np.int64] (Simply Image Shape: (W, H))
        image_scale: list[float]            [W_scale, H_scale] if image is resized
    
    CameraPose:
        camera_pose: list[np.ndarray]   [3 x 4] (np.float) Camera pose for each corresponding frame. Each pose is 3x4 (R, T)
        rotations: list[np.ndarray]     [3 x 3] (np.float) Rotation matrices for each corresponding frame (Derived from camera_pose)
        translations: list[np.ndarray]  [3 x 1] (np.float) Translation matrices for each corresponding frame (Derived from camera_pose)

Module Output - Handled Internally from SfMScene in the common API Workflow:
    Scene:
        points3D: Points3D 
            Points3D
                - points3D: np.ndarray      [N x 3] Point position in 3D space [x, y, z]
                - color: np.ndarray         [N x 3] Point Color [r, g, b]               
        cam_poses: list[np.ndarray]         [3 x 4] (np.float) Camera pose for each corresponding frame. Each pose is 3x4 (R, T)
        observations: np.ndarray            [M x 4] matrix for each point observation where M=num_of_observations, and each row = [frame, 3d_point_ind, pix_x, pix_y]
        depth_maps: list[np.ndarray]        List[[H x W]] List of Depth Maps per frame, formated as HeightxWidth of image shape
        sparse: bool                        Used to determine if current scene is sparse or dense
"""
        example = f"""
Initialization:
from modules.features import ...
from modules.featurematching import ...
from modules.camerapose import CamPoseEstimatorVGGTModel
from modules.scenereconstruction import {module_name}
from modules.baseclass import SfMScene

Function Use:
# Step 1: Read in Calibration/Image Data
reconstructed_scene = SfMScene(image_path = image_path, 
                               calibration_path = calibration_path)

# Step 2: Detect Features Prior to Step 5 (Data filled in SfMScene)

# Step 3: Detect Cam Poses (Must use VGGT prior to this step!)
reconstructed_scene.CamPoseEstimatorVGGTModel() 

# Step 4: Detect Feature Tracks Prior to Step 5 (Data filled in SfMScene)

# Step 5: Estimate Sparse Reconstruction using VGGT Module
reconstructed_scene.{module_name}(
    min_observe=3
)
"""
        super().__init__(cam_data = cam_data,
                         module_name=module_name,
                         description=description,
                         example=example)
        
        # Initialize Model
        if os.name == 'nt':
            WEIGHT_MODULE = str(os.path.dirname(__file__)) + "\\models\\sfm_models\\vggt\\weights\\model.pt"
        elif os.name == 'posix':
            WEIGHT_MODULE = str(os.path.dirname(__file__)) + "/models/sfm_models/vggt/weights/model.pt"

        # WEIGHT_MODULE = "/work/model_weights/model.pt"
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cuda":
            # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
            self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            self.dtype = torch.float32

        self.model = VGGT().to(self.device)
        self.model.load_state_dict(torch.load(WEIGHT_MODULE, weights_only=True))
        self.model.eval()

        self.height, self.width = self.image_list[0].shape[:2]
         # Load Images in correct format for VGGT inference
        to_tensor = TF.ToTensor()
        tensor_img_list = [to_tensor(img) for img in self.image_list]

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
    

###########################################################################################################
############################################ CLASSICAL MODULES ############################################

class Sparse3DReconstructionStereo(SparseSceneEstimation):
    def __init__(self,
                 cam_data: CameraData):
        module_name = "Sparse3DReconstructionMono"
        description = f"""
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
        example = f"""
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
    
# Mono Camera Reconstruction

class Sparse3DReconstructionMono(SparseSceneEstimation):
    def __init__(self, cam_data: CameraData, 
                 multi_view: bool = True,
                 reproj_error: float = 3.0,
                 min_observe: int = 3,
                 min_angle: float = 1.0):

        module_name = "Sparse3DReconstructionMono"
        description = f"""
Sparsely reconstructs a 3D scene utilizing pre-processed information of camera poses and
detected features tracked across the scene. Camera Poses are estimated prior to this module
through the camera pose estimation module. Features are matched, or tracked, prior to this module 
through the feature matching/tracking module. 

This module can reconstruct sparse 3D scenes specifically using a monocular camera as primary sensor. 
This module can reconstruct sparse 3D scenes either through multi-view or two-view triangulation.
This is determined by the method used to find matching features.
Features that are Tracked (Hence a FeatureTracking module is called prior to this step), set multi-view
to True. If Features are Matched (a FeatureMatching module is called prior with no tracking module called), 
set multi_view to False.

Use this module when specified for sparse reconstruction and calibration data is provided,
with the camera being used is a monocular camera, and when enough features are detected in the scene. 
This can apply for scenes with high textured with good lighting, but also scenes that do not apply if 
the prerequisite for enough features detected are met. The module is for reconstructing the 
scene using the direct mathematical (Classical) approach.

Initialization/Function Parameters:
- view: Method used to trace feature points across frames (Two-View [Corresponding Pairs] or Multi-View [Tracking])
    - Default (bool): True
- min_observe: The minimum number of observations (number of tracked feature points) needed to conduct a 3D 
point estimation. Note: this must be greater than 2
    - Default (int): 3 
- min_angle: The minimum angle required between bearing rays from paired 2D feature point to accept a 3D point 
estimation from the set of corresponding 2D feature points. Used for the Triangulation Angle Test. The larger the angle
the more accurate the 3D point (Maximum of 4.0)
    - Default (float): 1.0 (Typically 1.0 - 3.0 [Number represents angle degree])
- reproj_error: Maximum reprojection error accepted for a potential 3D point estimation to keep in a point cloud. Error is measured in pixel coordinates.
    - Default (float): 3.0

Function Call Inputs - Handled Internally from SfMScene in the common API Workflow:
- cam_poses (CameraPose): Estimated camera poses for the given scene. Poses are estimated prior to this function call, 
specifically from the CameraPoseEstimation modules. 
- tracked_features (PointsMatched): Feature points tracked across multiple frames to allow Multi-View 3D point estimation. Feature Tracks are 
estimated from the FeatureTracking modules.

Module Input - Handled Internally from SfMScene in the common API Workflow:
    PointsMatched (Matched Features across image pairs)
        # General Data Information for Feature Matches
        image_size: np.ndarray              [1 x 2] [np.int64] (Simply Image Shape: (W, H))
        image_scale: list[float]            [W_scale, H_scale] if image is resized
        multi_view: bool                    Determine if Pairwise/Feature Matching
        stereo_cam: bool                    Deterine if the camera utilized is a stereo camera for feature matching/tracking

        # Tracked Data Features
        data_matrix: np.ndarray             [N x 4] Data Structure to store corresponding points. In the form of Nx4 -> [track_id, frame_num, x, y]
        track_map: dict                     Used to aid in the feature matching process.
        point_count: int                    Based on track_id max count -> tells us how many 3D points exist
    
    CameraPose:
        camera_pose: list[np.ndarray]   [3 x 4] (np.float) Camera pose for each corresponding frame. Each pose is 3x4 (R, T)
        rotations: list[np.ndarray]     [3 x 3] (np.float) Rotation matrices for each corresponding frame (Derived from camera_pose)
        translations: list[np.ndarray]  [3 x 1] (np.float) Translation matrices for each corresponding frame (Derived from camera_pose)

Module Output - Handled Internally from SfMScene in the common API Workflow:
    Scene:
        points3D: Points3D 
            Points3D
                - points3D: np.ndarray      [N x 3] Point position in 3D space [x, y, z]
                - color: np.ndarray         [N x 3] Point Color [r, g, b]               
        cam_poses: list[np.ndarray]         [3 x 4] (np.float) Camera pose for each corresponding frame. Each pose is 3x4 (R, T)
        observations: np.ndarray            [M x 4] matrix for each point observation where M=num_of_observations, and each row = [frame, 3d_point_ind, pix_x, pix_y]
        depth_maps: list[np.ndarray]        List[[H x W]] List of Depth Maps per frame, formated as HeightxWidth of image shape
        sparse: bool                        Used to determine if current scene is sparse or dense
"""

        example = f"""
Initialization:
from modules.features import ...
from modules.featurematching import ... (Pair Module), ... (Tracking Module)
from modules.camerapose import ...
from modules.scenereconstruction import {module_name}
from modules.baseclass import SfMScene

Function Use:
# Step 1: Read in Calibration/Image Data
reconstructed_scene = SfMScene(image_path = image_path, 
                               calibration_path = calibration_path)

# Step 2: Detect Features Prior to Step 3 (Data filled in SfMScene)

# Step 3: Detect Feature Pairwise Matches Prior to Step 4 (Data filled in SfMScene)

# Step 4: Detect Cam Poses Prior to Step 5 Using a Pose Modules

# Step 5: Detect Feature Tracks Prior to Step 6 (Data filled in SfMScene)

# Step 6: Estimate Sparse Reconstruction using VGGT Module
reconstructed_scene.{module_name}(
    min_observe=3
)
"""
        
        super().__init__(cam_data = cam_data,
                         module_name=module_name,
                         description=description,
                         example=example)
        
        self.multi_view = multi_view
        self.minimum_observation = min_observe # N-view functionality only
        self.min_angle = min_angle
        self.reproj_error_min = reproj_error


    def build_reconstruction(self, 
                 points: PointsMatched, 
                 camera_poses: CameraPose) -> Scene:
        
        points_3d = Points3D()

        # BAL File for Optimization Module
        num_observations = 0 
        observations = []
        observations_pix = []

        if self.multi_view: # Multi-view
            if not points.multi_view:
                message = 'Error: features are not tracked. Use set multi_view to FALSE instead to use this Reconstruction Module for pairwise feature matching.'
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
                    norm_pts = self._normalize_points(views)#views[:, 1:])
                    observation = np.hstack((np.vstack(views[:,0]), point_ind, norm_pts))#views[:,1:]))
                    observation_pix = np.hstack((np.vstack(views[:,0]), point_ind, views[:, 1:]))
                    observations_pix.append(observation_pix)
                    observations.append(observation)
                    num_observations += views.shape[0] # Number of observations

                    # Keep 3D point here
                    points_3d.update_points(point)

                    point_index += 1 # Successfully Estimated Point

            # # Build BAL data
            # ba_data = BundleAdjustmentData(num_cameras=num_cameras, 
            #                                num_points=points_3d.points3D.shape[0],
            #                                num_observations=num_observations,
            #                                observations=observations,
            #                                cameras=camera_poses,
            #                                points=points_3d.points3D,
            #                                dist=[self.dist],
            #                                mono=True)
            try:
                count_of_points = points_3d.points3D.shape[0]
            except:
                message = 'Error: no 3D points are calculated. Try reducing min_observe to 3 (Default). If its already set to 3, set min_angle to 1.0. If its already set to 1.0, use VGGT pipeline for robustness.'
                raise Exception(message)

            scene = Scene(points3D = points_3d,
                          cam_poses = camera_poses.camera_pose, 
                          observations = np.vstack(observations_pix),
                          representation = "point cloud",
                        #   bal_data=ba_data,
                          sparse=True)
            print(np.vstack(observations_pix).shape)
            print(points.data_matrix.shape)
            print(point_index)
            return scene
        else: # Two-View
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
    

###########################################################################################################
###################################### DENSE RECONSTRUCTION MODULES #######################################

class Dense3DReconstructionVGGT(DenseSceneEstimation):
    def __init__(self,
                 cam_data: CameraData,
                 min_observe: int = 3):

        module_name = "Dense3DReconstructionVGGT"
        description = f"""
Densely reconstructs a 3D scene utilizing pre-processed information of camera poses and
images of the scene (SKIP THE SPARSE RECONSTRUCTION STEP - DO NOT USE SPARSE VGGT AND INSTEAD REPLACE WITH THIS MODULE). 

Camera Poses are estimated prior to thie module through the camera pose estimation  module, specifically from VGGT 
pose estimation. Features do NOT need to be tracked or matched between frames.

This module can reconstruct dense 3D scenes specifically using a monocular camera. 
This module can reconstruct dense 3D scenes either through single view or multi-view scenes.
This is determined by the how many images exist in the scene and how many poses were estimated from the previous
module using the VGGT pose estimation tool specifically.

Use this module when specified for dense reconstruction and the scene doesn't allow for many features to be detected
from classical feature detectors (SIFT or ORB), or ML Detectors. Utilize this module in conjuction with the VGGT pose 
estimation module in these cases where feature detection is low. This module is for reconstructing the scene using 
the deep learning approach. 
Computation time should matter when invoking this tool, KEEP IN MIND of system constraints such as GPU memory prior
to USING THIS TOOL.

Initialization Parameters:
None - Not applicable here

Function Call Parameters - Handled Internally from SfMScene in the common API Workflow with Pose Estimation Module:
- camera_poses (CameraPose): Estimated camera poses for the given scene. Poses are estimated prior to this function call, 
specifically from the CameraPoseEstimation modules. 

Module Input - Handled Internally from SfMScene in the common API Workflow:
    sparse_scene (Scene):
        points3D: Points3D 
            Points3D
                - points3D: np.ndarray      [N x 3] Point position in 3D space [x, y, z]
                - color: np.ndarray         [N x 3] Point Color [r, g, b]               
        cam_poses: list[np.ndarray]         [3 x 4] (np.float) Camera pose for each corresponding frame. Each pose is 3x4 (R, T)
        point_to_pose: np.ndarray           [N x 2] List of corresponding camera poses to 3D points [cam_frame, point_index]
        depth_maps: list[np.ndarray]        list[H x W] Depth Maps per frame, formated as HeightxWidth of image shape
    cam_poses (CameraPose):
        camera_pose: list[np.ndarray]   [3 x 4] (np.float) Camera pose for each corresponding frame. Each pose is 3x4 (R, T)
        rotations: list[np.ndarray]     [3 x 3] (np.float) Rotation matrices for each corresponding frame (Derived from camera_pose)
        translations: list[np.ndarray]  [3 x 1] (np.float) Translation matrices for each corresponding frame (Derived from camera_pose)

Module Output: Scene (Densely Reconstructed)
"""
        example = f"""
Initialization:
from modules.camerapose import CamPoseEstimatorVGGTModel
from modules.scenereconstruction import {module_name}
from modules.baseclass import SfMScene

Function Use:
# With Global Optimization
# Step 1: Read in Calibration/Image Data
reconstructed_scene = SfMScene(image_path = image_path, 
                               calibration_path = calibration_path)

# Step 2: Detect Features - Not Needed

# Step 3: Detect Cam Poses (Must use VGGT prior to this step!)
reconstructed_scene.CamPoseEstimatorVGGTModel() 

# Step 4: Detect Feature Tracks - Not Needed

# Step 5: Estimate Dense Reconstruction using VGGT Module (Don't need Sparse Here)
reconstructed_scene.{module_name}()
"""
        super().__init__(cam_data = cam_data,
                         module_name=module_name,
                         description=description,
                         example=example)
        
        # Initialize Model
        if os.name == 'nt':
            WEIGHT_MODULE = str(os.path.dirname(__file__)) + "\\models\\sfm_models\\vggt\\weights\\model.pt"
        elif os.name == 'posix':
            WEIGHT_MODULE = str(os.path.dirname(__file__)) + "/models/sfm_models/vggt/weights/model.pt"

        # WEIGHT_MODULE = "/work/model_weights/model.pt"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cuda":
            # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
            self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            self.dtype = torch.float32

        self.model = VGGT().to(self.device)
        self.model.load_state_dict(torch.load(WEIGHT_MODULE, weights_only=True))
        self.model.eval()

        self.height, self.width = self.image_list[0].shape[:2]
        # Load Images in correct format for VGGT inference
        to_tensor = TF.ToTensor()
        tensor_img_list = []
        for ind in range(len(self.image_list)):
            tensor_img_list.append(to_tensor(self.image_list[ind]))
        self.images = torch.stack(tensor_img_list).to(self.device) 

        self.minimum_observation = min_observe

    def build_reconstruction(self, 
                             sparse_scene: Scene | None = None,
                             cam_poses: CameraPose | None = None) -> Scene:
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
            torch.cuda.empty_cache() #Empty GPU cache

            point_map = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                                                                ext_torch, 
                                                                int_torch)
        
        num_cameras = len(cam_poses.camera_pose)

        print("DEPTH CONFIDENCE MAP", depth_conf.shape)
        depth_conf_np = depth_conf.squeeze(0).detach().cpu().numpy()
        depth_map_np = depth_map.squeeze(0).detach().cpu().numpy() 
        print(point_map.shape)
        print(depth_conf.shape) 
        points_3d = self.export_vggt_dense_ply(
            point_map,   # preferred
            conf=depth_conf,
            conf_threshold=0.5,
            stride=2,
        )
        # points_3d = self.collect_PM_points(point_maps=point_map, 
        #                                    conf_maps=depth_conf)
        
        # points_3d = self.voxel_downsample(points=points_3d)
        
        torch.cuda.empty_cache() #Empty GPU cache
        pts = Points3D()
        pts.set_all_points(points = points_3d)
        scene = Scene(points3D = pts,
                      cam_poses = cam_poses.camera_pose,
                      depth_maps = depth_map_np,
                      sparse = False)
        # print(points_3d.shape)

        # val = self.point_density(points=points_3d)
        # print("POINT DENSITY", val)
        # val = self.coverage(points = points_3d)
        # print("OCCUPANCY GRID", val)
        # val = self.depth_consistency(depth_maps=depth_map)
        # print("DEPTH CONSISTENCY", val)
        return scene
    # Helper Function to grab proper dense point cloud
    def to_numpy(self, x):
        import torch
        import numpy as np

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)
    def save_ply_xyzrgb(self, path, points_xyz, colors_rgb=None):
        """
        points_xyz: (M, 3) float array
        colors_rgb: optional (M, 3) uint8 array
        """
        points_xyz = np.asarray(points_xyz, dtype=np.float32)

        if colors_rgb is not None:
            colors_rgb = np.asarray(colors_rgb, dtype=np.uint8)
            assert len(points_xyz) == len(colors_rgb)

        with open(path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points_xyz)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")

            if colors_rgb is not None:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")

            f.write("end_header\n")

            if colors_rgb is None:
                for x, y, z in points_xyz:
                    f.write(f"{x} {y} {z}\n")
            else:
                for (x, y, z), (r, g, b) in zip(points_xyz, colors_rgb):
                    f.write(f"{x} {y} {z} {r} {g} {b}\n")

    def export_vggt_dense_ply(self,
        point_maps,          # N,H,W,3
        images_rgb=None,     # optional N,H,W,3 uint8
        conf=None,           # optional N,H,W or N,H,W,1
        conf_threshold=None,
        max_depth=None,
        stride=1,
    ):
        dense_path = os.path.join(self.cam_data.logging_dir, str(self.cam_data.script_id), f"fused_dense_vggt.ply")
        points = np.asarray(point_maps)

        if points.ndim != 4 or points.shape[-1] != 3:
            raise ValueError(f"Expected point_maps with shape (N,H,W,3), got {points.shape}")

        # Optional spatial subsampling. Useful because dense VGGT clouds can be huge.
        points = points[:, ::stride, ::stride, :]

        colors = None
        if images_rgb is not None:
            colors = np.asarray(images_rgb)[:, ::stride, ::stride, :]

        valid = np.isfinite(points).all(axis=-1)

        # Optional confidence filtering
        if conf is not None and conf_threshold is not None:
            conf_arr = self.to_numpy(conf)
            # depth_conf from VGGT: 1,N,H,W -> N,H,W
            if conf_arr.ndim == 4 and conf_arr.shape[0] == 1:
                conf_arr = conf_arr[0]
            elif conf_arr.ndim == 4 and conf_arr.shape[-1] == 1:
                conf_arr = conf_arr[..., 0]
            conf_arr = conf_arr[:, ::stride, ::stride]
            valid &= conf_arr >= conf_threshold

        # Optional depth/radius filtering to remove far junk
        if max_depth is not None:
            radius = np.linalg.norm(points, axis=-1)
            valid &= radius <= max_depth

        flat_points = points[valid].reshape(-1, 3)

        flat_colors = None
        if colors is not None:
            flat_colors = colors[valid].reshape(-1, 3).astype(np.uint8)

        self.save_ply_xyzrgb(dense_path, flat_points, flat_colors)

        print(flat_points.shape)
        return flat_points

    def point_density(self, points):
            mins = points.min(axis=0)
            maxs = points.max(axis=0)
            volume = np.prod(maxs - mins)
            return len(points) / volume
        
    def coverage(self, points, voxel=0.05):
        voxels = np.floor(points / voxel).astype(int)
        return len(np.unique(voxels, axis=0))
    
    def depth_consistency(self, depth_maps):
        """
        depth_maps: list of (H, W) depth arrays
        """
        diffs = []
        for i in range(depth_maps.shape[0] - 1):
            d1, d2 = depth_maps[i], depth_maps[i+1]
            mask = np.isfinite(d1) & np.isfinite(d2)
            diffs.append(np.abs(d1[mask] - d2[mask]))

        return np.mean(np.concatenate(diffs))
    
class Dense3DReconstructionMono(DenseSceneEstimation):
    def __init__(self, 
                 cam_data: CameraData,
                 use_gpu: bool = True,
                 reproj_error: float = 3.0,
                 min_triangulation_angle: float = 1.0,
                 num_samples: int = 15,
                 num_iterations: int = 5):

        module_name = "Dense3DReconstructionMono"
        description = f"""
Densely reconstructs a 3D scene utilizing pre-processed information of the sparsely reconstructed scene
(Depends on Sparse Reconstruction Module). Camera Poses are estimated prior to thie module through the camera 
pose estimation  module. The sparse scene is reconstructed using the Sparse Reconstruction Modules, with the inclusion
of Feature Tracking and Pose estimation data being processed prior to full scene reconstruction.
Use this module when specified for dense reconstruction. Utilize this module in conjuction with the Camera Pose estimation
module, feature tracking module, and sparse scene reconstruction modules.
Computation time should partially matter when invoking this tool, KEEP IN MIND of system constraints such as GPU memory prior
to USING THIS TOOL (Less GPU memory is not a constraint here, but it is a longer runtime). 

Use this module if utilizing the classical approach for scene reconstruction as the methodology. 

Initialization/Function Parameters:
- use_gpu: Whether to use GPU or not.
    - default (bool): True,
- reproj_error: Maximum geometric consistency cost in terms of the forward-backward reprojection 
error in pixels.
    - default (float): 3.0,
- min_triangulation_angle: Minimum triangulation angle in degrees for usable 3D points. 
    - default (float): 1.0,
- num_samples: Number of random samples to draw in Monte Carlo sampling (Patch Match Stereo - Colmap).
    - default (int): 15,
- num_iterations: Number of coordinate descent iterations.
    - default (int): 5

Function Call Parameters - Handled Internally from SfMScene in the common API Workflow:
- camera_poses (CameraPose): Estimated camera poses for the given scene. Poses are estimated prior to this function call, 
specifically from the CameraPoseEstimation modules. 
- sparse_scene (Scene): Estimated scene containing information of the sparsely reconstructed scene. Estimated prior to this 
function call specifically from the SparseReconstruction modules.

Module Input - Handled Internally from SfMScene in the common API Workflow:
    sparse_scene (Scene):
        points3D: Points3D 
            Points3D
                - points3D: np.ndarray      [N x 3] Point position in 3D space [x, y, z]
                - color: np.ndarray         [N x 3] Point Color [r, g, b]               
        cam_poses: list[np.ndarray]         [3 x 4] (np.float) Camera pose for each corresponding frame. Each pose is 3x4 (R, T)
        point_to_pose: np.ndarray           [N x 2] List of corresponding camera poses to 3D points [cam_frame, point_index]
        depth_maps: list[np.ndarray]        list[H x W] Depth Maps per frame, formated as HeightxWidth of image shape
    cam_poses (CameraPose):
        camera_pose: list[np.ndarray]   [3 x 4] (np.float) Camera pose for each corresponding frame. Each pose is 3x4 (R, T)
        rotations: list[np.ndarray]     [3 x 3] (np.float) Rotation matrices for each corresponding frame (Derived from camera_pose)
        translations: list[np.ndarray]  [3 x 1] (np.float) Translation matrices for each corresponding frame (Derived from camera_pose)

Module Output - Handled Internally from SfMScene in the common API Workflow:
Scene (Densely reconstructed scene)
"""
        example = f"""
Initialization:
from modules.features import ...
from modules.featurematching import ... (Pair Module), ... (Tracking Module)
from modules.camerapose import ...
from modules.scenereconstruction import ... (Sparse), {module_name} (Dense)
from modules.baseclass import SfMScene

Function Use:
# Step 1: Read in Calibration/Image Data
reconstructed_scene = SfMScene(image_path = image_path, 
                               calibration_path = calibration_path)

# Step 2: Detect Features Prior to Step 3 (Data filled in SfMScene)

# Step 3: Detect Feature Pairwise Matches Prior to Step 4 (Data filled in SfMScene)

# Step 4: Detect Cam Poses Prior to Step 5 Using a Pose Modules

# Step 5: Detect Feature Tracks Prior to Step 6 (Data filled in SfMScene)

# Step 6: Estimate Sparse Reconstruction module using the Classical Method for Step 7

# Step 7: Run Global Optimizer to build Colmap Workspace Piror to step 8

# Step 8: Run Dense Reconstruction Algorithm
reconstructed_scene.{module_name}(reproj_error=3.0,
                                  min_triangulation_angle=1.0,
                                  num_samples=15,
                                  num_iterations=3)
"""
        super().__init__(cam_data = cam_data,
                         module_name=module_name,
                         description=description,
                         example=example)
        
        self.opts = pycolmap.PatchMatchOptions()
        if use_gpu:
            self.opts.gpu_index = "0"
        self.opts.geom_consistency_max_cost = reproj_error
        self.opts.min_triangulation_angle = min_triangulation_angle
        self.opts.num_samples = num_samples
        self.opts.num_iterations = num_iterations

        # self.workspace_path = "C:\\Users\\Anthony\\Documents\\Projects\\scene_agent\\breadth_agent\\results\\workspace"
        # self.workspace_path = Path(__file__).resolve().parents[2] / "results" / "workspace"
        self.workspace_path = Path(self.cam_data.logging_dir + f"/{self.cam_data.script_id}/workspace")

        # Create Dense Directory
        dense_path = str(self.workspace_path / "dense") #f"{self.workspace_path}\\dense"
        fused_path = str(self.workspace_path / "dense_fused") #f"{self.workspace_path}\\dense_fused"

        if os.path.exists(dense_path):
            # Delete the directory and all its contents
            shutil.rmtree(dense_path)
            
        if os.path.exists(fused_path):
            # Delete the directory and all its contents
            shutil.rmtree(fused_path)

        # Recreate an empty directory
        os.makedirs(dense_path)
        os.makedirs(fused_path)

    def build_reconstruction(self,
                             sparse_scene: Scene | None = None,
                             cam_poses: CameraPose | None = None):
        
        # Step 1: Undistort Images
        pycolmap.undistort_images(
            output_path=str(self.workspace_path / "dense"),#f"{self.workspace_path}\\dense",
            input_path=str(self.workspace_path / "sparse"),#f"{self.workspace_path}\\sparse",
            image_path=str(self.workspace_path / "images"),#f"{self.workspace_path}\\images",
            output_type="COLMAP",
        )
        
        # Step 2: Run Patch Match Stereo to create per-image depth maps
        pycolmap.patch_match_stereo(
            workspace_path=str(self.workspace_path / "dense"),#f"{self.workspace_path}\\dense",
            options=self.opts
        )

        # Step 3: Fuse the Depth Maps
        pycolmap.stereo_fusion(
            workspace_path = str(self.workspace_path / "dense"),#f"{self.workspace_path}\\dense",
            output_path = str(self.workspace_path / "dense_fused")#f"{self.workspace_path}\\dense_fused"
        )

        # Grab 
        dense_model_path = str(self.workspace_path / "dense_fused") #os.path.join(self.workspace_path, "dense_fused")
        recon = pycolmap.Reconstruction(dense_model_path)

        # Obtain 3D Points from Dense Reconstruction
        points = np.array([p.xyz for p in recon.points3D.values()])
        colors = np.array([p.color / 255.0 for p in recon.points3D.values()])
        # pcd = o3d.io.read_point_cloud(f"{self.workspace_path}\\dense\\fused.ply")
        
        # points = np.asarray(pcd.points)      # (N, 3)
        print("NUMBER OF POINTS", points.shape)

        # Obtain Depth Maps from 3D Reconstruction
        depth_maps = self.read_colmap_depth()
        print("Number of maps", len(depth_maps))
        print("Depth Map Shape", depth_maps[0].shape)

        # Obtain Poses from SfM reconstruction
        poses = []
        for _, img in recon.images.items():
            if not img.has_pose:  # only registered images have valid poses
                continue

            poses.append(img.cam_from_world().matrix()) # Pose from SfM Reconstruction     
        
        # Construct Scene for Dense Reconstruction
        pts = Points3D()
        pts.set_all_points(points = points,
                           color = colors)
        scene = Scene(points3D = pts,
                      cam_poses = poses,
                      depth_maps = depth_maps,
                      sparse = False)
        dense_path = os.path.join(self.cam_data.logging_dir, str(self.cam_data.script_id), f"fused_dense.ply")
        recon.export_PLY(dense_path) #str(self.workspace_path / "dense" / "fused.ply")) #(os.path.join(self.workspace_path, "dense", "fused.ply"))
        return scene #super().build_reconstruction(sparse_scene)
    
    def read_colmap_depth(self) -> list[np.ndarray]:
        depth_dir = str(self.workspace_path / "dense" / "stereo" / "depth_maps")#os.path.join(self.workspace_path, "dense", "stereo", "depth_maps")
        depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.photometric.bin")))
        # print(depth_files)
        depth_maps = []
        for file in depth_files:
            with open(file, "rb") as f:
                # Read until newline (end of the ASCII header)
                # ---- Read the text header ----
                header_bytes = b""
                count = 0
                while True:
                    byte = f.read(1)
                    if not byte:
                        raise IOError(f"EOF before header finished")
                    header_bytes += byte
                    if byte == b"&":
                        count += 1
                        if count == 3:
                            # The header ends with an extra ampersand
                            # after channels, e.g. "1600&1200&1&"
                            break

                header_str = header_bytes.decode("ascii")
                parts = header_str.strip("&").split("&")

                # Now read the numeric values that follow
                if len(parts) != 3:
                    raise ValueError(f"Unexpected header format: {header_str}")

                width = int(parts[0])
                height = int(parts[1])
                channels = int(parts[2])

                # ---- Read the float32 depth data ----
                num_values = width * height
                depth_data = np.fromfile(f, dtype=np.float32, count=num_values)

                if depth_data.size != num_values:
                    raise ValueError(
                        f"Depth size mismatch, "
                        f"expected {num_values}, got {depth_data.size}"
                    )
                
                depth = depth_data.reshape((height, width))
                depth_maps.append(depth)

        return depth_maps #data.reshape((height, width))