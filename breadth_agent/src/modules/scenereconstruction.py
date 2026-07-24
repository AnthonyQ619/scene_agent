import cv2
import numpy as np
from modules.baseclass import SparseSceneEstimation, DenseSceneEstimation
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms as TF
from itertools import combinations
from typing import Optional
from scipy.optimize import least_squares
import gtsam
import os
import shutil
import struct
import open3d as o3d
import glob
from modules.models.sfm_models.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from modules.models.sfm_models.vggt.utils.geometry import unproject_depth_map_to_point_map
from modules.models.sfm_models.vggt.models.vggt import VGGT
from modules.models.sfm_models.vggt.utils.load_fn import load_and_preprocess_images
from modules.models.sfm_models.vggt.dependency.track_predict import predict_tracks
from modules.models.sfm_models.vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap
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

torch.manual_seed(42)

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
        self.device = f"cuda:{self.cam_data.gpu_num}" if torch.cuda.is_available() else "cpu"

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

        self.width, self.height = self.image_list[0].size
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

class Sparse3DReconstructionVGGTNoFeatures(SparseSceneEstimation):
    def __init__(self,
                cam_data: CameraData,
                ):
        
        module_name = "Sparse3DReconstructionVGGTNoFeatures"
        description = f"""
Sparsely reconstructs a 3D scene utilizing pre-processed information of camera poses and
images of the scene. Camera Poses are estimated prior to thie module through the camera pose estimation 
module, specifically from VGGT pose estimation. This moduls is specifically the case where features detected 
are too sparsely estimated or inaccurate for feature tracking, so this is no-feature-detection module to estimate the 
point maps of VGGT.

This module can reconstruct sparse 3D scenes specifically using a monocular camera. 
This module can reconstruct sparse 3D scenes either through single view or multi-view scenes.
This is determined by the how many images exist in the scene and how many poses were estimated from the previous
module using the VGGT pose estimation tool specifically.

Use this module when specified for ONLY SPARSE reconstruction and the scene doesn't allow for ANY features to be detected
from any of the supported Feature Detections (SIFT, ORB, SuperPoitn). Utilize this module in conjuction with the VGGT pose 
estimation module in these cases where feature detection too low or innacurate for good feature tracks to be detected! 
This module is for reconstructing the scene using the deep learning approach. 
Computation time should not matter when invoking this tool, but keep in mind of system constraints such as GPU memory.

Use this module for cases where feature detection is too unrelieable for bundle adjustment, leaving for large reprojection
errors in the scene, so we build the scene with no detectors as a prior, and estimate the feature tracks after points are 
estimated with VGGT!

Initialization/Function Parameters:
- No Initial Parameterization
    - Reasoning: we detect the features in this module using a deep learning feature tracking with 3D points as a prior, which we 
      estimate with VGGT in this module.

Function Call Parameters - Handled Internally from SfMScene in the common API Workflow:
- cam_poses (CameraPose): Estimated camera poses for the given scene. Poses are estimated prior to this function call, 
specifically from the CameraPoseEstimationVGGT module. 
- tracked_features (PointsMatched): Feature points are NOT tracked prior to this module!
    - Input: None

Module Input - Handled Internally from SfMScene in the common API Workflow:
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
        recon: Pycolmap.Reconstruction      We estimate the Pycolmap.Reconstruction structure in this module with the esitmated 3D points and Feature Tracks
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

# Step 2: Feature Detection is skipped at this stage

# Step 3: Detect Cam Poses (Must use VGGT prior to this step!)
reconstructed_scene.CamPoseEstimatorVGGTModel() 

# Step 4: Feature Tracking is Skipped at this stage

# Step 5: Estimate Sparse Reconstruction using VGGT Module -> 3D Points and Feature Tracks are estimated in this module for Global Optimization!
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

        # WEIGHT_MODULE = "/workspace/model_weights/model.pt"
            
        self.device = f"cuda:{self.cam_data.gpu_num}" if torch.cuda.is_available() else "cpu"

        if self.device == f"cuda:{self.cam_data.gpu_num}":
            # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
            self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            self.dtype = torch.float32

        self.model = VGGT().to(self.device)
        self.model.load_state_dict(torch.load(WEIGHT_MODULE, weights_only=True))
        self.model.eval()

        self.width, self.height = self.image_list[0].size
         # Load Images in correct format for VGGT inference
        to_tensor = TF.ToTensor()
        tensor_img_list = [to_tensor(img) for img in self.image_list]

        self.images = torch.stack(tensor_img_list).to(self.device) 
        self.frame_nums = len(cam_data.image_names)
        self.detector_free_modules.append(module_name)

    def build_reconstruction(self, 
                             tracked_features: PointsMatched | None, 
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

        depth_conf_np = depth_conf.squeeze(0).detach().cpu().numpy()
        depth_map_np = depth_map.squeeze(0).detach().cpu().numpy() 
        extrinsics_np = np.array(cam_poses.camera_pose)
        intrinsics_np = np.array(self.K_mat)
        image_size = np.array(self.images.shape[-2:])

        # Here we use the ext, int, depth_map, and point_map (points3D) to initialize the sparse reconstruction with tracked feature points
        # scene = self.match_tracks_to_point_maps(tracked_features=tracked_features,
        #                                         point_maps = point_map,
        #                                         conf_maps = depth_conf,
        #                                         minimum_observation = self.minimum_observation,
        #                                         img_width = self.width,
        #                                         num_cameras = num_cameras,
        #                                         camera_poses=cam_poses)
        print()
        print("IMAGESIZE", image_size)
        print("IMAGESHAPE", self.images.shape)
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                # Predicting Tracks
                # Using VGGSfM tracker instead of VGGT tracker for efficiency
                # VGGT tracker requires multiple backbone runs to query different frames (this is a problem caused by the training process)
                # Will be fixed in VGGT v2

                # You can also change the pred_tracks to tracks from any other methods
                # e.g., from COLMAP, from CoTracker, or by chaining 2D matches from Lightglue/LoFTR.
                pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
                    self.images,
                    conf=depth_conf_np,
                    points_3d=point_map,
                    masks=None,
                    max_query_pts=4096,
                    query_frame_num=self.frame_nums,
                    keypoint_extractor="sp",
                    fine_tracking=True,
                )

            # torch.cuda.empty_cache()
        torch.cuda.empty_cache() #Empty GPU cache

        track_mask = pred_vis_scores > 0.1

        reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
            points_3d,
            extrinsics_np,
            intrinsics_np,
            pred_tracks,
            image_size,
            masks=track_mask,
            max_reproj_error=8.0,
            shared_camera=True,
            camera_type="SIMPLE_PINHOLE",
            points_rgb=points_rgb,
            image_names=self.cam_data.image_names
        )

        points3D = Points3D()
        points3D.set_all_points(points_3d, points_rgb)

        scene = Scene(points3D = points3D,
                      cam_poses = cam_poses.camera_pose,
                    #   observations= np.vstack(observations_pix),
                      representation = "point cloud",
                    #   bal_data=ba_data,
                      sparse=True,
                      recon=reconstruction)
        
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

        # WEIGHT_MODULE = "/workspace/model_weights/model.pt"
            
        self.device = f"cuda:{self.cam_data.gpu_num}" if torch.cuda.is_available() else "cpu"

        if self.device == f"cuda:{self.cam_data.gpu_num}":
            # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
            self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            self.dtype = torch.float32

        self.model = VGGT().to(self.device)
        self.model.load_state_dict(torch.load(WEIGHT_MODULE, weights_only=True))
        self.model.eval()

        self.width, self.height = self.image_list[0].size
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

# Global Pose and 3D Reconstruction
# class SparseSceneEstimationGLOMAP(SparseSceneEstimation):
#     """
#     GLOMAP-backed global sparse scene estimator.

#     This module belongs under SparseSceneEstimation, not CameraPoseEstimatorClass,
#     because GLOMAP estimates:
#         1. global camera poses
#         2. sparse 3D points
#         3. point tracks / observations

#     Input:
#         PointsMatched feature_pairs

#     Output:
#         Scene with:
#             scene.points3D
#             scene.cam_poses
#             scene.observations

#     Requirements:
#         - GLOMAP installed as a command-line executable.
#         - pycolmap installed.
#         - Pairwise matches stored in PointsMatched.
#         - Intrinsics provided by CameraData.
#     """

#     detector_free_modules = ["SparseSceneEstimationGLOMAP"]

#     def __init__(
#         self,
#         cam_data: CameraData,
#         glomap_bin: str = "glomap",
#         work_dir: str | None = None,
#         min_track_len: int = 2,
#         min_num_matches: int = 30,
#         geometric_verification: bool = True,
#         keep_work_dir: bool = False,
#         track_builder: FeatureTrackFromPairsUnionFind | None = None,
#     ):
#         super().__init__(
#             cam_data=cam_data,
#             module_name="SparseSceneEstimationGLOMAP",
#             description=(
#                 "Global SfM sparse reconstruction using GLOMAP. "
#                 "Builds a COLMAP-compatible database from custom PointsMatched "
#                 "pairwise correspondences, runs GLOMAP, and imports the sparse "
#                 "scene back into the framework."
#             ),
#             example="scene.SparseSceneEstimationGLOMAP()",
#         )

#         self.glomap_bin = glomap_bin
#         self.work_dir = work_dir
#         self.min_track_len = min_track_len
#         self.min_num_matches = min_num_matches
#         self.geometric_verification = geometric_verification
#         self.keep_work_dir = keep_work_dir
#         self.track_builder = track_builder

#         # GLOMAP estimates camera poses, so this module should not require
#         # state.camera_poses from a previous CameraPoseEstimatorClass module.
#         self.requires_camera_poses = False

#     # -------------------------------------------------------------------------
#     # Override SparseSceneEstimation.run_from_state
#     # -------------------------------------------------------------------------

#     def run_from_state(self, state: SceneState) -> Scene:
#         """
#         GLOMAP estimates poses and scene jointly, so unlike the base
#         SparseSceneEstimation class, this should not require state.camera_poses.
#         """

#         feature_pairs = state.feature_pairs

#         if feature_pairs is None:
#             raise RuntimeError(
#                 "[SparseSceneEstimationGLOMAP Error]\n"
#                 "GLOMAP scene estimation requires state.feature_pairs.\n"
#                 "Run a pairwise feature matching module before this module."
#             )

#         try:
#             return self(feature_pairs)
#         except Exception as e:
#             raise RuntimeError(
#                 "[SparseSceneEstimationGLOMAP Error]\n"
#                 "GLOMAP sparse scene estimation failed.\n\n"
#                 "Likely causes:\n"
#                 "- GLOMAP is not installed or not visible on PATH.\n"
#                 "- The COLMAP database export failed.\n"
#                 "- Pairwise feature matches are too weak or too sparse.\n"
#                 "- The image names used in the database do not match the image folder.\n"
#                 "- There are not enough geometrically verified image pairs.\n\n"
#                 "Action needed:\n"
#                 "- Confirm that `glomap mapper` runs from the command line.\n"
#                 "- Check that feature_pairs.pairwise_matches and pairwise_obs_ids are populated.\n"
#                 "- Improve matching or increase the number of reliable pairwise matches.\n"
#                 "- Check the exported database and sparse reconstruction folder.\n\n"
#                 f"Original error: {type(e).__name__}: {e}"
#             ) from e

#     def __call__(self, feature_pairs: PointsMatched) -> Scene:
#         return self.build_reconstruction(feature_pairs, cam_poses=None)

#     # -------------------------------------------------------------------------
#     # Main reconstruction method
#     # -------------------------------------------------------------------------

#     def build_reconstruction(
#         self,
#         tracked_features: PointsMatched,
#         cam_poses: CameraPose | None = None,
#     ) -> Scene:
#         """
#         Main GLOMAP scene estimation entry point.

#         Note:
#             cam_poses is intentionally unused because GLOMAP estimates poses.
#         """

#         feature_pairs = tracked_features

#         if len(feature_pairs.pairwise_matches) == 0:
#             raise RuntimeError("No pairwise matches found for GLOMAP.")

#         if len(feature_pairs.pairwise_obs_ids) != len(feature_pairs.pairwise_matches):
#             raise RuntimeError(
#                 "feature_pairs.pairwise_obs_ids must be populated. "
#                 "Use PointsMatched.set_matching_pair(...) before GLOMAP."
#             )

#         # Build multi-view tracks for your framework-level Scene observations.
#         # GLOMAP itself consumes pairwise matches, but your later global optimizer
#         # likely benefits from Scene observations.
#         if not feature_pairs.multi_view:
#             if self.track_builder is not None:
#                 tracked = self.track_builder.build_tracks_from_pairs(feature_pairs)
#             else:
#                 tracked = FeatureTrackFromPairsUnionFind(
#                     cam_data=self.cam_data,
#                     min_track_len=self.min_track_len,
#                 ).build_tracks_from_pairs(feature_pairs)
#         else:
#             tracked = feature_pairs

#         work_dir = self._prepare_work_dir()
#         database_path = work_dir / "database.db"
#         image_dir = work_dir / "images"
#         sparse_dir = work_dir / "sparse"

#         image_dir.mkdir(parents=True, exist_ok=True)
#         sparse_dir.mkdir(parents=True, exist_ok=True)

#         try:
#             self._stage_images(image_dir=image_dir)
#             self._export_colmap_database(
#                 database_path=database_path,
#                 feature_pairs=feature_pairs,
#             )
#             self._run_glomap(
#                 database_path=database_path,
#                 image_dir=image_dir,
#                 output_dir=sparse_dir,
#             )

#             reconstruction = self._load_first_reconstruction(sparse_dir)

#             scene = self._convert_reconstruction_to_scene(
#                 reconstruction=reconstruction,
#                 tracked_features=tracked,
#             )

#             self._write_metrics(scene=scene, reconstruction=reconstruction)

#             return scene

#         finally:
#             if not self.keep_work_dir and self.work_dir is None:
#                 shutil.rmtree(work_dir, ignore_errors=True)

#     # -------------------------------------------------------------------------
#     # Work directory / images
#     # -------------------------------------------------------------------------

#     def _prepare_work_dir(self) -> Path:
#         if self.work_dir is not None:
#             work_dir = Path(self.work_dir)
#             work_dir.mkdir(parents=True, exist_ok=True)
#             return work_dir

#         return Path(tempfile.mkdtemp(prefix="scene_glomap_"))

#     def _stage_images(self, image_dir: Path) -> None:
#         """
#         GLOMAP/COLMAP requires images on disk.

#         This assumes cam_data.image_list stores PIL Images or image-like objects.
#         If your CameraData already has image file paths, replace this function
#         with symlinks/copies from those paths.
#         """

#         for image_id, image in enumerate(self.image_list):
#             out_path = image_dir / f"{image_id:06d}.png"

#             if hasattr(image, "save"):
#                 image.save(out_path)
#             else:
#                 arr = np.asarray(image)

#                 if arr.ndim == 3 and arr.shape[2] == 3:
#                     arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

#                 cv2.imwrite(str(out_path), arr)

#     # -------------------------------------------------------------------------
#     # COLMAP database export
#     # -------------------------------------------------------------------------

#     def _export_colmap_database(
#         self,
#         database_path: Path,
#         feature_pairs: PointsMatched,
#     ) -> None:
#         """
#         Export custom PointsMatched data into a COLMAP database.

#         Important:
#             COLMAP/GLOMAP databases use per-image keypoint indices.
#             Your PointsMatched data uses stable obs IDs.

#         Therefore:
#             obs_id -> per-image kp_idx
#             pairwise_obs_ids -> pairwise local match indices
#         """

#         if database_path.exists():
#             database_path.unlink()

#         db = COLMAPDatabase.connect(str(database_path))
#         db.create_tables()

#         camera_id = self._add_camera_to_database(db)
#         image_id_map = self._add_images_to_database(db, camera_id)

#         obs_to_kp_idx = self._add_keypoints_to_database(
#             db=db,
#             feature_pairs=feature_pairs,
#             image_id_map=image_id_map,
#         )

#         self._add_matches_to_database(
#             db=db,
#             feature_pairs=feature_pairs,
#             image_id_map=image_id_map,
#             obs_to_kp_idx=obs_to_kp_idx,
#         )

#         db.commit()
#         db.close()

#     def _add_camera_to_database(self, db) -> int:
#         K = self.K_mat
#         fx, fy = float(K[0, 0]), float(K[1, 1])
#         cx, cy = float(K[0, 2]), float(K[1, 2])

#         width = int(self.image_list[0].size[0])
#         height = int(self.image_list[0].size[1])

#         if self.dist is None:
#             model = "PINHOLE"
#             params = np.array([fx, fy, cx, cy], dtype=np.float64)
#         else:
#             d = np.asarray(self.dist, dtype=np.float64).ravel()

#             if len(d) >= 4:
#                 model = "OPENCV"
#                 params = np.array(
#                     [fx, fy, cx, cy, d[0], d[1], d[2], d[3]],
#                     dtype=np.float64,
#                 )
#             else:
#                 model = "PINHOLE"
#                 params = np.array([fx, fy, cx, cy], dtype=np.float64)

#         camera_id = db.add_camera(
#             model=model,
#             width=width,
#             height=height,
#             params=params,
#             prior_focal_length=True,
#         )

#         return int(camera_id)

#     def _add_images_to_database(self, db, camera_id: int) -> dict[int, int]:
#         """
#         Returns:
#             image_id_map[framework_image_id] = colmap_image_id
#         """

#         image_id_map = {}

#         for image_id in range(len(self.image_list)):
#             name = f"{image_id:06d}.png"

#             colmap_image_id = db.add_image(
#                 name=name,
#                 camera_id=camera_id,
#                 image_id=image_id + 1,
#             )

#             image_id_map[image_id] = int(colmap_image_id)

#         return image_id_map

#     def _add_keypoints_to_database(
#         self,
#         db,
#         feature_pairs: PointsMatched,
#         image_id_map: dict[int, int],
#     ) -> dict[int, int]:
#         """
#         Add one keypoint table per image.

#         Returns:
#             obs_to_kp_idx[obs_id] = local keypoint index in that image
#         """

#         image_to_obs_ids: dict[int, list[int]] = {}

#         for obs_id in feature_pairs.obs_xy.keys():
#             image_id = feature_pairs.get_obs_image(obs_id)
#             image_to_obs_ids.setdefault(image_id, []).append(int(obs_id))

#         obs_to_kp_idx = {}

#         for image_id, obs_ids in image_to_obs_ids.items():
#             obs_ids = sorted(obs_ids)

#             keypoints = []

#             for kp_idx, obs_id in enumerate(obs_ids):
#                 xy = np.asarray(
#                     feature_pairs.get_obs_xy(obs_id),
#                     dtype=np.float32,
#                 ).reshape(2)

#                 # COLMAP keypoint format can be Nx2, Nx4, or Nx6.
#                 # Nx2 is sufficient for imported custom keypoints.
#                 keypoints.append([float(xy[0]), float(xy[1])])

#                 obs_to_kp_idx[obs_id] = kp_idx

#             keypoints = np.asarray(keypoints, dtype=np.float32)

#             colmap_image_id = image_id_map[image_id]
#             db.add_keypoints(colmap_image_id, keypoints)

#         return obs_to_kp_idx

#     def _add_matches_to_database(
#         self,
#         db,
#         feature_pairs: PointsMatched,
#         image_id_map: dict[int, int],
#         obs_to_kp_idx: dict[int, int],
#     ) -> None:
#         """
#         Convert pairwise obs IDs into local keypoint-index matches.
#         """

#         for pair_idx, obs_pairs in enumerate(feature_pairs.pairwise_obs_ids):
#             if obs_pairs.shape[0] < self.min_num_matches:
#                 continue

#             # Infer image pair from the first correspondence.
#             obs_i0 = int(obs_pairs[0, 0])
#             obs_j0 = int(obs_pairs[0, 1])

#             image_i = feature_pairs.get_obs_image(obs_i0)
#             image_j = feature_pairs.get_obs_image(obs_j0)

#             colmap_i = image_id_map[image_i]
#             colmap_j = image_id_map[image_j]

#             matches = []

#             for obs_i, obs_j in obs_pairs:
#                 obs_i = int(obs_i)
#                 obs_j = int(obs_j)

#                 if obs_i not in obs_to_kp_idx or obs_j not in obs_to_kp_idx:
#                     continue

#                 kp_i = obs_to_kp_idx[obs_i]
#                 kp_j = obs_to_kp_idx[obs_j]

#                 matches.append([kp_i, kp_j])

#             if len(matches) < self.min_num_matches:
#                 continue

#             matches = np.asarray(matches, dtype=np.uint32)

#             db.add_matches(colmap_i, colmap_j, matches)

#             if self.geometric_verification:
#                 self._add_two_view_geometry(
#                     db=db,
#                     image_id1=colmap_i,
#                     image_id2=colmap_j,
#                     matches=matches,
#                     feature_pairs=feature_pairs,
#                     obs_pairs=obs_pairs,
#                 )

#     def _add_two_view_geometry(
#         self,
#         db,
#         image_id1: int,
#         image_id2: int,
#         matches: np.ndarray,
#         feature_pairs: PointsMatched,
#         obs_pairs: np.ndarray,
#     ) -> None:
#         """
#         Add verified two-view geometry.

#         This uses OpenCV Essential matrix verification from your existing
#         custom correspondences. GLOMAP benefits from verified image pairs.
#         """

#         pts1 = []
#         pts2 = []

#         for obs_i, obs_j in obs_pairs:
#             obs_i = int(obs_i)
#             obs_j = int(obs_j)

#             pts1.append(feature_pairs.get_obs_xy(obs_i))
#             pts2.append(feature_pairs.get_obs_xy(obs_j))

#         pts1 = np.asarray(pts1, dtype=np.float64).reshape(-1, 2)
#         pts2 = np.asarray(pts2, dtype=np.float64).reshape(-1, 2)

#         if len(pts1) < self.min_num_matches:
#             return

#         E, mask = cv2.findEssentialMat(
#             pts1,
#             pts2,
#             self.K_mat,
#             method=cv2.RANSAC,
#             prob=0.999,
#             threshold=1.0,
#         )

#         if E is None or mask is None:
#             return

#         inlier_mask = mask.ravel().astype(bool)

#         if int(inlier_mask.sum()) < self.min_num_matches:
#             return

#         verified_matches = matches[inlier_mask].astype(np.uint32)

#         try:
#             db.add_two_view_geometry(
#                 image_id1,
#                 image_id2,
#                 verified_matches,
#                 E=E.astype(np.float64),
#                 config=2,  # calibrated two-view geometry
#             )
#         except TypeError:
#             # pycolmap / COLMAP database wrappers vary by version.
#             # If your add_two_view_geometry signature differs, adapt this call.
#             db.add_two_view_geometry(
#                 image_id1,
#                 image_id2,
#                 verified_matches,
#             )

#     # -------------------------------------------------------------------------
#     # Run GLOMAP
#     # -------------------------------------------------------------------------

#     def _run_glomap(
#         self,
#         database_path: Path,
#         image_dir: Path,
#         output_dir: Path,
#     ) -> None:
#         """
#         Run external GLOMAP mapper.

#         Typical command:
#             glomap mapper
#                 --database_path database.db
#                 --image_path images
#                 --output_path sparse
#         """

#         cmd = [
#             self.glomap_bin,
#             "mapper",
#             "--database_path",
#             str(database_path),
#             "--image_path",
#             str(image_dir),
#             "--output_path",
#             str(output_dir),
#         ]

#         result = subprocess.run(
#             cmd,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             text=True,
#         )

#         if result.returncode != 0:
#             raise RuntimeError(
#                 "GLOMAP failed.\n\n"
#                 f"Command:\n{' '.join(cmd)}\n\n"
#                 f"STDOUT:\n{result.stdout}\n\n"
#                 f"STDERR:\n{result.stderr}"
#             )

#     # -------------------------------------------------------------------------
#     # Import GLOMAP/COLMAP sparse reconstruction
#     # -------------------------------------------------------------------------

#     def _load_first_reconstruction(self, sparse_dir: Path) -> pycolmap.Reconstruction:
#         """
#         GLOMAP writes a COLMAP sparse reconstruction.

#         Usually this appears as:
#             sparse/0
#         but this function finds the first valid reconstruction directory.
#         """

#         candidates = []

#         for child in sparse_dir.iterdir():
#             if child.is_dir():
#                 candidates.append(child)

#         # Some versions/tools may write directly into sparse_dir.
#         candidates.append(sparse_dir)

#         for candidate in candidates:
#             try:
#                 recon = pycolmap.Reconstruction(str(candidate))

#                 if recon.num_reg_images() > 0:
#                     return recon
#             except Exception:
#                 continue

#         raise RuntimeError(f"No valid reconstruction found in: {sparse_dir}")

#     def _convert_reconstruction_to_scene(
#         self,
#         reconstruction: pycolmap.Reconstruction,
#         tracked_features: PointsMatched,
#     ) -> Scene:
#         """
#         Convert pycolmap.Reconstruction into your framework Scene.

#         This uses:
#             - GLOMAP poses and 3D points from reconstruction
#             - observations from pycolmap tracks when possible
#             - fallback observations from tracked_features.data_matrix
#         """

#         num_images = len(self.image_list)

#         camera_poses = [None for _ in range(num_images)]

#         for image_id in reconstruction.reg_image_ids():
#             img = reconstruction.image(image_id)
#             T = img.cam_from_world()

#             R = T.rotation.matrix()
#             t = np.asarray(T.translation).reshape(3, 1)

#             framework_image_id = int(image_id) - 1

#             if 0 <= framework_image_id < num_images:
#                 camera_poses[framework_image_id] = np.hstack([R, t]).astype(np.float64)

#         # Fill missing poses with identity or skip, depending on your downstream expectations.
#         # I prefer explicit identity only as a placeholder.
#         for i in range(num_images):
#             if camera_poses[i] is None:
#                 camera_poses[i] = np.array(
#                     [
#                         [1.0, 0.0, 0.0, 0.0],
#                         [0.0, 1.0, 0.0, 0.0],
#                         [0.0, 0.0, 1.0, 0.0],
#                     ],
#                     dtype=np.float64,
#                 )

#         points3d_container = Points3D()
#         observations = []

#         point_index = 0

#         for point3D_id in reconstruction.point3D_ids():
#             p3d = reconstruction.point3D(point3D_id)
#             xyz = np.asarray(p3d.xyz, dtype=np.float64).reshape(3)

#             points3d_container.update_points(xyz)

#             for elem in p3d.track.elements:
#                 framework_image_id = int(elem.image_id) - 1
#                 point2D_idx = int(elem.point2D_idx)

#                 if not reconstruction.exists_image(elem.image_id):
#                     continue

#                 img = reconstruction.image(elem.image_id)

#                 if point2D_idx < 0 or point2D_idx >= len(img.points2D):
#                     continue

#                 xy = np.asarray(img.points2D[point2D_idx].xy, dtype=np.float64)

#                 observations.append(
#                     [
#                         framework_image_id,
#                         point_index,
#                         float(xy[0]),
#                         float(xy[1]),
#                     ]
#                 )

#             point_index += 1

#         if len(observations) == 0:
#             observations_arr = np.empty((0, 4), dtype=np.float64)
#         else:
#             observations_arr = np.asarray(observations, dtype=np.float64)

#         scene = Scene(
#             points3D=points3d_container,
#             cam_poses=camera_poses,
#             observations=observations_arr,
#             representation="point cloud",
#             sparse=True,
#         )

#         return scene


class Sparse3DReconstructionIncremental(SparseSceneEstimation):
    """Sparse reconstruction from known poses and pre-built feature tracks.

    The main path follows a COLMAP/GLOMAP-like structure:

        N-view triangulation
        -> point-only robust refinement
        -> per-observation reprojection/depth filtering
        -> retriangulation and refinement until stable
        -> final track-length and triangulation-angle filtering

    A two-view consensus initializer is retained only as a fallback for tracks
    that fail direct N-view initialization because their observations contain
    one or more mismatches.

    Pose convention:
        X_camera = R @ X_world + t
    """

    def __init__(
        self,
        cam_data: CameraData,
        max_reproj_error: float = 3.0,
        reproj_threshold: float = 1.5,
        min_observe: int = 3,
        min_angle: float = 1.0,
        min_inlier_ratio: float = 0.60,
        max_filter_iterations: int = 5,
        use_ransac_fallback: bool = True,
        max_pair_hypotheses: int = 100,
        robust_loss: str = "huber",
        loss_scale: float = 1.0,
        landmark_distance_threshold: float = 10.0,
    ):
        module_name = "Sparse3DReconstruction"
        example = f"""
Initialization:
from modules.scenereconstruction import {module_name}

Function Use:
reconstructed_scene.{module_name}(
    min_observe=3,
    reproj_threshold=1.5,
    max_reproj_error=3.0,
    min_angle=1.0,
)
"""
        super().__init__(
            cam_data=cam_data,
            module_name=module_name,
            description="Sparse 3D reconstruction from known poses and tracks",
            example=example,
        )

        self.minimum_observation = min_observe
        self.min_angle = min_angle
        self.reproj_error_min = max_reproj_error
        self.reprojection_threshold = reproj_threshold
        self.min_inlier_ratio = min_inlier_ratio
        self.max_filter_iterations = max_filter_iterations
        self.use_ransac_fallback = use_ransac_fallback
        self.max_pair_hypotheses = max_pair_hypotheses
        self.robust_loss = robust_loss
        self.loss_scale = loss_scale
        self.landmark_distance_threshold = landmark_distance_threshold

    # ------------------------------------------------------------------
    # Camera helpers
    # ------------------------------------------------------------------

    def _camera_center(self, pose: np.ndarray) -> np.ndarray:
        # pose = self._pose_3x4(pose)
        R = pose[:, :3]
        t = pose[:, 3]
        return -R.T @ t

    def _depth(self, xyz: np.ndarray, pose: np.ndarray) -> float:
        # pose = self._pose_3x4(pose)
        xyz_cam = pose[:, :3] @ xyz + pose[:, 3]
        return float(xyz_cam[2])

    def _project_point(
        self,
        xyz: np.ndarray,
        camera_id: int,
        pose: np.ndarray,
    ) -> Optional[np.ndarray]:

        R = pose[:, :3]
        t = pose[:, 3]
        xyz_cam = R @ xyz + t

        if not np.all(np.isfinite(xyz_cam)) or xyz_cam[2] <= 1e-10:
            return None

        # K, dist = self._camera_parameters(camera_id)
        rvec, _ = cv2.Rodrigues(R)
        projected, _ = cv2.projectPoints(
            objectPoints=np.asarray(xyz, dtype=np.float64).reshape(1, 1, 3),
            rvec=rvec,
            tvec=t.reshape(3, 1),
            cameraMatrix=self.K_mat,
            distCoeffs=self.dist,
        )
        return projected.reshape(2)

    def _normalized_point(self, xy: np.ndarray, camera_id: int) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64).reshape(1, 1, 2)
        return cv2.undistortPoints(xy, self.K_mat, self.dist).reshape(2)

    # ------------------------------------------------------------------
    # Triangulation and projection error
    # ------------------------------------------------------------------
    def _triangulate_linear(
        self,
        views: np.ndarray,
        camera_poses: list[np.ndarray],
    ) -> tuple[Optional[np.ndarray], float]:
        num_views = views.shape[0]
        if num_views < 2:
            return None, np.inf

        A = np.zeros((2 * num_views, 4), dtype=np.float64)

        for row_idx, view in enumerate(views):
            camera_id = int(view[0])
            pose = camera_poses[camera_id]
            x, y = self._normalized_point(view[1:3], camera_id)
            A[2 * row_idx] = x * pose[2] - pose[0]
            A[2 * row_idx + 1] = y * pose[2] - pose[1]

        row_norms = np.linalg.norm(A, axis=1, keepdims=True)
        valid_rows = row_norms[:, 0] > 1e-12
        if np.count_nonzero(valid_rows) < 4:
            return None, np.inf
        A[valid_rows] /= row_norms[valid_rows]

        try:
            _, singular_values, Vt = np.linalg.svd(A)
        except np.linalg.LinAlgError:
            return None, np.inf

        X_h = Vt[-1]
        if not np.all(np.isfinite(X_h)) or abs(X_h[3]) < 1e-12:
            return None, np.inf

        xyz = X_h[:3] / X_h[3]
        if not np.all(np.isfinite(xyz)):
            return None, np.inf

        return xyz

    # Helpers for LOST Implementation 
    def _check_landmark_distance(
        self,
        xyz: np.ndarray,
        views: np.ndarray,
        camera_poses: list[np.ndarray],
    ) -> bool:
        """
        GTSAM-style far-landmark rejection.

        Returns True when the landmark lies within the maximum allowed
        distance from every camera observing the point.
        """

        if self.landmark_distance_threshold <= 0.0:
            return True

        camera_ids = np.unique(
            views[:, 0].astype(np.int64)
        )

        for camera_id in camera_ids:
            camera_center = self._camera_center(
                camera_poses[camera_id]
            )

            landmark_distance = np.linalg.norm(
                xyz - camera_center
            )

            if landmark_distance > self.landmark_distance_threshold:
                return False

        return True

    
    def _triangulate_lost(
        self,
        views: np.ndarray,
        camera_poses: list[np.ndarray],
        ) -> tuple[Optional[np.ndarray], float]:

        # Vars to add if this is successful!
        rank_tol: float = 1e-9
        optimize: bool = True
        use_lost: bool = True
        measurement_sigma: float = 1.0

        # ------------------------------------------------------------
        # Build GTSAM camera set and measurement vector
        # ------------------------------------------------------------
        cameras = gtsam.CameraSetCal3_S2()
        measurements = gtsam.Point2Vector()

        for view in views:
            camera_id = int(view[0])
            pixel_xy = view[1:3]

            # --------------------------------------------------------
            # Get camera-specific calibration if using multi-camera.
            # --------------------------------------------------------
            if self.multi_cam:
                K = np.asarray(
                    self.K_mat[camera_id],
                    dtype=np.float64,
                )

                dist = (
                    None
                    if not np.any(self.dist[camera_id])
                    else np.asarray(
                        self.dist[camera_id],
                        dtype=np.float64,
                    )
                )
            else:
                K = np.asarray(
                    self.K_mat,
                    dtype=np.float64,
                )

                dist = (
                    None
                    if not np.any(self.dist)
                    else np.asarray(
                        self.dist,
                        dtype=np.float64,
                    )
                )

            # --------------------------------------------------------
            # Project convention:
            #
            #     X_cam = R_cw @ X_world + t_cw
            #
            # Convert world->camera to camera->world for GTSAM.
            # --------------------------------------------------------
            pose_cw = np.asarray(
                camera_poses[camera_id],
                dtype=np.float64,
            )

            if pose_cw.shape == (4, 4):
                R_cw = pose_cw[:3, :3]
                t_cw = pose_cw[:3, 3]

            elif pose_cw.shape == (3, 4):
                R_cw = pose_cw[:, :3]
                t_cw = pose_cw[:, 3]

            # Invert world -> camera:
            #
            # R_wc = R_cw.T
            #
            # t_wc = -R_cw.T @ t_cw
            #
            R_wc = R_cw.T
            t_wc = -R_cw.T @ t_cw

            gtsam_pose = gtsam.Pose3(
                gtsam.Rot3(R_wc),
                gtsam.Point3(
                    float(t_wc[0]),
                    float(t_wc[1]),
                    float(t_wc[2]),
                ),
            )

            # -----------------------------------------------------------
            # Handle Camera Distortion Case: Case 1 = None, Case 2 = Dist
            # -----------------------------------------------------------
            if dist is None:
                fx = float(K[0, 0])
                fy = float(K[1, 1])
                skew = float(K[0, 1])
                cx = float(K[0, 2])
                cy = float(K[1, 2])

                calibration = gtsam.Cal3_S2(
                    fx,
                    fy,
                    skew,
                    cx,
                    cy,
                )

                measurement_xy = pixel_xy
            else:
                normalized_xy = cv2.undistortPoints(
                    src=pixel_xy.reshape(1, 1, 2),
                    cameraMatrix=K,
                    distCoeffs=dist,
                ).reshape(2)

                calibration = gtsam.Cal3_S2(
                    1.0,  # fx
                    1.0,  # fy
                    0.0,  # skew
                    0.0,  # cx
                    0.0,  # cy
                )

                measurement_xy = normalized_xy

            # --------------------------------------------------------
            # Construct GTSAM pinhole camera.
            # --------------------------------------------------------
            camera = gtsam.PinholeCameraCal3_S2(
                gtsam_pose,
                calibration,
            )

            cameras.append(camera)

            measurements.append(
                gtsam.Point2(
                    float(measurement_xy[0]),
                    float(measurement_xy[1]),
                )
            )

        # ------------------------------------------------------------
        # Measurement uncertainty
        #
        # LOST uses the measurement-noise model in its weighting.
        # Optimization also requires the noise model.
        # ------------------------------------------------------------
        # noise_model = gtsam.noiseModel.Isotropic.Sigma(
        #     2,
        #     float(measurement_sigma),
        # )
        if dist is None:
            sigma = float(measurement_sigma)
        else:
            # Convert an approximate pixel-domain sigma to normalized units.
            #
            # This uses the average focal length. For multi-camera tracks,
            # cameras may have different focal lengths, so this is only an
            # approximation because triangulatePoint3 accepts a single noise
            # model for the measurement set.
            focal_lengths = []

            for view in views:
                camera_id = int(view[0])

                if self.multi_cam:
                    K_i = np.asarray(
                        self.K_mat[camera_id],
                        dtype=np.float64,
                    )
                else:
                    K_i = np.asarray(
                        self.K_mat,
                        dtype=np.float64,
                    )

                focal_lengths.extend([
                    float(K_i[0, 0]),
                    float(K_i[1, 1]),
                ])

            mean_focal = float(np.mean(focal_lengths))

            if mean_focal <= 0.0:
                raise ValueError(
                    "Invalid focal length encountered while converting "
                    "measurement sigma to normalized coordinates."
                )

            sigma = float(measurement_sigma) / mean_focal

        noise_model = gtsam.noiseModel.Isotropic.Sigma(
            2,
            sigma,
        )

        # ------------------------------------------------------------
        # GTSAM triangulation
        # ------------------------------------------------------------
        try:
            point = gtsam.triangulatePoint3(
                cameras,
                measurements,
                rank_tol,
                optimize,
                noise_model,
                use_lost,
            )
        except RuntimeError:
            # Depending on GTSAM build/configuration this can include
            # rank deficiency or cheirality-related triangulation failure.
            return None 

        xyz = np.asarray(point, dtype=np.float64).reshape(3)

        if not np.all(np.isfinite(xyz)):
            return None 

        return xyz #, condition
        

    def _reprojection_errors_per_view(
        self,
        xyz: np.ndarray,
        views: np.ndarray,
        camera_poses: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        errors = np.full(views.shape[0], np.inf, dtype=np.float64)
        positive_depth = np.zeros(views.shape[0], dtype=bool)

        for idx, view in enumerate(views):
            camera_id = int(view[0])
            positive_depth[idx] = self._depth(
                xyz, camera_poses[camera_id]
            ) > 1e-10
            if not positive_depth[idx]:
                continue

            projected_xy = self._project_point(
                xyz, camera_id, camera_poses[camera_id]
            )
            if projected_xy is None:
                continue

            errors[idx] = np.linalg.norm(projected_xy - view[1:3])

        return errors, positive_depth

    def _refine_point(
        self,
        xyz_initial: np.ndarray,
        views: np.ndarray,
        camera_poses: list[np.ndarray],
        max_iterations: int = 50,
    ) -> Optional[np.ndarray]:
        loss = self.robust_loss
        f_scale = self.loss_scale 

        def residual_function(xyz: np.ndarray) -> np.ndarray:
            residuals = np.empty(2 * views.shape[0], dtype=np.float64)
            for idx, view in enumerate(views):
                camera_id = int(view[0])
                projected_xy = self._project_point(
                    xyz, camera_id, camera_poses[camera_id]
                )
                if projected_xy is None:
                    residuals[2 * idx : 2 * idx + 2] = 1e3
                else:
                    residuals[2 * idx : 2 * idx + 2] = (
                        projected_xy - view[1:3]
                    )
            return residuals

        result = least_squares(
            residual_function,
            x0=np.asarray(xyz_initial, dtype=np.float64),
            method="trf",
            loss=self.robust_loss,
            f_scale=self.loss_scale,
            max_nfev=max_iterations,
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            return None
        return result.x

    # ------------------------------------------------------------------
    # Geometry checks
    # ------------------------------------------------------------------
    def _triangulation_angle_deg(
        self,
        xyz: np.ndarray,
        pose_a: np.ndarray,
        pose_b: np.ndarray,
    ) -> float:
        ray_a = xyz - self._camera_center(pose_a)
        ray_b = xyz - self._camera_center(pose_b)
        norm_a = np.linalg.norm(ray_a)
        norm_b = np.linalg.norm(ray_b)
        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0
        cos_angle = np.dot(ray_a, ray_b) / (norm_a * norm_b)
        return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))

    def _maximum_triangulation_angle_deg(
        self,
        xyz: np.ndarray,
        views: np.ndarray,
        camera_poses: list[np.ndarray],
    ) -> float:
        camera_ids = np.unique(views[:, 0].astype(np.int64))
        if camera_ids.size < 2:
            return 0.0

        return max(
            self._triangulation_angle_deg(
                xyz,
                camera_poses[int(camera_a)],
                camera_poses[int(camera_b)],
            )
            for camera_a, camera_b in combinations(camera_ids, 2)
        )

    # ------------------------------------------------------------------
    # Optional fallback initializer for unstable tracks
    # ------------------------------------------------------------------
    def _pair_consensus_initialization(
        self,
        views: np.ndarray,
        camera_poses: list[np.ndarray],
        reproj_threshold: float,
        min_observations: int,
        min_tri_angle_deg: float,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """Find a two-view point hypothesis supported by the complete track."""
        candidates = []

        for idx_a, idx_b in combinations(range(views.shape[0]), 2):
            pair_views = views[[idx_a, idx_b]]
            #xyz = self._triangulate_linear(pair_views, camera_poses)
            xyz = self._triangulate_lost(pair_views, camera_poses)
            if xyz is None:
                continue

            camera_a = int(views[idx_a, 0])
            camera_b = int(views[idx_b, 0])
            if self._depth(xyz, camera_poses[camera_a]) <= 1e-10:
                continue
            if self._depth(xyz, camera_poses[camera_b]) <= 1e-10:
                continue

            angle = self._triangulation_angle_deg(
                xyz, camera_poses[camera_a], camera_poses[camera_b]
            )
            if angle < min_tri_angle_deg:
                continue

            candidates.append((idx_a, idx_b, angle))#, condition))

        if not candidates:
            return None, None, np.inf

        # This cap is only a fallback-speed heuristic, not COLMAP behavior.
        candidates.sort(key=lambda item: item[2], reverse=True)

        if len(candidates) > self.max_pair_hypotheses:
            rng = np.random.default_rng(41)
            strong_count = self.max_pair_hypotheses // 2
            selected_candidates = candidates[:strong_count]
            remaining = candidates[strong_count:]
            random_count = self.max_pair_hypotheses - strong_count
            if len(remaining) > random_count:
                chosen = rng.choice(len(remaining), random_count, replace=False)
                selected_candidates.extend(remaining[idx] for idx in chosen)
            else:
                selected_candidates.extend(remaining)
            candidates = selected_candidates

        best_xyz = None
        best_mask = None
        # best_condition = np.inf
        best_score = None

        for idx_a, idx_b, angle in candidates:
            xyz = self._triangulate_lost( #self._triangulate_linear(
                views[[idx_a, idx_b]], camera_poses
            )
            if xyz is None:
                continue

            errors, positive_depth = self._reprojection_errors_per_view(
                xyz, views, camera_poses
            )
            mask = positive_depth & np.isfinite(errors) & (
                errors <= reproj_threshold
            )
            count = int(np.count_nonzero(mask))
            if count < min_observations:
                continue

            score = (
                count,
                -float(np.median(errors[mask])),
                -float(np.mean(errors[mask])),
                angle,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_xyz = xyz
                best_mask = mask
                # best_condition = condition

        return best_xyz, best_mask#, best_condition

    # ------------------------------------------------------------------
    # COLMAP/GLOMAP-style track triangulation and observation filtering
    # ------------------------------------------------------------------
    def robust_triangulate_track(
        self,
        views: np.ndarray,
        camera_poses: list[np.ndarray],
    ) -> tuple[np.ndarray]:#TrackTriangulationResult]:

        reproj_threshold = self.reprojection_threshold   
        max_reproj_error = self.reproj_error_min
        min_observations = self.minimum_observation
            
        min_inlier_ratio = self.min_inlier_ratio   
        min_tri_angle_deg = self.min_angle 

        max_filter_iterations = self.max_filter_iterations
        use_ransac_fallback = self.use_ransac_fallback
            

        views = np.asarray(views, dtype=np.float64)

        if views.shape[0] < min_observations:
            return None
        if not np.all(np.isfinite(views)):
            return None

        # views = self._remove_duplicate_camera_observations(views)
        if views is None:
            return None

        original_count = views.shape[0]
        active_mask = np.ones(original_count, dtype=bool)
        initialization = "n_view"

        # 1. Direct N-view initialization: preferred path.
        xyz = self._triangulate_lost(views, camera_poses) #self._triangulate_linear(views, camera_poses)
        direct_valid = xyz is not None

        # print("ORIGINAL POINT EST.", xyz)
        if direct_valid:
            errors, positive_depth = self._reprojection_errors_per_view(
                xyz, views, camera_poses
            )
            direct_mask = positive_depth & np.isfinite(errors) & (
                errors <= max_reproj_error
            )
            direct_valid = np.count_nonzero(direct_mask) >= min_observations

        # 2. Pair-consensus fallback only when the direct track is not viable.
        if not direct_valid:
            if not use_ransac_fallback:
                return None
            xyz, active_mask = (
                self._pair_consensus_initialization(
                    views=views,
                    camera_poses=camera_poses,
                    reproj_threshold=max_reproj_error,
                    min_observations=min_observations,
                    min_tri_angle_deg=min_tri_angle_deg,
                )
            )
            if xyz is None or active_mask is None:
                return None
            initialization = "pair_consensus_fallback"
        else:
            active_mask = direct_mask

        # 3. Iterative point-only refinement and observation-level pruning.
        #    A relaxed threshold is used first; the strict threshold is applied
        #    only after the inlier set and point estimate stabilize.
        for _ in range(max_filter_iterations):
            if np.count_nonzero(active_mask) < min_observations:
                return None

            active_views = views[active_mask]
            xyz_linear = self._triangulate_lost(#self._triangulate_linear(
                active_views, camera_poses
            )
            if xyz_linear is None:
                return None

            xyz_refined = self._refine_point(
                xyz_initial=xyz_linear,
                views=active_views,
                camera_poses=camera_poses,
            )
            if xyz_refined is None:
                break
            
            # xyz = xyz_linear if xyz_refined is None else xyz_refined

            errors, positive_depth = self._reprojection_errors_per_view(
                xyz_refined, views, camera_poses
            )
            updated_mask = positive_depth & np.isfinite(errors) & (
                errors <= max_reproj_error
            )

            if np.count_nonzero(updated_mask) < min_observations:
                return None
            if np.array_equal(updated_mask, active_mask):
                active_mask = updated_mask
                break
            active_mask = updated_mask

        # 4. Final refinement with stabilized observations.
        final_views = views[active_mask]
        xyz_linear = self._triangulate_lost(#self._triangulate_linear(
            final_views, camera_poses
        )
        if xyz_linear is None:
            return None

        xyz_refined = self._refine_point(
            xyz_initial=xyz_linear,
            views=final_views,
            camera_poses=camera_poses,
        )
        xyz = xyz_linear if xyz_refined is None else xyz_refined

        errors, positive_depth = self._reprojection_errors_per_view(
            xyz, views, camera_poses
        )

        # 5. Strict final observation filter.
        final_mask = active_mask & positive_depth & np.isfinite(errors) & (
            errors <= reproj_threshold
        )

        # Avoid deleting a geometrically valid point merely because the strict
        # threshold is too aggressive. Fall back to the configured hard limit,
        # but still enforce median quality below.
        if np.count_nonzero(final_mask) < min_observations:
            final_mask = active_mask & positive_depth & np.isfinite(errors) & (
                errors <= max_reproj_error
            )

        num_inliers = int(np.count_nonzero(final_mask))
        if num_inliers < min_observations:
            return None
        if num_inliers / original_count < min_inlier_ratio:
            return None

        final_views = views[final_mask]

        # Re-optimize after the final observation removal.
        xyz_linear = self._triangulate_lost(#self._triangulate_linear(
            final_views, camera_poses
        )
        if xyz_linear is None:
            return None
        xyz_refined = self._refine_point(
            xyz_initial=xyz_linear,
            views=final_views,
            camera_poses=camera_poses,
        )
        xyz = xyz_linear if xyz_refined is None else xyz_refined

        final_errors, final_positive_depth = self._reprojection_errors_per_view(
            xyz, views, camera_poses
        )
        final_mask = final_mask & final_positive_depth & np.isfinite(final_errors) & (
            final_errors <= max_reproj_error
        )

        if np.count_nonzero(final_mask) < min_observations:
            return None

        final_views = views[final_mask]
        inlier_errors = final_errors[final_mask]
        max_tri_angle = self._maximum_triangulation_angle_deg(
            xyz, final_views, camera_poses
        )

        if max_tri_angle < min_tri_angle_deg:
            return None
        if float(np.median(inlier_errors)) > reproj_threshold:
            return None
        if float(np.max(inlier_errors)) > max_reproj_error:
            return None
        if not self._check_landmark_distance(
            xyz=xyz,
            views=final_views,
            camera_poses=camera_poses,
        ):
            return None

        return (xyz, final_views, final_mask)
        # return TrackTriangulationResult(
        #     xyz=xyz,
        #     inlier_views=final_views,
        #     inlier_mask=final_mask,
        #     reprojection_errors=final_errors,
        #     median_error=float(np.median(inlier_errors)),
        #     max_error=float(np.max(inlier_errors)),
        #     max_tri_angle_deg=max_tri_angle,
        #     condition_number=condition_number,
        #     initialization=initialization,
        # )

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------
    def build_reconstruction(
        self,
        points: PointsMatched,
        camera_poses: CameraPose,
    ) -> Scene:

        # if not self.multi_view:
        if not points.multi_view:
            return self._build_two_view_reconstruction(points, camera_poses)

        if not points.multi_view:
            raise ValueError(
                "Features are not tracked. Set multi_view=False for "
                "pairwise-only reconstruction."
            )

        points_3d = Points3D()
        observations_pixel = []
        accepted_track_ids = []
        track_quality = []

        # rejection_counts = {
        #     "too_short": 0,
        #     "triangulation_failed": 0,
        #     "accepted": 0,
        #     "ransac_fallback": 0,
        # }

        point_index = 0
        for track_id in tqdm(range(points.point_count)):
            views = np.asarray(points.access_point3D(track_id), dtype=np.float64)

            if views.shape[0] < self.minimum_observation:
                # rejection_counts["too_short"] += 1
                continue

            result = self.robust_triangulate_track(
                views=views,
                camera_poses=camera_poses.camera_pose,
            )
            if result is None:
                # rejection_counts["triangulation_failed"] += 1
                continue

            final_xyz, inlier_views, mask = result[:]
            point_indices = np.full(
                (inlier_views.shape[0], 1),
                point_index,
                dtype=np.int64,
            )
            camera_indices = inlier_views[:, 0:1].astype(np.int64)
            observation_pixel = np.hstack(
                (
                    camera_indices,
                    point_indices,
                    inlier_views[:, 1:3],
                )
            )
            observations_pixel.append(observation_pixel)
            points_3d.update_points(final_xyz)
            accepted_track_ids.append(track_id)

            # track_quality.append(
            #     {
            #         "track_id": track_id,
            #         "point_index": point_index,
            #         "initial_track_length": int(views.shape[0]),
            #         "inlier_track_length": int(result.inlier_views.shape[0]),
            #         "inlier_ratio": float(
            #             result.inlier_views.shape[0] / views.shape[0]
            #         ),
            #         "median_reprojection_error": result.median_error,
            #         "max_reprojection_error": result.max_error,
            #         "max_triangulation_angle_deg": result.max_tri_angle_deg,
            #         "condition_number": result.condition_number,
            #         "initialization": result.initialization,
            #     }
            # )

            # rejection_counts["accepted"] += 1
            # if result.initialization == "pair_consensus_fallback":
            #     rejection_counts["ransac_fallback"] += 1
            point_index += 1

        if point_index == 0:
            raise RuntimeError(
                "No valid 3D points were reconstructed. Inspect pose convention, "
                "intrinsics, tracks, reprojection thresholds, and triangulation angles."
            )

        print("NUMBer OF POINTS")
        print(points_3d.points3D.shape)
        scene = Scene(
            points3D=points_3d,
            cam_poses=camera_poses.camera_pose,
            observations=np.vstack(observations_pixel),
            representation="point cloud",
            sparse=True,
        )

        # Retain these only if Scene permits dynamic metadata attributes.
        # scene.accepted_track_ids = np.asarray(accepted_track_ids, dtype=np.int64)
        # scene.track_quality = track_quality
        # scene.reconstruction_stats = rejection_counts

        return scene

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
This is especially useful for cases where feature tracking fails, even with robust matchers/trackers and feature detectors.
Opt for this module in those cases!

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

        # WEIGHT_MODULE = "/workspace/model_weights/model.pt"
        self.device = f"cuda:{self.cam_data.gpu_num}" if torch.cuda.is_available() else "cpu"

        if self.device == f"cuda:{self.cam_data.gpu_num}":
            # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
            self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            self.dtype = torch.float32

        self.model = VGGT().to(self.device)
        self.model.load_state_dict(torch.load(WEIGHT_MODULE, weights_only=True))
        self.model.eval()

        self.width, self.height = self.image_list[0].size
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
### EXAMPLE 1 - CLASSICAL FEATURE BASED###
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

### EXAMPLE 2 - VGGT PIPELINE BASED###
# Step 1: Read in Calibration/Image Data
reconstructed_scene = SfMScene(image_path = image_path, 
                               calibration_path = calibration_path)

# Step 2: Detect Features Prior to Step 3 (Data filled in SfMScene)

# Step 3: Don't need Feature pairs for VGGT Pose Estimation pipeline (Ignore)

# Step 4: Detect/Estimate Camera Poses (Utilizing VGGT)
reconstructed_scene.CamPoseEstimatorVGGTModel()

# Step 5: Detect Feature Tracks Prior to Step 6 (Data filled in SfMScene)

# Step 6: Estimate Sparse Reconstruction
reconstructed_scene.Sparse3DReconstructionVGGT(min_observe=4)

# Step 7: Run Optimization Global Optimizer to build Colmap Workspace Piror to step 8

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
            self.opts.gpu_index = str(self.cam_data.gpu_num)
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