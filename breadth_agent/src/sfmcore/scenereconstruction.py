import cv2
import numpy as np
from sfmcore.baseclass import SparseSceneEstimation, DenseSceneEstimation
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms as TF
from itertools import combinations
from typing import Optional, Any
from scipy.optimize import least_squares
import gtsam
import os
import shutil
import struct
import open3d as o3d
import glob
import tempfile
from sfmcore.models.sfm_models.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from sfmcore.models.sfm_models.vggt.utils.geometry import unproject_depth_map_to_point_map
from sfmcore.models.sfm_models.vggt.models.vggt import VGGT
from sfmcore.models.sfm_models.vggt.utils.load_fn import load_and_preprocess_images
from sfmcore.models.sfm_models.vggt.dependency.track_predict import predict_tracks
from sfmcore.models.sfm_models.vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap
from mapanything.models import MapAnything
from mapanything.utils.geometry import closed_form_pose_inverse, depthmap_to_world_frame
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT
from pathlib import Path

from sfmcore.DataTypes.pointDT import Points2D, Points3D
from sfmcore.DataTypes.cameraDT import CameraData
from sfmcore.DataTypes.cameraposeDT import CameraPose
from sfmcore.DataTypes.featmatchDT import PointsMatched
from sfmcore.DataTypes.sceneDT import Scene

torch.manual_seed(42)

import pycolmap

##########################################################################################################
############################################### ML MODULES ###############################################

class Sparse3DReconstructionMapAnything(SparseSceneEstimation):
    requires_camera_poses = False
    def __init__(self,
                 cam_data: CameraData,
                 min_observe: int = 3,
                 update_intrinsics = False):
    
        module_name = "Sparse3DReconstructionMapAnything"

        description = f"""
Reconstructs a sparse 3D scene using the learned MapAnything model. MapAnything predicts
metric scene geometry from one or more images and can optionally use available camera
intrinsics and camera poses as geometric inputs. Unlike classical sparse reconstruction,
this module does not require reliable feature detection, matching, or triangulation to
generate the initial scene geometry.

USE THIS MODULE when sparse features are insufficient, classical pose estimation or
triangulation is unreliable, the scene contains weakly textured regions, or rapid learned
reconstruction is preferred. It is also useful when camera calibration or previously
estimated poses are available and should be used to condition the reconstruction.
Classical incremental or global SfM is preferable when highly accurate feature tracks,
well-optimized poses, and strict geometric consistency are available.

This module supports single-view and multi-view reconstruction. GPU memory should be
considered when processing large images or many views.

Initialization Parameters:
- min_observe: Minimum number of image observations required for a predicted 3D point to
  be retained in the sparse scene. Increasing this value keeps points supported across
  more views, improving multi-view reliability while reducing point count.
    - Default (int): 3
- update_intrinsics: Whether the camera intrinsics stored in CameraData should be updated
  using the intrinsics predicted by MapAnything. Set to False to preserve known calibrated
  intrinsics. Enable when calibration is unavailable or the existing intrinsics are
  considered unreliable.
    - Default (bool): False

Function Call Parameters - Handled Internally by SfMScene:
- cam_poses (CameraPose): Optional camera poses used to condition MapAnything. These may
  come from any compatible CameraPoseEstimator module and are not restricted to VGGT.
- tracked_features (PointsMatched): Optional feature tracks used to select or associate
  predicted 3D points. MapAnything itself does not require classical feature tracks to
  infer scene geometry.

Module Input - Handled Internally by SfMScene:
    CameraData:
        images: Input image set used for reconstruction.
        calibration: Optional camera intrinsics supplied to MapAnything.
        image_size: Image dimensions represented as [width, height].
        image_scale: Scale factors applied when images are resized.

    CameraPose:
        camera_pose: list[np.ndarray]       Camera poses represented as [R | t].
        rotations: list[np.ndarray]         Camera rotation matrices.
        translations: list[np.ndarray]      Camera translation vectors.

    PointsMatched:
        data_matrix: np.ndarray             Multi-view feature observations, when available.
        track_map: dict                     Mapping between tracks and image observations.
        image_size: np.ndarray              Original image dimensions.
        image_scale: list[float]            Image resizing scale factors.

Module Output - Handled Internally by SfMScene:
    Scene:
        points3D: Points3D
            points3D: np.ndarray            Predicted 3D point positions [x, y, z].
            color: np.ndarray               RGB color associated with each 3D point.
        cam_poses: list[np.ndarray]          Input or MapAnything-predicted camera poses.
        observations: np.ndarray             Retained image observations associated with
                                             sparse 3D points.
        depth_maps: list[np.ndarray]         Predicted depth map for each input view.
        sparse: bool                         Set to True for the returned sparse scene.
"""

        example = f"""
Initialization:
from sfmcore.camerapose import CamPoseEstimatorVGGTModel
from sfmcore.scenereconstruction import {module_name}
from sfmcore.baseclass import SfMScene

Function Use:
# Step 1: Load image and optional calibration data.
reconstructed_scene = SfMScene(
    image_path=image_path,
    calibration_path=calibration_path
)
# Step 2: Detect Features Prior to Step 5 (Data filled in SfMScene)

# Step 3: Optionally estimate camera poses before reconstruction.
# MapAnything can also operate without externally estimated poses.
reconstructed_scene.CamPoseEstimatorVGGTModel()

# Step 4: Detect Feature Tracks Prior to Step 5 (Data filled in SfMScene)

# Step 3: Reconstruct the sparse scene using MapAnything.
reconstructed_scene.{module_name}(
    min_observe=3,
    update_intrinsics=False
)
"""
        super().__init__(cam_data = cam_data,
                         module_name=module_name,
                         description=description,
                         example=example)

        dtype = (
        torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        )
        self.device = f"cuda:{self.cam_data.gpu_num}" if torch.cuda.is_available() else "cpu"

        self.model = MapAnything.from_pretrained("facebook/map-anything").to(self.device)
        self.model.eval()        

        data_norm_type = self.model.encoder.data_norm_type

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
            int_torch = torch.from_numpy(np.array(self.K_mat.astype(np.float32))).to(self.device)
            int_torch[:2, :] *= (518/self.width)

            views = []
            for view_idx in range(images.shape[0]):
                view = {
                    "img": images[view_idx][None],  # Add batch dimension
                    "intrinsics": int_torch[None],
                    "data_norm_type": [self.model.encoder.data_norm_type],
                }
                views.append(view)
        else:
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
            all_extrinsics.append(extrinsic[:3, :])
            all_intrinsics.append(intrinsic)
            all_depth_maps.append(depth_map)
            all_depth_confs.append(depth_conf)
            all_pts3d.append(pts3d)

        # Stack results into arrays
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

USE THIS MODULE when sparse features are insufficient, viewpoint or illumination changes weaken classical matching, 
or rapid reconstruction is needed without a complete feature-to-pose pipeline. It is a strong fallback for weakly 
textured or casually captured scenes, but classical reconstruction with bundle adjustment may be preferable when 
precise geometric consistency is required.

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
from sfmcore.features import ...
from sfmcore.featurematching import ...
from sfmcore.camerapose import CamPoseEstimatorVGGTModel
from sfmcore.scenereconstruction import {module_name}
from sfmcore.baseclass import SfMScene

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

class SparseSceneEstimationCOLMAPGlobal(SparseSceneEstimation):
    requires_camera_poses = False
    detector_free_modules = [
        "SparseSceneEstimationCOLMAPGlobal",
    ]

    def __init__(
        self,
        cam_data: CameraData,
        min_track_len: int = 3,
        min_num_matches: int = 30,
        max_epipolar_error: float = 1.0,
        min_inlier_ratio: float = 0.25,
        verification_confidence: float = 0.999,
        keep_work_dir: bool = False,
        calibrate_view_graph: bool = False,
        num_threads: int = -1,
        random_seed: int = -1,
        refine_focal_length: bool = False,
        refine_principal_point: bool = False,
        refine_extra_params: bool = False,
        min_tri_angle_deg: float = 1.0,
        max_angular_reproj_error_deg: float = 1.0,
        max_normalized_reproj_error: float = 0.01,
        ba_num_iterations: int = 3
    ):

        module_name = "SparseSceneEstimationCOLMAPGlobal"
        description = f"""
Globally estimates camera poses and sparse 3D structure from feature correspondences using view-graph calibration, 
global camera positioning, triangulation, and bundle adjustment. Camera poses should not be estimated before this 
module, because pose estimation is an internal part of the reconstruction process.

USE THIS MODULE for large, unordered, or wide-baseline image collections with sufficient texture, reliable pairwise 
matches, and a well-connected view graph. It is especially useful when incremental reconstruction would be too slow 
or sensitive to its initialization order. Avoid it when feature matching is sparse or unreliable, image groups are 
disconnected, or scenes contain severe texturelessness, repeated patterns, dynamic objects, or poor illumination. 
GLOMAP provides accuracy competitive with incremental COLMAP while offering substantially better scalability.

This can apply for scenes with enough point correspondences between image pairs (minimum should be at least 30 matches)
in image pairs. Scene does not need to be perfect for detected features, but enough to properly conduct feature tracks and 
camera poses with 30 feature matches between image pairs.

Initialization/Function Parameters:
- min_track_len: The minimum number of image observations required for a feature track to be used during global reconstruction and bundle adjustment.
    - Default (int): 3
- min_num_matches: The minimum number of feature matches required between two images for the image pair to be included in the global view graph.
    - Default (int): 30
- max_epipolar_error: The maximum epipolar error allowed for a feature correspondence to be considered an inlier during two-view geometric verification.
    - Default (float): 1.0 pixels
- min_inlier_ratio: The minimum percentage of feature matches that must pass geometric verification for an image pair to be accepted.
    - Default (float): 0.25
    - Default Meaning: At least 25% of matches must be inliers.
- verification_confidence: The confidence level used by RANSAC during two-view geometric verification. Higher values increase verification reliability but may require more iterations.
    - Default (float): 0.999
- keep_work_dir: Determines whether the temporary COLMAP database, staged images, and reconstruction files are kept after processing.
    - Default (bool): False
- calibrate_view_graph: Determines whether COLMAP should refine camera calibration using the verified image-pair graph before global reconstruction.
    - Default (bool): False
    - Note: Usually disabled when accurate camera intrinsics are already provided.
- num_threads: The number of CPU threads used during global reconstruction and optimization.
    - Default (int): -1
    - Default Meaning: Automatically use the available CPU threads.
- random_seed: The random seed used by RANSAC and other randomized reconstruction processes.
    - Default (int): -1
    - Default Meaning: Use a non-fixed or automatically selected seed.
- refine_focal_length: Determines whether the camera focal length is refined during bundle adjustment.
    - Default (bool): False
- refine_principal_point: Determines whether the camera principal point is refined during bundle adjustment.
    - Default (bool): False
- refine_extra_params: Determines whether additional camera distortion parameters are refined during bundle adjustment.
    - Default (bool): False
- min_tri_angle_deg: The minimum triangulation angle required between camera observations to accept a reconstructed 3D point. Larger angles generally produce more stable depth estimates.
    - Default (float): 1.0 degree
- max_angular_reproj_error_deg: The maximum angular reprojection error allowed when validating and filtering reconstructed 3D points.
    - Default (float): 1.0 degree
- max_normalized_reproj_error: The maximum reprojection error allowed after image coordinates are normalized by the camera intrinsics.
    - Default (float): 0.01
- ba_num_iterations: The number of global bundle-adjustment refinement rounds performed during the reconstruction pipeline.
    - Default (int): 3

Function Call Inputs - Handled Internally from SfMScene in the common API Workflow:
- cam_poses (CameraPose): Estimated camera poses for the given scene. Poses are estimated prior to this function call, 
specifically from the CameraPoseEstimation modules. 
    - Required: Set to (None)
- tracked_features (PointsMatched): Feature points tracked across multiple frames to allow Multi-View 3D point estimation. Feature Tracks are 
estimated from the FeatureTracking modules. (Feature Matches also exist in this datatype)

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
from sfmcore.features import ...
from sfmcore.featurematching import ... (Pair Module), ... (Tracking Module)
from sfmcore.camerapose import ...
from sfmcore.scenereconstruction import {module_name}
from sfmcore.baseclass import SfMScene

Function Use:
# Step 1: Read in Calibration/Image Data
reconstructed_scene = SfMScene(image_path = image_path, 
                               calibration_path = calibration_path)

# Step 2: Detect Features Prior to Step 3 (Data filled in SfMScene)

# Step 3: Detect Feature Pairwise Matches Prior to Step 4 (Data filled in SfMScene)

# Step 4 (IGNORE): Detect Cam Poses (Not Needed - Will be set to None)

# Step 5: Detect Feature Tracks Prior to Step 6 (Data filled in SfMScene)

# Step 6: Estimate Sparse Reconstruction using VGGT Module
reconstructed_scene.{module_name}(
    min_observe=3
)
"""

        super().__init__(
            cam_data=cam_data,
            module_name="SparseSceneEstimationCOLMAPGlobal",
            description=(
                "Global sparse reconstruction using COLMAP's integrated "
                "global SfM implementation through PyCOLMAP."
            ),
            example="scene.SparseSceneEstimationCOLMAPGlobal()",
        )

        if min_track_len < 2:
            raise ValueError("min_track_len must be at least 2.")

        if min_num_matches < 5:
            raise ValueError("min_num_matches must be at least 5.")

        if max_epipolar_error <= 0:
            raise ValueError("max_epipolar_error must be positive.")

        if not 0.0 <= min_inlier_ratio <= 1.0:
            raise ValueError(
                "min_inlier_ratio must be in the range [0, 1]."
            )

        if not 0.0 < verification_confidence < 1.0:
            raise ValueError(
                "verification_confidence must be in the range (0, 1)."
            )

        ws_dir = f"{self.cam_data.logging_dir}/{self.cam_data.script_id}/workspace_global/sparse"
        self.work_dir = Path(ws_dir).expanduser().resolve()

        self.min_track_len = int(min_track_len)
        self.min_num_matches = int(min_num_matches)

        self.max_epipolar_error = float(max_epipolar_error)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.verification_confidence = float(
            verification_confidence
        )

        self.keep_work_dir = bool(keep_work_dir)
        self.calibrate_view_graph = bool(calibrate_view_graph)

        self.num_threads = int(num_threads)
        self.random_seed = int(random_seed)

        self.refine_focal_length = bool(refine_focal_length)
        self.refine_principal_point = bool(
            refine_principal_point
        )
        self.refine_extra_params = bool(refine_extra_params)

        self.min_tri_angle_deg = float(min_tri_angle_deg)
        self.max_angular_reproj_error_deg = float(
            max_angular_reproj_error_deg
        )
        self.max_normalized_reproj_error = float(
            max_normalized_reproj_error
        )
        self.ba_num_iterations = int(ba_num_iterations)

        # self.track_builder = track_builder

        self.last_work_dir: Path | None = None
        self.last_database_path: Path | None = None
        self.last_reconstruction: pycolmap.Reconstruction | None = None
        self.last_reconstructions: dict[
            int,
            pycolmap.Reconstruction,
        ] | None = None

    # ------------------------------------------------------------------
    # Public reconstruction entry point
    # ------------------------------------------------------------------

    def build_reconstruction(
        self,
        tracked_features: PointsMatched,
        cam_poses: CameraPose | None = None,
    ) -> Scene:

        self.work_dir.mkdir(parents=True, exist_ok=True)
        database_path = self.work_dir / "database.db"
        image_dir = self.work_dir / "images"
        sparse_dir = self.work_dir / "sparse"

        image_dir.mkdir(parents=True, exist_ok=True)
        sparse_dir.mkdir(parents=True, exist_ok=True)

        self.last_work_dir = self.work_dir
        self.last_database_path = database_path

        should_remove_work_dir = (
            not self.keep_work_dir and self.work_dir is None
        )

        try:
            # Set up images
            for framework_image_id, image in enumerate(self.image_list):
                output_path = image_dir / self._image_name(framework_image_id)
                self._write_image(image=image, output_path=output_path)

            self._export_pycolmap_database(
                database_path=database_path,
                feature_pairs=tracked_features,
            )

            if self.calibrate_view_graph:
                self._calibrate_view_graph(database_path)

            reconstructions = self._run_global_mapping(
                database_path=database_path,
                image_dir=image_dir,
                output_dir=sparse_dir,
            )

            reconstruction = self._select_reconstruction(
                reconstructions
            )

            self.last_reconstructions = reconstructions
            self.last_reconstruction = reconstruction

            scene = self._convert_reconstruction_to_scene(
                reconstruction=reconstruction,
                tracked_features=tracked_features,
            )

            if hasattr(self, "_write_metrics"):
                self._write_metrics(
                    scene=scene,
                    reconstruction=reconstruction,
                )

            return scene

        finally:
            if should_remove_work_dir:
                shutil.rmtree(work_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Image staging
    # ------------------------------------------------------------------

    @staticmethod
    def _write_image(image: Any, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if hasattr(image, "save"):
            image.save(output_path)
            return

        image_array = np.asarray(image)

        if image_array.ndim == 2:
            success = cv2.imwrite(str(output_path), image_array)
        elif (image_array.ndim == 3
            and image_array.shape[2] == 3
        ):
            # CameraData previously stored PIL/RGB images. Convert RGB to BGR
            # before writing with OpenCV.
            bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            success = cv2.imwrite(str(output_path), bgr_image)

    @staticmethod
    def _image_name(framework_image_id: int) -> str:
        return f"{framework_image_id:06d}.png"

    # ------------------------------------------------------------------
    # PyCOLMAP database export
    # ------------------------------------------------------------------

    def _export_pycolmap_database(self,
                                  database_path: Path,
                                  feature_pairs: PointsMatched
                                 ) -> None:
        """
        Serialize the custom feature graph using pycolmap.Database.

        The database contains:

            - camera calibration
            - image records
            - per-image keypoints
            - raw pairwise matches
            - verified two-view geometries
        """

        if database_path.exists():
            database_path.unlink()

        image_to_obs_ids = self._group_observations_by_image(feature_pairs)

        with pycolmap.Database.open(database_path) as database:
            camera_id = self._write_camera(database)

            image_id_map = self._write_images(
                database=database,
                camera_id=camera_id,
            )

            obs_to_kp_idx = self._write_keypoints(
                database=database,
                feature_pairs=feature_pairs,
                image_id_map=image_id_map,
                image_to_obs_ids=image_to_obs_ids,
            )

            self._write_matches_and_geometries(
                database=database,
                feature_pairs=feature_pairs,
                image_id_map=image_id_map,
                obs_to_kp_idx=obs_to_kp_idx,
            )

    def _group_observations_by_image(
        self,
        feature_pairs: PointsMatched,
    ) -> dict[int, list[int]]:
        image_to_obs_ids: dict[int, list[int]] = {}

        for raw_obs_id in feature_pairs.obs_xy:
            obs_id = int(raw_obs_id)
            image_id = int(
                feature_pairs.get_obs_image(obs_id)
            )

            if not 0 <= image_id < len(self.image_list):
                raise IndexError(
                    f"Observation {obs_id} references invalid "
                    f"image index {image_id}."
                )

            image_to_obs_ids.setdefault(
                image_id,
                [],
            ).append(obs_id)

        for obs_ids in image_to_obs_ids.values():
            obs_ids.sort()

        return image_to_obs_ids

    def _write_camera(self, database: pycolmap.Database) -> int:
        K = np.asarray(self.K_mat, dtype=np.float64)
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])

        distortion = (
            np.asarray(self.dist, dtype=np.float64).reshape(-1)
            if self.dist is not None
            else np.empty(0, dtype=np.float64,)
        )

        if distortion.size >= 4:
            camera_model = "OPENCV"
            camera_params = np.array(
                [fx, fy, cx, cy,
                 float(distortion[0]),
                 float(distortion[1]),
                 float(distortion[2]),
                 float(distortion[3]),
                ], dtype=np.float64,
            )

        else:
            camera_model = "PINHOLE"
            camera_params = np.array([fx, fy, cx, cy], dtype=np.float64)

        camera = pycolmap.Camera(
            camera_id=1,
            model=camera_model,
            width=self.width,
            height=self.height,
            params=camera_params,
            has_prior_focal_length=True,
        )

        return int(database.write_camera(camera, use_camera_id=True))

    def _write_images(self, 
                      database: pycolmap.Database, 
                      camera_id: int) -> dict[int, int]:
        """
        Returns:
            Mapping from framework image index to COLMAP image ID.
        """

        image_id_map: dict[int, int] = {}

        for framework_image_id in range(len(self.image_list)):
            colmap_image_id = framework_image_id + 1
            image = pycolmap.Image(
                name=self._image_name(framework_image_id),
                camera_id=camera_id,
                image_id=colmap_image_id,
            )

            written_image_id = database.write_image(image, use_image_id=True)
            image_id_map[framework_image_id] = int(written_image_id)

        return image_id_map

    def _write_keypoints(
        self,
        database: pycolmap.Database,
        feature_pairs: PointsMatched,
        image_id_map: dict[int, int],
        image_to_obs_ids: dict[int, list[int]],
    ) -> dict[int, int]:
        """
        Convert globally stable observation IDs to local COLMAP keypoint
        indices.

        COLMAP keypoint indices are local to each image.
        """

        obs_to_kp_idx: dict[int, int] = {}

        for framework_image_id in range(len(self.image_list)):
            obs_ids = image_to_obs_ids.get(framework_image_id,[])
            keypoints = np.empty((len(obs_ids), 2), dtype=np.float32,)

            for keypoint_index, obs_id in enumerate(obs_ids):
                xy = np.asarray(
                    feature_pairs.get_obs_xy(obs_id),
                    dtype=np.float32,
                ).reshape(2)

                keypoints[keypoint_index] = xy
                obs_to_kp_idx[obs_id] = keypoint_index

            database.write_keypoints(image_id_map[framework_image_id], keypoints)

        return obs_to_kp_idx

    def _write_matches_and_geometries(
        self,
        database: pycolmap.Database,
        feature_pairs: PointsMatched,
        image_id_map: dict[int, int],
        obs_to_kp_idx: dict[int, int],
    ) -> None:
        num_written_pairs = 0

        for pair_index, raw_obs_pairs in enumerate(feature_pairs.pairwise_obs_ids):
            obs_pairs = np.asarray(raw_obs_pairs, dtype=np.int64)
            if obs_pairs.size == 0:
                continue

            if (obs_pairs.ndim != 2) or (obs_pairs.shape[1] != 2):
                raise ValueError(
                    "Each pairwise_obs_ids element must have "
                    f"shape [N, 2]. Pair {pair_index} has shape "
                    f"{obs_pairs.shape}."
                )
            first_obs_i = int(obs_pairs[0, 0])
            first_obs_j = int(obs_pairs[0, 1])
            image_i = int(feature_pairs.get_obs_image(first_obs_i))
            image_j = int(feature_pairs.get_obs_image(first_obs_j))
            
            matches, valid_obs_pairs = (
                self._convert_observation_pairs(
                    obs_pairs=obs_pairs,
                    obs_to_kp_idx=obs_to_kp_idx,
                )
            )

            if len(matches) < self.min_num_matches:
                continue

            colmap_image_i = image_id_map[image_i]
            colmap_image_j = image_id_map[image_j]

            database.write_matches(
                colmap_image_i,
                colmap_image_j,
                matches,
            )

            # TODO: Come back to here to see if it's necessary?
            geometry = self._estimate_two_view_geometry(
                feature_pairs=feature_pairs,
                obs_pairs=valid_obs_pairs,
                matches=matches,
                image_i=image_i,
                image_j=image_j,
            )

            if geometry is None:
                continue

            if (
                len(geometry.inlier_matches)
                < self.min_num_matches
            ):
                continue

            database.write_two_view_geometry(
                colmap_image_i,
                colmap_image_j,
                geometry,
            )

            num_written_pairs += 1

        if num_written_pairs == 0:
            raise RuntimeError(
                "No image pair passed the match-count and geometric "
                "verification requirements."
            )

    @staticmethod
    def _convert_observation_pairs(
        obs_pairs: np.ndarray,
        obs_to_kp_idx: dict[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        matches: list[list[int]] = []
        valid_obs_pairs: list[list[int]] = []

        for raw_obs_i, raw_obs_j in obs_pairs:
            obs_i = int(raw_obs_i)
            obs_j = int(raw_obs_j)

            keypoint_i = obs_to_kp_idx.get(obs_i)
            keypoint_j = obs_to_kp_idx.get(obs_j)

            if keypoint_i is None or keypoint_j is None:
                continue

            matches.append([keypoint_i, keypoint_j,])
            valid_obs_pairs.append([obs_i, obs_j])

        return (np.asarray(matches, dtype=np.uint32).reshape(-1, 2),
                np.asarray(valid_obs_pairs, dtype=np.int64).reshape(-1, 2))

    # ------------------------------------------------------------------
    # Two-view verification
    # ------------------------------------------------------------------

    def _estimate_two_view_geometry(
        self,
        feature_pairs: PointsMatched,
        obs_pairs: np.ndarray,
        matches: np.ndarray,
        image_i: int,
        image_j: int,
    ) -> pycolmap.TwoViewGeometry | None:
        """
        Estimate a calibrated two-view geometry suitable for COLMAP global mapping.

        A geometry is returned only when COLMAP successfully estimates an
        essential matrix and enough geometrically verified correspondences.
        """

        if len(matches) < self.min_num_matches:
            return None

        points_i, obs_to_local_i = self._get_image_observation_array(
            feature_pairs=feature_pairs,
            image_id=image_i,
        )

        points_j, obs_to_local_j = self._get_image_observation_array(
            feature_pairs=feature_pairs,
            image_id=image_j,
        )

        estimator_matches = []
        estimator_to_database_match = []

        for match_index, (raw_obs_i, raw_obs_j) in enumerate(obs_pairs):
            obs_i = int(raw_obs_i)
            obs_j = int(raw_obs_j)

            local_i = obs_to_local_i.get(obs_i)
            local_j = obs_to_local_j.get(obs_j)

            if local_i is None or local_j is None:
                continue

            estimator_matches.append([local_i, local_j])
            estimator_to_database_match.append(match_index)

        estimator_matches = np.asarray(estimator_matches, dtype=np.uint32).reshape(-1, 2)

        if len(estimator_matches) < self.min_num_matches:
            return None

        camera_i = self._make_pycolmap_camera(
            camera_id=1,
            image=self.image_list[image_i],
        )

        camera_j = self._make_pycolmap_camera(
            camera_id=2,
            image=self.image_list[image_j],
        )

        options = pycolmap.TwoViewGeometryOptions()
        options.compute_relative_pose = True

        options.ransac.max_error = self.max_epipolar_error
        options.ransac.min_inlier_ratio = self.min_inlier_ratio
        options.ransac.confidence = self.verification_confidence
        options.ransac.random_seed = self.random_seed

        if hasattr(options.ransac, "num_threads"):
            options.ransac.num_threads = self.num_threads

        geometry = pycolmap.estimate_calibrated_two_view_geometry(
            camera1=camera_i,
            points1=points_i,
            camera2=camera_j,
            points2=points_j,
            matches=estimator_matches,
            options=options,
        )

        if geometry is None:
            return None

        # Critical validation for COLMAP global mapping.
        if geometry.E is None:
            return None

        E = np.asarray(geometry.E, dtype=np.float64)

        if E.shape != (3, 3):
            return None

        if not np.all(np.isfinite(E)):
            return None

        if len(geometry.inlier_matches) < self.min_num_matches:
            return None

        # Map estimator-local match indices back to the keypoint indices stored
        # in the database.
        estimator_pair_to_database_match = {
            (
                int(estimator_matches[k, 0]),
                int(estimator_matches[k, 1]),
            ): int(estimator_to_database_match[k])
            for k in range(len(estimator_matches))
        }

        verified_database_matches = []

        for estimator_match in geometry.inlier_matches:
            pair = (int(estimator_match[0]), int(estimator_match[1]))

            original_match_index = estimator_pair_to_database_match.get(pair)

            if original_match_index is None:
                continue

            verified_database_matches.append(matches[original_match_index])

        verified_database_matches = np.asarray(
            verified_database_matches,
            dtype=np.uint32,
        ).reshape(-1, 2)

        if len(verified_database_matches) < self.min_num_matches:
            return None

        geometry.config = pycolmap.TwoViewGeometryConfiguration.CALIBRATED
        geometry.E = E
        geometry.inlier_matches = verified_database_matches

        return geometry

    def _get_image_observation_array(
        self,
        feature_pairs: PointsMatched,
        image_id: int,
    ) -> tuple[np.ndarray, dict[int, int]]:
        obs_ids = [
            int(obs_id)
            for obs_id in feature_pairs.obs_xy
            if int(feature_pairs.get_obs_image(int(obs_id))) == image_id
        ]

        obs_ids.sort()
        points = np.empty((len(obs_ids), 2), dtype=np.float64)
        obs_to_local: dict[int, int] = {}

        for local_index, obs_id in enumerate(obs_ids):
            points[local_index] = np.asarray(
                feature_pairs.get_obs_xy(obs_id),
                dtype=np.float64,
            ).reshape(2)

            obs_to_local[obs_id] = local_index

        return points, obs_to_local

    def _make_pycolmap_camera(
        self,
        camera_id: int,
        image: Any,
    ) -> pycolmap.Camera:
        K = np.asarray(self.K_mat, dtype=np.float64)

        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])

        # width, height = self._get_image_size(image)

        distortion = (
            np.asarray(
                self.dist,
                dtype=np.float64,
            ).reshape(-1)
            if self.dist is not None
            else np.empty(
                0,
                dtype=np.float64,
            )
        )

        if distortion.size >= 4:
            return pycolmap.Camera(
                camera_id=camera_id,
                model="OPENCV",
                width=self.width,
                height=self.height,
                params=np.array(
                    [fx, fy, cx, cy,
                     distortion[0],
                     distortion[1],
                     distortion[2],
                     distortion[3],
                    ],
                    dtype=np.float64,
                ),
                has_prior_focal_length=True,
            )

        return pycolmap.Camera(
            camera_id=camera_id,
            model="PINHOLE",
            width=self.width,
            height=self.height,
            params=np.array([fx, fy, cx, cy], dtype=np.float64),
            has_prior_focal_length=True,
        )

    # ------------------------------------------------------------------
    # View-graph calibration
    # ------------------------------------------------------------------

    def _calibrate_view_graph(
        self,
        database_path: Path,
    ) -> None:
        options = pycolmap.ViewGraphCalibrationOptions()

        options.random_seed = self.random_seed
        options.relpose_max_error = self.max_epipolar_error
        options.relpose_min_num_inliers = self.min_num_matches
        options.relpose_min_inlier_ratio = self.min_inlier_ratio

        success = pycolmap.calibrate_view_graph(
            database_path=database_path,
            options=options,
        )

        if not success:
            raise RuntimeError(
                "PyCOLMAP view-graph calibration failed."
            )

    # ------------------------------------------------------------------
    # Official in-process global mapper
    # ------------------------------------------------------------------

    def _run_global_mapping(
        self,
        database_path: Path,
        image_dir: Path,
        output_dir: Path,
    ) -> dict[int, pycolmap.Reconstruction]:
        if not hasattr(pycolmap, "global_mapping"):
            version = getattr(
                pycolmap,
                "__version__",
                "unknown",
            )

            raise RuntimeError(
                "The installed PyCOLMAP build does not expose "
                "pycolmap.global_mapping(). "
                f"Installed version: {version}."
            )

        options = self._create_global_mapping_options()

        reconstructions = pycolmap.global_mapping(
            database_path=database_path,
            image_path=image_dir,
            output_path=output_dir,
            options=options,
        )

        if not reconstructions:
            raise RuntimeError(
                "PyCOLMAP global mapping completed without "
                "producing a reconstruction."
            )

        return reconstructions

    def _create_global_mapping_options(
        self,
    ) -> pycolmap.GlobalPipelineOptions:
        options = pycolmap.GlobalPipelineOptions()

        options.min_num_matches = self.min_num_matches
        options.num_threads = self.num_threads
        options.random_seed = self.random_seed

        mapper = options.mapper

        mapper.num_threads = self.num_threads
        mapper.random_seed = self.random_seed

        mapper.track_min_num_views_per_track = self.min_track_len
        mapper.min_tri_angle_deg = self.min_tri_angle_deg
        mapper.max_angular_reproj_error_deg = self.max_angular_reproj_error_deg
        mapper.max_normalized_reproj_error = self.max_normalized_reproj_error
        mapper.ba_num_iterations = self.ba_num_iterations

        mapper.global_positioning.random_seed = self.random_seed
        mapper.global_positioning.min_num_view_per_track = self.min_track_len

        bundle_options = mapper.bundle_adjustment
        bundle_options.refine_focal_length = self.refine_focal_length
        bundle_options.refine_principal_point = self.refine_principal_point
        bundle_options.refine_extra_params = self.refine_extra_params
        bundle_options.min_track_length = self.min_track_len
        bundle_options.ceres.solver_options.num_threads = self.num_threads

        return options

    # ------------------------------------------------------------------
    # Reconstruction selection
    # ------------------------------------------------------------------

    @staticmethod
    def _select_reconstruction(
        reconstructions: dict[
            int,
            pycolmap.Reconstruction,
        ],
    ) -> pycolmap.Reconstruction:
        valid_reconstructions = [
            reconstruction
            for reconstruction in reconstructions.values()
            if reconstruction.num_reg_images() > 0
        ]

        if not valid_reconstructions:
            raise RuntimeError(
                "Global mapping did not register any images."
            )

        return max(
            valid_reconstructions,
            key=lambda reconstruction: (
                reconstruction.num_reg_images(),
                reconstruction.num_points3D(),
            ),
        )

    # ------------------------------------------------------------------
    # Convert PyCOLMAP reconstruction into framework Scene
    # ------------------------------------------------------------------

    def _convert_reconstruction_to_scene(
        self,
        reconstruction: pycolmap.Reconstruction,
        tracked_features: PointsMatched,
    ) -> Scene:
        """
        Convert COLMAP world-to-camera poses and sparse points into Scene.
        """

        num_images = len(self.image_list)

        camera_poses = [None for _ in range(num_images)]

        image_name_to_framework_id = {
            self._image_name(image_id): image_id
            for image_id in range(num_images)
        }

        for colmap_image_id in (reconstruction.reg_image_ids()):
            image = reconstruction.image(colmap_image_id)
            framework_image_id = image_name_to_framework_id.get(image.name)

            if framework_image_id is None:
                continue

            cam_from_world = image.cam_from_world()

            rotation = np.asarray(
                cam_from_world.rotation.matrix(),
                dtype=np.float64,
            )

            translation = np.asarray(
                cam_from_world.translation,
                dtype=np.float64,
            ).reshape(3, 1)

            camera_poses[framework_image_id] = np.hstack([rotation, translation])

        # Preserve explicit missing-camera information when supported by your
        # CameraPose/Scene types. Identity is retained here for compatibility
        # with the original implementation.
        for image_id, pose in enumerate(camera_poses):
            if pose is None:
                camera_poses[image_id] = np.array([[1.0, 0.0, 0.0, 0.0],
                                                   [0.0, 1.0, 0.0, 0.0],
                                                   [0.0, 0.0, 1.0, 0.0]],
                                                  dtype=np.float64)

        points3d_container = Points3D()
        observations: list[list[float]] = []

        point_index = 0

        for point3d_id in reconstruction.point3D_ids():
            point3d = reconstruction.point3D(point3d_id)
            xyz = np.asarray(point3d.xyz, dtype=np.float64).reshape(3)

            if not np.all(np.isfinite(xyz)):
                continue

            points3d_container.update_points(xyz)

            for track_element in point3d.track.elements:
                colmap_image_id = int(track_element.image_id)
                point2d_index = int(track_element.point2D_idx)

                if not reconstruction.exists_image(colmap_image_id):
                    continue

                image = reconstruction.image(colmap_image_id)
                framework_image_id = image_name_to_framework_id.get(image.name)

                if framework_image_id is None:
                    continue

                if not (0 <= point2d_index < len(image.points2D)):
                    continue

                xy = np.asarray(image.points2D[point2d_index].xy,dtype=np.float64).reshape(2)

                observations.append([float(framework_image_id),
                                     float(point_index),
                                     float(xy[0]),
                                     float(xy[1])])
            point_index += 1

        observations_array = (
            np.asarray(
                observations,
                dtype=np.float64,
            ).reshape(-1, 4)
            if observations
            else np.empty(
                (0, 4),
                dtype=np.float64,
            )
        )

        return Scene(
            points3D=points3d_container,
            cam_poses=camera_poses,
            observations=observations_array,
            representation="point cloud",
            sparse=True,
        )


class Sparse3DReconstructionIncremental(SparseSceneEstimation):
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
        module_name = "Sparse3DReconstructionIncremental"
        description = f"""
Sparsely reconstructs a 3D scene utilizing pre-processed information of camera poses and
detected features tracked across the scene. Camera Poses are estimated prior to this module
through the camera pose estimation module. Features are matched, or tracked, prior to this module 
through the feature matching/tracking module. 

This module can reconstruct sparse 3D scenes specifically using a monocular camera as primary sensor. 
This module can reconstruct sparse 3D scenes either through multi-view or two-view triangulation.
This is determined internally bby the method used to find matching features.
If features are tracked, the multi-view triangulation algorithm will be utilized. 
If features are only matched for corresponding pairs, the two-view triangulation algorithm will be used.

USE THIS MODULE when images have sufficient texture, reliable feature matches, accurate camera calibration, 
strong overlap, and enough parallax for geometric triangulation. It is best for well-lit, mostly static 
scenes where geometric accuracy, bundle-adjustment refinement, and explainable failure checks are more 
important than runtime. Avoid it when feature detection, tracking, or pose registration is unreliable.

This can apply for scenes with high textured with good lighting, but also scenes that do not apply if 
the prerequisite for enough features detected are met. The module is for reconstructing the 
scene using the direct mathematical (Classical) approach.

Initialization/Function Parameters:
- max_reproj_error: The maximum reprojection error allowed for a triangulated 3D point to remain in the reconstructed point cloud. Error is measured in pixel coordinates.
    - Default (float): 3.0 pixels
- reproj_threshold: The reprojection error threshold used to determine whether an individual 2D observation is an inlier during triangulation and point refinement.
    - Default (float): 1.5 pixels
- min_observe: The minimum number of tracked 2D observations required to estimate a 3D point.
    - Default (int): 3
    - Note: Must be at least 2.
- min_angle: The minimum angle required between two camera bearing rays to accept a triangulated 3D point. Larger angles generally produce more accurate depth estimates.
    - Default (float): 1.0 degree
    - Typical Range: 1.0–3.0 degrees
- min_inlier_ratio: The minimum percentage of observations that must agree with the estimated 3D point for it to be accepted.
    - Default (float): 0.60
    - Default Meaning: At least 60% of observations must be inliers.
- max_filter_iterations: The maximum number of iterations used to remove outlier observations and refine the estimated 3D point.
    - Default (int): 5
- use_ransac_fallback: Determines whether RANSAC triangulation is attempted when the primary triangulation method fails.
    - Default (bool): True
- max_pair_hypotheses: The maximum number of observation pairs evaluated as possible triangulation hypotheses for a feature track.
    - Default (int): 100
- robust_loss: The robust loss function used during 3D point refinement to reduce the effect of inaccurate observations.
    - Default (str): "huber"
- loss_scale: Controls how strongly the robust loss function reduces the influence of large reprojection errors.
    - Default (float): 1.0
- landmark_distance_threshold: The maximum accepted distance of a reconstructed 3D point from the expected scene or camera region. Used to reject unstable or extremely distant points.
    - Default (float): 10.0
    - Note: Uses the same scale as the camera poses and reconstructed scene.

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
from sfmcore.features import ...
from sfmcore.featurematching import ... (Pair Module), ... (Tracking Module)
from sfmcore.camerapose import ...
from sfmcore.scenereconstruction import {module_name}
from sfmcore.baseclass import SfMScene

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
    min_observe=3,
    min_angle=2.0,
    max_reproj_error=1.5,
    reproj_threshold=1.0,
    max_filter_iterations=5
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

            # Convert world->camera to camera->world for GTSAM.
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

        if dist is None:
            sigma = float(measurement_sigma)
        else:
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

        return xyz 
        

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
        best_score = None

        for idx_a, idx_b, angle in candidates:
            xyz = self._triangulate_lost( 
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

        return best_xyz, best_mask

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
        xyz = self._triangulate_lost(views, camera_poses) 
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
            xyz_linear = self._triangulate_lost(active_views, camera_poses)
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
        xyz_linear = self._triangulate_lost(final_views, camera_poses)

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
        xyz_linear = self._triangulate_lost(
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

            point_index += 1

        if point_index == 0:
            raise RuntimeError(
                "No valid 3D points were reconstructed. Inspect pose convention, "
                "intrinsics, tracks, reprojection thresholds, and triangulation angles."
            )

        print("NumBer OF POINTS")
        print(points_3d.points3D.shape)
        scene = Scene(
            points3D=points_3d,
            cam_poses=camera_poses.camera_pose,
            observations=np.vstack(observations_pixel),
            representation="point cloud",
            sparse=True,
        )

        return scene

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

Directly predicts dense point maps and depth from one or more images using a learned feed-forward geometry model. 
USE THIS MODULE when rapid reconstruction is required, sparse features are insufficient, classical pose estimation 
fails, or the scene contains weakly textured regions and challenging viewpoint changes.

VGGT does not require a successful sparse feature-matching pipeline and can estimate camera parameters and dense 
geometry jointly. It is therefore a strong fallback for difficult or casually captured image sets. Prefer 
COLMAP MVS when highly accurate, explicitly geometry-optimized reconstruction is required, because learned 
feed-forward predictions may contain greater local or global geometric inconsistency.

Note: 
- Utilize this module in conjuction with the VGGT pose estimation module in these cases where feature detection 
  is low. 
- This is especially useful for cases where feature tracking fails, even with robust matchers/trackers and 
  feature detectors.

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
from sfmcore.camerapose import CamPoseEstimatorVGGTModel
from sfmcore.scenereconstruction import {module_name}
from sfmcore.baseclass import SfMScene

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
    
class Dense3DReconstructionMVS(DenseSceneEstimation):
    def __init__(self, 
                 cam_data: CameraData,
                 use_gpu: bool = True,
                 reproj_error: float = 3.0,
                 min_triangulation_angle: float = 1.0,
                 num_samples: int = 15,
                 num_iterations: int = 5):

        module_name = "Dense3DReconstructionMVS"
        description = f"""
Densely reconstructs a scene using COLMAPs PatchMatch Stereo and depth-map fusion. Camera Poses are 
estimated prior to thie module through the camera pose estimation module. The sparse scene is 
reconstructed using the Sparse Reconstruction Modules, with the inclusion of Feature Tracking and Pose 
estimation data being processed prior to full scene reconstruction.

USE THIS MODULE when the images have accurate geometrically estimated camera poses, strong multi-view 
overlap, sufficient parallax, and consistent visual texture. It is best for well-lit, mostly static 
scenes where fine geometric accuracy is more important than runtime.

Avoid it when sparse matching cannot register enough images, poses have high reprojection error, or the 
scene contains large textureless, reflective, transparent, dynamic, or strongly illumination-varying regions. 
These conditions weaken the photometric and geometric consistency required by classical MVS. Local or global 
bundle adjustment should be applied before dense reconstruction when pose accuracy is uncertain.
Again, computation time should partially matter when invoking this tool, KEEP IN MIND of system constraints 
such as GPU memory prior to USING THIS TOOL (Less GPU memory is not a constraint here, but it is a longer runtime). 

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
from sfmcore.features import ...
from sfmcore.featurematching import ... (Pair Module), ... (Tracking Module)
from sfmcore.camerapose import ...
from sfmcore.scenereconstruction import ... (Sparse), {module_name} (Dense)
from sfmcore.baseclass import SfMScene

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