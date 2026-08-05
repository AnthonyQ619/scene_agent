import numpy as np
import cv2
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms as TF
import json

from sfmcore.baseclass import CameraPoseEstimatorClass, module_metric
from sfmcore.optimization import BundleAdjustmentOptimizerLocal

from sfmcore.DataTypes.cameraposeDT import CameraPose
from sfmcore.DataTypes.featmatchDT import PointsMatched
from sfmcore.DataTypes.cameraDT import CameraData
from sfmcore.DataTypes.sceneDT import IncrementalSfMState

from sfmcore.models.sfm_models.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from sfmcore.models.sfm_models.vggt.utils.geometry import unproject_depth_map_to_point_map
from sfmcore.models.sfm_models.vggt.models.vggt import VGGT
from sfmcore.models.sfm_models.vggt.utils.load_fn import load_and_preprocess_images

import glob

torch.manual_seed(42)
##########################################################################################################
############################################### ML MODULES ###############################################

class CamPoseEstimatorVGGTModel(CameraPoseEstimatorClass):
    def __init__(self, 
                 cam_data: CameraData):
        
        module_name = "CamPoseEstimatorVGGTModel"
        description = f"""
Estimates the camera pose for each frame in a set of images for a monocular camera. The 
process of this module is to estimate the camera pose utilizing the Visual Geometry 
Grounded Transformer (VGGT) Model, a feed-forward neural network that directly infers 
all key 3D attributes of a scene, including camera parameters, point maps, depth maps, 
and 3D point tracks, from one, a few, or hundreds of its views. However, this module
only utilizes the pose estimation feature with intrinsic estimation. This module can estimate
the camera poses from just images alone, without features needing to be detected prior.

USE THIS MODULE (VGGT) for rapid, learned global pose initialization or recovery when correspondence-based 
pose estimation is unreliable. Refine its predictions with bundle adjustment when high geometric precision 
is required.

Note:
BBecause the VGGT model reasons across the complete image set, it is useful for:
- Initializing camera poses before triangulation or bundle adjustment.
- Scenes with weak texture, repetitive patterns, or imperfect pairwise feature matches.
- Quickly producing approximate poses for downstream dense reconstruction.
- Serving as a fallback when Essential-matrix or PnP-based pose estimation fails.

If calibration is provided and we utilize this module due to lack of good feature correspondences for
pose estimation, make note that we do rewrite calibration with this module and update it. 

Initialization Parameters:
- None -> Handled internally through the SfMScene object

Function Call Parameters:
- None

Module Input:
- None
    
Module Output - HANDLED INTERNALLY, DO NOT USE IF SfMScene IS IN USE:
    CameraPose:
        camera_pose: list[np.ndarray]   [3 x 4] (np.float) Camera pose for each corresponding frame. Each pose is 3x4 (R, T)
        rotations: list[np.ndarray]     [3 x 3] (np.float) Rotation matrices for each corresponding frame (Derived from camera_pose)
        translations: list[np.ndarray]  [3 x 1] (np.float) Translation matrices for each corresponding frame (Derived from camera_pose)
"""

        example = f"""
Initialization modules
from sfmcore.baseclass import SfMScene
from sfmcore.features import ....
from sfmcore.featurematching import ....
from sfmcore.camerapose import {module_name}

# Start SfM Pipeline 
# Step 1: Read in Calibration/Image Data
reconstructed_scene = SfMScene(image_path = image_path, 
                            calibration_path = calibration_path)

# Step 2 and 3: Not needed for this module (Don't need to detect Features or conduct Corresponding matches)
# Step 3: 
reconstructed_scene.{module_name}() # Images read in previous step (1)
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

        device = f"cuda:{self.cam_data.gpu_num}" if torch.cuda.is_available() else "cpu"

        if device == f"cuda:{self.cam_data.gpu_num}":
            # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
            self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            self.dtype = torch.float32

        self.model = VGGT().to(device)
        self.model.load_state_dict(torch.load(WEIGHT_MODULE, weights_only=True))
        self.model.eval()

        # Load Images in correct format for VGGT inference
        to_tensor = TF.ToTensor()
        tensor_img_list = [to_tensor(img) for img in self.image_list]

        self.images = torch.stack(tensor_img_list).to(device) 

        self.img_shape = self.image_list[0].size #shape[:2] # Images 
        self.use_base_metrics = False

    def _estimate_camera_poses(self,
                               feature_pairs: PointsMatched | None = None) -> None:
        
        assert self.img_shape[0] == self.img_shape[1], (
            "Input images must be square size, or Height must equal Width. "
            "Must reshape images to a square size, such as (1024, 1024)"
        )

        # VGGT Fixed Resolution to 518 for Inference
        images = F.interpolate(self.images, size=(518, 518), mode="bilinear", align_corners=False)
        new_scale = self.img_shape[0] / 518 # Get change of scale from old shape to new smaller shape

        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=self.dtype):
                images = images[None]  # add batch dimension
                aggregated_tokens_list, ps_idx = self.model.aggregator(images)

            img_shape = images.shape
            # Predict Cameras
            pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
            # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            
        intrinsic_np = intrinsic.squeeze(0).detach().cpu().numpy()
        extrinsic_np = extrinsic.squeeze(0).detach().cpu().numpy()

        for i in range(extrinsic_np.shape[0]):
            self.camera_poses.camera_pose.append(extrinsic_np[i, :, :])
            self.camera_poses.rotations.append(extrinsic_np[i, :, :3])
            self.camera_poses.translations.append(extrinsic_np[i, :, 3:])
        
        # Store Intrinsics -> Reset camerapose to multi_cam approach
        intrins = []
        dists = []   # Assume Camera image were undistorted for now

        intrinsic_np[:, :2, :] *=  new_scale
        print(new_scale)
        for i in range(intrinsic_np.shape[0]):
            intrins.append(intrinsic_np[i, :, :])
            dists.append(np.zeros((1,5), dtype=float))
        
        # Metric Variables
        self.pred_intrinsics = intrins

        self.cam_data.apply_new_calibration(intrins, dists)

        self._save_camera_poses(extrinsic, intrinsic, self.cam_data.image_names, (518, 518))

        torch.cuda.empty_cache() #Empty GPU cache
    
    def _save_camera_poses(self,
        extrinsic,
        intrinsic,
        image_names=None,
        image_size=None,
        extra_metadata=None,
    ):
        """
        Save VGGT camera predictions for pose evaluation.

        Parameters
        ----------
        output_path:
            Path to output .npz file.

        extrinsic:
            VGGT extrinsic output, shape (N, 3, 4).
            Convention: world-to-camera / camera-from-world.

        intrinsic:
            VGGT intrinsic output, shape (N, 3, 3).

        image_names:
            Optional list of image filenames in the same order used by VGGT.

        image_size:
            Optional tuple/list: (H, W).

        extra_metadata:
            Optional dict of extra scalar/string metadata.
        """
        out_path = os.path.join(self.cam_data.logging_dir, str(self.cam_data.script_id), "vggt_poses.npz")

        extrinsics_w2c = self.to_numpy(extrinsic)

        if extrinsics_w2c.ndim == 4 and extrinsics_w2c.shape[0] == 1:
            extrinsics_w2c = extrinsics_w2c[0]

        intrinsics = self.to_numpy(intrinsic)

        if intrinsics.ndim == 4 and intrinsics.shape[0] == 1:
            intrinsics = intrinsics[0]

        if extrinsics_w2c.shape[-2:] != (3, 4):
            raise ValueError(f"Expected extrinsics shape (N,3,4), got {extrinsics_w2c.shape}")

        if intrinsics.shape[-2:] != (3, 3):
            raise ValueError(f"Expected intrinsics shape (N,3,3), got {intrinsics.shape}")

        if len(extrinsics_w2c) != len(intrinsics):
            raise ValueError("Number of extrinsics and intrinsics does not match.")

        R_w2c = extrinsics_w2c[:, :3, :3]
        t_w2c = extrinsics_w2c[:, :3, 3]

        extrinsics_c2w, cam_centers_world = self.w2c_3x4_to_c2w_4x4(extrinsics_w2c)

        save_dict = {
            "extrinsics_w2c": extrinsics_w2c.astype(np.float64),
            "intrinsics": intrinsics.astype(np.float64),
            "R_w2c": R_w2c.astype(np.float64),
            "t_w2c": t_w2c.astype(np.float64),
            "extrinsics_c2w": extrinsics_c2w.astype(np.float64),
            "cam_centers_world": cam_centers_world.astype(np.float64),
            "pose_convention": np.array("OpenCV camera-from-world / world-to-camera"),
        }

        if image_names is not None:
            save_dict["image_names"] = np.asarray(image_names)

        if image_size is not None:
            save_dict["image_size"] = np.asarray(image_size, dtype=np.int32)

        if extra_metadata is not None:
            for k, v in extra_metadata.items():
                save_dict[k] = np.asarray(v)

        np.savez_compressed(out_path, **save_dict)

    @module_metric
    def _metric_pose_matrix_quality(self) -> dict:
        if len(self.camera_poses.camera_pose) == 0:
            return {}

        ortho_errors = []
        det_values = []
        trans_norms = []

        for pose in self.camera_poses.camera_pose:
            R = pose[:, :3]
            t = pose[:, 3:]

            ortho_errors.append(float(np.linalg.norm(R.T @ R - np.eye(3), ord="fro")))
            det_values.append(float(np.linalg.det(R)))
            trans_norms.append(float(np.linalg.norm(t)))

        return {
            "Validity Analysis of Rotation Matrices in Pose Estimation":{
                "Average Rotation Orthonormality Error": float(np.mean(ortho_errors)),
                "Max Rotation Orthonormality Error": float(np.max(ortho_errors)),
                "Average det(R)": float(np.mean(det_values)),
                "Mean Abs det(R)-1": float(np.mean(np.abs(np.array(det_values) - 1.0)))},
            "Translation Analysis for Stable Trajectory Estimation":{
                "Translation Norm Std": float(np.std(trans_norms)),
                "Min Translation Norm": float(np.min(trans_norms)),
                "Max Translation Norm": float(np.max(trans_norms)),
                "Median Translation Norm": float(np.median(trans_norms))}
        }

    # Helper Functions for Poses Recording
    def to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().float().numpy()
        return np.asarray(x)


    def w2c_3x4_to_c2w_4x4(self, extrinsics_w2c):
        """
        Convert OpenCV-style world-to-camera extrinsics [R|t]
        into camera-to-world 4x4 matrices.

        extrinsics_w2c: (N, 3, 4)
        returns: (N, 4, 4)
        """
        extrinsics_w2c = np.asarray(extrinsics_w2c, dtype=np.float64)

        R_w2c = extrinsics_w2c[:, :3, :3]
        t_w2c = extrinsics_w2c[:, :3, 3]

        R_c2w = np.transpose(R_w2c, (0, 2, 1))
        cam_centers_world = -np.einsum("nij,nj->ni", R_c2w, t_w2c)

        N = extrinsics_w2c.shape[0]
        c2w = np.tile(np.eye(4, dtype=np.float64)[None], (N, 1, 1))
        c2w[:, :3, :3] = R_c2w
        c2w[:, :3, 3] = cam_centers_world

        return c2w, cam_centers_world
                
###########################################################################################################
############################################ CLASSICAL MODULES ############################################

class CamPoseEstimatorEssentialToPnP(CameraPoseEstimatorClass):
    def __init__(self, 
                 cam_data: CameraData,
                 iteration_count: int = 200,
                 reprojection_error: float = 3.0,
                 confidence: float = 0.99,
                 ba_per_frame: int = 4,
                 RANSAC_thresh: float = 1.0,
                 optimizer: BundleAdjustmentOptimizerLocal | None = None,
                 min_pnp_inliers: int = 30,
                 min_track_len_triangulate: int = 2,
                 min_stable_track_len: int = 3,
                 max_tri_reproj_error: float = 4.0,
                 min_tri_angle_deg: float = 1.0,
                 initial_pair_search_window: int = 5,
                 ):

        module_name = "CamPoseEstimatorEssentialToPnP"
        description = f"""
Estimates monocular camera poses using an incremental Structure-from-Motion approach. The module estimates an 
Essential matrix from the initial image pair, recovers their relative pose up to an arbitrary scale, triangulates 
initial 3D points, and registers subsequent frames using PnP with existing 2D-to-3D correspondences. This follows 
the standard incremental reconstruction strategy used by COLMAP. (COLMAP)

USE THIS MODULE for calibrated monocular image sequences with gradual camera movement, consistent frame overlap, 
sufficient parallax, and reliable feature tracks. It is less robust to unordered images, low-overlap viewpoint 
changes, pure rotation, weak texture, or dynamic scenes. In those cases, use VGGT for pose initialization or 
recovery, or GLOMAP for globally connected image collections if geometric accuracy is still necessary.

Key Points for when to use this module:
- consistent overlap between consecutive or nearby frames;
- sufficient parallax for reliable initialization and triangulation; mostly static scene content;
- reliable feature tracks connecting new frames to existing 3D points;
- gradual or moderate camera motion, such as video, robotic navigation, or ordered image capture.

Enable the Optimizer when poses produce high reprojection error, unstable triangulation, limited PnP inliers, or 
accumulated trajectory drift. The optimizer applies local bundle adjustment to jointly refine recent camera poses 
and 3D points. This is especially useful with noisy feature tracks or challenging lighting, regardless of the 
selected feature detector. (Hartley and Zisserman)

Initialization/Function Parameters:
- iteration_count: Number of iterations to run the Levenberg-Marquardt algorithm for Pose Estimation with PnP
    - Default (int): 200,
- reprojection_error: Inlier threshold value used by the RANSAC procedure. The parameter value is the maximum allowed distance between the observed and computed point projections to consider it an inlier.
    - Default (float): 3.0
- confidence: The probability that the algorithm produces a useful result. 
    - Default (float): 0.99
- ba_per_frame: The number of frames that are used to estimate poses before a local bundle adjustment optimization is executed
    - Default (int): 4
- optimizer: Optimization parameter to pass in, where in cases of initial poses will lead to poor results, and need more robust
pose estimates for more accurate initial sparse reconstruction estimates.
    - Default (BundleAdjustmentOptimizerLocal): None (Pass BundleAdjustmentOptimizerLocal object that is initialized to activate local optimization.)

Function Call Parameters - HANDLED INTERNALLY, DO NOT USE IF SFMCORE IN USE:
- feature_pairs (PointsMatched): Data Type containing the detected feature correspondences of image pairs
estimated from the feature matcher modules.

Module Input:
    PointsMatched (Matched Features across image pairs)
        pairwise_matches: list[np.ndarray]  [N x 4] -> [x1, y1, x2, y2]. Data Structure to store Pairwise feature matches.
        multi_view: bool                    Determine if Pairwise/Multi-View Feature Matching (Should be False for Pairwise in this function)
        image_size: np.ndarray              [1 x 2] [np.int64] (Simply Image Shape: (W, H))
        image_scale: list[float]            [W_scale, H_scale] if image is resized
    
Module Output: 
    CameraPose:
        camera_pose: list[np.ndarray]   [3 x 4] (np.float) Camera pose for each corresponding frame. Each pose is 3x4 (R, T)
        rotations: list[np.ndarray]     [3 x 3] (np.float) Rotation matrices for each corresponding frame (Derived from camera_pose)
        translations: list[np.ndarray]  [3 x 1] (np.float) Translation matrices for each corresponding frame (Derived from camera_pose)
"""

        example = f"""
Initialization modules
from sfmcore.baseclass import SfMScene
from sfmcore.features import ....
from sfmcore.featurematching import ....
from sfmcore.camerapose import {module_name}

# Start SfM Pipeline 
# Step 1: Read in Calibration/Image Data
reconstructed_scene = SfMScene(image_path = image_path, 
                            calibration_path = calibration_path)

# Step 2: Detect Features must be completed prior!
# Step 3: Feature Matching Pairs module must be completed prior!
# Step 4: Detect Camera Poses

# With Local Bundle Adjustment 
reconstructed_scene.{module_name}(
    iteration_count = 150,
    reprojection_error = 3.0,
    ba_per_frame = 4,
    optimizer = ("BundleAdjustmentOptimizerLocal", {{
        "max_num_iterations": 25,
        "robust_loss": True,
        "use_gpu": False
    }})
)

# WITHOUT local bundle adjustment
reconstructed_scene.{module_name}(
    iteration_count = 150,
    reprojection_error = 3.0
)
"""     
        super().__init__(cam_data = cam_data,
                         module_name=module_name,
                         description=description,
                         example=example)

        self.reproj_error = reprojection_error
        self.iteration_ct = iteration_count
        self.confidence = confidence
        self.ba_per_frame = ba_per_frame
        self.optimizer = optimizer
        self.RANSAC_thresh = RANSAC_thresh

        self.min_pnp_inliers = min_pnp_inliers
        self.min_track_len_triangulate = min_track_len_triangulate
        self.min_stable_track_len = min_stable_track_len
        self.max_tri_reproj_error = max_tri_reproj_error
        self.min_tri_angle_deg = min_tri_angle_deg
        self.initial_pair_search_window = initial_pair_search_window

    def _estimate_camera_poses(self, feature_pairs: PointsMatched) -> None:
        assert feature_pairs.multi_view is False, (
            "Features passed must be two-view correspondences. "
            "Ensure to invoke Feature Matching Two View tools prior to this call."
        )

        assert len(feature_pairs.pairwise_matches) > 0, (
            "No pairwise matches found in PointsMatched."
        )

        assert len(feature_pairs.pairwise_obs_ids) == len(feature_pairs.pairwise_matches), (
            "pairwise_obs_ids must be populated. Use PointsMatched.set_matching_pair(...) "
            "so pairwise matches are normalized into stable observation IDs."
        )

        W, H = self.image_list[0].size

        state = IncrementalSfMState(
            self.K_mat,
            self.dist,
            width=W,
            height=H,
        )

        # Tracks are stored as:
        #   state.tracks[track_id] = list[(image_id, obs_id)]
        #
        # This preserves image_id while using stable observation IDs.
        state.poses = []
        state.points3D = {}
        state.tracks = {}

        # ---------------------------------------------------------------------
        # 1. Choose and initialize a robust first pair.
        # ---------------------------------------------------------------------
        init_pair_idx = 0
        if init_pair_idx is None:
            raise RuntimeError(
                "Could not find a valid initial pair. "
                "Try improving matching, lowering the initial-pair threshold, "
                "or using a sequence with more parallax."
            )

        pose0, pose1, init_inlier_mask = self.estimate_first_pair(
            feature_pairs=feature_pairs,
            pair_index=init_pair_idx,
        )

        # If init_pair_idx > 0, this simple implementation still anchors the
        # selected pair as the first registered pair. For strictly sequential
        # output, prefer init_pair_idx == 0 or add logic to grow both directions.
        if init_pair_idx != 0:
            print(
                f"[Warning] Initial pair selected at index {init_pair_idx}. "
                "This class currently grows forward from that pair. "
                "For strictly frame-0 anchored output, force init_pair_idx=0."
            )

        state.poses.append(pose0)
        state.poses.append(pose1)

        self.camera_poses.camera_pose = list(state.poses)

        # Initialize tracks from the selected pair's inlier obs IDs.
        self.initialize_tracks_from_pair(
            state=state,
            feature_pairs=feature_pairs,
            pair_index=init_pair_idx,
            inlier_mask=init_inlier_mask,
        )

        # Initial triangulation.
        self.update_structure_from_tracks(
            state=state,
            feature_pairs=feature_pairs,
            min_len=self.min_track_len_triangulate,
            frame_id=1,
            force_refresh=True,
        )

        self.filter_points3D(
            state=state,
            feature_pairs=feature_pairs,
            max_reproj_error=self.max_tri_reproj_error,
            min_track_len=self.min_track_len_triangulate,
        )

        # Optional BA right after initialization.
        if self.optimizer is not None:
            state = self.run_local_ba_and_refresh(
                state=state,
                feature_pairs=feature_pairs,
                new_image_id=1,
            )

        # ---------------------------------------------------------------------
        # 2. Incrementally register the rest of the images.
        # ---------------------------------------------------------------------
        start_pair = init_pair_idx + 1

        for pair_index in tqdm(
            range(start_pair, len(feature_pairs.pairwise_matches)),
            desc="Estimating Camera Poses",
        ):
            new_img_id = pair_index + 1

            # Extend tracks using observation IDs.
            obs_prev_curr = feature_pairs.pairwise_obs_ids[pair_index - 1]
            obs_curr_next = feature_pairs.pairwise_obs_ids[pair_index]

            self.extend_tracks_from_obs_ids(
                obs_prev_curr=obs_prev_curr,
                obs_curr_next=obs_curr_next,
                state=state,
                feature_pairs=feature_pairs,
            )

            # Build 2D-3D correspondences for the new image.
            obj_pts, img_pts = self.build_pnp_correspondences_from_obs(
                state=state,
                feature_pairs=feature_pairs,
                image_id=new_img_id,
            )

            new_pose = None

            if obj_pts is not None and len(obj_pts) >= self.min_pnp_inliers:
                new_pose = self.estimate_pose_pnp(
                    point_cloud=obj_pts,
                    pts2=img_pts,
                    prev_pose=state.poses[-1],
                    min_inliers=self.min_pnp_inliers,
                )

            # Fallback to pairwise geometry if PnP failed.
            if new_pose is None:
                new_pose = self.estimate_pose_pairwise_fallback_obs(
                    pair_index=pair_index,
                    feature_pairs=feature_pairs,
                    state=state,
                )

            if new_pose is None:
                print(
                    f"[Warning] Failed to register image {new_img_id}. "
                    "Skipping this frame."
                )
                continue

            state.poses.append(new_pose)
            self.camera_poses.camera_pose = list(state.poses)

            # Record residuals if PnP correspondences were available.
            if obj_pts is not None and img_pts is not None:
                self._record_residual_metric(obj_pts, img_pts, new_pose)

            # Triangulate newly eligible tracks.
            self.update_structure_from_tracks(
                state=state,
                feature_pairs=feature_pairs,
                min_len=self.min_track_len_triangulate,
                frame_id=new_img_id,
                force_refresh=False,
            )

            # Filter bad 3D points before BA.
            self.filter_points3D(
                state=state,
                feature_pairs=feature_pairs,
                max_reproj_error=self.max_tri_reproj_error,
                min_track_len=self.min_track_len_triangulate,
            )

            # Local BA.
            should_run_ba = (
                self.optimizer is not None
                and new_img_id >= 2
                and (
                    new_img_id <= 10
                    or new_img_id % self.ba_per_frame == 0
                )
            )

            if should_run_ba:
                state = self.run_local_ba_and_refresh(
                    state=state,
                    feature_pairs=feature_pairs,
                    new_image_id=new_img_id,
                )

        self.camera_poses.camera_pose = list(state.poses)

    def triangulate_track_best_pair(self, 
                                    track_obs: list[tuple], 
                                    state: IncrementalSfMState, 
                                    cur_img_id: int):
        """
        track_obs: list[(image_id, kp_idx)] with >=2 entries
        returns xyz (3,) or None
        """
        # Choose pair with largest baseline / angle proxy
        # Simple baseline proxy: ||C_i - C_j|| in world coordinates (from poses)
        best = None
        best_score = -1.0

        # Precompute camera centers in world: C = -R^T t  (pose is cam_from_world)
        centers = {}
        for (im, _) in track_obs:
            if im > cur_img_id:
                continue
            P = state.poses[im]                 # 3x4 cam_from_world
            R = P[:, :3]
            t = P[:, 3]
            C = -R.T @ t
            centers[im] = C

        obs_list = track_obs
 
        for a in range(len(obs_list) - 1):
            for b in range(a + 1, len(obs_list) - 1):
                i, _ = obs_list[a]
                j, _ = obs_list[b]
                score = np.linalg.norm(centers[i] - centers[j])
                if score > best_score:
                    best_score = score
                    best = (obs_list[a], obs_list[b])

        if best is None or best_score < 1e-6:
            return None

        # Convert to Observation
        (i1, obsv1), (i2, obsv2) = best

        x1 = state.keypoints[i1][kp1].reshape(2, 1)
        x2 = state.keypoints[i2][kp2].reshape(2, 1)

        # Use pixel projection matrices
        # Normalize Points
        pt1 = cv2.undistortPoints(x1, self.K_mat, self.dist)
        pt2 = cv2.undistortPoints(x2, self.K_mat, self.dist)
        
        P1mtx = np.eye(3) @ state.poses[i1]
        P2mtx = np.eye(3) @ state.poses[i2]

        X_h = cv2.triangulatePoints(P1mtx, P2mtx, pt1, pt2)
        X = (X_h[:3] / X_h[3]).reshape(3,)

        # Basic sanity checks
        if not np.all(np.isfinite(X)):
            return None

        return X

    def three_view_tracking_indices(
                                    self,
                                    matches_prev_curr: np.ndarray,  # (M1,2): kp_{k-1} -> kp_k
                                    matches_curr_next: np.ndarray,  # (M2,2): kp_k -> kp_{k+1}
                                    frame_k: int,
                                    frame_k1: int,
                                    state: IncrementalSfMState,
                                    ):
        
        """
        Updates state.tracks in-place.

        matches_prev_curr[:,0] = kp idx in frame k-1
        matches_prev_curr[:,1] = kp idx in frame k

        matches_curr_next[:,0] = kp idx in frame k
        matches_curr_next[:,1] = kp idx in frame k+1
        """

        # Build fast lookup: kp_k -> kp_{k+1}
        curr_to_next = {}
        for obs_k, obs_k1 in matches_curr_next: 
            curr_to_next[int(obs_k)] = int(obs_k1)

        # Map from (frame, kp_idx) to track_id
        obs_to_track = {}
        for track_id, observ in state.tracks.items():
            for (f_i, obs_i) in observ:#(f, kp) in obs:
                #kp_to_track[(f, kp)] = track_id
                obs_to_track[(f_i, obs_i)] = track_id

        # used_next_kps = set()
        used_next_obs = set()

        # 1) Extend existing tracks
        for obs_prev, obs_curr in matches_prev_curr:
            obs_prev = int(obs_prev)
            obs_curr = int(obs_curr)

            # Is kp_curr observed again in next frame?
            if obs_curr not in curr_to_next:
                continue

            obs_next = curr_to_next[obs_curr]

            # Does this correspondence belong to an existing track?
            key = (frame_k, obs_curr)
            if key in obs_to_track:
                track_id = obs_to_track[key]

                # Append next observation if not already present
                obs = state.tracks[track_id]
                if (frame_k1, obs_next) not in obs:
                    obs.append((frame_k1, obs_next))
                    used_next_obs.add(obs_next)

        # 2) Start new tracks for unmatched correspondences
        for obs_curr, obs_next in matches_curr_next:
            obs_curr = int(obs_curr)
            obs_next = int(obs_next)

            if obs_next in used_next_obs:
                continue

            # If kp_curr not already tracked, start a new track
            if (frame_k, obs_curr) not in obs_to_track:
                new_track_id = len(state.tracks)
                state.tracks[new_track_id] = [
                    (frame_k, obs_curr),
                    (frame_k1, obs_next),
                ]

    def update_structure_from_tracks(self,
                                     state: IncrementalSfMState, 
                                     min_len: int = 2, 
                                     refresh_every: int = 5, 
                                     frame_id: int | None =None):
        """
        Triangulate tracks that have become eligible.
        Optionally refresh points occasionally using best pair if poses changed.
        """
        for track_id, obs in state.tracks.items():
            if len(obs) < min_len:
                continue

            if track_id not in state.points3D:
                X = self.triangulate_track_best_pair(obs, state, frame_id)
                if X is not None:
                    state.points3D[track_id] = X
            else:
                # optional refresh: if BA updated poses, re-triangulate sometimes
                if frame_id is not None and (frame_id % refresh_every) == 0:
                    X = self.triangulate_track_best_pair(obs, state, frame_id)
                    if X is not None:
                        state.points3D[track_id] = X

    def estimate_first_pair(
        self,
        feature_pairs: PointsMatched,
        pair_index: int = 0,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate first two camera poses.

        Pose convention:
            pose0 = [I | 0]
            pose1 = [R | t]

        where pose maps world coordinates into camera coordinates.
        """

        pts1, pts2 = feature_pairs.access_matching_pair(pair_index)

        pts1 = pts1.astype(np.float64)
        pts2 = pts2.astype(np.float64)

        E, mask = cv2.findEssentialMat(
            pts1,
            pts2,
            self.K_mat,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=self.RANSAC_thresh,
        )

        if E is None or mask is None:
            raise RuntimeError(f"Essential matrix failed for pair {pair_index}.")

        _, R, T, pose_mask = cv2.recoverPose(
            points1 = pts2, 
            points2 = pts1, 
            cameraMatrix = self.K_mat, 
            E = E)

        pose0 = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )

        pose1 = np.hstack([R, T.reshape(3, 1)]).astype(np.float64)

        inlier_mask = (pose_mask.ravel() > 0)

        return pose0, pose1, inlier_mask

    def estimate_pose_pnp(
        self,
        point_cloud: np.ndarray,
        pts2: np.ndarray,
        prev_pose: np.ndarray | None = None,
        min_inliers: int = 30,
        ) -> np.ndarray | None:

        object_points = np.asarray(point_cloud, dtype=np.float64).reshape(-1, 1, 3)
        image_points = np.asarray(pts2, dtype=np.float64).reshape(-1, 1, 2)

        if object_points.shape[0] < min_inliers:
            return None

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=object_points,
            imagePoints=image_points,
            cameraMatrix=self.K_mat,
            distCoeffs=self.dist,
            useExtrinsicGuess=False,
            iterationsCount=max(self.iteration_ct, 1000),
            reprojectionError=self.reproj_error,
            confidence=max(self.confidence, 0.999),
            flags=cv2.SOLVEPNP_EPNP,
        )

        if not success or inliers is None or len(inliers) < min_inliers:
            return None

        inlier_idx = inliers.ravel()
        inlier_3d = object_points.reshape(-1, 3)[inlier_idx]
        inlier_2d = image_points.reshape(-1, 2)[inlier_idx]

        # Optional second-stage iterative refinement with previous pose as guess.
        if prev_pose is not None:
            rvec_guess, _ = cv2.Rodrigues(prev_pose[:, :3])
            tvec_guess = prev_pose[:, 3:4].copy()

            success_iter, rvec_iter, tvec_iter = cv2.solvePnP(
                objectPoints=inlier_3d,
                imagePoints=inlier_2d,
                cameraMatrix=self.K_mat,
                distCoeffs=self.dist,
                rvec=rvec_guess,
                tvec=tvec_guess,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            if success_iter:
                rvec, tvec = rvec_iter, tvec_iter

        rvec, tvec = cv2.solvePnPRefineLM(
            inlier_3d,
            inlier_2d,
            self.K_mat,
            self.dist,
            rvec,
            tvec,
        )

        R, _ = cv2.Rodrigues(rvec)
        pose = np.hstack([R, tvec.reshape(3, 1)])

        return pose
        
    def two_view_triangulation(self, 
                               pose_1: np.ndarray, 
                               pose_2: np.ndarray, 
                               pts1: np.ndarray, 
                               pts2: np.ndarray) -> np.ndarray:

        # Normalize Points
        pt1 = cv2.undistortPoints(pts1.T, self.K_mat, self.dist)
        pt2 = cv2.undistortPoints(pts2.T, self.K_mat, self.dist)
        
        P1mtx = np.eye(3) @ pose_1
        P2mtx = np.eye(3) @ pose_2

        # cloud = cv2.triangulatePoints(proj_1, proj_2, pts1.T, pts2.T)
        cloud = cv2.triangulatePoints(P1mtx, P2mtx, pt1, pt2)
        cloud/=cloud[3]

        cloud=cv2.convertPointsFromHomogeneous(cloud.T)
        
        return cloud
    
    def estimate_pose_pairwise_fallback(self, pair_index: int, feature_pairs: PointsMatched, camera_poses: CameraPose) -> np.ndarray:
        """
        Estimate pose for image (pair_index+1) using only pairwise geometry.
        Requires that poses up to pair_index are already in camera_poses.
        """

        # We want pose for image j = pair_index+1 using (pair_index-1 -> pair_index) to triangulate
        if pair_index == 0:
            # shouldn't happen inside the loop (you already initialized first pair)
            raise ValueError("pair_index must be >= 1 for fallback")

        # Triangulate from (pair_index-1, pair_index)
        pts_im1, pts_i = feature_pairs.access_matching_pair(pair_index - 1)
        cloud = self.two_view_triangulation(
            camera_poses.camera_pose[pair_index - 1],
            camera_poses.camera_pose[pair_index],
            pts_im1, pts_i
        )

        # Use correspondences between i and i+1 and find common points with i
        pts_i2, pts_ip1 = feature_pairs.access_matching_pair(pair_index)
        idx, pts_i_common, pts_ip1_common, _, _ = self.three_view_tracking(pts_i, pts_i2, pts_ip1)

        # Call your existing PnP function
        return self.estimate_pose_pnp(cloud[idx], pts_ip1_common)

    def initialize_tracks_from_pair(
        self,
        state: IncrementalSfMState,
        feature_pairs: PointsMatched,
        pair_index: int,
        inlier_mask: np.ndarray | None = None,
        ) -> None:
        obs_pairs = feature_pairs.pairwise_obs_ids[pair_index]

        if inlier_mask is not None:
            obs_pairs = obs_pairs[inlier_mask]

        for obs0, obs1 in obs_pairs:
            obs0 = int(obs0)
            obs1 = int(obs1)

            im0 = feature_pairs.get_obs_image(obs0)
            im1 = feature_pairs.get_obs_image(obs1)

            tid = len(state.tracks)
            state.tracks[tid] = [(im0, obs0), (im1, obs1)]

    # -------------------------------------------------------------------------
    # Track construction
    # -------------------------------------------------------------------------

    def extend_tracks_from_obs_ids(
        self,
        obs_prev_curr: np.ndarray,
        obs_curr_next: np.ndarray,
        state: IncrementalSfMState,
        feature_pairs: PointsMatched,
        ) -> None:
        """
        Extend tracks using stable observation IDs.

        obs_prev_curr:
            Nx2 array [obs in frame k-1, obs in frame k]

        obs_curr_next:
            Mx2 array [obs in frame k, obs in frame k+1]
        """

        curr_to_next = {
            int(obs_curr): int(obs_next)
            for obs_curr, obs_next in obs_curr_next
        }

        obs_to_track = {}

        for track_id, obs_list in state.tracks.items():
            for _, obs_id in obs_list:
                obs_to_track[int(obs_id)] = track_id

        used_next_obs = set()

        # Extend existing tracks.
        for _, obs_curr in obs_prev_curr:
            obs_curr = int(obs_curr)

            if obs_curr not in curr_to_next:
                continue

            obs_next = curr_to_next[obs_curr]

            if obs_curr in obs_to_track:
                track_id = obs_to_track[obs_curr]

                image_next = feature_pairs.get_obs_image(obs_next)
                obs_tuple = (image_next, obs_next)

                if obs_tuple not in state.tracks[track_id]:
                    state.tracks[track_id].append(obs_tuple)
                    used_next_obs.add(obs_next)

        # Start new tracks.
        for obs_curr, obs_next in obs_curr_next:
            obs_curr = int(obs_curr)
            obs_next = int(obs_next)

            if obs_next in used_next_obs:
                continue

            if obs_curr not in obs_to_track:
                image_curr = feature_pairs.get_obs_image(obs_curr)
                image_next = feature_pairs.get_obs_image(obs_next)

                track_id = len(state.tracks)
                state.tracks[track_id] = [
                    (image_curr, obs_curr),
                    (image_next, obs_next),
                ]

    # -------------------------------------------------------------------------
    # PnP
    # -------------------------------------------------------------------------

    def build_pnp_correspondences_from_obs(
        self,
        state: IncrementalSfMState,
        feature_pairs: PointsMatched,
        image_id: int,
        max_points: int = 5000,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        obj = []
        img = []

        for track_id, obs_list in state.tracks.items():
            if track_id not in state.points3D:
                continue

            for im, obs_id in obs_list:
                if im == image_id:
                    obj.append(state.points3D[track_id])
                    img.append(feature_pairs.get_obs_xy(obs_id))
                    break

        if len(obj) == 0:
            return None, None

        obj = np.asarray(obj, dtype=np.float64).reshape(-1, 3)
        img = np.asarray(img, dtype=np.float64).reshape(-1, 2)

        if obj.shape[0] > max_points:
            idx = np.random.choice(obj.shape[0], max_points, replace=False)
            obj = obj[idx]
            img = img[idx]

        return obj, img

    def estimate_pose_pnp(
        self,
        point_cloud: np.ndarray,
        pts2: np.ndarray,
        prev_pose: np.ndarray | None = None,
        min_inliers: int = 30,
    ) -> np.ndarray | None:
        object_points = np.asarray(point_cloud, dtype=np.float64).reshape(-1, 1, 3)
        image_points = np.asarray(pts2, dtype=np.float64).reshape(-1, 1, 2)

        if object_points.shape[0] < min_inliers:
            return None

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=object_points,
            imagePoints=image_points,
            cameraMatrix=self.K_mat,
            distCoeffs=self.dist,
            iterationsCount=max(self.iteration_ct, 1000),
            reprojectionError=self.reproj_error,
            confidence=max(self.confidence, 0.999),
            flags=cv2.SOLVEPNP_EPNP,
        )

        if not success or inliers is None or len(inliers) < min_inliers:
            return None

        inlier_idx = inliers.ravel()

        inlier_3d = object_points.reshape(-1, 3)[inlier_idx]
        inlier_2d = image_points.reshape(-1, 2)[inlier_idx]

        # Optional second-stage iterative refinement initialized from previous pose.
        if prev_pose is not None:
            rvec_guess, _ = cv2.Rodrigues(prev_pose[:, :3])
            tvec_guess = prev_pose[:, 3:4].copy()

            success_iter, rvec_iter, tvec_iter = cv2.solvePnP(
                objectPoints=inlier_3d,
                imagePoints=inlier_2d,
                cameraMatrix=self.K_mat,
                distCoeffs=self.dist,
                rvec=rvec_guess,
                tvec=tvec_guess,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            if success_iter:
                rvec = rvec_iter
                tvec = tvec_iter

        # Correct refinement call.
        rvec, tvec = cv2.solvePnPRefineLM(
            inlier_3d,
            inlier_2d,
            self.K_mat,
            self.dist,
            rvec,
            tvec,
        )

        R, _ = cv2.Rodrigues(rvec)
        pose = np.hstack([R, tvec.reshape(3, 1)]).astype(np.float64)

        median_error = self.compute_pose_reprojection_error(
            inlier_3d,
            inlier_2d,
            pose,
            reduce="median",
        )

        if not np.isfinite(median_error) or median_error > self.reproj_error * 2.0:
            return None

        return pose

    def estimate_pose_pairwise_fallback_obs(
        self,
        pair_index: int,
        feature_pairs: PointsMatched,
        state: IncrementalSfMState,
    ) -> np.ndarray | None:
        """
        Fallback pose estimate using pairwise essential geometry.

        This is less accurate than PnP from stable 3D tracks, but useful when
        the track map is still weak.
        """

        pts_i, pts_j = feature_pairs.access_matching_pair(pair_index)

        if len(pts_i) < 20:
            return None

        E, mask = cv2.findEssentialMat(
            pts_i.astype(np.float64),
            pts_j.astype(np.float64),
            self.K_mat,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )

        if E is None or mask is None:
            return None

        try:
            _, R_rel, t_rel, pose_mask = cv2.recoverPose(
                E,
                pts_i.astype(np.float64),
                pts_j.astype(np.float64),
                self.K_mat,
                mask=mask,
            )
        except cv2.error:
            return None

        if len(state.poses) == 0:
            return None

        prev_pose = state.poses[-1]

        R_prev = prev_pose[:, :3]
        t_prev = prev_pose[:, 3:4]

        # Compose cam_j_from_world = cam_j_from_cam_i @ cam_i_from_world
        R_new = R_rel @ R_prev
        t_new = R_rel @ t_prev + t_rel.reshape(3, 1)

        return np.hstack([R_new, t_new]).astype(np.float64)

    # -------------------------------------------------------------------------
    # Triangulation
    # -------------------------------------------------------------------------

    def update_structure_from_tracks(
        self,
        state: IncrementalSfMState,
        feature_pairs: PointsMatched,
        min_len: int = 2,
        frame_id: int | None = None,
        force_refresh: bool = False,
        refresh_every: int = 5,
    ) -> None:
        for track_id, obs_list in state.tracks.items():
            if len(obs_list) < min_len:
                continue

            should_refresh = (
                force_refresh
                or track_id not in state.points3D
                or (
                    frame_id is not None
                    and frame_id % refresh_every == 0
                )
            )

            if not should_refresh:
                continue

            X = self.triangulate_track_best_pair(
                track_obs=obs_list,
                state=state,
                feature_pairs=feature_pairs,
                cur_img_id=frame_id,
            )

            if X is not None:
                state.points3D[track_id] = X

    def triangulate_track_best_pair(
        self,
        track_obs: list[tuple[int, int]],
        state: IncrementalSfMState,
        feature_pairs: PointsMatched,
        cur_img_id: int | None,
    ) -> np.ndarray | None:
        """
        Triangulate a track using the observation pair with the largest baseline.

        track_obs:
            list[(image_id, obs_id)]
        """

        valid_obs = []

        for image_id, obs_id in track_obs:
            if image_id >= len(state.poses):
                continue

            if cur_img_id is not None and image_id > cur_img_id:
                continue

            valid_obs.append((image_id, obs_id))

        if len(valid_obs) < 2:
            return None

        centers = {}

        for image_id, _ in valid_obs:
            pose = state.poses[image_id]
            R = pose[:, :3]
            t = pose[:, 3]
            centers[image_id] = -R.T @ t

        best_pair = None
        best_score = -1.0

        for a in range(len(valid_obs) - 1):
            for b in range(a + 1, len(valid_obs)):
                i1, obs1 = valid_obs[a]
                i2, obs2 = valid_obs[b]

                if i1 == i2:
                    continue

                baseline = np.linalg.norm(centers[i1] - centers[i2])

                if baseline > best_score:
                    best_score = baseline
                    best_pair = (valid_obs[a], valid_obs[b])

        if best_pair is None or best_score < 1e-8:
            return None

        (i1, obs1), (i2, obs2) = best_pair

        x1 = feature_pairs.get_obs_xy(obs1).reshape(1, 1, 2).astype(np.float64)
        x2 = feature_pairs.get_obs_xy(obs2).reshape(1, 1, 2).astype(np.float64)

        pt1 = cv2.undistortPoints(x1, self.K_mat, self.dist)
        pt2 = cv2.undistortPoints(x2, self.K_mat, self.dist)

        P1 = state.poses[i1].astype(np.float64)
        P2 = state.poses[i2].astype(np.float64)

        X_h = cv2.triangulatePoints(P1, P2, pt1, pt2)
        X = (X_h[:3] / X_h[3]).reshape(3)

        if not np.all(np.isfinite(X)):
            return None

        # Basic single-point filtering.
        pts1_px = feature_pairs.get_obs_xy(obs1).reshape(1, 2)
        pts2_px = feature_pairs.get_obs_xy(obs2).reshape(1, 2)

        X_batch = X.reshape(1, 3)

        mask = self.filter_triangulated_points_mask(
            X=X_batch,
            pts1=pts1_px,
            pts2=pts2_px,
            pose1=P1,
            pose2=P2,
            max_reproj_px=self.max_tri_reproj_error,
            min_tri_angle_deg=self.min_tri_angle_deg,
        )

        if not bool(mask[0]):
            return None

        return X

    def two_view_triangulation(
        self,
        pose_1: np.ndarray,
        pose_2: np.ndarray,
        pts1: np.ndarray,
        pts2: np.ndarray,
    ) -> np.ndarray:
        pts1 = np.asarray(pts1, dtype=np.float64).reshape(-1, 1, 2)
        pts2 = np.asarray(pts2, dtype=np.float64).reshape(-1, 1, 2)

        pt1 = cv2.undistortPoints(pts1, self.K_mat, self.dist)
        pt2 = cv2.undistortPoints(pts2, self.K_mat, self.dist)

        P1 = pose_1.astype(np.float64)
        P2 = pose_2.astype(np.float64)

        X_h = cv2.triangulatePoints(P1, P2, pt1, pt2)
        X = (X_h[:3] / X_h[3]).T

        return X.astype(np.float64)

    # -------------------------------------------------------------------------
    # Filtering and reprojection utilities
    # -------------------------------------------------------------------------

    def project_points(
        self,
        X: np.ndarray,
        pose: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)

        R = pose[:, :3]
        t = pose[:, 3:4]

        X_cam = (R @ X.T + t).T
        z = X_cam[:, 2]

        x_norm = X_cam[:, :2] / (z[:, None] + 1e-12)

        fx = self.K_mat[0, 0]
        fy = self.K_mat[1, 1]
        cx = self.K_mat[0, 2]
        cy = self.K_mat[1, 2]

        uv = np.empty((X.shape[0], 2), dtype=np.float64)
        uv[:, 0] = fx * x_norm[:, 0] + cx
        uv[:, 1] = fy * x_norm[:, 1] + cy

        return uv, z

    def triangulation_angle(
        self,
        X: np.ndarray,
        pose1: np.ndarray,
        pose2: np.ndarray,
    ) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)

        R1 = pose1[:, :3]
        t1 = pose1[:, 3]

        R2 = pose2[:, :3]
        t2 = pose2[:, 3]

        C1 = -R1.T @ t1
        C2 = -R2.T @ t2

        v1 = C1.reshape(1, 3) - X
        v2 = C2.reshape(1, 3) - X

        v1 = v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + 1e-12)
        v2 = v2 / (np.linalg.norm(v2, axis=1, keepdims=True) + 1e-12)

        cosang = np.sum(v1 * v2, axis=1)
        cosang = np.clip(cosang, -1.0, 1.0)

        return np.degrees(np.arccos(cosang))

    def filter_triangulated_points_mask(
        self,
        X: np.ndarray,
        pts1: np.ndarray,
        pts2: np.ndarray,
        pose1: np.ndarray,
        pose2: np.ndarray,
        max_reproj_px: float,
        min_tri_angle_deg: float,
    ) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        pts1 = np.asarray(pts1, dtype=np.float64).reshape(-1, 2)
        pts2 = np.asarray(pts2, dtype=np.float64).reshape(-1, 2)

        uv1, z1 = self.project_points(X, pose1)
        uv2, z2 = self.project_points(X, pose2)

        err1 = np.linalg.norm(uv1 - pts1, axis=1)
        err2 = np.linalg.norm(uv2 - pts2, axis=1)

        angle = self.triangulation_angle(X, pose1, pose2)

        mask = (
            np.isfinite(X).all(axis=1)
            & np.isfinite(err1)
            & np.isfinite(err2)
            & (z1 > 1e-8)
            & (z2 > 1e-8)
            & (err1 <= max_reproj_px)
            & (err2 <= max_reproj_px)
            & (angle >= min_tri_angle_deg)
        )

        return mask

    def compute_pose_reprojection_error(
        self,
        X: np.ndarray,
        pts2: np.ndarray,
        pose: np.ndarray,
        reduce: str = "mean",
    ) -> float:
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        pts2 = np.asarray(pts2, dtype=np.float64).reshape(-1, 2)

        uv, z = self.project_points(X, pose)

        valid = z > 1e-8

        if valid.sum() == 0:
            return float("inf")

        err = np.linalg.norm(uv[valid] - pts2[valid], axis=1)

        if reduce == "median":
            return float(np.median(err))

        if reduce == "mean":
            return float(np.mean(err))

        raise ValueError(f"Unknown reduce type: {reduce}")

    def filter_points3D(
        self,
        state: IncrementalSfMState,
        feature_pairs: PointsMatched,
        max_reproj_error: float = 4.0,
        min_track_len: int = 2,
    ) -> None:
        """
        Remove points that have high reprojection error or invalid depth.

        This is intentionally conservative. It evaluates every observation in
        the track whose pose is already registered.
        """

        bad_track_ids = []

        for track_id, X in list(state.points3D.items()):
            if track_id not in state.tracks:
                bad_track_ids.append(track_id)
                continue

            obs_list = state.tracks[track_id]

            if len(obs_list) < min_track_len:
                bad_track_ids.append(track_id)
                continue

            errors = []
            valid_depth_count = 0

            for image_id, obs_id in obs_list:
                if image_id >= len(state.poses):
                    continue

                pose = state.poses[image_id]
                xy = feature_pairs.get_obs_xy(obs_id).reshape(1, 2)

                uv, z = self.project_points(X.reshape(1, 3), pose)

                if z[0] > 1e-8:
                    valid_depth_count += 1

                err = np.linalg.norm(uv.reshape(1, 2) - xy, axis=1)[0]

                if np.isfinite(err):
                    errors.append(err)

            if valid_depth_count < 2:
                bad_track_ids.append(track_id)
                continue

            if len(errors) == 0:
                bad_track_ids.append(track_id)
                continue

            median_error = float(np.median(errors))

            if median_error > max_reproj_error:
                bad_track_ids.append(track_id)

        for track_id in bad_track_ids:
            if track_id in state.points3D:
                del state.points3D[track_id]

    # -------------------------------------------------------------------------
    # BA hook
    # -------------------------------------------------------------------------
    def build_ba_state_from_obs_state(
        self,
        state: IncrementalSfMState,
        feature_pairs: PointsMatched,
        min_track_len: int = 2,
    ) -> IncrementalSfMState:
        """
        Convert the pose-estimation state into a BA-compatible state.

        Pose-estimation state:
            state.tracks[track_id] = [(image_id, obs_id), ...]

        BA-compatible state:
            ba_state.tracks[track_id] = [(image_id, kp_idx), ...]

        where kp_idx is the local index into ba_state.keypoints[image_id].
        """

        ba_state = IncrementalSfMState(
            K=state.K.copy(),
            dist=None if state.dist is None else state.dist.copy(),
            width=state.width,
            height=state.height,
        )

        # Copy poses.
        ba_state.poses = [p.copy() for p in state.poses]

        # Copy 3D points.
        ba_state.points3D = {
            int(track_id): np.asarray(X, dtype=np.float64).reshape(3).copy()
            for track_id, X in state.points3D.items()
        }

        # -------------------------------------------------------------------------
        # 1. Collect all obs_ids that are needed by valid 3D tracks.
        # -------------------------------------------------------------------------
        image_to_obs_ids: dict[int, set[int]] = {
            image_id: set()
            for image_id in range(len(state.poses))
        }

        valid_track_ids = []

        for track_id, obs_list in state.tracks.items():
            if track_id not in state.points3D:
                continue

            # Only keep observations whose image has a registered pose.
            obs_registered = [
                (int(image_id), int(obs_id))
                for image_id, obs_id in obs_list
                if int(image_id) < len(state.poses)
            ]

            if len(obs_registered) < min_track_len:
                continue

            valid_track_ids.append(int(track_id))

            for image_id, obs_id in obs_registered:
                image_to_obs_ids[image_id].add(obs_id)

        # -------------------------------------------------------------------------
        # 2. Build local keypoint arrays per image.
        # -------------------------------------------------------------------------
        obs_to_kp_idx: dict[tuple[int, int], int] = {}

        for image_id in range(len(state.poses)):
            obs_ids = sorted(image_to_obs_ids.get(image_id, set()))

            kps = []

            for local_kp_idx, obs_id in enumerate(obs_ids):
                xy = feature_pairs.get_obs_xy(obs_id)
                xy = np.asarray(xy, dtype=np.float64).reshape(2)

                kps.append(xy)
                obs_to_kp_idx[(image_id, obs_id)] = local_kp_idx

            if len(kps) == 0:
                ba_state.keypoints[image_id] = np.empty((0, 2), dtype=np.float64)
            else:
                ba_state.keypoints[image_id] = np.asarray(kps, dtype=np.float64)

        # -------------------------------------------------------------------------
        # 3. Convert tracks from obs_id to local kp_idx.
        # -------------------------------------------------------------------------
        for track_id in valid_track_ids:
            obs_list = state.tracks[track_id]

            ba_obs = []

            for image_id, obs_id in obs_list:
                image_id = int(image_id)
                obs_id = int(obs_id)

                if image_id >= len(state.poses):
                    continue

                key = (image_id, obs_id)

                if key not in obs_to_kp_idx:
                    continue

                kp_idx = obs_to_kp_idx[key]
                ba_obs.append((image_id, kp_idx))

            if len(ba_obs) >= min_track_len:
                ba_state.tracks[track_id] = ba_obs

        return ba_state


    def run_local_ba_and_refresh(
        self,
        state: IncrementalSfMState,
        feature_pairs: PointsMatched,
        new_image_id: int,
    ) -> IncrementalSfMState:
        """
        Converts obs-ID tracks to BA-compatible local kp-index tracks,
        runs pycolmap local BA, then copies optimized poses back.
        """

        if self.optimizer is None:
            return state

        # Convert obs_id state -> BA-compatible kp_idx state.
        ba_state = self.build_ba_state_from_obs_state(
            state=state,
            feature_pairs=feature_pairs,
            min_track_len=self.min_track_len_triangulate,
        )

        # If there are too few BA tracks, skip safely.
        if len(ba_state.points3D) == 0 or len(ba_state.tracks) == 0:
            return state

        # Run local BA on the converted state.
        ba_state = self.optimizer(
            ba_state,
            new_image_id=new_image_id,
        )

        # Copy optimized poses back into the original obs-ID state.
        for image_id in range(min(len(state.poses), len(ba_state.poses))):
            state.poses[image_id] = ba_state.poses[image_id].copy()

        self.camera_poses.camera_pose = list(state.poses)

        # Refresh 3D structure using the original obs-ID tracks.
        self.update_structure_from_tracks(
            state=state,
            feature_pairs=feature_pairs,
            min_len=self.min_track_len_triangulate,
            frame_id=new_image_id,
            force_refresh=True,
        )

        # Filter bad points after pose refinement.
        self.filter_points3D(
            state=state,
            feature_pairs=feature_pairs,
            max_reproj_error=self.max_tri_reproj_error,
            min_track_len=self.min_track_len_triangulate,
        )

        self.camera_poses.camera_pose = list(state.poses)

        return state