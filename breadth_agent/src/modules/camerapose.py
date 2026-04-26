import numpy as np
import cv2
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms as TF
import json

from modules.baseclass import CameraPoseEstimatorClass, module_metric
from modules.optimization import BundleAdjustmentOptimizerLocal
from modules.DataTypes.datatype import Points2D, Calibration, Points3D, CameraPose, PointsMatched, CameraData, IncrementalSfMState
from modules.models.sfm_models.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from modules.models.sfm_models.vggt.utils.geometry import unproject_depth_map_to_point_map
from modules.models.sfm_models.vggt.models.vggt import VGGT
from modules.models.sfm_models.vggt.utils.load_fn import load_and_preprocess_images

import glob
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

Utilize this module in cases where images do not have extreme overlap, scale is needed for a 
monocular camera setup, GPU memory is accessible, or calibration is not provided, and we need 
to estimate camera pose and calibration parameters to reconstruct the scene.

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
from modules.baseclass import SfMScene
from modules.features import ....
from modules.featurematching import ....
from modules.camerapose import {module_name}

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

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda":
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
        # tensor_img_list = []
        # for ind in range(len(self.image_list)):
        #     tensor_img_list.append(to_tensor(self.image_list[ind]))

        self.images = torch.stack(tensor_img_list).to(device) 

        self.img_shape = self.image_list[0].shape[:2] # Images 
        self.use_base_metrics = False

    # def __call__(self, features: list[Points2D] | None = None) -> CameraPose:
    def _estimate_camera_poses(self,
                               feature_pairs: PointsMatched | None = None) -> None:
        
        assert self.img_shape[0] == self.img_shape[1], (
            "Input images must be square size, or Height must equal Width. "
            "Must reshape images to a square size, such as (1024, 1024)"
        )
        # return super()._estimate_camera_poses(camera_poses, feature_pairs)
        # cam_poses = CameraPose()

        # VGGT Fixed Resolution to 518 for Inference
        images = F.interpolate(self.images, size=(518, 518), mode="bilinear", align_corners=False)
        new_scale = self.img_shape[0] / 518 # Get change of scale from old shape to new smaller shape

        # images = self.images
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

        self.cam_data.apply_new_calibration(intrins, dists)

        # print("Image Shape", img_shape)
        # print(camera_poses.camera_pose)
        torch.cuda.empty_cache() #Empty GPU cache
        # return camera_poses
    
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
            "Average Rotation Orthonormality Error": float(np.mean(ortho_errors)),
            "Average det(R)": float(np.mean(det_values)),
            "Average Translation Norm": float(np.mean(trans_norms)),
        }

    @module_metric
    def _metric_intrinsics_summary(self) -> dict:
        if len(self._pred_intrinsics) == 0:
            return {}

        fx_vals = [float(K[0, 0]) for K in self._pred_intrinsics]
        fy_vals = [float(K[1, 1]) for K in self._pred_intrinsics]
        cx_vals = [float(K[0, 2]) for K in self._pred_intrinsics]
        cy_vals = [float(K[1, 2]) for K in self._pred_intrinsics]

        return {
            "Average fx": float(np.mean(fx_vals)),
            "Average fy": float(np.mean(fy_vals)),
            "Average cx": float(np.mean(cx_vals)),
            "Average cy": float(np.mean(cy_vals)),
        }
                
###########################################################################################################
############################################ CLASSICAL MODULES ############################################

class CamPoseEstimatorEssentialToPnP(CameraPoseEstimatorClass):
    def __init__(self, 
                 cam_data: CameraData,
                 iteration_count: int = 200,
                 reprojection_error: float = 3.0,
                 confidence: float = 0.99,
                 optimizer: BundleAdjustmentOptimizerLocal | None = None
                 ):

        module_name = "CamPoseEstimatorEssentialToPnP"
        description = f"""
Estimates the camera pose for each frame in a set of images for a monocular camera. The
process of this module is to estimate the essential matrix for the first pair of images 
of the image set, recover the camera pose that's up-to-scale, then use the PnP algorithm
to estimate the rest of the camera's pose for each following images so each trajectory is 
in scale to the first pose estimation. Use this module for Monocular cameras that are 
calibrated for a given image set. USE THIS MODULE when image sets have ONLY INCREMENTAL 
camera movement. Any large view changes, this module will not be robust to. In case of those
failures, be sure to utilize VGGT instead. Otherwise, this module is good for incremental, 
consistent camera movement.

If intial pose estimates lead to poor 3D reprojection results, or are using a less accurate feature
detector (Such as ORB or SuperPoint), then you MUST use the Optimizer option. The optimizer option
utilizes a local bundle adjustment procedure for more robust pose estimation. Could use with SIFT 
in image cases where lighting is not the best but texture is good enough for SIFT features.

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
from modules.baseclass import SfMScene
from modules.features import ....
from modules.featurematching import ....
from modules.camerapose import {module_name}

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
        self.optimizer = optimizer
        
    def _estimate_camera_poses(self,
                               feature_pairs: PointsMatched,
                               ba_per_frame: int = 4) -> None:
        assert(feature_pairs.multi_view == False), (
            "Features passed must be two view correspondences. "
            "Ensure to invoke Feature Matching Two View tools prior to this call."
        )

        if self.optimizer is not None:
            state = IncrementalSfMState(self.K_mat, self.dist,
                                        width=self.cam_data.image_list[0].shape[1], 
                                        height=self.cam_data.image_list[0].shape[0],
                                        )
            for i in range(len(feature_pairs.img_features)):
                state.keypoints[i] = feature_pairs.img_features[i]
        else:
            state = None

        # Get First set of camera poses (Initial and 2nd Camera)
        pts1, pts2 = feature_pairs.access_matching_pair(0)

        # cam_poses = self.estimate_first_pair(pts1, pts2) # First two poses defined here
        # self.cam_poses = cam_poses
        self.estimate_first_pair(pts1, pts2) # First two poses defined here

        # cloud = self.two_view_triangulation(camera_poses.camera_pose[0], camera_poses.camera_pose[1], pts1, pts2)
        
        if state is not None:
            state.poses = self.camera_poses.camera_pose

            # Set up First Tracks
            first_matches = feature_pairs.pairwise_indices[0]
            for m in first_matches:
                kp0, kp1 = int(m[0]), int(m[1])
                tid = len(state.tracks)
                state.tracks[tid] = [(0, kp0), (1, kp1)]

            # triangulate initial structure from those tracks
            # self.update_structure_from_tracks(state, min_len=2, frame_id=1)

            for i in tqdm(range(1, len(feature_pairs.pairwise_matches)), 
                        desc='Estimating Camera Poses'):
                new_img_id = i + 1
                curr_img_id = i

                # 1) Update tracks (requires pairwise_indices!)
                matches_prev_curr = feature_pairs.pairwise_indices[i - 1]  # (i-1 -> i)
                matches_curr_next = feature_pairs.pairwise_indices[i]      # (i -> i+1)
                self.three_view_tracking_indices(matches_prev_curr, matches_curr_next, curr_img_id, new_img_id, state)

                # 2) Update / create 3D points from tracks
                self.update_structure_from_tracks(state, min_len=2, frame_id=curr_img_id)

                # 3) Try track-based PnP
                obj_pts, img_pts = self.build_pnp_correspondences(state=state, image_id=new_img_id)

                if obj_pts is not None and len(obj_pts) >= 20:
                    # Track-based PnP
                    new_pose = self.estimate_pose_pnp(
                        point_cloud=obj_pts.reshape(-1, 1, 3),
                        # pts1=None,
                        pts2=img_pts.reshape(-1, 1, 2),
                        # prev_pose=camera_poses.camera_pose[-1],
                    )
                else:
                    # 4) Fallback (self-contained)
                    # Ensure camera_poses already has pose for curr_img_id before using fallback
                    # camera_poses currently has poses up to curr_img_id
                    new_pose = self.estimate_pose_pairwise_fallback(
                        pair_index=i,
                        feature_pairs=feature_pairs,
                        camera_poses=self.camera_poses,
                    )

                # 5) Append pose to BOTH state and camera_poses (keep in sync)
                # state.poses.append(new_pose)
                self.camera_poses.camera_pose.append(new_pose)
                
                # Store Residual Error for Metric Recording
                self._record_residual_metric(obj_pts, img_pts, new_pose)
                # mean_error, median_error = self._metric_calculation_residuals(obj_pts, img_pts, new_pose)
                # reprojection_error.append(mean_error)
                # median_reproj_error.append(median_error)

                # 6) Local BA every ba_per_frame frames
                if ((new_img_id) % ba_per_frame) == 0:
                    state = self.optimizer(state, new_image_id=new_img_id)

                    # copy refined poses back
                    self.camera_poses.camera_pose = state.poses

                    # refresh structure after pose changes (important!)
                    self.update_structure_from_tracks(state, min_len=2, refresh_every=1, frame_id=new_img_id)

        else: 
            cloud = self.two_view_triangulation(
                self.camera_poses.camera_pose[0], 
                self.camera_poses.camera_pose[1], 
                pts1, 
                pts2
            )

            for i in tqdm(range(1, len(feature_pairs.pairwise_matches)), 
                        desc='Estimating Camera Poses'):
                
                if i > 1:
                    cloud = self.two_view_triangulation(pose1, pose2, pts1, pts2)

                # pts3_t = features[i+2]
                # pts2_3, pts3 = self.match_pairs(features[i + 1], pts3_t)
                pts2_3, pts3 = feature_pairs.access_matching_pair(i)
                index, pts2_3_com, pts3_com, pts2_3_new, pts3_new = self.three_view_tracking(pts2, pts2_3, pts3)

                # print("Prev PAIR", pts2_3_com.shape)
                # print("Current POINTS", pts3_com.shape)
                # new_pose = self.estimate_pose_pnp(cloud[index], pts2_3_com, pts3_com, camera_poses.camera_pose[-1])
                new_pose = self.estimate_pose_pnp(cloud[index], pts3_com)

                pose1 = self.camera_poses.camera_pose[-1]
                pose2 = new_pose
                pts1 = pts2_3
                pts2 = pts3

                self.camera_poses.camera_pose.append(new_pose)

                # if state is not None:
                #     state.poses.append(new_pose)

                # Store Residual Error for Metric Recording
                self._record_residual_metric(cloud[index], pts3_com, new_pose)
                # mean_error, median_error = self._metric_calculation_residuals(cloud[index], pts3_com, new_pose)
                # # reprojection_error.append(self._metric_calculation_residuals(cloud[index], pts3_com, new_pose))
                # reprojection_error.append(mean_error)
                # median_reproj_error.append(median_error)

                # # Local BA refinement hook w/ updating poses in window!
                # if optimizer is not None and state is not None and ((i + 1) % ba_per_frame) == 0:
                #     state = optimizer.optimize(state, new_image_id=i + 1)
                #     camera_poses.camera_pose = list(state.poses)

        # # Report the reprojection metric
        # residual_error = np.array(reprojection_error).mean()
        # residual_median = np.median(median_reproj_error)
        # event_msg = {"Average Reprojection Error per Frame": float(residual_error),
        #              "Average Median Reprojection Error per Frame": float(residual_median)}
        # print(json.dumps(event_msg), flush=True)

        # return camera_poses

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
        # print(len(obs_list))
        # print(obs_list)
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

        (i1, kp1), (i2, kp2) = best

        x1 = state.keypoints[i1][kp1].reshape(2, 1)
        x2 = state.keypoints[i2][kp2].reshape(2, 1)

        # Use pixel projection matrices
        # Normalize Points
        pt1 = cv2.undistortPoints(x1, self.K_mat, self.dist)
        pt2 = cv2.undistortPoints(x2, self.K_mat, self.dist)
        
        P1mtx = np.eye(3) @ state.poses[i1]
        P2mtx = np.eye(3) @ state.poses[i2]

        # P1 = state.K @ state.poses[i1]
        # P2 = state.K @ state.poses[i2]

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
        for kp_k, kp_k1 in matches_curr_next:
            curr_to_next[int(kp_k)] = int(kp_k1)

        # Map from (frame, kp_idx) to track_id
        kp_to_track = {}
        for track_id, obs in state.tracks.items():
            for (f, kp) in obs:
                kp_to_track[(f, kp)] = track_id

        used_next_kps = set()

        # 1) Extend existing tracks
        for kp_prev, kp_curr in matches_prev_curr:
            kp_prev = int(kp_prev)
            kp_curr = int(kp_curr)

            # Is kp_curr observed again in next frame?
            if kp_curr not in curr_to_next:
                continue

            kp_next = curr_to_next[kp_curr]

            # Does this correspondence belong to an existing track?
            key = (frame_k, kp_curr)
            if key in kp_to_track:
                track_id = kp_to_track[key]

                # Append next observation if not already present
                obs = state.tracks[track_id]
                if (frame_k1, kp_next) not in obs:
                    obs.append((frame_k1, kp_next))
                    used_next_kps.add(kp_next)

        # 2) Start new tracks for unmatched correspondences
        for kp_curr, kp_next in matches_curr_next:
            kp_curr = int(kp_curr)
            kp_next = int(kp_next)

            if kp_next in used_next_kps:
                continue

            # If kp_curr not already tracked, start a new track
            if (frame_k, kp_curr) not in kp_to_track:
                new_track_id = len(state.tracks)
                state.tracks[new_track_id] = [
                    (frame_k, kp_curr),
                    (frame_k1, kp_next),
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
                # print(obs)
                # print(state.poses)  
                X = self.triangulate_track_best_pair(obs, state, frame_id)
                if X is not None:
                    state.points3D[track_id] = X
            else:
                # optional refresh: if BA updated poses, re-triangulate sometimes
                if frame_id is not None and (frame_id % refresh_every) == 0:
                    X = self.triangulate_track_best_pair(obs, state, frame_id)
                    if X is not None:
                        state.points3D[track_id] = X

    def build_pnp_correspondences(self, 
                                  state: IncrementalSfMState, 
                                  image_id: int, 
                                  max_points: int = 2000):
        obj = []
        img = []
        for track_id, obs in state.tracks.items():
            if track_id not in state.points3D:
                continue
            # find if this track is observed in this image
            for (im, kp) in obs:
                if im == image_id:
                    obj.append(state.points3D[track_id])
                    img.append(state.keypoints[im][kp])
                    break

        if len(obj) == 0:
            return None, None

        obj = np.asarray(obj, dtype=np.float64).reshape(-1, 3)
        img = np.asarray(img, dtype=np.float64).reshape(-1, 2)

        # Optional: subsample for speed
        if obj.shape[0] > max_points:
            idx = np.random.choice(obj.shape[0], max_points, replace=False)
            obj, img = obj[idx], img[idx]

        return obj, img

    # Simple Pair View Pose Estimation
    def three_view_tracking(self, pts2: np.ndarray, pts2_3: np.ndarray, pts3: np.ndarray):
        #pts2 is the set of keypoints obtained from image(n-1) and image(n)
        #pts2_3 and pts3 are the set of keypoints obtained from image(n) and image(n+1)
        
        # Finding Commmon Points
        index1=[]
        index2=[]
        for i in range(pts2.shape[0]):
            if (pts2[i,:] == pts2_3).any():
                index1.append(i)

            idx2_3 = np.where(pts2_3 == pts2[i,:])[0]

            if idx2_3.size != 0:
                index2.append(idx2_3[0])
        
        #Finding New Points
        pts3_new=[]
        pts2_3_new=[]
        
        for i in range(pts3.shape[0]):
            if i not in index2:
                pts3_new.append(list(pts3[i,:]))
                pts2_3_new.append(list(pts2_3[i,:]))
        
        index1=np.array(index1)
        index2=np.array(index2)
        pts2_3_common=pts2_3[index2]
        pts3_common=pts3[index2]
                
        return index1,pts2_3_common,pts3_common,np.array(pts2_3_new),np.array(pts3_new)

    def estimate_first_pair(self, pts1: np.ndarray, pts2: np.ndarray) -> None: #CameraPose: #pts1: Points2D, pts2: Points2D) -> CameraPose:
        initial_pose = np.array(
            [[1,0,0,0],
             [0,1,0,0],
             [0,0,1,0]], 
             dtype=float,
             )
        self.camera_poses.camera_pose.append(initial_pose)

        E, _ = cv2.findEssentialMat(pts1, pts2, self.K_mat, method=cv2.RANSAC, prob=0.999, threshold=0.3)

        _, R, T, _ = cv2.recoverPose(points1 = pts2, points2 = pts1, cameraMatrix = self.K_mat, E = E)

        new_pose = np.hstack((R, T.reshape(3, 1)))
        self.camera_poses.camera_pose.append(new_pose)

    # def estimate_pose_pnp(self, point_cloud: np.ndarray, pts1: np.ndarray, pts2: np.ndarray, prev_pose: np.ndarray) -> np.ndarray:
    def estimate_pose_pnp(self, point_cloud: np.ndarray, pts2: np.ndarray) -> np.ndarray:
        #cv2.solvePnPRansac(point_cloud, pts2, self.K1, self.dist1, cv2.SOLVEPNP_ITERATIVE)
        _,rot,trans,inliers= cv2.solvePnPRansac(objectPoints=point_cloud, 
                                          imagePoints=pts2, 
                                          cameraMatrix=self.K_mat, 
                                          distCoeffs=self.dist, 
                                          useExtrinsicGuess=False,
                                          reprojectionError= self.reproj_error,
                                          iterationsCount= self.iteration_ct,
                                          confidence=self.confidence,
                                          flags=cv2.SOLVEPNP_ITERATIVE)

        # Set inlier Points
        inlier_3dPoints = point_cloud[inliers][:,0,:,:]
        inlier_2dPoints = pts2[inliers][:,0,:]

        rot,trans - cv2.solvePnPRefineLM(inlier_3dPoints,
                                        inlier_2dPoints,
                                        self.K_mat,
                                        self.dist,
                                        rot,
                                        trans)

        rot,_=cv2.Rodrigues(rot)

        new_pose = np.hstack((rot, trans))

        # new_pose = np.empty((3,4))
        # new_pose[:3,:3] = rot @ self.cam_poses.camera_pose[-1][:3,:3]
        # new_pose[:3,3]  = self.cam_poses.camera_pose[-1][:3, 3] + self.cam_poses.camera_pose[-1][:3, :3] @ trans

        # self.cam_poses.camera_pose.append(new_pose)

        return new_pose
        
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
