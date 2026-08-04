import sys
############ TEMP SOLUTION FOR NOW #################
import json
import cv2
import copy
import numpy as np
import glob
from tqdm import tqdm
from omegaconf import OmegaConf
from torchvision import transforms as TF

from modules.DataTypes.pointDT import Points2D
from modules.DataTypes.featmatchDT import PointsMatched
from modules.DataTypes.cameraDT import CameraData

from modules.models.matchers import LightGlue, SuperGlue
from romatch import roma_outdoor, roma_indoor
from modules.models.sfm_models.vggt.dependency.vggsfm_utils import generate_rank_by_dino
from vggsfm.models.track_predictor import TrackerPredictor
from vggsfm.utils.utils import (
    calculate_index_mappings,
    switch_tensor_order,
    generate_rank_by_interval,
    generate_rank_by_midpoint,
)
from vggsfm.models import (
    TrackerPredictor,
    BasicEncoder,
    ShallowEncoder,
    BaseTrackerPredictor,
)
from tapnet.torch import tapir_model

from modules.baseclass import FeatureMatching, FeatureTracking, FeatureTrackingBase, module_metric
from collections.abc import Callable
import kornia as K
import kornia.feature as KF
import torch
from PIL import Image, ImageOps
import piexif
from typing import Optional
from types import SimpleNamespace

#############################################
############## Helper Function ##############
class UnionFindTracks:
    def __init__(self, obs_image: dict[int, int]):
        self.parent = {}
        self.rank = {}
        self.component_images = {}
        self.obs_image = obs_image

    def add(self, obs_id: int):
        if obs_id not in self.parent:
            self.parent[obs_id] = obs_id
            self.rank[obs_id] = 0
            self.component_images[obs_id] = {self.obs_image[obs_id]}

    def find(self, x: int):
        self.add(x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def can_union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return True

        images_a = self.component_images[ra]
        images_b = self.component_images[rb]

        # If both components already contain the same image,
        # merging would create an invalid SfM track.
        return len(images_a.intersection(images_b)) == 0

    def union(self, a: int, b: int) -> bool:
        self.add(a)
        self.add(b)

        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return True

        if not self.can_union(ra, rb):
            return False

        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra
        self.component_images[ra].update(self.component_images[rb])

        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

        return True

class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x):
        self.add(x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1
#############################################
#############################################

class FeatureTrackingTapir(FeatureTrackingBase):
    requires_features = True
    requires_feature_pairs = False
    direct_tracker = False
    
    def __init__(
        self,
        cam_data: CameraData,
        min_track_len: int = 2,
        query_fram_num: int = 5,
        query_selection: str = "dino",
        max_query_pts: int = 2048,
        query_chunk_size: int = 64,
        visibility_threshold: float = 0.5,
        score_threshold: float = 0.0,
        pyramid_level: int = 1,
        resize_to: tuple[int, int] | None = (512, 512),
        **kwargs,
    ):
        model_path = "/home/anthonyq/projects/repos/tapnet/checkpoints/bootstapir_checkpoint_v2.pt"

        module_name = "FeatureTrackingTapir"

        description = f"""
Tracks detected feature points across an image sequence using the learned
BootsTAPIR point tracker. Features from selected query frames are tracked
throughout the sequence and converted into persistent multi-view tracks.

Tracks selected query points throughout an ordered video using learned temporal 
matching and refinement, including visibility estimation and recovery after temporary 
occlusion. USE THIS MODULE for continuous video with smooth-to-moderate motion, 
changing illumination, motion blur, temporary occlusions, or low-texture regions 
where pairwise descriptor matching creates fragmented tracks. 
Query points may come from SIFT, ORB, SuperPoint, or ALIKED, although well-distributed 
points from SuperPoint or ALIKED are generally preferable. Avoid using tracks on 
independently moving objects for static-scene SfM.

Note: Detector Free feature pairs must use UnionFind, this module does not support 
RoMa and Loftr Features!

Initialization/Function Parameters:

- min_track_len: Minimum number of visible frames required to retain a track.
    - Default (int): 2
- query_fram_num: Number of frames used to initialize feature tracks.
    - Default (int): 5
- query_selection: Method used to select query frames. Supported values are "dino",
  "interval", and "midpoint".
    - Default (str): "dino"
- max_query_pts: Maximum number of detected points tracked from each query frame.
    - Default (int): 2048
- query_chunk_size: Number of query points processed together during inference.
    - Default (int): 64
- visibility_threshold: Minimum predicted visibility required to keep an observation.
    - Default (float): 0.5
- score_threshold: Minimum confidence required to keep an observation. A value of zero
  disables additional score filtering.
    - Default (float): 0.0
- pyramid_level: Feature pyramid level used by the BootsTAPIR tracker.
    - Default (int): 1
- resize_to: Image resolution used during tracking. Set to None to use the original
  image resolution.
    - Default (tuple[int, int] | None): (512, 512)

Function Call Parameters - HANDLED INTERNALLY, DO NOT USE IF SFMCORE IS IN USE:

- img_features: list[Points2D]
    Detected feature points for each input frame.
"""

        example = f"""
from modules.baseclass import SfMScene
from modules.features import FeatureDetectionSIFT
from modules.featuretracking import {module_name}

# Step 1: Load image and calibration data.
reconstructed_scene = SfMScene(
    image_path=image_path,
    calibration_path=calibration_path,
)

# Step 2: Detect features before tracking.
reconstructed_scene.FeatureDetectionSIFT()

# Step 3: Track features using BootsTAPIR.
reconstructed_scene.{module_name}(
    min_track_len=3,
    query_fram_num=5,
    query_selection="dino",
    max_query_pts=2048,
    query_chunk_size=64,
    visibility_threshold=0.5,
    score_threshold=0.0,
    pyramid_level=1,
    resize_to=(512, 512),
)
"""

        super().__init__(
            detector="features",
            cam_data=cam_data,
            module_name=module_name,
            description=description,
            example=example,
            RANSAC_threshold=kwargs.get("RANSAC_threshold", 3.0),
            RANSAC_conf=kwargs.get("RANSAC_conf", 0.99),
            RANSAC_homography=kwargs.get("RANSAC_homography", False),
        )
        self.device = f"cuda:{self.cam_data.gpu_num}" if torch.cuda.is_available() else "cpu"

        self.min_track_length = min_track_len
        self.query_frame_num = query_fram_num
        self.query_selection = query_selection

        self.max_query_pts = max_query_pts
        self.query_chunk_size = query_chunk_size

        self.visibility_threshold = visibility_threshold
        self.score_threshold = score_threshold

        self.pyramid_level = pyramid_level
        self.resize_to = resize_to

        # Original image size.
        # Your class uses PIL images, where .size = (width, height).
        self.width, self.height = self.image_list[0].size

        # Build video tensor for TAPIR.
        self.images = self._prepare_images()

        # Build BootsTAPIR model.
        self.track_predictor = self._build_tracker(
            model_path=model_path,
        )
    
    # ================================================================
    # Model loading
    # ================================================================

    def _build_tracker(self, model_path: str | None = None):
        """
        Build BootsTAPIR.

        BootsTAPIR uses TAPIR with extra_convs=True.
        """

        model = tapir_model.TAPIR(
            pyramid_level=self.pyramid_level,
            extra_convs=True,
        )

        if model_path is None:
            # Official BootsTAPIR PyTorch checkpoint.
            model_url = (
                "https://storage.googleapis.com/dm-tapnet/bootstap/"
                "bootstapir_checkpoint_v2.pt"
            )

            state_dict = torch.hub.load_state_dict_from_url(model_url, map_location="cpu")
        else:
            state_dict = torch.load(model_path, map_location="cpu")

        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        missing, unexpected = model.load_state_dict(
            state_dict,
            strict=False,
        )

        model = model.to(self.device)
        model.eval()

        return model

    # ================================================================
    # Image preparation
    # ================================================================

    def _prepare_images(self) -> torch.Tensor:
        """
        Prepare video for TAPIR.

        Your input:
            list[PIL.Image] or image-like frames

        Internal TAPIR tensor:
            [1, T, H, W, 3]

        TAPIR uses channel-last video tensors.
        """

        to_tensor = TF.ToTensor()
        tensor_img_list = [to_tensor(img) for img in self.image_list]

        images_chw = torch.stack(tensor_img_list, dim=0)

        # [T,3,H,W]
        images_chw = images_chw.to(device=self.device, dtype=torch.float32)

        if self.resize_to is not None:
            target_h, target_w = self.resize_to

            images_chw = torch.nn.functional.interpolate(
                images_chw,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )

        # TAPIR expects channel-last:
        #
        # [T,3,H,W] -> [T,H,W,3]
        images_thwc = images_chw.permute(0, 2, 3, 1)

        # TAPIR convention is usually [-1,1].
        images_thwc = images_thwc * 2.0 - 1.0

        # [T,H,W,3] -> [1,T,H,W,3]
        images_bthwc = images_thwc.unsqueeze(0)

        return images_bthwc.contiguous()

    # ================================================================
    # Query frame selection
    # ================================================================
    def _select_query_frames(self, frame_num: int) -> list[int]:
        num_queries = min(self.query_frame_num, frame_num)

        if self.query_selection == "dino":
            if generate_rank_by_dino is None:
                raise ImportError(
                    "query_selection='dino' requires VGGT's "
                    "generate_rank_by_dino. Could not import from "
                    "vggt.dependency.vggsfm_utils."
                )

            # self.images is [1, T, 3, H, W]
            # VGGT generate_rank_by_dino expects [T, 3, H, W]
            images_for_dino = self.images[0]

            # TAPIR class likely stores video in [-1,1].
            # Convert back to [0,1] for DINO.
            if images_for_dino.min() < 0:
                images_for_dino = (images_for_dino + 1.0) / 2.0

            images_for_dino = images_for_dino.clamp(0.0, 1.0)

            # [T,H,W,3] -> [T,3,H,W]
            images_for_dino = images_for_dino.permute(0, 3, 1, 2).contiguous()

            ranking = generate_rank_by_dino(
                images=images_for_dino,
                query_frame_num=num_queries,
                device=self.device,
            )
            print(ranking)

        elif self.query_selection == "interval":
            interval = max(1, frame_num // num_queries)
            ranking = generate_rank_by_interval(frame_num, interval)

        elif self.query_selection == "midpoint":
            ranking = generate_rank_by_midpoint(frame_num)

        else:
            raise ValueError(
                "query_selection must be 'dino', 'interval', or 'midpoint'."
            )

        ranking = list(ranking)

        # Always include frame 0, matching the VGGT demo behavior.
        if 0 in ranking:
            ranking.remove(0)

        query_frames = [0, *ranking]

        return query_frames[:num_queries]

    # ================================================================
    # Query points
    # ================================================================
    def _get_query_points_for_frame(
        self,
        features: Points2D,
        query_index: int,
    ) -> torch.Tensor:
        """
        Convert your Points2D for one frame into TAPIR query format.

        Input:
            features.points2D or features.keypoints:
                [N,2] in [x,y] pixel coordinates from original image size

        Output:
            [N,3] in [t,y,x]
        """

        points_xy = features.points2D
        points_xy = torch.as_tensor(
            points_xy,
            device=self.device,
            dtype=torch.float32,
        )

        if len(points_xy) == 0:
            return torch.empty(
                (0, 3),
                device=self.device,
                dtype=torch.float32,
            )

        # Keep strongest detector features if scores exist.
        if len(points_xy) > self.max_query_pts:
            scores = getattr(features, "scores", None)

            if scores is not None:
                scores = torch.as_tensor(
                    scores,
                    device=self.device,
                    dtype=torch.float32,
                ).reshape(-1)

                keep = torch.topk(scores, k=self.max_query_pts).indices
                points_xy = points_xy[keep]
            else:
                points_xy = points_xy[: self.max_query_pts]

        # If TAPIR image was resized, scale coordinates into resized space.
        if self.resize_to is not None:
            target_h, target_w = self.resize_to

            scale_x = target_w / float(self.width)
            scale_y = target_h / float(self.height)

            points_xy = points_xy.clone()
            points_xy[:, 0] *= scale_x
            points_xy[:, 1] *= scale_y

        # TAPIR query format:
        #
        # [t, y, x]
        query_tyx = torch.empty(
            (points_xy.shape[0], 3),
            device=self.device,
            dtype=torch.float32,
        )

        query_tyx[:, 0] = float(query_index)
        query_tyx[:, 1] = points_xy[:, 1]
        query_tyx[:, 2] = points_xy[:, 0]

        return query_tyx

    # ================================================================
    # TAPIR output confidence / visibility
    # ================================================================

    @staticmethod
    def _tapir_visibility_from_logits(
        occlusion: torch.Tensor,
        expected_dist: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        TAPIR returns occlusion logits and expected-distance logits.

        Convert them into:
            confidence in [0,1]
            visibility mask
        """

        occ_prob = torch.sigmoid(occlusion)
        dist_prob = torch.sigmoid(expected_dist)

        # Same logic used in many TAPIR demos:
        # visible if not occluded and expected distance is reliable.
        confidence = (1.0 - occ_prob) * (1.0 - dist_prob)
        return confidence

    # ================================================================
    # Track query chunks
    # ================================================================

    @torch.inference_mode()
    def _predict_tracks_in_chunks( 
        self,
        query_points_tyx: torch.Tensor,
    ):
        """
        query_points_tyx:
            [N,3]

        Returns
        -------
        tracks_xy:
            [1,N,T,2]

        visibility_scores:
            [1,N,T]

        confidence_scores:
            [1,N,T]
        """

        if query_points_tyx.shape[0] == 0:
            raise ValueError(
                "Cannot track zero TAPIR query points."
            )

        track_chunks = []
        vis_score_chunks = []
        conf_chunks = []

        for start in range(0, 
                           query_points_tyx.shape[0],
                           self.query_chunk_size,
        ):
            end = min(start + self.query_chunk_size,
                      query_points_tyx.shape[0],
            )

            query_chunk = query_points_tyx[start:end].unsqueeze(0)

            outputs = self.track_predictor(
                video=self.images,
                query_points=query_chunk,
                is_training=False,
                query_chunk_size=self.query_chunk_size,
            )

            tracks = outputs["tracks"]
            occlusion = outputs["occlusion"]
            expected_dist = outputs["expected_dist"]

            confidence = self._tapir_visibility_from_logits(occlusion, expected_dist)
            # visibility_score = confidence

            track_chunks.append(tracks)
            vis_score_chunks.append(confidence)
            conf_chunks.append(confidence)

        pred_tracks = torch.cat(track_chunks, dim=1)
        pred_vis_scores = torch.cat(vis_score_chunks, dim=1)
        pred_conf_scores = torch.cat(conf_chunks, dim=1)

        return pred_tracks, pred_vis_scores, pred_conf_scores

    # ================================================================
    # Multi-query-frame video tracking
    # ================================================================

    @torch.inference_mode()
    def _track_video(self, img_features: list[Points2D]):
        """
        Run BootsTAPIR over selected query frames.

        Returns
        -------
        pred_tracks:
            [1,N_total,T,2]

        pred_vis_scores:
            [1,N_total,T]

        pred_conf_scores:
            [1,N_total,T]

        query_frame_ids:
            [N_total]
        """

        frame_num = self.images.shape[1]
        query_frames = self._select_query_frames(frame_num)

        all_tracks = []
        all_vis_scores = []
        all_conf_scores = []
        all_query_frames = []

        for query_index in query_frames:
            if query_index >= len(img_features):
                continue
            features = img_features[query_index]

            query_points_tyx = (
                self._get_query_points_for_frame(features, query_index)
            )

            if len(query_points_tyx) == 0:
                continue

            pred_tracks, pred_vis_scores, pred_conf_scores = (
                self._predict_tracks_in_chunks(query_points_tyx)
            )

            all_tracks.append(pred_tracks)
            all_vis_scores.append(pred_vis_scores)
            all_conf_scores.append(pred_conf_scores)

            all_query_frames.append(
                torch.full(
                    (query_points_tyx.shape[0],),
                    query_index,
                    dtype=torch.long,
                    device=self.device,
                )
            )

        pred_tracks = torch.cat(all_tracks, dim=1)
        pred_vis_scores = torch.cat(all_vis_scores, dim=1)
        pred_conf_scores = torch.cat(all_conf_scores, dim=1)
        query_frame_ids = torch.cat(all_query_frames, dim=0)

        return pred_tracks, pred_vis_scores, pred_conf_scores, query_frame_ids

    # ================================================================
    # Convert TAPIR output -> PointsMatched representation
    # ================================================================

    def _tracks_to_data_matrix(
        self,
        pred_tracks: torch.Tensor,
        pred_vis_scores: torch.Tensor,
        pred_conf_scores: Optional[torch.Tensor],
    ):
        """
        Convert TAPIR:

            pred_tracks:
                [1,N,T,2]

        into your API format:

            data_matrix:
                [track_id, frame_id, x, y]
        """

        tracks = pred_tracks[0]
        vis_scores = pred_vis_scores[0]

        # [N,T,2]
        # [N,T]

        if pred_conf_scores is not None:
            conf_scores = pred_conf_scores[0]
        else:
            conf_scores = None

        # If TAPIR ran on resized images, convert coordinates
        # back to the original image resolution.
        if self.resize_to is not None:
            target_h, target_w = self.resize_to

            scale_x = float(self.width) / float(target_w)
            scale_y = float(self.height) / float(target_h)

            tracks = tracks.clone()
            tracks[..., 0] *= scale_x
            tracks[..., 1] *= scale_y

        rows = []
        new_track_id = 0

        track_num = tracks.shape[0]

        for track_idx in range(track_num):
            visible = vis_scores[track_idx] >= self.visibility_threshold

            if (conf_scores is not None) and (self.score_threshold > 0):
                visible &= conf_scores[track_idx] >= self.score_threshold

            frame_ids = torch.nonzero(visible, as_tuple=False).flatten()

            if frame_ids.numel() < self.min_track_length:
                continue

            for frame_id in frame_ids.tolist():
                x, y = tracks[track_idx, frame_id]

                rows.append([new_track_id, frame_id, float(x), float(y)])

            new_track_id += 1

        if not rows:
            return np.empty((0, 4), dtype=np.float32)

        return np.asarray(rows, dtype=np.float32)

    # ================================================================
    # Build track map
    # ================================================================

    @staticmethod
    def _build_track_map(
        data_matrix: np.ndarray,
    ):
        """
        Representation:

            track_map[track_id][frame_id] = xy
        """

        track_map = {}

        for row in data_matrix:
            track_id = int(row[0])
            frame_id = int(row[1])

            xy = row[
                2:4
            ].astype(
                np.float32
            )

            track_map.setdefault(
                track_id,
                {}
            )[frame_id] = xy

        return track_map

    # ================================================================
    # Main API call
    # ================================================================

    def build_tracks_from_features(self, 
                                   img_features: list[Points2D],
                                   ) -> PointsMatched:
        """
        Build PointsMatched from existing per-frame features.

        Parameters
        ----------
        img_features:
            list[Points2D]

        Returns
        -------
        PointsMatched
            Populated with:
                data_matrix
                track_map
                point_count
        """

        pred_tracks, pred_vis_scores, pred_conf_scores, query_frame_ids = (
            self._track_video(img_features=img_features)
        )

        data_matrix = self._tracks_to_data_matrix(pred_tracks, 
                                                  pred_vis_scores, 
                                                  pred_conf_scores)
        track_map = self._build_track_map(data_matrix)

        # Adjust this constructor to your actual PointsMatched signature.
        matched_points = PointsMatched(image_size=(self.width, self.height),
                                       multi_view=True,
                                       stereo_cam=False)

        matched_points.img_features = img_features
        matched_points.data_matrix = data_matrix
        matched_points.track_map = track_map
        matched_points.point_count = len(track_map)

        return matched_points

class FeatureTrackingVGGSfM(FeatureTrackingBase):
    requires_features = True
    requires_feature_pairs = False
    direct_tracker = False

    def _build_tracker(self):
        vggsfm_cfg = SimpleNamespace(
            MODEL=SimpleNamespace(
                TRACK=SimpleNamespace(
                    efficient_corr=self.efficient_corr,
                )
            )
        )
        coarse_cfg = OmegaConf.create({
            "stride": self.coarse_stride,
            "down_ratio": self.coarse_down_ratio,

            "FEATURENET": {
                "_target_": "vggsfm.models.BasicEncoder",
            },

            "PREDICTOR": {
                "_target_": "vggsfm.models.BaseTrackerPredictor",
            },
        })

        fine_cfg = OmegaConf.create({
            "FEATURENET": {
                "_target_": "vggsfm.models.ShallowEncoder",
            },

            "PREDICTOR": {
                "_target_": "vggsfm.models.BaseTrackerPredictor",

                "depth": self.fine_depth,
                "corr_levels": self.fine_corr_levels,
                "corr_radius": self.fine_corr_radius,
                "latent_dim": self.fine_latent_dim,
                "hidden_size": self.fine_hidden_size,

                "fine": True,
                "use_spaceatt": self.fine_use_spaceatt,
            },
        })

        tracker = TrackerPredictor(
            COARSE=coarse_cfg,
            FINE=fine_cfg,
            cfg=vggsfm_cfg,
        )

        return tracker.to(device=self.device).eval()

    def __init__(
        self,
        cam_data: CameraData,
        model_path: str | None = None,
        min_track_len: int = 2,
        query_fram_num: int = 5, 
        query_selection: str = "dino",
        max_points_num: int = 163840,
        fine_tracking: bool = True,
        coarse_iters: int = 6,
        visibility_threshold: float = 0.5,
        score_threshold: float = 0.0,
        # -------------------------
        # Coarse tracker
        # -------------------------
        coarse_stride: int = 4,
        coarse_down_ratio: int = 2,
        efficient_corr: bool = False,
        # -------------------------
        # Fine tracker
        # -------------------------
        fine_depth: int = 4,
        fine_corr_levels: int = 3,
        fine_corr_radius: int = 3,
        fine_latent_dim: int = 32,
        fine_hidden_size: int = 256,
        fine_use_spaceatt: bool = False,
        **kwargs,
    ):  

        module_name = "FeatureTrackingVGGSfM"

        description = f"""
Tracks detected feature points across an image sequence using the learned
VGGSfM tracker. Features from selected query frames are propagated across the
remaining frames and converted into persistent multi-view tracks.

Creates learned multi-view feature tracks specifically for camera-pose estimation, 
triangulation, and bundle adjustment. USE THIS MODULE for SfM reconstruction when 
images contain substantial viewpoint or baseline changes, imperfect illumination, 
or cases where sequential pairwise matching does not maintain sufficiently long tracks. 
It is best used with GPU processing and can refine query points initialized from sparse 
detectors such as SIFT or SuperPoint. Choose it when reconstruction accuracy and 
multi-view consistency are more important than lightweight runtime.

Note: Detector Free feature pairs must use UnionFind, this module does not support 
RoMa and Loftr Features!

Initialization/Function Parameters:

- model_path: Optional path to VGGSfM tracker weights. The default pretrained weights are
  downloaded when no path is provided.
    - Default: None
- min_track_len:Minimum number of visible frames required to retain a track.
    - Default (int): 2
- query_fram_num: Number of frames used to initialize feature trajectories.
    - Default (int): 5
- query_selection: Method used to select query frames. Supported values are "dino", "interval", and "midpoint".
    - Default (str): "dino"
- max_points_num: Controls the number of points processed at once to limit memory usage.
    - Default (int): 163840
- fine_tracking: Enables the VGGSfM fine-tracking refinement stage.
    - Default (bool): True
- coarse_iters: Number of coarse tracking refinement iterations.
    - Default (int): 6
- visibility_threshold: Minimum predicted visibility required to keep an observation.
    - Default (float): 0.5
- score_threshold: Minimum tracking score required to keep an observation. A value of zero
  disables score filtering.
    - Default (float): 0.0

Coarse Tracker Parameters:

- coarse_stride: Feature-map stride used by the coarse tracker.
    - Default (int): 4
- coarse_down_ratio: Image downsampling ratio used during coarse tracking.
    - Default (int): 2
- efficient_corr: Enables the memory-efficient correlation implementation.
    - Default (bool): False

Fine Tracker Parameters:

- fine_depth: Number of fine tracker refinement layers.
    - Default (int): 4
- fine_corr_levels: Number of correlation pyramid levels.
    - Default (int): 3
- fine_corr_radius: Search radius used by the fine correlation stage.
    - Default (int): 3
- fine_latent_dim: Latent feature dimension used by the fine tracker.
    - Default (int): 32
- fine_hidden_size: Hidden feature dimension used by the fine tracker.
    - Default (int): 256
- fine_use_spaceatt: Enables spatial attention in the fine tracker.
    - Default (bool): False

Function Call Parameters - HANDLED INTERNALLY, DO NOT USE IF SFMCORE IS IN USE:

- img_features: list[Points2D]
    Detected feature points for each input frame.
"""

        example = f"""
from modules.baseclass import SfMScene
from modules.features import FeatureDetectionSIFT
from modules.featuretracking import {module_name}

# Step 1: Load image and calibration data.
reconstructed_scene = SfMScene(
    image_path=image_path,
    calibration_path=calibration_path,
)

# Step 2: Detect features before tracking.
reconstructed_scene.FeatureDetectionSIFT()

# Step 3: Track features using VGGSfM.
reconstructed_scene.{module_name}(
    min_track_len=3,
    query_fram_num=5,
    query_selection="dino",
    fine_tracking=True,
    visibility_threshold=0.5,
    score_threshold=0.0,
)
"""


        super().__init__(
            detector="features",
            cam_data=cam_data,
            module_name=module_name,
            description=description,
            example=example,
            RANSAC_threshold=kwargs.get("RANSAC_threshold", 3.0),
            RANSAC_conf=kwargs.get("RANSAC_conf", 0.99),
            RANSAC_homography=kwargs.get("RANSAC_homography", False),
            )

        self.device = f"cuda:{self.cam_data.gpu_num}" if torch.cuda.is_available() else "cpu"
        

        self.fine_tracking = fine_tracking
        self.coarse_iters = coarse_iters
        self.visibility_threshold = visibility_threshold
        self.score_threshold = score_threshold
        self.query_selection = query_selection
        self.query_frame_num = query_fram_num
        self.min_track_length = min_track_len
        self.max_points_num = max_points_num

        # Instantiate only the learned VGGSfM tracker.

        # PARAMS
        self.efficient_corr = efficient_corr
        self.coarse_stride = coarse_stride
        self.coarse_down_ratio = coarse_down_ratio

        self.fine_depth = fine_depth
        self.fine_corr_levels = fine_corr_levels
        self.fine_corr_radius = fine_corr_radius
        self.fine_latent_dim = fine_latent_dim
        self.fine_hidden_size = fine_hidden_size
        self.fine_use_spaceatt = fine_use_spaceatt

        self.track_predictor = self._build_tracker()
        if model_path is None:
            default_url = "https://huggingface.co/facebook/VGGSfM/resolve/main/vggsfm_v2_tracker.pt"
            self.track_predictor.load_state_dict(torch.hub.load_state_dict_from_url(default_url))
        else:
            self.track_predictor.load_state_dict(torch.load(model_path))

        # self.track_predictor = (
        #     self.track_predictor
        #     .to(self.device)
        #     .eval()
        # )

        self.width, self.height = self.image_list[0].size
        # Load Images in correct format for VGGSfM Inference!
        to_tensor = TF.ToTensor()
        tensor_img_list = [to_tensor(img) for img in self.image_list]

        self.images = torch.stack(tensor_img_list, dim=0)
        # [T, C, H, W] -> [1, T, C, H, W]
        self.images = self.images.unsqueeze(0)
        self.images = self.images.to(self.device)

        # self.images = torch.stack(tensor_img_list).to(self.device) 
        print(self.images.shape)
        # self.minimum_observation = min_observe

    # ================================================================
    # Query frame selection
    # ================================================================
    def _select_query_frames(self, frame_num: int) -> list[int]:
        num_queries = min(self.query_frame_num, frame_num)

        if self.query_selection == "dino":
            if generate_rank_by_dino is None:
                raise ImportError(
                    "query_selection='dino' requires VGGT's "
                    "generate_rank_by_dino. Could not import from "
                    "vggt.dependency.vggsfm_utils."
                )

            # self.images is [1, T, 3, H, W]
            # VGGT generate_rank_by_dino expects [T, 3, H, W]
            images_for_dino = self.images[0]

            ranking = generate_rank_by_dino(
                images=images_for_dino,
                query_frame_num=num_queries,
                device=self.device,
            )
            print(ranking)

        elif self.query_selection == "interval":
            interval = max(1, frame_num // num_queries)
            ranking = generate_rank_by_interval(frame_num, interval)

        elif self.query_selection == "midpoint":
            ranking = generate_rank_by_midpoint(frame_num)

        else:
            raise ValueError(
                "query_selection must be 'dino', 'interval', or 'midpoint'."
            )

        ranking = list(ranking)

        # Always include frame 0, matching the VGGT demo behavior.
        if 0 in ranking:
            ranking.remove(0)

        query_frames = [0, *ranking]

        return query_frames[:num_queries]


    # ================================================================
    # Chunked VGGSfM tracking
    # ================================================================

    def _predict_tracks_in_chunks(
        self,
        images_feed: torch.Tensor,
        query_points: torch.Tensor,
        fmaps_feed: torch.Tensor,
    ):
        """
        Memory-aware wrapper around TrackerPredictor.

        images_feed:
            [1,T,3,H,W]

        query_points:
            [1,N,2]

        output:
            track:
                [1,T,N,2]

            visibility:
                [1,T,N]

            scores:
                [1,T,N]
        """

        frame_num = images_feed.shape[1]
        point_num = query_points.shape[1]

        if point_num == 0:
            raise ValueError( "Cannot track zero query points.")

        # Approximate VGGSfM memory control:
        #
        # T * points_per_chunk <= max_points_num

        points_per_chunk = max(1, self.max_points_num // frame_num)

        track_list = []
        vis_list = []
        score_list = []

        for start in range(0, point_num, points_per_chunk):

            end = min(start + points_per_chunk, point_num)
            points_chunk = query_points[:,start:end]

            track, _ , pred_vis, pred_score = self.track_predictor(
                images_feed,
                points_chunk,
                fmaps=fmaps_feed,
                coarse_iters=self.coarse_iters,
                inference=True,
                fine_tracking=self.fine_tracking,
            )

            track_list.append(track)
            vis_list.append(pred_vis)

            if pred_score is not None:
                score_list.append(
                    pred_score
                )

        pred_track = torch.cat(track_list, dim=2)
        pred_vis = torch.cat(vis_list, dim=2)
        pred_score = (torch.cat(score_list, dim=2)
                      if score_list
                      else None
                     )

        return pred_track, pred_vis, pred_score

    # ================================================================
    # Track one selected query frame
    # ================================================================

    def _track_query_frame(
        self,
        # images: torch.Tensor,
        fmaps_for_tracker: torch.Tensor,
        query_points: torch.Tensor,
        query_index: int,
    ):
        """
        Track query points originating from query_index.

        VGGSfM's TrackerPredictor assumes that query_points belong
        to image 0.

        Therefore:

            [0,1,2,3,4]

        for query_index=3 becomes:

            [3,1,2,0,4]

        before inference.

        Results are then restored to original frame order.
        """

        frame_num = self.images.shape[1]

        new_order = calculate_index_mappings(query_index, frame_num, device=self.images.device)

        images_feed, fmaps_feed = switch_tensor_order(
            [self.images, fmaps_for_tracker],
            new_order,
            dim=1,
        )

        # N x 2
        #
        # ->
        #
        # 1 x N x 2

        query_points = query_points.unsqueeze(0)

        pred_track, pred_vis, pred_score = self._predict_tracks_in_chunks(
            images_feed,
            query_points,
            fmaps_feed,
        )

        # ------------------------------------------------------------
        # Restore the original frame ordering.
        # ------------------------------------------------------------

        output_tensors = [pred_track, pred_vis]

        if pred_score is not None:
            pred_track, pred_vis, pred_score = switch_tensor_order(
                [pred_track, pred_vis, pred_score],
                new_order,
                dim=1,
            )
        else:
            pred_track, pred_vis = switch_tensor_order([pred_track, pred_vis],
                                                       new_order,
                                                       dim=1
                                                     )

        return pred_track, pred_vis, pred_score

    # ================================================================
    # Multi-query-frame video tracking
    # ================================================================

    @torch.inference_mode()
    def _track_video(self, img_features: list[Points2D]):
        """
        Main VGGSfM video tracking operation.
        """

        frame_num = self.images.shape[1]
        query_frames = self._select_query_frames(frame_num)

        # ------------------------------------------------------------
        # VGGSfM computes this once for the entire scene.
        # ------------------------------------------------------------

        fmaps_for_tracker = self.track_predictor.process_images_to_fmaps(self.images)

        all_tracks = []
        all_vis = []
        all_scores = []
        all_query_frames = []

        for query_index in query_frames:
            if query_index >= len(img_features):
                continue

            features = img_features[query_index]

            if features is None:
                continue

            query_points = features.points2D

            query_points = torch.as_tensor(
                query_points,
                device=self.device
            )

            if len(query_points) == 0:
                continue

            pred_track, pred_vis, pred_score = self._track_query_frame(
                # images=images,
                fmaps_for_tracker=fmaps_for_tracker,
                query_points=query_points,
                query_index=query_index,
            )

            all_tracks.append(pred_track)
            all_vis.append(pred_vis)

            if pred_score is not None:
                all_scores.append(pred_score)

            all_query_frames.append(
                torch.full(
                    (query_points.shape[0],),
                    query_index,
                    dtype=torch.long,
                    device=self.device,
                )
            )

        if not all_tracks:
            raise RuntimeError("VGGSfM could not generate any tracks.")

        pred_tracks = torch.cat(all_tracks, dim=2)
        pred_vis = torch.cat(all_vis, dim=2)
        pred_scores = (
            torch.cat(all_scores, dim=2)
            if all_scores
            else None
        )
        query_frame_ids = torch.cat(all_query_frames, dim=0)

        return pred_tracks, pred_vis, pred_scores, query_frame_ids

    # ================================================================
    # Convert VGGSfM output -> PointsMatched representation
    # ================================================================

    def _tracks_to_data_matrix(
        self,
        pred_tracks: torch.Tensor,
        pred_vis: torch.Tensor,
        pred_scores: Optional[torch.Tensor],
    ):
        """
        Convert:

            tracks:
                [1,T,N,2]

        to:

            data_matrix:
                [track_id, frame_id, x, y]
        """

        tracks = pred_tracks[0]
        visibility = pred_vis[0]

        # T x N x 2

        if pred_scores is not None:
            scores = pred_scores[0]
        else:
            scores = None

        rows = []
        new_track_id = 0
        track_num = tracks.shape[1]

        for track_idx in range(track_num):

            visible = (visibility[:, track_idx] >= 
                       self.visibility_threshold)

            if (scores is not None
                and self.score_threshold > 0
            ):
                visible &= (scores[:, track_idx] >= self.score_threshold)

            frame_ids = torch.nonzero(visible, as_tuple=False).flatten()

            if (frame_ids.numel() < self.min_track_length):
                continue

            for frame_id in frame_ids.tolist():
                x, y = tracks[frame_id, track_idx]
                rows.append([new_track_id, frame_id, float(x), float(y)])

            new_track_id += 1

        if not rows:
            return np.empty((0, 4), dtype=np.float32)

        return np.asarray(rows, dtype=np.float32,)

    # ================================================================
    # Build track map
    # ================================================================

    @staticmethod
    def _build_track_map(data_matrix: np.ndarray):
        """
        Representation:

            track_map[track_id][frame_id] = xy
        """

        track_map = {}

        for row in data_matrix:

            track_id = int(row[0])
            frame_id = int(row[1])

            xy = row[2:4].astype(np.float32)

            track_map.setdefault(track_id,{})[frame_id] = xy

        return track_map

    # ================================================================
    # Main API call
    # ================================================================

    def build_tracks_from_features(self, img_features: list[Points2D]) -> PointsMatched:
        """
        Parameters
        ----------
        camera_data:
            Your CameraData object containing the image sequence.

        matched_points:
            Your PointsMatched object.

            Expected:
                matched_points.img_features[frame_id]

            with each entry containing:
                keypoints
                scores (optional)

        Returns
        -------
        matched_points:
            Same PointsMatched object with:

                data_matrix
                track_map
                point_count

            populated from VGGSfM trajectories.
        """
        img_size = img_features[0].image_size
        img_scale = img_features[0].reshape_scale
        matched_points = PointsMatched(image_size=img_size, 
                                         multi_view=True,
                                         image_scale=img_scale)

        # ------------------------------------------------------------
        # 2. Track features across video
        # ------------------------------------------------------------

        pred_tracks, pred_vis, pred_scores, query_frame_ids = self._track_video(
            # images=images,
            img_features=img_features,#matched_points.img_features,
        )

        # ------------------------------------------------------------
        # 3. Convert to persistent track representation
        # ------------------------------------------------------------

        print(pred_tracks)
        data_matrix = self._tracks_to_data_matrix(pred_tracks,
                                                  pred_vis,
                                                  pred_scores
                                                  )
        print(data_matrix)
        track_map = self._build_track_map(data_matrix)

        # ------------------------------------------------------------
        # 4. Update your PointsMatched object
        # ------------------------------------------------------------

        matched_points.data_matrix = data_matrix
        matched_points.track_map = track_map
        matched_points.point_count = len(track_map)
        print(matched_points.data_matrix)

        return matched_points

class FeatureTrackFromPairsUnionFind(FeatureTrackingBase):
    requires_features = False
    requires_feature_pairs = True
    direct_tracker = False

    def __init__(
        self,
        cam_data: CameraData,
        min_track_len: int = 2,
        allow_pair_local_merge: bool = True,
        **kwargs,
    ):

        module_name = "FeatureTrackFromPairsUnionFind"

        description = f"""
Builds multi-view feature tracks from existing pairwise point correspondences.
This module follows a global track-construction approach similar to the
Union-Find track-building stage used in global Structure-from-Motion pipelines.

Unlike direct feature trackers, this module does not detect features or estimate
new correspondences between images. Pairwise feature matches must already have
been generated by a feature-matching module. USE THIS MODULE for classical sparse 
SfM when reliable features and pairwise matches already exist. It works well with 
SIFT for textured, wide-baseline image collections and with SuperPoint or ALIKED 
plus LightGlue for more challenging appearance changes. ORB is better suited to 
well-lit, sequential, high-overlap imagery. This is the fastest and most 
lightweight option.

Each pairwise match indicates that two observations likely correspond to the
same physical 3D point. Union-Find combines these pairwise relationships into
connected components, where each connected component becomes a candidate
feature track spanning multiple images.

Use this module after pairwise feature matching when globally consistent
multi-view tracks are required for camera pose estimation, triangulation, scene
reconstruction, or bundle adjustment. This module is especially useful when
the pairwise correspondences were generated by matchers such as FLANN,
Brute-Force, LightGlue, SuperGlue, LoFTR, RoMa, or another pairwise matching
method supported by SfMCore.

This is not a direct tracking module and does not process image sequences using
optical flow or a learned temporal tracking network.

Initialization/Function Parameters:
- min_track_len: Minimum number of unique image observations required for a connected
  component to be retained as a valid feature track.
    - Default (int): 2
- allow_pair_local_merge: Determines whether observations generated by pair-local matchers may be
  merged across image pairs when constructing global tracks.
    - Default (bool): True

- RANSAC_homography: Determines whether Homography or Fundamental matrix geometry is used for
  pairwise correspondence outlier rejection.
    - Default (bool): False
    - True: Use a Homography model
    - False: Use a Fundamental matrix model

- RANSAC_threshold: Maximum geometric residual allowed for a correspondence to be considered
  an inlier during RANSAC-based geometric verification. For Fundamental matrix estimation, 
  this typically represents the maximum point-to-epipolar-line error in pixel coordinates. 
  For Homography estimation, it typically represents the maximum reprojection error.
    - Default (float): 3.0

- RANSAC_conf: Desired confidence that the model estimated by RANSAC is valid.
  Higher values may require more RANSAC iterations but reduce the probability
  that the selected model was generated from an insufficient inlier sample.
    - Default (float): 0.99

Function Call Parameters - HANDLED INTERNALLY, DO NOT USE IF SFMCORE IS IN USE:

- feature_pairs: PointsMatched
    Existing pairwise feature correspondences produced by a feature-matching
    module.
    The PointsMatched object must contain:
    - pairwise_obs_ids:
        Observation ID pairs associated with each accepted pairwise match.
    - obs_image:
        Mapping from each observation ID to its source image.
    - obs_xy:
        Mapping from each observation ID to its 2D image coordinates.

Module Output:

- PointsMatched: The input PointsMatched object is updated with multi-view feature-track
    information.

    Updated fields include:
    - data_matrix:
        Float array with shape [N, 4], where each row contains:
        [track_id, image_id, x, y]
    - track_map:
        Dictionary mapping each track ID to its image observations.
    - point_count:
        Number of valid multi-view tracks.
    - multi_view:
        Set to True after track construction.
"""


        example = f"""
# Initialization
from modules.baseclass import SfMScene
from modules.features import FeatureDetectionSIFT
from modules.featurematching import FeatureMatchFlann
from modules.featuretracking import {module_name}

# Start SfM Pipeline

# Step 1: Read calibration and image data.
reconstructed_scene = SfMScene(
    image_path=image_path,
    calibration_path=calibration_path,
)

# Step 2: Detect image features.
reconstructed_scene.FeatureDetectionSIFT()

# Step 3: Estimate pairwise feature correspondences.
reconstructed_scene.FeatureMatchFlann(
    detector="sift",
    k=2,
    lowes_thresh=0.78,
    RANSAC_homography=False,
    RANSAC_threshold=2.0,
    RANSAC_conf=0.999,
)

# Step 4: Convert pairwise correspondences into multi-view tracks.
reconstructed_scene.{module_name}(
    min_track_len=2,
    allow_pair_local_merge=True,
    RANSAC_homography=False,
    RANSAC_threshold=3.0,
    RANSAC_conf=0.99,
)
"""

        super().__init__(
            detector="pairwise",
            cam_data=cam_data,
            module_name=module_name,
            description=description,
            example=example,
            RANSAC_threshold=kwargs.get("RANSAC_threshold", 3.0),
            RANSAC_conf=kwargs.get("RANSAC_conf", 0.99),
            RANSAC_homography=kwargs.get("RANSAC_homography", False),
        )

        self.min_track_len = min_track_len

    def build_tracks_from_pairs(self, feature_pairs: PointsMatched) -> PointsMatched:
        uf = UnionFindTracks(feature_pairs.obs_image)

        for pair_idx in range(len(feature_pairs.pairwise_obs_ids)):
            obs_pairs = feature_pairs.pairwise_obs_ids[pair_idx]

            for k in range(obs_pairs.shape[0]):
                obs_i = int(obs_pairs[k, 0])
                obs_j = int(obs_pairs[k, 1])

                uf.add(obs_i)
                uf.add(obs_j)
                uf.union(obs_i, obs_j)

        components = {}

        for obs_id in feature_pairs.obs_xy.keys():
            root = uf.find(obs_id)
            components.setdefault(root, []).append(obs_id)

        rows = []
        track_map = {}
        track_id = 0

        for _, obs_ids in components.items():
            obs_ids = self._dedupe_one_obs_per_image(obs_ids, feature_pairs)

            if len(obs_ids) < self.min_track_len:
                continue

            track_map[track_id] = []

            for obs_id in obs_ids:
                image_id = feature_pairs.get_obs_image(obs_id)
                xy = feature_pairs.get_obs_xy(obs_id)

                row = [
                    track_id,
                    image_id,
                    float(xy[0]),
                    float(xy[1]),
                ]

                rows.append(row)
                track_map[track_id].append(row)

            track_id += 1

        if len(rows) == 0:
            data_matrix = np.empty((0, 4), dtype=np.float32)
        else:
            data_matrix = np.array(rows, dtype=np.float32)

        feature_pairs.data_matrix = data_matrix
        feature_pairs.track_map = track_map
        feature_pairs.point_count = track_id
        feature_pairs.multi_view = True

        return feature_pairs

    def _dedupe_one_obs_per_image(
        self,
        obs_ids: list[int],
        feature_pairs: PointsMatched,
    ) -> list[int]:
        """
        Valid SfM tracks should have at most one observation per image.
        If a component contains multiple observations from the same image,
        keep one.
        """
        selected = {}

        for obs_id in obs_ids:
            image_id = feature_pairs.get_obs_image(obs_id)

            if image_id not in selected:
                selected[image_id] = obs_id
            else:
                # Later you can choose by confidence, residual, or track support.
                pass

        return list(selected.values())