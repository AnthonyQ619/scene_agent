import sys
############ TEMP SOLUTION FOR NOW #################
import json
import cv2
import copy
import numpy as np
import glob
from tqdm import tqdm
from modules.DataTypes.datatype import Points2D, PointsMatched, CameraData
from modules.models.matchers import LightGlue, SuperGlue
from romatch import roma_outdoor, roma_indoor
from vggsfm.models.track_predictor import TrackerPredictor
from vggsfm.utils.utils import (
    calculate_index_mappings,
    switch_tensor_order,
    generate_rank_by_interval,
    generate_rank_by_midpoint,
)

from modules.baseclass import FeatureMatching, FeatureTracking, FeatureTrackingBase, module_metric
from collections.abc import Callable
import kornia as K
import kornia.feature as KF
import torch
from PIL import Image, ImageOps
import piexif

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

class FeatureTrackVGGT(FeatureTrackingBase):
    requires_features = True
    requires_feature_pairs = False
    direct_tracker = False

    def __init__(
        self,
        cam_data: CameraData,
        model_path: str | None = None,
        min_track_len: int = 2,
        query_fram_num: int = 5, 
        query_selection: str = "interval",
        fine_tracking: bool = True,
        coarse_iters: int = 6,
        visibility_threshold: float = 0.5,
        score_threshold: float = 0.0,
    ):  

        super().__init__(
            cam_data=cam_data,
            module_name="FeatureTrackFromPairsUnionFind",
            description="Build feature tracks from existing pairwise correspondences using Union-Find.",
            example="scene.FeatureTrackFromPairsUnionFind()")

        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.fine_tracking = fine_tracking
        self.coarse_iters = coarse_iters
        self.visibility_threshold = visibility_threshold
        self.score_threshold = score_threshold
        self.query_selection = query_selection
        self.query_fram_num = query_fram_num
        self.min_track_length = min_track_length

        # Instantiate only the learned VGGSfM tracker.
        self.track_predictor = TrackerPredictor()

        if model_path is None:
            default_url = "https://huggingface.co/facebook/VGGSfM/resolve/main/vggsfm_v2_tracker.pt"
            self.track_predictor.load_state_dict(torch.hub.load_state_dict_from_url(default_url))
        else:
            self.track_predictor.load_state_dict(torch.load(model_path))

        self.track_predictor = (
            self.track_predictor
            .to(self.device)
            .eval()
        )

        self.width, self.height = self.image_list[0].size
        # Load Images in correct format for VGGSfM Inference!
        to_tensor = TF.ToTensor()
        tensor_img_list = [to_tensor(img) for img in self.image_list]

        self.images = torch.stack(tensor_img_list).to(self.device) 

        self.minimum_observation = min_observe

    # ================================================================
    # Image preparation
    # ================================================================

    def _prepare_images(
        self,
        images,
    ) -> torch.Tensor:
        """
        Convert input images to:

            [1, T, 3, H, W]

        float RGB in [0,1].

        VGGSfM TrackerPredictor expects exactly this representation.
        """

        processed = []

        for image in images:

            if isinstance(image, np.ndarray):

                image = torch.from_numpy(image)

                if (
                    image.ndim == 3
                    and image.shape[-1] == 3
                ):
                    image = image.permute(
                        2,
                        0,
                        1,
                    )

            elif isinstance(image, torch.Tensor):

                image = image.detach()

                if (
                    image.ndim == 3
                    and image.shape[-1] == 3
                ):
                    image = image.permute(
                        2,
                        0,
                        1,
                    )

            else:
                # Support PIL images.
                image = np.asarray(image)

                image = torch.from_numpy(
                    image
                ).permute(
                    2,
                    0,
                    1,
                )

            image = image.float()

            if image.max() > 1.0:
                image /= 255.0

            processed.append(image)

        if not processed:
            raise ValueError(
                "No images provided to VGGSfM tracker."
            )

        first_shape = processed[0].shape

        for frame_id, image in enumerate(processed):

            if image.shape != first_shape:
                raise ValueError(
                    "VGGSfM requires all tracking images to "
                    "have the same dimensions. "
                    f"Frame 0={first_shape}, "
                    f"frame {frame_id}={image.shape}."
                )

        images = torch.stack(
            processed,
            dim=0,
        )

        # T C H W
        #
        # ->
        #
        # 1 T C H W

        images = images.unsqueeze(0)

        return images.to(
            device=self.device,
            dtype=self.dtype,
        )

    # ================================================================
    # Query frame selection
    # ================================================================

    def _select_query_frames(
        self,
        frame_num: int,
    ) -> list[int]:

        num_queries = min(
            self.query_frame_num,
            frame_num,
        )

        if self.query_selection == "interval":

            # Spread query frames across the sequence.
            #
            # Similar to:
            #
            # 0 ----- 20 ----- 40 ----- 60 ----- 80

            interval = max(
                1,
                frame_num // num_queries,
            )

            ranking = generate_rank_by_interval(
                frame_num,
                interval,
            )

        elif self.query_selection == "midpoint":

            ranking = generate_rank_by_midpoint(
                frame_num
            )

        else:
            raise ValueError(
                "query_selection must be "
                "'interval' or 'midpoint'."
            )

        ranking = list(ranking)

        # ------------------------------------------------------------
        # Always include frame zero.
        #
        # This follows the behavior you wanted from the VGGT demo.
        # ------------------------------------------------------------

        if 0 in ranking:
            ranking.remove(0)

        query_frames = [
            0,
            *ranking,
        ]

        return query_frames[:num_queries]

    # ================================================================
    # Query feature extraction
    # ================================================================

    def _get_query_points(
        self,
        features,
    ) -> torch.Tensor:
        """
        Extract strongest query points from your existing Points2D.

        Expected:
            features.keypoints -> Nx2
            features.scores    -> N, optional
        """

        keypoints = features.keypoints

        keypoints = torch.as_tensor(
            keypoints,
            device=self.device,
            dtype=self.dtype,
        )

        if (
            keypoints.ndim != 2
            or keypoints.shape[1] != 2
        ):
            raise ValueError(
                "Feature keypoints must have shape [N,2]."
            )

        if len(keypoints) == 0:
            return keypoints

        # ------------------------------------------------------------
        # Keep strongest features if detector confidence exists.
        # ------------------------------------------------------------

        if len(keypoints) > self.max_query_pts:

            scores = getattr(
                features,
                "scores",
                None,
            )

            if scores is not None:

                scores = torch.as_tensor(
                    scores,
                    device=self.device,
                ).reshape(-1)

                keep = torch.topk(
                    scores,
                    k=self.max_query_pts,
                ).indices

                keypoints = keypoints[keep]

            else:

                keypoints = keypoints[
                    : self.max_query_pts
                ]

        return keypoints

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
            raise ValueError(
                "Cannot track zero query points."
            )

        # Approximate VGGSfM memory control:
        #
        # T * points_per_chunk <= max_points_num

        points_per_chunk = max(
            1,
            self.max_points_num // frame_num,
        )

        track_list = []
        vis_list = []
        score_list = []

        for start in range(
            0,
            point_num,
            points_per_chunk,
        ):

            end = min(
                start + points_per_chunk,
                point_num,
            )

            points_chunk = query_points[
                :,
                start:end,
            ]

            (
                fine_track,
                coarse_track,
                pred_vis,
                pred_score,
            ) = self.track_predictor(
                images_feed,
                points_chunk,
                fmaps=fmaps_feed,
                coarse_iters=self.coarse_iters,
                inference=True,
                fine_tracking=self.fine_tracking,
            )

            track = (
                fine_track
                if self.fine_tracking
                else coarse_track
            )

            track_list.append(track)

            vis_list.append(
                pred_vis
            )

            if pred_score is not None:
                score_list.append(
                    pred_score
                )

        pred_track = torch.cat(
            track_list,
            dim=2,
        )

        pred_vis = torch.cat(
            vis_list,
            dim=2,
        )

        pred_score = (
            torch.cat(
                score_list,
                dim=2,
            )
            if score_list
            else None
        )

        return (
            pred_track,
            pred_vis,
            pred_score,
        )

    # ================================================================
    # Track one selected query frame
    # ================================================================

    def _track_query_frame(
        self,
        images: torch.Tensor,
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

        frame_num = images.shape[1]

        new_order = calculate_index_mappings(
            query_index,
            frame_num,
            device=images.device,
        )

        (
            images_feed,
            fmaps_feed,
        ) = switch_tensor_order(
            [
                images,
                fmaps_for_tracker,
            ],
            new_order,
            dim=1,
        )

        # N x 2
        #
        # ->
        #
        # 1 x N x 2

        query_points = query_points.unsqueeze(0)

        (
            pred_track,
            pred_vis,
            pred_score,
        ) = self._predict_tracks_in_chunks(
            images_feed,
            query_points,
            fmaps_feed,
        )

        # ------------------------------------------------------------
        # Restore the original frame ordering.
        # ------------------------------------------------------------

        output_tensors = [
            pred_track,
            pred_vis,
        ]

        if pred_score is not None:

            (
                pred_track,
                pred_vis,
                pred_score,
            ) = switch_tensor_order(
                [
                    pred_track,
                    pred_vis,
                    pred_score,
                ],
                new_order,
                dim=1,
            )

        else:

            (
                pred_track,
                pred_vis,
            ) = switch_tensor_order(
                [
                    pred_track,
                    pred_vis,
                ],
                new_order,
                dim=1,
            )

        return (
            pred_track,
            pred_vis,
            pred_score,
        )

    # ================================================================
    # Multi-query-frame video tracking
    # ================================================================

    @torch.inference_mode()
    def _track_video(
        self,
        images: torch.Tensor,
        img_features,
    ):
        """
        Main VGGSfM video tracking operation.
        """

        frame_num = images.shape[1]

        query_frames = self._select_query_frames(
            frame_num
        )

        # ------------------------------------------------------------
        # VGGSfM computes this once for the entire scene.
        # ------------------------------------------------------------

        fmaps_for_tracker = (
            self.track_predictor
            .process_images_to_fmaps(
                images
            )
        )

        all_tracks = []
        all_vis = []
        all_scores = []
        all_query_frames = []

        for query_index in query_frames:

            if query_index >= len(img_features):
                continue

            features = img_features[
                query_index
            ]

            if features is None:
                continue

            query_points = (
                self._get_query_points(
                    features
                )
            )

            if len(query_points) == 0:
                continue

            (
                pred_track,
                pred_vis,
                pred_score,
            ) = self._track_query_frame(
                images=images,
                fmaps_for_tracker=fmaps_for_tracker,
                query_points=query_points,
                query_index=query_index,
            )

            all_tracks.append(
                pred_track
            )

            all_vis.append(
                pred_vis
            )

            if pred_score is not None:
                all_scores.append(
                    pred_score
                )

            all_query_frames.append(
                torch.full(
                    (
                        query_points.shape[0],
                    ),
                    query_index,
                    dtype=torch.long,
                    device=self.device,
                )
            )

        if not all_tracks:

            raise RuntimeError(
                "VGGSfM could not generate any tracks."
            )

        # ------------------------------------------------------------
        # Combine trajectories seeded by different query frames.
        #
        # Each:
        #     [1,T,N_i,2]
        #
        # Combined:
        #     [1,T,N_total,2]
        # ------------------------------------------------------------

        pred_tracks = torch.cat(
            all_tracks,
            dim=2,
        )

        pred_vis = torch.cat(
            all_vis,
            dim=2,
        )

        pred_scores = (
            torch.cat(
                all_scores,
                dim=2,
            )
            if all_scores
            else None
        )

        query_frame_ids = torch.cat(
            all_query_frames,
            dim=0,
        )

        return (
            pred_tracks,
            pred_vis,
            pred_scores,
            query_frame_ids,
        )

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

            visible = (
                visibility[:, track_idx]
                >= self.visibility_threshold
            )

            if (
                scores is not None
                and self.score_threshold > 0
            ):
                visible &= (
                    scores[:, track_idx]
                    >= self.score_threshold
                )

            frame_ids = torch.nonzero(
                visible,
                as_tuple=False,
            ).flatten()

            if (
                frame_ids.numel()
                < self.min_track_length
            ):
                continue

            for frame_id in frame_ids.tolist():

                x, y = tracks[
                    frame_id,
                    track_idx,
                ]

                rows.append(
                    [
                        new_track_id,
                        frame_id,
                        float(x),
                        float(y),
                    ]
                )

            new_track_id += 1

        if not rows:

            return np.empty(
                (0, 4),
                dtype=np.float32,
            )

        return np.asarray(
            rows,
            dtype=np.float32,
        )

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

    def __call__(
        self,
        camera_data,
        matched_points,
    ):
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

        # ------------------------------------------------------------
        # 1. Prepare video
        # ------------------------------------------------------------

        images = self._prepare_images(
            camera_data.images
        )

        # ------------------------------------------------------------
        # 2. Track features across video
        # ------------------------------------------------------------

        (
            pred_tracks,
            pred_vis,
            pred_scores,
            query_frame_ids,
        ) = self._track_video(
            images=images,
            img_features=matched_points.img_features,
        )

        # ------------------------------------------------------------
        # 3. Convert to persistent track representation
        # ------------------------------------------------------------

        data_matrix = (
            self._tracks_to_data_matrix(
                pred_tracks,
                pred_vis,
                pred_scores,
            )
        )

        track_map = (
            self._build_track_map(
                data_matrix
            )
        )

        # ------------------------------------------------------------
        # 4. Update your PointsMatched object
        # ------------------------------------------------------------

        matched_points.data_matrix = (
            data_matrix
        )

        matched_points.track_map = (
            track_map
        )

        matched_points.point_count = len(
            track_map
        )

        # Optional — useful for debugging / metrics.
        matched_points.vggsfm_tracks = (
            pred_tracks
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        matched_points.vggsfm_visibility = (
            pred_vis
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        matched_points.vggsfm_scores = (
            pred_scores
            .detach()
            .float()
            .cpu()
            .numpy()
            if pred_scores is not None
            else None
        )

        matched_points.vggsfm_query_frames = (
            query_frame_ids
            .detach()
            .cpu()
            .numpy()
        )

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
        super().__init__(
            detector="pairwise",
            cam_data=cam_data,
            module_name="FeatureTrackFromPairsUnionFind",
            description="Build feature tracks from existing pairwise correspondences using Union-Find.",
            example="scene.FeatureTrackFromPairsUnionFind()",
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