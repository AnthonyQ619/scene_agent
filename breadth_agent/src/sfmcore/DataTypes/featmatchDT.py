from __future__ import annotations
from PIL import Image
import numpy as np
import torch
import cv2
import os

import inspect
from abc import ABC
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Optional, Tuple, Union, Dict, Any

Obs = Tuple[int, int]  # (image_id, kp_idx)

@dataclass
class PointsMatched:
    # General Data Information
    image_size: np.ndarray | None = None
    image_scale: list[float] = field(default_factory=lambda: [1.0, 1.0])
    multi_view: bool = False
    stereo_cam: bool = False

    # Tracked Data Features
    data_matrix: np.ndarray | None = None      # Nx4 -> [track_id, frame_num, x, y]
    track_map: dict = field(default_factory=dict)
    point_count: int = 0

    # Pairwise Data
    pairwise_matches: list[np.ndarray] = field(default_factory=list)  # Nx4 -> [x1, y1, x2, y2]
    pairwise_indices: list[np.ndarray] = field(default_factory=list)  # Nx2 -> [idx1, idx2]
    pairwise_obs_ids: list[np.ndarray] = field(default_factory=list)  # Nx2 -> stable observation ids
    pairwise_confidence: list[np.ndarray | None] = field(default_factory=list)

    # Existing image-global feature storage
    img_features: list[np.ndarray] = field(default_factory=list)

    # New lightweight metadata; avoids new classes
    pairwise_meta: list[dict] = field(default_factory=list)

    # Optional pair-local features for RoMa / LoFTR
    pair_features: dict = field(default_factory=dict)

    # Observation registry
    # obs_id -> image_id, xy
    obs_image: dict[int, int] = field(default_factory=dict)
    obs_xy: dict[int, np.ndarray] = field(default_factory=dict)

    # For sparse global feature detectors:
    # (image_id, feature_idx) -> obs_id
    feature_to_obs_id: dict[tuple[int, int], int] = field(default_factory=dict)

    # For detector-free / pair-local matchers:
    # image_id -> grid cell -> list[obs_id]
    image_obs_grid: dict[int, dict[tuple[int, int], list[int]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list))
    )

    next_obs_id: int = 0

    # Controls coordinate merging for LoFTR/RoMa pseudo observations
    pseudo_merge_eps_px: float = 1.5

    def set_matched_matrix(self, data: list[list]) -> None:
        self.data_matrix = np.array(data, dtype=np.float32)
        self.multi_view = True

    def set_matching_pair(
        self,
        data: np.ndarray,
        idx_data: np.ndarray | None = None,
        image_pair: tuple[int, int] | None = None,
        index_type: str = "global",
        matcher_name: str = "unknown",
        confidence: np.ndarray | None = None,
        # img1_feats: np.ndarray | None = None,
        # img2_feats: np.ndarray | None = None,
    ) -> None:
        """
        Store a pairwise correspondence set and immediately normalize it into
        stable observation IDs.

        Parameters
        ----------
        data:
            Nx4 array: [x1, y1, x2, y2]

        idx_data:
            Nx2 array.

            If index_type == "global":
                idx_data[:, 0] indexes the feature table of image_i.
                idx_data[:, 1] indexes the feature table of image_j.

            If index_type == "pair_local":
                idx_data is pair-local. If None, it is automatically set to
                [0..N-1, 0..N-1].

        image_pair:
            Tuple: (image_i, image_j)

        index_type:
            "global" for SIFT/SuperPoint/LightGlue.
            "pair_local" for LoFTR/RoMa.
        """
        assert data.ndim == 2 and data.shape[1] == 4, (
            "data must be Nx4: [x1, y1, x2, y2]"
        )

        assert index_type in {"global", "pair_local"}, (
            "index_type must be either 'global' or 'pair_local'"
        )

        if image_pair is None:
            image_pair = (len(self.pairwise_matches), len(self.pairwise_matches) + 1)

        image_i, image_j = image_pair

        data = data.astype(np.float32)
        N = data.shape[0]

        if idx_data is None:
            idx_data = np.stack(
                [
                    np.arange(N, dtype=np.int64),
                    np.arange(N, dtype=np.int64),
                ],
                axis=1,
            )

        idx_data = np.asarray(idx_data, dtype=np.int64)

        assert idx_data.ndim == 2 and idx_data.shape[1] == 2, (
            "idx_data must be Nx2"
        )
        assert idx_data.shape[0] == N, (
            "idx_data and data must have the same number of rows"
        )

        pts_i = data[:, :2] # Convert this from data to global features!
        pts_j = data[:, 2:] # Convert this from data (matching pair pts) to global features!

        if index_type == "global":
            # pts_i = img1_feats #pts in Nx2 for img1 -> Global Features
            # pts_j = img2_feats
            obs_ids = self._obs_ids_from_global_indices(
                image_i=image_i,
                image_j=image_j,
                pts_i=pts_i,
                pts_j=pts_j,
                idx_data=idx_data,
            )
        else:
            obs_ids = self._obs_ids_from_pair_local_points(
                image_i=image_i,
                image_j=image_j,
                pts_i=pts_i,
                pts_j=pts_j,
            )

        self.pairwise_matches.append(data)
        self.pairwise_indices.append(idx_data)
        self.pairwise_obs_ids.append(obs_ids)
        self.pairwise_confidence.append(confidence)

        self.multi_view = False

    def _obs_ids_from_global_indices(
        self,
        image_i: int,
        image_j: int,
        pts_i: np.ndarray,
        pts_j: np.ndarray,
        idx_data: np.ndarray,
    ) -> np.ndarray:
        obs_i = np.empty(idx_data.shape[0], dtype=np.int64)
        obs_j = np.empty(idx_data.shape[0], dtype=np.int64)

        for k in range(idx_data.shape[0]):
            feat_i = int(idx_data[k, 0])
            feat_j = int(idx_data[k, 1])

            obs_i[k] = self._get_or_create_global_obs(
                image_id=int(image_i),
                feature_idx=feat_i,
                xy=pts_i[k],#[feat_i],
            )

            obs_j[k] = self._get_or_create_global_obs(
                image_id=int(image_j),
                feature_idx=feat_j,
                xy=pts_j[k],#[feat_j],
            )

        return np.stack([obs_i, obs_j], axis=1)

    def _get_or_create_global_obs(
        self,
        image_id: int,
        feature_idx: int,
        xy: np.ndarray,
    ) -> int:
        key = (int(image_id), int(feature_idx))

        if key in self.feature_to_obs_id:
            return self.feature_to_obs_id[key]

        obs_id = self._new_obs_id(image_id=image_id, xy=xy)
        self.feature_to_obs_id[key] = obs_id

        return obs_id

    def _obs_ids_from_pair_local_points(
        self,
        image_i: int,
        image_j: int,
        pts_i: np.ndarray,
        pts_j: np.ndarray,
    ) -> np.ndarray:
        obs_i = self._assign_pair_local_obs_ids_fast(
            image_id=int(image_i),
            points=pts_i,
        )

        obs_j = self._assign_pair_local_obs_ids_fast(
            image_id=int(image_j),
            points=pts_j,
        )

        return np.stack([obs_i, obs_j], axis=1)

    def _assign_pair_local_obs_ids_fast(
        self,
        image_id: int,
        points: np.ndarray,
    ) -> np.ndarray:
        """
        Fast pseudo-observation assignment for detector-free matchers.

        Uses a per-image spatial hash grid instead of scanning all existing
        observations. Average-case cost is approximately O(N), assuming points
        are reasonably distributed.
        """
        eps = float(self.pseudo_merge_eps_px)
        grid = self.image_obs_grid[int(image_id)]

        obs_ids = np.empty(points.shape[0], dtype=np.int64)

        for k, xy in enumerate(points):
            obs_ids[k] = self._get_or_create_pair_local_obs_hash(
                image_id=int(image_id),
                xy=xy,
                grid=grid,
                eps=eps,
            )

        return obs_ids

    def _get_or_create_pair_local_obs_hash(
        self,
        image_id: int,
        xy: np.ndarray,
        grid: dict,
        eps: float,
    ) -> int:
        x = float(xy[0])
        y = float(xy[1])

        cell_x = int(np.floor(x / eps))
        cell_y = int(np.floor(y / eps))

        best_obs = None
        best_dist2 = eps * eps

        # Check current cell and neighboring cells.
        # This is enough because any point within eps must lie in one of these.
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cell = (cell_x + dx, cell_y + dy)

                for obs_id in grid.get(cell, []):
                    old_xy = self.obs_xy[obs_id]

                    diff_x = x - float(old_xy[0])
                    diff_y = y - float(old_xy[1])
                    dist2 = diff_x * diff_x + diff_y * diff_y

                    if dist2 <= best_dist2:
                        best_dist2 = dist2
                        best_obs = obs_id

        if best_obs is not None:
            return best_obs

        obs_id = self._new_obs_id(image_id=image_id, xy=np.array([x, y], dtype=np.float32))
        grid[(cell_x, cell_y)].append(obs_id)

        return obs_id

    def _new_obs_id(self, image_id: int, xy: np.ndarray) -> int:
        obs_id = int(self.next_obs_id)
        self.next_obs_id += 1

        self.obs_image[obs_id] = int(image_id)
        self.obs_xy[obs_id] = np.asarray(xy, dtype=np.float32)

        return obs_id

    def access_point3D(self, track_id: int) -> np.ndarray:
        indices = np.where(self.data_matrix[:, 0] == track_id)[0]
        return self.data_matrix[indices, 1:]

    def access_matching_pair(self, pair_index: int) -> tuple[np.ndarray, np.ndarray]:
        data = self.pairwise_matches[pair_index]
        return data[:, :2], data[:, 2:]

    def access_matching_pair_with_indices(
        self,
        pair_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pts = self.pairwise_matches[pair_index]
        idx = self.pairwise_indices[pair_index]
        return pts[:, :2], pts[:, 2:], idx[:, 0], idx[:, 1]

    def access_matching_indices(
        self,
        pair_index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        idx = self.pairwise_indices[pair_index]
        return idx[:, 0], idx[:, 1]

    def access_matching_obs_ids(
        self,
        pair_index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        obs = self.pairwise_obs_ids[pair_index]
        return obs[:, 0], obs[:, 1]

    def access_pair_image_ids(self, pair_index: int) -> tuple[int, int]:
        return self.pairwise_image_ids[pair_index]

    def get_obs_xy(self, obs_id: int) -> np.ndarray:
        return self.obs_xy[int(obs_id)]

    def get_obs_image(self, obs_id: int) -> int:
        return self.obs_image[int(obs_id)]