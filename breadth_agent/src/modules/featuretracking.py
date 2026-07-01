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