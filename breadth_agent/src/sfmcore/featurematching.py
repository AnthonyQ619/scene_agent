import sys
# import
# sys.path.append("C:\\Users\\Anthony\\Documents\\Projects\\Matchers\\RoMa\\romatch")
############ TEMP SOLUTION FOR NOW #################
import json
import cv2
import copy
# from cv2.xfeatures2d import matchGMS
import numpy as np
import glob
from tqdm import tqdm

from sfmcore.DataTypes.pointDT import Points2D
from sfmcore.DataTypes.featmatchDT import PointsMatched
from sfmcore.DataTypes.cameraDT import CameraData

from sfmcore.models.matchers import LightGlue, SuperGlue
from romatch import roma_outdoor, roma_indoor

from sfmcore.baseclass import FeatureMatching, FeatureTracking, FeatureTrackingBase, module_metric
from collections.abc import Callable
from torchvision import transforms as TF
import kornia as K
import kornia.feature as KF
from kornia.feature.loftr.loftr import default_cfg
import torch
from PIL import Image, ImageOps
import piexif


##########################################################################################################
############################################# DETECTOR--FREE #############################################

class FeatureMatchRoMAPair(FeatureMatching):
    use_base_metrics = False
    
    def __init__(self, 
                 cam_data: CameraData, 
                 setting: str = "indoor", 
                 pseudo_merge_eps_px: float = 1.5,
                 RANSAC_homography: bool = False,
                 RANSAC_threshold: float = 3.0,
                 RANSAC_conf: float = 0.99,
                 max_keypoints: int = 10000):

        module_name = "FeatureMatchRoMAPair"

        description = f"""
Detects dense point correspondences directly between sequential image pairs using the 
detector-free RoMa model. RoMa jointly detects and matches points without requiring features 
from a separate FeatureDetection module. It is designed to remain robust under large changes 
in viewpoint, scale, illumination, and texture, making it useful for wide-baseline pairs, 
low-texture regions, and image pairs where sparse detectors produce too few reliable correspondences.

USE THIS MODULE for challenging two-view matching, camera-pose estimation, or dense correspondence 
generation. Because RoMa processes images pairwise, additional track merging is required to form 
consistent multi-view feature tracks across successive image pairs. Therefore,
utilize this module in cases where detector free model is needed:
- in cases where the scene or image data has extreme view changes, 
- or when the scene has many textureless regions.

Initialization Parameters:

- setting: Model configuration for either "indoor" or "outdoor" scenes.
    - Default (str): "indoor"
- pseudo_merge_eps_px: Maximum pixel distance used to merge pair-local observations that likely 
  represent the same feature in a shared frame.
    - Default (float): 1.5
- RANSAC_homography: Whether to reject matches using homography-based RANSAC. Enable for approximately 
  planar scenes or camera rotation; avoid for general scenes with significant depth variation.
    - Default (bool): False
- RANSAC_threshold: Maximum reprojection error, in pixels, for a match to be considered a RANSAC inlier.
    - Default (float): 3.0
- RANSAC_conf: Confidence used when estimating the RANSAC geometric model.
    - Default (float): 0.99
- max_keypoints: Maximum number of RoMa correspondences retained for each image pair. Lower values reduce 
  memory and computation, while higher values provide denser scene coverage.
    - Default (int): 10000
""" 
        
        example = f"""
Initialization: 
# Determine the detector that was used previously and initialize module with said detector

# Feature Matcher Module initialized with default parameters
feature_matcher = FeatureMatchRoMAPair(image_path=image_path) # Initialized image_path with destination to image

# Feature Matcher Module initialized with outdoor parameter
feature_matcher = FeatureMatchRoMAPair(image_path=image_path, setting="outdoor") 

# Feature Matcher Module initialized with outdoor parameter and no image reshaping
feature_matcher = FeatureMatchRoMAPair(image_path=image_path, setting="outdoor", img_reshape=False) 

Example Usage in Script:  
tracked_features = feature_matcher() # Features are not needed as this matcher detects features when matching
"""

        SETTINGS = {"indoor": "indoor",
                    "outdoor": "outdoor",
                    "inside": "indoor",
                    "outside": "outdoor"}

        if setting.lower() not in SETTINGS:
            message = 'Error: setting is not supported. Use one of ' + str(SETTINGS) + ' instead to use this Detector-Free Matcher.'
            raise Exception(message)

        super().__init__(cam_data=cam_data,
                         module_name=module_name,
                         description=description,
                         example=example,
                         RANSAC_conf=RANSAC_conf,
                         RANSAC_homography=RANSAC_homography,
                         RANSAC_threshold=RANSAC_threshold)
        
        self.setting = SETTINGS[setting.lower()]

        self.device = torch.device(f"cuda:{self.cam_data.gpu_num}" if torch.cuda.is_available() else 'cpu')
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')

        if self.setting == "outdoor":
            self.roma_model = roma_outdoor(device=self.device)
        else:
            self.roma_model = roma_indoor(device=self.device)

        self.detector_free = True
        self.pseudo_merge_eps_px = pseudo_merge_eps_px
        self.max_keypoints = max_keypoints
    
    def find_correspondences(self, features: list[Points2D] | None) -> PointsMatched:

        matched_points = PointsMatched(pairwise_matches=[], 
                                       pairwise_indices=[],
                                       image_size=np.array(self.cam_data.image_shape_new),
                                       image_scale=np.array(self.cam_data.image_scale),
                                       img_features=[],
                                       pseudo_merge_eps_px=self.pseudo_merge_eps_px )

        W_1, H_1 = self.cam_data.image_shape_new
        for scene in tqdm(range(0, len(self.image_list) - 1)): 
            img1 = self.image_list[scene]
            img2 = self.image_list[scene + 1]

            # Match
            warp, certainty = self.roma_model.match(img1, img2, device=self.device)
            # Sample matches for estimation
            matches, certainty = self.roma_model.sample(warp, certainty, num=self.max_keypoints)
            kpts1, kpts2 = self.roma_model.to_pixel_coordinates(matches, H_1, W_1, H_1, W_1)

            # Bring to CPU and convert to Numpy vectors / Build Index List
            kpts1, kpts2 = kpts1.cpu().numpy(), kpts2.cpu().numpy()
            pts1 = Points2D(points2D = kpts1,
                            descriptors = None,
                            scores = certainty,
                            image_size = img1.size,
                            reshape_scale = self.cam_data.image_scale)
            pts2 = Points2D(points2D = kpts2,
                            descriptors = None,
                            scores = certainty,
                            image_size = img1.size,
                            reshape_scale = self.cam_data.image_scale)

            N = kpts1.shape[0]
            idx1, idx2 = np.arange(N, dtype=np.int64), np.arange(N, dtype=np.int64)
            
            inlier_pts1, inlier_pts2, idx1_inliers, idx2_inliers, M = self.outlier_reject(pts1, pts2, idx1, idx2, scene)
            
            feat_pair = np.hstack((inlier_pts1.points2D, inlier_pts2.points2D))
            idx_pair = np.hstack((np.vstack(idx1_inliers), np.vstack(idx2_inliers)))

            matched_points.set_matching_pair(
                data=feat_pair,
                idx_data=idx_pair,
                image_pair=(scene, scene + 1),
                index_type="pair_local",
                matcher_name="roma",
            )

        return matched_points

    @module_metric
    def calculate_detector_free_metrics(self, matching_points: PointsMatched):
        raw_counts = []
        inlier_ratios = []
        median_sampson_errors = []
        spatial_coverages = []

        gric_score_F, gric_score_H = self.evaluate_models(matching_points)

        for pair_idx, pair_matches in enumerate(matching_points.pairwise_matches):
            pts1 = pair_matches[:, :2]
            pts2 = pair_matches[:, 2:]

            raw_counts.append(len(pair_matches))

            F, mask = cv2.findFundamentalMat(
                pts1,
                pts2,
                cv2.USAC_MAGSAC,
                ransacReprojThreshold=self.ransac_threshold,
                maxIters=10000,
                confidence=self.ransac_conf,
            )

            if mask is None or F is None:
                inlier_ratios.append(0.0)
                median_sampson_errors.append(float("inf"))
            else:
                mask = mask.ravel().astype(bool)
                inlier_ratios.append(float(mask.mean()))

                if mask.sum() >= 8:
                    errors = self.fundamental_error(pts1[mask], pts2[mask], F)
                    median_sampson_errors.append(float(np.median(errors)))
                else:
                    median_sampson_errors.append(float("inf"))

            spatial_coverages.append(
                self._calculate_pair_spatial_coverage(
                    pts1=pts1,
                    pts2=pts2,
                    image_size=matching_points.image_size,
                )
            )

        return {
            "Matcher Type": "detector_free_pair_local",
            "Average Corresponding Features": float(np.mean(raw_counts)),
            "Median Corresponding Features": float(np.median(raw_counts)),
            "Average Geometric Inlier Ratio": float(np.mean(inlier_ratios)),
            "Median Geometric Inlier Ratio": float(np.median(inlier_ratios)),
            "Median Sampson Error": float(np.median(median_sampson_errors)),
            "Average Spatial Coverage": float(np.mean(spatial_coverages)),
            "Gric Score - Fundamental": gric_score_F,
            "Gric Score - Homography": gric_score_H,
        }

        # Helper for Metric Function
    def _calculate_pair_spatial_coverage(
            self,
            pts1: np.ndarray,
            pts2: np.ndarray,
            image_size: np.ndarray,
            grid_size: int = 8,
        ) -> float:
        W, H = image_size[:]

        def coverage_for_points(pts):
            if len(pts) == 0:
                return 0.0

            x = np.clip((pts[:, 0] / W * grid_size).astype(int), 0, grid_size - 1)
            y = np.clip((pts[:, 1] / H * grid_size).astype(int), 0, grid_size - 1)

            occupied = set(zip(x, y))
            return len(occupied) / float(grid_size * grid_size)

        cov1 = coverage_for_points(pts1)
        cov2 = coverage_for_points(pts2)

        return min(cov1, cov2)
    
class FeatureMatchLoftrPair(FeatureMatching):
    use_base_metrics = False
    
    def __init__(self, 
                 cam_data: CameraData, 
                 setting: str = "indoor", 
                 pseudo_merge_eps_px: float = 1.5,
                 coarse_thr: float = 0.2,
                 border_rm: int = 2,
                 min_confidence: float | None = None,
                 max_matches: int | None = None,
                 RANSAC_homography: bool = False,
                 RANSAC_threshold: float = 3.0,
                 RANSAC_conf: float = 0.99):

        module_name = "FeatureMatchLoftrPair"

        description = f"""
Detects point correspondences directly between image pairs using the detector-free
LoFTR deep learning matcher. Unlike traditional feature matching methods, LoFTR
does not require a separate feature detector or descriptor and instead estimates
matches from learned coarse-to-fine image features.

Use this module when:
- images contain low-texture or weakly textured regions where traditional
- keypoint detectors may produce too few repeatable features, matching is required across 
  moderate viewpoint, scale, or illumination changes, dense and spatially distributed correspondences 
  are preferred over a smaller set of sparse keypoints, or traditional detector-and-descriptor combinations 
  such as SIFT, SuperPoint, or ORB do not provide enough reliable matches.

LoFTR is particularly useful for indoor scenes, architectural environments, and
other scenes containing large smooth surfaces or repeated structures. 

Model is trained both for indoor and outdoor setting. When not specified, assume indoor
setting to properly initialize the model.

Initialization Parameters:
- setting: Selects LoFTR weights trained for "indoor" or "outdoor" imagery.
    - Default (str): "indoor"
- pseudo_merge_eps_px: Maximum pixel distance for merging pair-local correspondences that 
  represent the same observation in a shared frame.
    - Default (float): 1.5
- coarse_thr: Minimum confidence required for a candidate match during LoFTR's coarse matching 
  stage. Increase to retain fewer, more reliable matches; decrease to improve match coverage.
    - Default (float): 0.2
- border_rm: Number of coarse feature-map cells excluded along each image border to remove unreliable 
  edge matches.
    - Default (int): 2
- min_confidence: Optional post-processing confidence threshold applied to LoFTR's final matches. When 
  None, no additional confidence filtering is performed.
    - Default: None
- max_matches: Optional maximum number of matches retained per image pair. When specified, the 
  highest-confidence matches are kept.
    - Default: None
- RANSAC_homography: Whether to reject matches using homography-based RANSAC. Enable for approximately 
  planar scenes or rotation-dominant image pairs; avoid for general 3D scenes with significant depth variation.
    - Default (bool): False
- RANSAC_threshold: Maximum reprojection error, in pixels, for a match to be accepted as a RANSAC inlier.
    - Default (float): 3.0
- RANSAC_conf: Confidence used when estimating the RANSAC geometric model.
    - Default (float): 0.99

Higher coarse_thr and min_confidence values improve precision but may leave too few correspondences for pose 
estimation. LoFTR can provide strong coverage in texture-poor regions, but pairwise matches must still be merged 
to form consistent multi-view tracks.
""" 
        
        example = f"""
Initialization: 
# Determine the detector that was used previously and initialize module with said detector

# Feature Matcher Module initialized with default parameters
feature_matcher = FeatureMatchLoftrPair(image_path=image_path) # Initialized image_path with destination to image

# Feature Matcher Module initialized with outdoor parameter
feature_matcher = FeatureMatchLoftrPair(image_path=image_path, setting="outdoor") 

# Feature Matcher Module initialized with outdoor parameter and no image reshaping
feature_matcher = FeatureMatchLoftrPair(image_path=image_path, setting="outdoor", img_reshape=False) 

Example Usage in Script:  
tracked_features = feature_matcher() # Features are not needed as this matcher detects features when matching
"""

        SETTINGS = {"indoor": "indoor_new",
                    "outdoor": "outdoor",
                    "inside": "indoor_new",
                    "outside": "outdoor"}

        if setting not in SETTINGS:
            message = 'Error: setting is not supported. Use one of ' + str(SETTINGS) + ' instead to use this Detector-Free Matcher.'
            raise Exception(message)
        
        super().__init__(cam_data=cam_data,
                         module_name=module_name,
                         description=description,
                         example=example,
                         RANSAC_conf=RANSAC_conf,
                         RANSAC_homography=RANSAC_homography,
                         RANSAC_threshold=RANSAC_threshold)
        
        weight = SETTINGS[setting.lower()]
        self.device = torch.device(f"cuda:{self.cam_data.gpu_num}" if torch.cuda.is_available() else 'cpu')
        cfg = copy.deepcopy(default_cfg)
        cfg["match_coarse"]["thr"] = coarse_thr
        cfg["match_coarse"]["border_rm"] = border_rm

        self.matcher = KF.LoFTR(
            pretrained=weight,
            config=cfg,
        ).to(self.device).eval()

        self.min_confidence = min_confidence
        self.max_matches = max_matches

        self.to_tensor = TF.ToTensor()
        self.detector_free = True
        

    def find_correspondences(self, features: list[Points2D] | None) -> PointsMatched:
        matched_points = PointsMatched(pairwise_matches=[], 
                                       pairwise_indices=[],
                                       image_size=np.array(self.cam_data.image_shape_new),
                                       image_scale=np.array(self.cam_data.image_scale),
                                       img_features=[])

        W_1, H_1 = self.cam_data.image_shape_new

        for scene in tqdm(range(0, len(self.image_list)-1)): 
            img1 = self.image_list[scene]
            img2 = self.image_list[scene + 1]

            image1 = self.pil_to_kornia_gray(img1, self.device)
            image2 = self.pil_to_kornia_gray(img2, self.device)
            print(image1.shape, image1.dtype, image1.min().item(), image1.max().item())

            input_dict = {
                "image0": image1,  # LofTR works on grayscale images only
                "image1": image2,
            }

            with torch.inference_mode():
                correspondences = self.matcher(input_dict)

            mkpts0 = correspondences["keypoints0"].cpu().numpy()
            mkpts1 = correspondences["keypoints1"].cpu().numpy()
            confidence = correspondences["confidence"].cpu().numpy()
            pts1 = Points2D(points2D = mkpts0,
                            descriptors = None,
                            scores = confidence,
                            image_size = img1.size,
                            reshape_scale = self.cam_data.image_scale)
            pts2 = Points2D(points2D = mkpts1,
                            descriptors = None,
                            scores = confidence,
                            image_size = img2.size,
                            reshape_scale = self.cam_data.image_scale)

            N = mkpts0.shape[0]
            idx1, idx2 = np.arange(N, dtype=np.int64), np.arange(N, dtype=np.int64)
            
            inlier_pts1, inlier_pts2, idx1_inliers, idx2_inliers, M = self.outlier_reject(pts1, pts2, idx1, idx2, scene)
            
            feat_pair = np.hstack((inlier_pts1.points2D, inlier_pts2.points2D))
            idx_pair = np.hstack((np.vstack(idx1_inliers), np.vstack(idx2_inliers)))

            matched_points.set_matching_pair(
                data=feat_pair,
                idx_data=idx_pair,
                image_pair=(scene, scene + 1),
                index_type="pair_local",
                matcher_name="loftr",
            )

        return matched_points

    def pil_to_kornia_gray(
        self,
        img: Image.Image,
        device: torch.device | str = "cuda",
    ) -> torch.Tensor:
        """
        Convert PIL image to Kornia/LoFTR-compatible grayscale tensor.

        Returns:
            Tensor of shape [1, 1, H, W], dtype float32, range [0, 1]
        """
        # Ensure RGB first, even if PIL image is L, RGBA, etc.
        img = img.convert("RGB")

        # PIL -> torch tensor [3, H, W], float32, range [0, 1]
        tensor = self.to_tensor(img)

        # Add batch dimension: [1, 3, H, W]
        tensor = tensor.unsqueeze(0).to(device)

        # RGB -> grayscale: [1, 1, H, W]
        gray = K.color.rgb_to_grayscale(tensor)

        return gray

    @module_metric
    def calculate_detector_free_metrics(self, matching_points: PointsMatched):
        raw_counts = []
        inlier_ratios = []
        median_sampson_errors = []
        spatial_coverages = []

        gric_score_F, gric_score_H = self.evaluate_models(matching_points)

        for pair_idx, pair_matches in enumerate(matching_points.pairwise_matches):
            pts1 = pair_matches[:, :2]
            pts2 = pair_matches[:, 2:]

            raw_counts.append(len(pair_matches))

            F, mask = cv2.findFundamentalMat(
                pts1,
                pts2,
                cv2.USAC_MAGSAC,
                ransacReprojThreshold=self.ransac_threshold,
                maxIters=10000,
                confidence=self.ransac_conf,
            )

            if mask is None or F is None:
                inlier_ratios.append(0.0)
                median_sampson_errors.append(float("inf"))
            else:
                mask = mask.ravel().astype(bool)
                inlier_ratios.append(float(mask.mean()))

                if mask.sum() >= 8:
                    errors = self.fundamental_error(pts1[mask], pts2[mask], F)
                    median_sampson_errors.append(float(np.median(errors)))
                else:
                    median_sampson_errors.append(float("inf"))

            spatial_coverages.append(
                self._calculate_pair_spatial_coverage(
                    pts1=pts1,
                    pts2=pts2,
                    image_size=matching_points.image_size,
                )
            )

        return {
            "Matcher Type": "detector_free_pair_local",
            "Average Corresponding Features": float(np.mean(raw_counts)),
            "Median Corresponding Features": float(np.median(raw_counts)),
            "Average Geometric Inlier Ratio": float(np.mean(inlier_ratios)),
            "Median Geometric Inlier Ratio": float(np.median(inlier_ratios)),
            "Median Sampson Error": float(np.median(median_sampson_errors)),
            "Average Spatial Coverage": float(np.mean(spatial_coverages)),
            "Gric Score - Fundamental": gric_score_F,
            "Gric Score - Homography": gric_score_H,
        }

        # Helper for Metric Function
    def _calculate_pair_spatial_coverage(
            self,
            pts1: np.ndarray,
            pts2: np.ndarray,
            image_size: np.ndarray,
            grid_size: int = 8,
        ) -> float:
        W, H = image_size[:]

        def coverage_for_points(pts):
            if len(pts) == 0:
                return 0.0

            x = np.clip((pts[:, 0] / W * grid_size).astype(int), 0, grid_size - 1)
            y = np.clip((pts[:, 1] / H * grid_size).astype(int), 0, grid_size - 1)

            occupied = set(zip(x, y))
            return len(occupied) / float(grid_size * grid_size)

        cov1 = coverage_for_points(pts1)
        cov2 = coverage_for_points(pts2)

        return min(cov1, cov2)
    
##########################################################################################################
############################################# DETECTOR-BASED #############################################
class FeatureMatchSuperGluePair(FeatureMatching):
    def __init__(self, 
                 cam_data: CameraData,
                 detector:str = 'superpoint', 
                 sinkhorn_iterations: int = 20, 
                 match_threshold: float = 0.2, 
                 descriptor_dim: int = 256,
                 setting: str = 'indoor',
                 RANSAC_homography: bool = False,
                 RANSAC_threshold: float = 3.0,
                 RANSAC_conf: float = 0.99):
        
        SUPPORTED_FEATURES = ["superpoint", "sp", "sift"]
        SUPPORTED_SETTINGS = ["indoor", "outdoor"]

        if setting not in SUPPORTED_SETTINGS:
            message = 'Error: setting is not supported. Use one of ' + str(SUPPORTED_SETTINGS) + ' instead to use this Feature Matcher.'
            raise Exception(message)

        if detector.lower() not in SUPPORTED_FEATURES:
            message = 'Error: detector is not supported. Use one of ' + str(SUPPORTED_FEATURES) + ' instead to use this Feature Matcher.'
            raise Exception(message)

        module_name = "FeatureMatchSuperGluePair"

        description = f"""
Detects point correspondance between two sequential frames at once to detect matching 
features across a set of images. The feature matching algorithm used is the SuperGlue deep
learning model trained as a feature matcher. Unless specified directly, assume the features 
are detected using the SuperPoint deep learning feature detector algorithm and initialize 
through the detector parameter. Matches sparse local features between sequential image pairs 
using a graph neural network and optimal-transport assignment to jointly identify correspondences 
and reject unmatched features.

USE THIS MODULE when matching accuracy is more important than runtime, or when reproducing 
pipelines and benchmarks built specifically around SuperGlue. It is suitable for challenging 
indoor or outdoor pairs with viewpoint and appearance changes, but is generally slower and 
more computationally expensive than LightGlue.

Model is trained both for indoor and outdoor setting. When not specified, assume indoor
setting to properly initialize the model.

At this time, this deep learning matcher is only usable with SuperPoint.

Initialization/Function Parameters:
- detector (str): Name of Feature Detector that was used to estimate the features provided.
    - default (str): superpoint
- sinkhorn_iterations: number iterations for running the Sinkhorn Algorithm in the model for optimal
  partial assignment of detected feature matches
    - default (int): 20
- match_threshold: confidence threshold (we choose 0.2) to retain some matches from the soft assignment stage
    - default (float): 0.2
- descriptor_dim: the dimensions for the estimated desciptor generated from the detector used
    - default (int): 256
- setting: the string to determine if the images are "indoor" or "outdoor"
    - default (str): indoor
- RANSAC_homography: Determines whether to use the Homography or Fundamental model for outlier rejection in 
  matching point correspondences. Homography is fit as a model in scenes where the major focus is of a planar 
  object, whereas fundamental matrix is a better model otherwise (Scenes that lack structure of planar objects).
    - Default (bool): False (True runs Homography model, False uses Fundamental model)
- RANSAC_threshold: Parameter used only for RANSAC. It is the maximum distance from a point to an epipolar line 
  in normalized pixel coordinates, beyond which the point is considered an outlier and is not used for computing 
  the final fundamental/homography matrix.
    - Default (float): 3.0
- RANSAC_conf: Parameter used for the RANSAC and LMedS methods only. It specifies a desirable level of confidence 
  (probability) that the estimated matrix is correct.
    - Default (float): 0.99

Function Call Parameters - HANDLED INTERNALLY, DO NOT USE IF SFMCORE IN USE:
- features (list[Points2D]): list of features detected per scene estimated from the feature detection module
""" # TODO: Fill in details for the matcher. Be precise as we want the agent to know when exactly to use this
        
        example = f"""
Initialization modules
from sfmcore.baseclass import SfMScene
from sfmcore.features import FeatureDetectionSP
from sfmcore.featurematching import {module_name}

# Start SfM Pipeline 
# Step 1: Read in Calibration/Image Data
reconstructed_scene = SfMScene(image_path = image_path, 
                            calibration_path = calibration_path)

# Step 2 (Detect Features) must be completed prior!
# Step 3: 
reconstructed_scene.{module_name}(
    detector="superpoint",
    setting="indoor",
    RANSAC_homography=False,
    RANSAC_threshold=1.0,
    RANSAC_conf=0.999
    )
"""

        super().__init__(cam_data=cam_data,
                         module_name=module_name,
                         description=description,
                         example=example,
                         RANSAC_conf=RANSAC_conf,
                         RANSAC_homography=RANSAC_homography,
                         RANSAC_threshold=RANSAC_threshold)
        
        self.detector = detector

        self.device = torch.device(f"cuda:{self.cam_data.gpu_num}" if torch.cuda.is_available() else "cpu") 

        config_settings = {
            'weights': setting,
            'sinkhorn_iterations': sinkhorn_iterations,
            'match_threshold': match_threshold,
            'descriptor_dim': descriptor_dim,
        }

        self.matcher = SuperGlue(config=config_settings).eval().to(self.device)
    
    def find_correspondences(self, features: list[Points2D]) -> PointsMatched:
        torch.set_grad_enabled(False)

        img_size = features[0].image_size
        img_scale = features[0].reshape_scale
        matched_points = PointsMatched(pairwise_matches=[], 
                                       pairwise_indices=[],
                                       image_size=img_size,
                                       image_scale=img_scale,
                                       img_features=[])

        for scene in tqdm(range(0, len(features) - 1), desc="Detecting Correspondences"):
            pt1 = features[scene]
            pt2 = features[scene + 1]

            feats0 = {"keypoints0": torch.from_numpy(pt1.points2D).unsqueeze(0).to(self.device),
                      "descriptors0": torch.from_numpy(pt1.descriptors).unsqueeze(0).to(self.device).permute(0, 2, 1),
                      "scores0": torch.from_numpy(pt1.scores.T).to(self.device),
                      "image_size0": torch.from_numpy(pt1.image_size).unsqueeze(0).to(self.device)}
            
            feats1 = {"keypoints1": torch.from_numpy(pt2.points2D).unsqueeze(0).to(self.device),
                      "descriptors1": torch.from_numpy(pt2.descriptors).unsqueeze(0).to(self.device).permute(0, 2, 1),
                      "scores1": torch.from_numpy(pt2.scores.T).to(self.device),
                      "image_size1": torch.from_numpy(pt2.image_size).unsqueeze(0).to(self.device)}

            idx1, idx2 = self.matcher_parser({**feats0, **feats1})
            inlier_idx1 = np.where(idx1)[0].tolist()
     
            new_pt1 = Points2D(**pt1.splice_2D_points(inlier_idx1)) # Previously just idx1 as input
            new_pt2 = Points2D(**pt2.splice_2D_points(idx2))
 
            inlier_pts1, inlier_pts2, idx1_inliers, idx2_inliers, F = self.outlier_reject(new_pt1, new_pt2, inlier_idx1, idx2, scene)
            
            feat_pair = np.hstack((inlier_pts1.points2D, inlier_pts2.points2D))
            idx_pair = np.hstack((np.vstack(idx1_inliers), np.vstack(idx2_inliers)))

            matched_points.set_matching_pair(
                data=feat_pair,
                idx_data=idx_pair,
                image_pair=(scene, scene + 1),
                index_type="global",
                matcher_name="superglue",
            )
            matched_points.img_features.append(pt1.points2D)

        # Get the last image feature set
        matched_points.img_features.append(features[-1].points2D)

        return matched_points

    def matcher_parser(self, feature_pair: dict = {}) -> tuple[list, list]:        
        pred = self.matcher(feature_pair)
        matches = pred["matches0"].detach().cpu().numpy()

        valid_idx1 = matches > -1
        valid_idx2 = copy.copy(matches[valid_idx1])

        return valid_idx1[0].tolist(), valid_idx2.tolist()
    
class FeatureMatchLightGluePair(FeatureMatching):
    def __init__(self,  
                 cam_data = CameraData,
                 detector:str = 'superpoint', 
                 n_layers: int = 9, 
                 flash: bool = True, 
                 mp:bool = False, 
                 depth_confidence: float = 0.95,
                 width_confidence: float = 0.99, 
                 filter_threshold: float = 0.1,
                 RANSAC_homography: bool = False,
                 RANSAC_threshold: float = 3.0,
                 RANSAC_conf: float = 0.99):
        
        SUPPORTED_FEATURES = ["superpoint", "sp", "sift", "aliked"]

        if detector.lower() not in SUPPORTED_FEATURES:
            message = 'Error: detector is not supported. Use one of ' + str(self.FORMATS) + ' instead to use this Feature Matcher.'
            raise Exception(message)

        module_name = "FeatureMatchLightGluePair"
        description = f"""
Detects point correspondance between two sequential frames at once to detect matching 
features across a set of images. The feature matching algorithm used is the LightGlue deep
learning model trained as a feature matcher. Unless specified directly, assume the features 
are detected using the SuperPoint deep learning feature detector algorithm and initialize 
through the detector parameter. 

Overall: Matches sparse local features between sequential image pairs using an adaptive 
deep feature matcher. LightGlue adjusts its computation according to image-pair difficulty, 
allowing easier pairs to be processed with fewer network layers and candidate features.

USE THIS MODULE as the preferred general-purpose learned matcher for SfM, visual localization, 
or SLAM when fast runtime, lower memory usage, and strong matching accuracy are required. 
Choose it over SuperGlue for large image sets, repeated sequential matching, or latency-sensitive pipelines.

Other supported detectors are: SIFT and SuperPoint

Initialization/Function Parameters:
- detector (str): Name of Feature Detector that was used to estimate the features provided.
    - Default (str): SIFT
- n_layers: Number of stacked self+cross attention layers. Reduce this value for faster inference 
  at the cost of accuracy (continuous red line in the plot above). 
    - Default (int): 9 (all layers).
- flash: Enable FlashAttention. Significantly increases the speed and reduces the memory consumption 
  without any impact on accuracy. 
    - Default (bool): True (LightGlue automatically detects if FlashAttention is available).
- mp: Enable mixed precision inference. 
    - Default (bool): False (off)
- depth_confidence: Controls the early stopping. A lower values stops more often at earlier layers. 
    - Default (float): 0.95, disable with -1.
- width_confidence: Controls the iterative point pruning. A lower value prunes more points earlier. 
    - Default (float): 0.99, disable with -1.
- filter_threshold: Match confidence. Increase this value to obtain less, but stronger matches. 
    - Default (float): 0.1
- RANSAC_homography: Determines whether to use the Homography or Fundamental model for outlier rejection in 
  matching point correspondences. Homography is fit as a model in scenes where the major focus is of a planar 
  object, whereas fundamental matrix is a better model otherwise (Scenes that lack structure of planar objects).
    - Default (bool): False (True runs Homography model, False uses Fundamental model)
- RANSAC_threshold: Parameter used only for RANSAC. It is the maximum distance from a point to an epipolar line 
  in normalized pixel coordinates, beyond which the point 
     is considered an outlier and is not used for computing the final fundamental/homography matrix.
    - Default (float): 3.0
- RANSAC_conf: Parameter used for the RANSAC and LMedS methods only. It specifies a desirable level of confidence 
  (probability) that the estimated matrix is correct.
    - Default (float): 0.99

Function Call Parameters - HANDLED INTERNALLY, DO NOT USE IF SFMCORE IN USE:
- features list[Points2D]: list of features detected per scene estimated from the feature detection module
""" 
        
        example = f"""
Initialization modules
from sfmcore.baseclass import SfMScene
from sfmcore.features import FeatureDetectionSP
from sfmcore.featurematching import {module_name}

# Start SfM Pipeline 
# Step 1: Read in Calibration/Image Data
reconstructed_scene = SfMScene(image_path = image_path, 
                            calibration_path = calibration_path)

# Step 2 (Detect Features) must be completed prior!
# Step 3: 
reconstructed_scene.{module_name}(
    detector="superpoint",
    RANSAC_homography=False,
    RANSAC_threshold=2.0,
    RANSAC_conf=0.999
    )
"""
        
        super().__init__(cam_data=cam_data,
                         module_name=module_name,
                         description=description,
                         example=example,
                         RANSAC_conf=RANSAC_conf,
                         RANSAC_homography=RANSAC_homography,
                         RANSAC_threshold=RANSAC_threshold)
        self.device = torch.device(f"cuda:{self.cam_data.gpu_num}" if torch.cuda.is_available() else "cpu")
        
        self.detector = detector
        
        self.matcher = LightGlue(features=self.detector, 
                                 n_layers = n_layers,
                                 flash = flash, 
                                 mp = mp, 
                                 depth_confidence = depth_confidence,
                                 width_confidence = width_confidence, 
                                 filter_threshold = filter_threshold).eval().to(self.device)

    
    def find_correspondences(self, features: list[Points2D]) -> PointsMatched:
        torch.set_grad_enabled(False)

        img_size = features[0].image_size
        img_scale = features[0].reshape_scale
        matched_points = PointsMatched(pairwise_matches=[], 
                                       pairwise_indices=[],
                                       image_size=img_size,
                                       image_scale=img_scale,
                                       img_features=[])

        for scene in tqdm(range(0, len(features) - 1), desc="Detecting Correspondences"):
            pt1 = features[scene]
            pt2 = features[scene + 1]

            if self.detector == 'sift':
                feats0 = {"keypoints": torch.from_numpy(pt1.points2D).unsqueeze(0).to(self.device),
                        "descriptors": torch.from_numpy(pt1.descriptors).unsqueeze(0).to(self.device),
                        "scales": torch.from_numpy(pt1.scale).to(self.device),
                        "oris": torch.from_numpy(pt1.orientation).to(self.device),
                        "image_size": torch.from_numpy(pt1.image_size).unsqueeze(0).to(self.device)}
                
                feats1 = {"keypoints": torch.from_numpy(pt2.points2D).unsqueeze(0).to(self.device),
                        "descriptors": torch.from_numpy(pt2.descriptors).unsqueeze(0).to(self.device),
                        "scales": torch.from_numpy(pt2.scale).to(self.device),
                        "oris": torch.from_numpy(pt2.orientation).to(self.device),
                        "image_size": torch.from_numpy(pt2.image_size).unsqueeze(0).to(self.device)}
            else:
                feats0 = {"keypoints": torch.from_numpy(pt1.points2D).unsqueeze(0).to(self.device),
                        "descriptors": torch.from_numpy(pt1.descriptors).unsqueeze(0).to(self.device),
                        "image_size": torch.from_numpy(pt1.image_size).unsqueeze(0).to(self.device)}
                
                feats1 = {"keypoints": torch.from_numpy(pt2.points2D).unsqueeze(0).to(self.device),
                        "descriptors": torch.from_numpy(pt2.descriptors).unsqueeze(0).to(self.device),
                        "image_size": torch.from_numpy(pt2.image_size).unsqueeze(0).to(self.device)}

            idx1, idx2 = self.matcher_parser({"image0": feats0, "image1": feats1})

            new_pt1 = Points2D(**pt1.splice_2D_points(idx1))
            new_pt2 = Points2D(**pt2.splice_2D_points(idx2))

            inlier_pts1, inlier_pts2, idx1_inliers, idx2_inliers, F = self.outlier_reject(new_pt1, new_pt2, idx1, idx2, scene)
            
            feat_pair = np.hstack((inlier_pts1.points2D, inlier_pts2.points2D))
            idx_pair = np.hstack((np.vstack(idx1_inliers), np.vstack(idx2_inliers)))

            matched_points.set_matching_pair(
                data=feat_pair,
                idx_data=idx_pair,
                image_pair=(scene, scene + 1),
                index_type="global",
                matcher_name="lightglue",
            )
            matched_points.img_features.append(pt1.points2D)

        # Get the last image feature set
        matched_points.img_features.append(features[-1].points2D)

        return matched_points

    def matcher_parser(self, feature_pair: dict = {}) -> tuple[list, list]:        
        matches = self.matcher(feature_pair)

        def rbd(data: dict) -> dict:
            """Remove batch dimension from elements in data"""
            return {
                k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
                for k, v in data.items()
                }
        
        matches = rbd(matches)
        
        matches_idx = matches['matches']

        return (matches_idx[..., 0].detach().cpu().numpy().tolist(), 
                matches_idx[..., 1].detach().cpu().numpy().tolist())

class FeatureMatchFlannPair(FeatureMatching):
    def __init__(self, 
                 cam_data: CameraData, 
                 detector:str = 'sift',
                 k: int = 2,
                 RANSAC_homography: bool = False,
                 RANSAC_threshold: float = 3.0,
                 RANSAC_conf: float = 0.99,
                 lowes_thresh: float = 0.75):

        module_name = "FeatureMatchFlannPair"
        description = f"""
Uses approximate nearest-neighbor search to efficiently match large collections of local descriptors. USE THIS MODULE 
when images contain many detected features, the scene is sufficiently textured, and faster matching is needed across 
a large image set. It is particularly suitable for SIFT descriptors in offline SfM pipelines with strong image 
overlap and moderate lighting or viewpoint changes.

Choose FLANN over Brute Force or ML matchers when CPU scalability and low memory overhead are priorities, while 
classical descriptors remain reliable. Because its search is approximate, it may return less accurate matches than 
Brute Force and should be combined with ratio filtering and geometric verification. It is less suitable for dark, 
textureless, repetitive, or strongly appearance-changing environments where the underlying descriptors are already 
unreliable. FLANN is specifically designed for fast approximate nearest-neighbor search and can outperform 
exhaustive matching for large descriptor collections.

Other supported detectors are: SIFT, ORB, and SuperPoint.

SuperPoint and Sift share the same parameters, whereas ORB contains different parameters.

Initialization/Function Parameters: 
- detector: String representing the name of the feature detector used for the features provided.
    - Default (str): SIFT
- k: Integer Number for consideration of nearest neighbor count of potential feature matchers before post-processing 
  with lowes threshold.
    - Default (int): 2
- RANSAC_homography: Determines whether to use the Homography or Fundamental model for outlier rejection in matching 
  point correspondences. Homography is fit as a model in scenes where the major focus is of a planar object, whereas 
  fundamental matrix is a better model otherwise (Scenes that lack structure of planar objects).
    - Default (bool): False (True runs Homography model, False uses Fundamental model)
- RANSAC_threshold: Parameter used only for RANSAC. It is the maximum distance from a point to an epipolar line in 
  normalized pixel coordinates, beyond which the point 
     is considered an outlier and is not used for computing the final fundamental/homography matrix.
    - Default (float): 3.0
- RANSAC_conf: Parameter used for the RANSAC and LMedS methods only. It specifies a desirable level of confidence 
  (probability) that the estimated matrix is correct.
    - Default (float): 0.99
- lowes_thresh: Threshold for Lowe's Ratio Test, accepting a match only if the ratio of the distance to the best 
  match to the distance of the second-best match is 
     below a specific threshold
    - Default (float): 0.75

Function Call Parameters - HANDLED INTERNALLY, DO NOT USE IF SFMCORE IN USE:
- features list[Points2D]: list of features detected per scene estimated from the feature detection module
"""
        example = f"""
Initialization modules
from sfmcore.baseclass import SfMScene
from sfmcore.features import FeatureDetectionSIFT
from sfmcore.featurematching import {module_name}

# Start SfM Pipeline 
# Step 1: Read in Calibration/Image Data
reconstructed_scene = SfMScene(image_path = image_path, 
                            calibration_path = calibration_path)

# Step 2 (Detect Features) must be completed prior!
# Step 3: 
reconstructed_scene.{module_name}(
    detector="sift",
    k=2,
    lowes_thresh=0.78,
    RANSAC_homography=False,
    RANSAC_threshold=2.0,
    RANSAC_conf=0.999
    )
"""     
        super().__init__(cam_data=cam_data,
                         module_name=module_name,
                         description=description,
                         example=example,
                         RANSAC_conf=RANSAC_conf,
                         RANSAC_homography=RANSAC_homography,
                         RANSAC_threshold=RANSAC_threshold)
        
        self.detector = detector

        if detector in ["sift", "sp", "superpoint", "aliked"]:
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        else: # Fast and Orb
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm = FLANN_INDEX_LSH,
                                table_number = 6, # 12
                                key_size = 12,     # 20
                                multi_probe_level = 2) #2
        
        search_params = dict(checks=50)   # or pass empty dictionary
        self.lowes_thresh = lowes_thresh
        self.k = k

        self.matcher = cv2.FlannBasedMatcher(index_params,
                                             search_params)

    
    def find_correspondences(self, features: list[Points2D]) -> PointsMatched:
        img_size = features[0].image_size
        img_scale = features[0].reshape_scale
        matched_points = PointsMatched(pairwise_matches=[], 
                                       pairwise_indices=[],
                                       image_size=img_size,
                                       image_scale=img_scale,
                                       img_features=[])

        for scene in tqdm(range(0, len(features) - 1), desc="Detecting Correspondences"):
            pt1 = features[scene]
            pt2 = features[scene + 1]

            idx1, idx2 = self.matcher_parser(pt1.descriptors, pt2.descriptors)

            new_pt1 = Points2D(**pt1.splice_2D_points(idx1))
            new_pt2 = Points2D(**pt2.splice_2D_points(idx2))

            inlier_pts1, inlier_pts2, idx1_inliers, idx2_inliers, F = self.outlier_reject(new_pt1, new_pt2, idx1, idx2, scene)

            feat_pair = np.hstack((inlier_pts1.points2D, inlier_pts2.points2D))
            idx_pair = np.hstack((np.vstack(idx1_inliers), np.vstack(idx2_inliers)))

            matched_points.set_matching_pair(
                data=feat_pair,
                idx_data=idx_pair,
                image_pair=(scene, scene + 1),
                index_type="global",
                matcher_name="flann",
            )
            matched_points.img_features.append(pt1.points2D)

        # Get the last image feature set
        matched_points.img_features.append(features[-1].points2D)

        return matched_points

    def matcher_parser(self, desc1: np.ndarray, desc2: np.ndarray) -> tuple[list, list]:        
        matches = self.matcher.knnMatch(desc1, desc2, k=self.k)
                 
        # Conduct Lowe's Test Here
        good = []
        for m,n in matches:
            if m.distance < self.lowes_thresh*n.distance:
                good.append(m)

        pts1_idx = [good[i].queryIdx for i in range(len(good))]
        pts2_idx = [good[i].trainIdx for i in range(len(good))]

        return pts1_idx, pts2_idx

class FeatureMatchBFPair(FeatureMatching):
    def __init__(self, 
                 cam_data: CameraData,
                 detector:str = 'sift',
                 k: int = 2,
                 cross_check: bool = True,
                 RANSAC_homography: bool = False,
                 RANSAC_threshold: float = 1.0,
                 RANSAC_conf: float = 0.99,
                 GMS: bool = False,
                 lowes_thresh: float = 0.75):

        module_name = "FeatureMatchBFPair"
        description = f"""
Compares every descriptor in one image against all descriptors in the paired image to find the closest matches. 
USE THIS MODULE for small-to-moderate feature sets when matching accuracy (especially with this 1-1 exhaustive 
matching method), deterministic behavior, and simple CPU execution are more important than speed. It works well 
with SIFT in well-lit, textured scenes and with ORB for high-overlap sequential images.

Choose Brute Force over ML matchers when GPU resources are unavailable, the image domain differs greatly from 
learned training data, or a lightweight and explainable classical SfM pipeline is preferred. Avoid it for very 
large feature sets or difficult scenes with severe lighting, viewpoint, or appearance changes, where learned 
matchers may reject ambiguous correspondences more effectively.

Initalization/Function Parameters:
- detector: String representing the name of the feature detector used for the features provided.
    - Default (str): SIFT
- k: Integer Number for consideration of nearest neighbor count of potential feature matchers before post-processing 
  with lowes threshold.
    - Default (int): 2
- cross_check: If it is false, this is will be default BFMatcher behaviour when it finds the k nearest neighbors for 
  each query descriptor. If True then the nearest neighbor method with k=1 will only return pairs (i,j) such that 
  for i-th query descriptor the j-th descriptor in the matcher's collection is the nearest and vice versa, i.e. the 
  BFMatcher will only return consistent pairs. Such technique usually produces best results with minimal number of 
  outliers when there are enough matches. i.e only use when there's are lot of feature points
    - Default (bool): False
- RANSAC_homography: Determines whether to use the Homography or Fundamental model for outlier rejection in matching 
  point correspondences. Homography is fit as a model in scenes where the major focus is of a planar object, 
  whereas fundamental matrix is a better model otherwise (Scenes that lack structure of planar objects).
    - Default (bool): False (True runs Homography model, False uses Fundamental model)
- RANSAC_threshold: Parameter used only for RANSAC. It is the maximum distance from a point to an epipolar line in 
  normalized pixel coordinates, beyond which the point 
  is considered an outlier and is not used for computing the final fundamental/homography matrix.
    - Default (float): 3.0
- RANSAC_conf: Parameter used for the RANSAC and LMedS methods only. It specifies a desirable level of confidence 
  (probability) that the estimated matrix is correct.
    - Default (float): 0.99
- lowes_thresh: Threshold for Lowe's Ratio Test, accepting a match only if the ratio of the distance to the best 
  match to the distance of the second-best match is 
     below a specific threshold
    - Default (float): 0.75

Function Call Parameters - HANDLED INTERNALLY, DO NOT USE IF SFMCORE IN USE:
- features list[Points2D]: list of features detected per scene estimated from the feature detection module
"""
        example = f"""
Initialization modules
from sfmcore.baseclass import SfMScene
from sfmcore.features import FeatureDetectionSIFT
from sfmcore.featurematching import {module_name}

# Start SfM Pipeline 
# Step 1: Read in Calibration/Image Data
reconstructed_scene = SfMScene(image_path = image_path, 
                            calibration_path = calibration_path)

# Step 2 (Detect Features) must be completed prior!
# Step 3: 
reconstructed_scene.{module_name}(
    detector="sift",
    k=2,
    lowes_thresh=0.78,
    RANSAC_homography=False,
    RANSAC_threshold=3.0,
    RANSAC_conf=0.999
    )
"""

        super().__init__(cam_data=cam_data,
                         module_name=module_name,
                         description=description,
                         example=example,
                         RANSAC_conf=RANSAC_conf,
                         RANSAC_homography=RANSAC_homography,
                         RANSAC_threshold=RANSAC_threshold)
        
        self.detector = detector

        if self.detector in ["sift", "sp", "superpoint", "aliked"]:
            norm_type = cv2.NORM_L2
        else:
            norm_type = cv2.NORM_HAMMING

        self.cross_check = cross_check
        self.gms = GMS
        self.k = k
        self.lowes_thresh = lowes_thresh

        self.matcher = cv2.BFMatcher(normType=norm_type, 
                                     crossCheck=self.cross_check)
    
    def find_correspondences(self, features: list[Points2D]) -> PointsMatched:
        img_size = features[0].image_size
        img_scale = features[0].reshape_scale
        matched_points = PointsMatched(pairwise_matches=[], 
                                       pairwise_indices=[],
                                       image_size=img_size,
                                       image_scale=img_scale,
                                       img_features=[])

        for scene in tqdm(range(0, len(features) - 1), desc="Detecting Correspondences"):
            pt1 = features[scene]
            pt2 = features[scene + 1]

            idx1, idx2 = self.matcher_parser(pt1.descriptors, pt2.descriptors)

            new_pt1 = Points2D(**pt1.splice_2D_points(idx1))
            new_pt2 = Points2D(**pt2.splice_2D_points(idx2))

            inlier_pts1, inlier_pts2, idx1_inliers, idx2_inliers, F = self.outlier_reject(new_pt1, new_pt2, idx1, idx2, scene)

            feat_pair = np.hstack((inlier_pts1.points2D, inlier_pts2.points2D))
            idx_pair = np.hstack((np.vstack(idx1_inliers), np.vstack(idx2_inliers)))

            # matched_points.set_matching_pair(feat_pair, idx_pair)
            matched_points.set_matching_pair(
                data=feat_pair,
                idx_data=idx_pair,
                image_pair=(scene, scene + 1),
                index_type="global",
                matcher_name="bruteforce",
            )
            matched_points.img_features.append(pt1.points2D)
        # Get the last image feature set
        matched_points.img_features.append(features[-1].points2D)

        return matched_points

    def matcher_parser(self, desc1: np.ndarray, desc2: np.ndarray) -> tuple[list, list]:

        if self.cross_check:
            matches = self.matcher.match(desc1,desc2)
        else:
            matches = self.matcher.knnMatch(desc1, desc2, k=self.k)
                
        if not self.cross_check:
            # Conduct Lowe's Test Here
            good = []
            for m,n in matches:
                if m.distance < self.lowes_thresh*n.distance:
                    good.append(m)
        # if not self.cross_check:
            pts1_idx = [good[i].queryIdx for i in range(len(good))]
            pts2_idx = [good[i].trainIdx for i in range(len(good))]
        else:
             
            pts1_idx = [matches[i].queryIdx for i in range(len(matches))]
            pts2_idx = [matches[i].trainIdx for i in range(len(matches))]

        return pts1_idx, pts2_idx