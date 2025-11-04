import sys
sys.path.append("C:\\Users\\Anthony\\Documents\\Projects\\Matchers\\RoMa\\romatch")
############ TEMP SOLUTION FOR NOW #################

import cv2
# from cv2.xfeatures2d import matchGMS
import numpy as np
import glob
from tqdm import tqdm
from .DataTypes.datatype import Points2D, PointsMatched, CameraData
from models.matchers import LightGlue, SuperGlue
from romatch import roma_outdoor, roma_indoor

from baseclass import FeatureMatching, FeatureTracking
from collections.abc import Callable
import kornia as K
import kornia.feature as KF
import torch
from PIL import Image, ImageOps
import piexif


############################################## HELPER CLASS ##############################################

# Convert points to normalized iamge coordinates!
# class Normalization():
#     def __init__(self, 
#                  K: np.ndarray | None = None,
#                  dist: np.ndarray | None = None):
#         if K is None:
#             self.K = None
#             self.dist = None
#             self.calibration = False
#         else:
#             self.K = K
#             self.dist = dist
#             self.calibration = True

#     def __call__(self, pts: Points2D) -> np.ndarray:
#         if self.calibration:
#             return self._calibrated(pts)
#         else:
#             return self._uncalibrated(pts)
        
#     def _calibrated(self, pts: Points2D) -> np.ndarray:
#         # print(pts.points2D.shape)
#         # print(self.dist1.shape)
#         # print(self.K1.shape)
#         pts_norm = cv2.undistortPoints(pts.points2D.T, self.K, self.dist)[:, 0, :]

#         return pts_norm
    

#     def _uncalibrated(self, pts:Points2D) -> Points2D:
#         pass 

# class FeatureTracker():
#     def __init__(self, 
#                  matcher_parser: Callable[[Points2D, Points2D], tuple],
#                  cam_data: CameraData
#                  ):

#         # Establish the data structures 
#         self.track_map = {}
#         self.next_track_id = 0
#         self.observations = []

#         # Set the Matcher
#         self.matcher = matcher_parser

#         self.ep_check = EpipoleChecker(pxl_min=25)
#         self.normalization = Normalization(cam_data=cam_data)
    
#     def tracking_points(self, frame_id: float, pts1: Points2D, pts2: Points2D) -> None:        
#         for i in range(pts1.points2D.shape[0]):
#             pt1_id = pts1.points2D[i, :].tobytes()
#             pt2_id = pts2.points2D[i, :].tobytes()

#             key1 = (frame_id, pt1_id)
#             key2 = (frame_id + 1, pt2_id)

#             if key2 in self.track_map:
#                 continue

#             if key1 in self.track_map:
#                 track_id = self.track_map[key1]
#             else:
#                 track_id = self.next_track_id
#                 self.next_track_id += 1
#                 pt1 = pts1.points2D[i, :]
#                 self.observations.append([float(track_id), frame_id, pt1[0], pt1[1]])
#                 self.track_map[key1] = track_id
        
#             pt2 = pts2.points2D[i, :] 
#             self.observations.append([float(track_id), frame_id + 1, pt2[0], pt2[1]])
#             self.track_map[key2] = track_id

#     def outlier_reject(self, pts1: Points2D, pts2: Points2D) -> tuple[Points2D, Points2D]:
#         pts_norm1 = self.normalization(pts1)
#         pts_norm2 = self.normalization(pts2)
        
#         # _, mask = cv2.findFundamentalMat(pts1.points2D, pts2.points2D, cv2.FM_LMEDS)
#         _, mask = cv2.findFundamentalMat(pts_norm1, pts_norm2, cv2.FM_LMEDS)

#         # Could update points2D to inlier points with Mask
#         inlier_pts1 = Points2D(**pts1.set_inliers(mask))
#         inlier_pts2 = Points2D(**pts2.set_inliers(mask))

#         # # print(matches)
#         # matches_np = np.array(matches)

#         # # print(inlier_pts1.points2D.shape)
#         # matches_inlier = matches_np[mask.ravel()==1].tolist()
#         # # print(len(matches_inlier))
#         # return matches_inlier, inlier_pts1, inlier_pts2
#         return inlier_pts1, inlier_pts2
    
#     def match_full(self, features: list[Points2D]) -> PointsMatched:
#         img_size = features[0].image_size
#         img_scale = features[0].reshape_scale
#         tracked_features = PointsMatched(image_size=img_size, 
#                                          multi_view=True,
#                                          image_scale=img_scale)
        
#         for scene in tqdm(range(0, len(features) - 1)):
#             pt1 = features[scene]
#             pt2 = features[scene + 1]

#             # matches, idx1, idx2 = self.matcher_parser(pt1, pt2) # Match and Lowe's Ratio Test
#             idx1, idx2 = self.matcher(pt1, pt2)

#             new_pt1 = Points2D(**pt1.splice_2D_points(idx1))
#             new_pt2 = Points2D(**pt2.splice_2D_points(idx2))

#             # Outlier Rejection Here
#             # matches_inlier, inlier_pts1, inlier_pts2 = self.outlier_reject(matches, new_pt1, new_pt2)
#             inlier_pts1, inlier_pts2 = self.outlier_reject(new_pt1, new_pt2)
#             # inlier_pts1, inlier_pts2 = self.ep_check(inlier_pts1, inlier_pts2)

#             # Feature Tracking algorithm here
#             self.tracking_points(scene, inlier_pts1, inlier_pts2) #, matches_inlier)

#             # matched_points.append([new_pt1, new_pt2])

#         tracked_features.set_matched_matrix(self.observations)
#         tracked_features.track_map = self.track_map
#         tracked_features.point_count = self.next_track_id - 1
        
#         return tracked_features

# class EpipoleChecker():
#     def __init__(self, frac: float = 0.02, pxl_min: int = 12):
#         self.frac = frac
#         self.pxl_min = pxl_min

#     def __call__(self, 
#                  points1: Points2D, 
#                  points2: Points2D, 
#                  F_mat: np.ndarray,
#                  normalize_func: Normalization
#                  ) -> tuple[Points2D, Points2D]:
#         h, w = points1.image_size[:]

#         # F_mat, _ = cv2.findFundamentalMat(points1=points1.points2D,
#         #                                points2=points2.points2D,
#         #                                method=cv2.FM_8POINT)
#         pts_norm1 = normalize_func(points1)
#         pts_norm2 = normalize_func(points2)
        
#         e1, e2 = self.compute_epipoles(F_mat)

#         d1, d2 = self.distances_to_epipoles(points1=points1.points2D,
#                                             points2=points2.points2D,
#                                             e1 = e1, e2 = e2)

#         # d1, d2 = self.distances_to_epipoles(points1=pts_norm1,
#         #                                     points2=pts_norm2,
#         #                                     e1 = e1, e2 = e2)

#         inliers = self.determine_thresh(d1, d2, w, h)

#         inlier_pts1 = Points2D(**points1.set_inliers(inliers))
#         inlier_pts2 = Points2D(**points2.set_inliers(inliers))

#         return inlier_pts1, inlier_pts2

#     def compute_epipoles(self, F: np.ndarray) -> tuple[np.ndarray]:
#         # Normalize F matrix!


#         # Epipole in image 2
#         U, S, Vt = np.linalg.svd(F)
#         e2 = Vt[-1]
#         e2 = e2[:2] / e2[2]

#         # Epipole in image 1
#         U, S, Vt = np.linalg.svd(F.T)
#         e1 = Vt[-1]
#         e1 = e1[:2] / e1[2]

#         return np.array(e1).reshape((2,1)), np.array(e2).reshape((2,1))

#     def distances_to_epipoles(self, points1: np.ndarray, points2: np.ndarray, 
#                               e1: np.ndarray, e2: np.ndarray) -> tuple[np.ndarray]:
#         points1_t = points1.T
#         points2_t = points2.T

#         print("EPIPOLE")
#         print(e1)
#         print("POINTS")
#         print(points1_t)
#         print("END OF POINTS")
#         d1 = np.linalg.norm(points1_t - e1, axis=0)
#         d2 = np.linalg.norm(points2_t - e2, axis=0)
#         return d1, d2
    
#     def determine_thresh(self, d1: np.ndarray, d2: np.ndarray, img_w: int, img_h: int):

#         diag = np.hypot(img_w, img_h)
#         thr = max(self.pxl_min, self.frac * diag)

#         print(thr)
#         print(d1)
#         thr_d1 = d1 <= thr
#         thr_d2 = d2 <= thr

#         inliers = thr_d1 + thr_d2
#         return inliers

##########################################################################################################

##########################################################################################################
############################################ TRACKING MODULES ############################################

class FeatureMatchSuperGlueTracking(FeatureTracking):
    def __init__(self, 
                 cam_data: CameraData,
                 RANSAC_threshold: float = 3.0,
                 RANSAC_conf: float = 0.99, 
                 detector:str = 'superpoint', 
                 sinkhorn_iterations: int = 20, 
                 match_threshold: float = 0.2, 
                 descriptor_dim: int = 256,
                 setting: str = 'indoor'):
        
        SUPPORTED_FEATURES = ["superpoint", "sp", "sift"]
        SUPPORTED_SETTINGS = ["indoor", "outdoor"]

        if setting not in SUPPORTED_SETTINGS:
            message = 'Error: setting is not supported. Use one of ' + str(SUPPORTED_SETTINGS) + ' instead to use this Feature Matcher.'
            raise Exception(message)

        if detector.lower() not in SUPPORTED_FEATURES:
            message = 'Error: detector is not supported. Use one of ' + str(SUPPORTED_FEATURES) + ' instead to use this Feature Matcher.'
            raise Exception(message)

        super().__init__(detector=detector.lower(),
                         cam_data=cam_data,
                         RANSAC_threshold=RANSAC_threshold,
                         RANSAC_conf=RANSAC_conf)
        
        self.module_name = "FeatureMatchSuperGlueTracking"

        self.description = f"""
Detects point correspondance across multiple frames for feature tracking in case of multi-view 
purposes. The feature matching algorithm used is the SuperGlue deep learning model trained 
as a feature matcher. Unless specified directly, assume the features are detected using 
the SuperPoint deep learning feature detector algorithm and initialize through the detector 
parameter. Use this matching module when features need to be matched in images where the 
scene contains low-lit enviornment or illumination changes occur in the scene 
(Outdoors for example), and run time is NOT A CONCERN, or when specified directly.
This matcher excels in Nighttime setting with SuperPoint as the detector, but other detectors
can be utilized for efficiency purposes.

Model is trained both for indoor and outdoor setting. When not specified, assume indoor
setting to properly initialize the model.

Other supported detectors are: SIFT and SuperPoint

Initialization Parameters:
- detector (str): Name of Feature Detector that was used to estimate the features provided.
- sinkhorn_iterations: number iterations for running the Sinkhorn Algorithm in the model for optimal
partial assignment of detected feature matches
    - default (int): 20
- match_threshold: confidence threshold (we choose 0.2) to retain some matches from the soft assignment
stage
    - default (float): 0.2
- descriptor_dim: the dimensions for the estimated desciptor generated from the detector used
    - default (int): 256
- setting: the string to determine if the images are "indoor" or "outdoor"
    - default (str): indoor

Function Call Parameters:
- features: list of features detected per scene estimated from the feature detection module
""" # TODO: Fill in details for the matcher. Be precise as we want the agent to know when exactly to use this
        
        self.example = f"""
Initialization: 
# Determine the detector that was used previously and initialize module with said detector

# Feature Matcher Module initialized with the SIFT detector
feature_tracker = FeatureMatchSuperGlueTracking(detector='sift') # Initialized with detector 'sift' for proper matching

# Feature Matcher Module intialized with the SuperPoint detector
feature_tracker = FeatureMatchSuperGlueTracking(detector='SuperPoint') # Initialized with detector 'orb' for proper matching

Example Usage in Script:  
features = feature_detector() # Call Feature Detector Module on image frames

tracked_features = feature_tracker(features=features) # Features used from Feature Detector Module are input to feature module
"""

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

        config_settings = {
            'weights': setting,
            'sinkhorn_iterations': sinkhorn_iterations,
            'match_threshold': match_threshold,
            'descriptor_dim': descriptor_dim,
        }

        self.matcher = SuperGlue(config=config_settings).eval().to(device)

        
    def __call__(self, features: list[Points2D]) -> PointsMatched:
        torch.set_grad_enabled(False)

        matched_points = self.feature_tracker.match_full(features)

        return matched_points
    
    
    def matcher_parser(self, pt1: Points2D, pt2: Points2D) -> tuple[list, list]:        
        # matches = self.matcher.knnMatch(desc1, desc2, k=2)
        feats0 = {"keypoints0": torch.from_numpy(pt1.points2D).unsqueeze(0).cuda(),
                      "descriptors0": torch.from_numpy(pt1.descriptors).unsqueeze(0).cuda().permute(0, 2, 1),
                      "scores0": torch.from_numpy(pt1.scores.T).cuda(),
                      "image_size0": torch.from_numpy(pt1.image_size).unsqueeze(0).cuda()}
            
        feats1 = {"keypoints1": torch.from_numpy(pt2.points2D).unsqueeze(0).cuda(),
                    "descriptors1": torch.from_numpy(pt2.descriptors).unsqueeze(0).cuda().permute(0, 2, 1),
                    "scores1": torch.from_numpy(pt2.scores.T).cuda(),
                    "image_size1": torch.from_numpy(pt2.image_size).unsqueeze(0).cuda()}

        feature_pair = {**feats0, **feats1}

        pred = self.matcher(feature_pair)
        matches = pred["matches0"].detach().cpu().numpy()

        valid_idx1 = matches > -1
        valid_idx2 = matches[valid_idx1]
        # print(valid_idx1)
        # print(valid_idx2)
        return valid_idx1[0].tolist(), valid_idx2.tolist()

class FeatureMatchLightGlueTracking(FeatureTracking):
    def __init__(self, 
                 cam_data: CameraData,
                 RANSAC_threshold: float = 3.0,
                 RANSAC_conf: float = 0.99, 
                 detector:str = 'superpoint', 
                 n_layers: int = 9, 
                 flash: bool = True, 
                 mp:bool = False, 
                 depth_confidence: float = 0.95,
                 width_confidence: float = 0.99, 
                 filter_threshold: float = 0.1):
        
        SUPPORTED_FEATURES = ["superpoint", "sift"]
      
        if detector.lower() not in SUPPORTED_FEATURES:
            message = 'Error: detector is not supported. Use one of ' + str(SUPPORTED_FEATURES) + ' instead to use this Feature Matcher.'
            raise Exception(message)

        super().__init__(detector=detector.lower(),
                         cam_data=cam_data,
                         RANSAC_threshold=RANSAC_threshold,
                         RANSAC_conf=RANSAC_conf)
        
        self.module_name = "FeatureMatchLightGlueTracking"
        
        self.description = f"""
Detects point correspondance across multiple frames to track features for multi-view purposes. 
The feature matching algorithm used is the LightGlue deep learning model trained as a feature 
matcher. Unless specified directly, assume the features are detected using the SuperPoint 
deep learning feature detector algorithm and initialized with the detector parameter. 
Use this matching module when features need to be matched in images where the scene contains 
low-lit enviornment or illumination changes occur in the scene (Outdoors for example), 
and faster run time is REQUIRED, or when specified directly in this scenario.

Other supported detectors are: SIFT and SuperPoint

Initialization Parameters:

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

Function Call Parameters:
- features: list of features detected per scene estimated from the feature detection module
""" # TODO: Fill in details for the matcher. Be precise as we want the agent to know when exactly to use this
        
        self.example = f"""
Initialization: 
# Determine the detector that was used previously and initialize module with said detector

# Feature Matcher Module initialized with the SIFT detector
feature_tracker = FeatureMatchLightGlueTracking(detector='sift') # Initialized with detector 'sift' for proper matching

# Feature Matcher Module intialized with the SuperPoint detector
feature_tracker = FeatureMatchLightGlueTracking(detector='SuperPoint') # Initialized with detector 'orb' for proper matching


Example Usage in Script:  
features = feature_detector() # Call Feature Detector Module on image frames

tracked_features = feature_tracker(features=features) # Features used from Feature Detector Module are input to feature module
"""

        self.matcher = LightGlue(features = self.detector, 
                                 n_layers = n_layers,
                                 flash = flash, 
                                 mp = mp, 
                                 depth_confidence = depth_confidence,
                                 width_confidence = width_confidence, 
                                 filter_threshold = filter_threshold).eval().cuda()

        
    def __call__(self, features: list[Points2D]) -> PointsMatched:
        torch.set_grad_enabled(False)

        # matched_points = self.match_full(features) # TODO: Edit how PointsMatched is Filled

        # feature_tracker = FeatureTracker(self.matcher_parser)

        matched_points = self.feature_tracker.match_full(features)

        return matched_points
    
    
    def matcher_parser(self, pt1: Points2D, pt2: Points2D) -> tuple[list, list]:        
        # matches = self.matcher.knnMatch(desc1, desc2, k=2)

        # feats0 = {"keypoints": torch.from_numpy(pt1.points2D).unsqueeze(0).cuda(),
        #               "descriptors": torch.from_numpy(pt1.descriptors).unsqueeze(0).cuda(),
        #               "image_size": torch.from_numpy(pt1.image_size).unsqueeze(0).cuda()}
            
        # feats1 = {"keypoints": torch.from_numpy(pt2.points2D).unsqueeze(0).cuda(),
        #             "descriptors": torch.from_numpy(pt2.descriptors).unsqueeze(0).cuda(),
        #             "image_size": torch.from_numpy(pt2.image_size).unsqueeze(0).cuda()}

        if self.detector == 'sift':
                feats0 = {"keypoints": torch.from_numpy(pt1.points2D).unsqueeze(0).cuda(),
                        "descriptors": torch.from_numpy(pt1.descriptors).unsqueeze(0).cuda(),
                        "scales": torch.from_numpy(pt1.scale).cuda(),
                        "oris": torch.from_numpy(pt1.orientation).cuda(),
                        "image_size": torch.from_numpy(pt1.image_size).unsqueeze(0).cuda()}
                
                feats1 = {"keypoints": torch.from_numpy(pt2.points2D).unsqueeze(0).cuda(),
                        "descriptors": torch.from_numpy(pt2.descriptors).unsqueeze(0).cuda(),
                        "scales": torch.from_numpy(pt2.scale).cuda(),
                        "oris": torch.from_numpy(pt2.orientation).cuda(),
                        "image_size": torch.from_numpy(pt2.image_size).unsqueeze(0).cuda()}
        else:
            feats0 = {"keypoints": torch.from_numpy(pt1.points2D).unsqueeze(0).cuda(),
                    "descriptors": torch.from_numpy(pt1.descriptors).unsqueeze(0).cuda(),
                    "image_size": torch.from_numpy(pt1.image_size).unsqueeze(0).cuda()}
            
            feats1 = {"keypoints": torch.from_numpy(pt2.points2D).unsqueeze(0).cuda(),
                    "descriptors": torch.from_numpy(pt2.descriptors).unsqueeze(0).cuda(),
                    "image_size": torch.from_numpy(pt2.image_size).unsqueeze(0).cuda()}

        feature_pair = {"image0": feats0, "image1": feats1}

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

class FeatureMatchFlannTracking(FeatureTracking):
    def __init__(self, 
                 cam_data: CameraData,
                 lowes_thresh: float = 0.75,
                 RANSAC_threshold: float = 3.0,
                 RANSAC_conf: float = 0.99, 
                 detector:str = 'sift'):

        super().__init__(detector.lower(), 
                         cam_data=cam_data,
                         RANSAC_conf=RANSAC_conf,
                         RANSAC_threshold=RANSAC_threshold)

        self.module_name = "FeatureMatchFlannTracking"

        self.description = f"""
Detects point correspondance across multiple frames to track features. The feature matching
algorithm used is the Flann feature detector. Unless specified directly, assume the features
are detected using the SIFT algorithm and initialize through the detector parameter. 
Other supported detectors are: SIFT, ORB, SuperPoint, and FAST.

SuperPoint and Sift share the same parameters, whereas ORB and FAST share the same parameters.

Use this module in the case a faster feature matcher is needed for tracking features, in which
this module utilizes a nearest neighbor method for fast matching.

Initialization Parameters: 
- detector: String representing the name of the feature detector used for the features provided.
    - Default (str): SIFT
    
Function Call Parameters:
- features: list of features detected per scene estimated from the feature detection module
"""

        self.example = f"""
Initialization: 
# Determine the detector that was used previously and initialize module with said detector

# Feature Tracker Module initialized with the SIFT detector
feature_tracker = FeatureMatchFlannTracking(detector='sift') # Initialized with detector 'sift' for proper matching

# Feature Tracker Module initialized with the ORB detector 
feature_tracker = FeatureMatchFlannTracking(detector='orb') # Initialized with detector 'orb' for proper matching

# Feature Tracker Module intialized with the FAST detector
feature_tracker = FeatureMatchFlannTracking(detector='fast') # Initialized with detector 'orb' for proper matching

Example Usage in Script:  
features = feature_detector() # Call Feature Detector Module on image frames

tracked_features = feature_tracker(features=features) # Features used from Feature Detector Module are input to feature module
"""

        if self.detector ==  self.DETECTORS[0]:
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        else: # Fast and Orb
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm = FLANN_INDEX_LSH,
                                table_number = 6, # 12
                                key_size = 12,     # 20
                                multi_probe_level = 2) #2
        
        search_params = dict(checks=50)   # or pass empty dictionary

        self.matcher = cv2.FlannBasedMatcher(index_params,search_params)
        self.lowes_thresh = lowes_thresh

        
    # def __call__(self, features: list[Points2D]) -> PointsMatched:
        
    #     # matched_points = self.match_full(features) # TODO: Edit how PointsMatched is Filled

    #     feature_tracker = FeatureTracker(self.matcher_parser, calibration=self.cam_data)

    #     matched_points = feature_tracker.match_full(features)

    #     return matched_points
    
    def matcher_parser(self, pt1: Points2D, pt2: Points2D) -> tuple[list, list]:        
        desc1 = pt1.descriptors
        desc2 = pt2.descriptors

        matches = self.matcher.knnMatch(desc1, desc2, k=2)
                 
        # if self.detector == self.DETECTORS[0]:
        # Conduct Lowe's Test Here
        good_matches = []
        for m,n in matches:
            if m.distance < self.lowes_thresh*n.distance:
                good_matches.append(m)

        pts1_idx = [good_matches[i].queryIdx for i in range(len(good_matches))]
        pts2_idx = [good_matches[i].trainIdx for i in range(len(good_matches))]

        return pts1_idx, pts2_idx
        # else:
        #     # print(matches)
        #     good_matches = []
        #     for m in matches:
        #         good_matches.append(m[0])
        #         # if m[0].distance < 0.75*m[1].distance:
        #         #     good_matches.append(m[0])

        #     pts1_idx = [matches[i][0].queryIdx for i in range(len(matches))]
        #     pts2_idx = [matches[i][0].trainIdx for i in range(len(matches))]

        #     #return good_matches, pts1_idx, pts2_idx
        #     return pts1_idx, pts2_idx

class FeatureMatchBFTracking(FeatureMatching):
    def __init__(self):
        pass
##########################################################################################################
############################################ TWO-VIEW MODULES ############################################


##########################################################################################################
############################################# DETECTOR--FREE #############################################

class FeatureMatchRoMAPair(FeatureMatching):
    def __init__(self, img_path:str, setting: str = "indoor", 
                 img_reshape: bool = True):

        super().__init__("None")

        self.module_name = "FeatureMatchLoftrPair"

        self.description = f"""
Detects point correspondance between two sequential frames at once to detect matching 
features across a set of images. This matching algorithm used is the detector free 
RoMA deep learning model trained as a feature matcher. This Feature Detection and matching
module does not take in features as it is a detector-free feature matching model. Therefore,
utilize this module in cases where detector free model is needed:
- in cases where the scene or image data has extreme view changes, 
- or when the scene has many textureless regions.
RoMA is specifically exceptional in two-view camera pose estimation due to the
quality of features detected.

Model is trained both for indoor and outdoor setting. When not specified, assume indoor
setting to properly initialize the model.

Initialization Parameters:
-img_path: the image path (str) in which the images are stored and to utilize for scene building
-setting: the string to determine if the images are "indoor" or "outdoor"
    - default (str): indoor
-img_reshape: parameter to determine whether to reshape image for model output.
    - Default (bool) = True (Reshape takes place by default for best model outcome)
""" 
        
        self.example = f"""
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

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')

        if setting.lower() not in SETTINGS:
            message = 'Error: setting is not supported. Use one of ' + str(SETTINGS) + ' instead to use this Detector-Free Matcher.'
            raise Exception(message)
        
        self.setting = SETTINGS[setting.lower()]

        if self.setting == "indoor":
            self.roma_model = roma_outdoor(device=self.device)
        else:
            self.roma_model = roma_indoor(device=self.device)

        self.image_path = sorted(glob.glob(img_path + "\\*"))

        self.img_shape = cv2.imread(self.image_path[0]).shape[:2] #HxWxC -> HxW

        self.ep_check = EpipoleChecker(pxl_min=25)
    
    def __call__(self):
        matched_points = PointsMatched(pairwise_matches=[], 
                                       image_size=np.array([self.img_shape[1], self.img_shape[0]]),
                                       image_scale=[1.0, 1.0])
        H_1, W_1 = self.img_shape[:2]
        for i in tqdm(range(8)): # len(self.image_path)
            img1_f = self.image_path[i]
            img2_f = self.image_path[i + 1]

            img1 = Image.open(img1_f)
            img2 = Image.open(img2_f)

            img1 = ImageOps.exif_transpose(img1)
            img2 = ImageOps.exif_transpose(img2)

            # Match
            warp, certainty = self.roma_model.match(img1, img2, device=self.device)
            # Sample matches for estimation
            matches, certainty = self.roma_model.sample(warp, certainty)
            kpts1, kpts2 = self.roma_model.to_pixel_coordinates(matches, H_1, W_1, H_1, W_1)

            # Bring to CPU and convert to Numpy vectors
            kpts1, kpts2 = kpts1.cpu().numpy(), kpts2.cpu().numpy()

            print(kpts1.shape)
            inlier_pts1, inlier_pts2 = self.outlier_reject(kpts1, kpts2)
            #inlier_pts1, inlier_pts2 = self.ep_check(inlier_pts1, inlier_pts2)

            matched_points.set_matching_pair(np.hstack((inlier_pts1, inlier_pts2)))

        return matched_points

    def outlier_reject(self, pts1: np.ndarray, pts2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # _, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
        _, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=2.0)

        # Could update points2D to inlier points with Mask
        inlier_pts1 = pts1[mask.ravel() == 1] #Points2D( #**pts1.set_inliers(mask))
        inlier_pts2 = pts2[mask.ravel() == 1] #Points2D( #**pts2.set_inliers(mask))


        return inlier_pts1, inlier_pts2
    
class FeatureMatchLoftrPair(FeatureMatching):
    def __init__(self, img_path:str, setting: str = "indoor", 
                 img_reshape: bool = True):
        super().__init__("None")

        self.module_name = "FeatureMatchLoftrPair"

        self.description = f"""
Detects point correspondance between two sequential frames at once to detect matching 
features across a set of images. The  matching algorithm used is the detector free 
LoFTR deep learning model trained as a feature matcher. This Feature Detection and Matching
module does not take in features as it is a detector-free feature matching model. Therefore,
utilize this module in cases where detector free model is needed: 
- in cases where the scene or image data has extreme view changes, 
- or when the scene has many textureless regions.
LoFTR is specifically exceptional in scens where there are textureless regions in the images
or illumination changes, especially in multi-view settings. 

Model is trained both for indoor and outdoor setting. When not specified, assume indoor
setting to properly initialize the model.

Initialization Parameters:
-img_path (str): the image path in which the images are stored and to utilize for scene building
-setting: the string to determine if the images are "indoor" or "outdoor"
    - default (str): indoor
-img_reshape: parameter to determine whether to reshape image for model output.
    - Default (bool) = True (Reshape takes place by default for best model outcome)
""" 
        
        self.example = f"""
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
        
        weight = SETTINGS[setting.lower()]

        self.matcher = KF.LoFTR(pretrained=weight)

        self.image_path = sorted(glob.glob(img_path + "\\*"))
        
        img_shape = cv2.imread(self.image_path[0]).shape[:2] #HxWxC -> HxW

        self.ep_check = EpipoleChecker(pxl_min=25)
        
        if "jpg" in self.image_path[0].lower():
            self.orientation = self.kornia_orientation_reader(self.image_path[0])
        else:
            self.orientation = False

        if (img_reshape == True) or (weight == "indoor_new"):
            self.img_shape = (480, 640) # HxW
            self.scale = [480/img_shape[0], 640/img_shape[1]]
        else: 
            print(img_shape)
            self.img_shape = img_shape
            self.scale = [1.0, 1.0]

    def __call__(self) -> PointsMatched:

        matched_points = PointsMatched(pairwise_matches=[], 
                                       image_size=np.array([self.img_shape[0], self.img_shape[1]]),
                                       image_scale=self.scale)
        
        for i in tqdm(range(5)): # len(self.image_path)
        # img in self.image_path:
            img1_f = self.image_path[i]
            img2_f = self.image_path[i + 1]

            img1 = self.kornia_img_reader(img1_f)
            img2 = self.kornia_img_reader(img2_f)

            input_dict = {
                "image0": img1,  # LofTR works on grayscale images only
                "image1": img2,
            }

            with torch.inference_mode():
                correspondences = self.matcher(input_dict)

            mkpts0 = correspondences["keypoints0"].cpu().numpy()
            mkpts1 = correspondences["keypoints1"].cpu().numpy()
            
            inlier_pts1, inlier_pts2 = self.outlier_reject(mkpts0, mkpts1)
            #inlier_pts1, inlier_pts2 = self.ep_check(inlier_pts1, inlier_pts2)
            
            matched_points.set_matching_pair(np.hstack((inlier_pts1, inlier_pts2)))

        return matched_points

    def outlier_reject(self, pts1: np.ndarray, pts2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

        # Could update points2D to inlier points with Mask
        inlier_pts1 = pts1[mask.ravel() == 1] #Points2D( #**pts1.set_inliers(mask))
        inlier_pts2 = pts2[mask.ravel() == 1] #Points2D( #**pts2.set_inliers(mask))


        return inlier_pts1, inlier_pts2
    
    def kornia_orientation_reader(self, img_path):
        img = Image.open(img_path)
        exif_dict = piexif.load(img.info.get('exif', b''))
        orientation = exif_dict["0th"].get(piexif.ImageIFD.Orientation, 1)

        return orientation
    
    def kornia_img_reader(self, img_path):
        img = K.io.load_image(img_path, K.io.ImageLoadType.RGB32)[None, ...]

        img = K.geometry.resize(img, self.img_shape, antialias=True)

        img_g = K.color.rgb_to_grayscale(img)

        if self.orientation:
            img_g = torch.rot90(img_g, k = 1, dims=(2,3))

        return img_g

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
                 RANSAC: bool = True,
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
        
        super().__init__(detector=detector.lower(), 
                         cam_data=cam_data,
                         RANSAC_conf=RANSAC_conf,
                         RANSAC=RANSAC,
                         RANSAC_threshold=RANSAC_threshold)

        self.module_name = "FeatureMatchSuperGluePair"

        self.description = f"""
Detects point correspondance between two sequential frames at once to detect matching 
features across a set of images. The feature matching algorithm used is the SuperGlue deep
learning model trained as a feature matcher. Unless specified directly, assume the features 
are detected using the SuperPoint deep learning feature detector algorithm and initialize 
through the detector parameter. Use this matching module when features need to be matched 
in images where the scene contains low-lit enviornment or illumination changes occur in 
the scene (Outdoors for example), and run time is NOT A CONCERN, or when specified directly.
This matcher excels in Nighttime setting with SuperPoint as the detector.

Model is trained both for indoor and outdoor setting. When not specified, assume indoor
setting to properly initialize the model.

Other supported detectors are: SIFT and SuperPoint.

Initialization Parameters:
- detector (str): Name of Feature Detector that was used to estimate the features provided.
- sinkhorn_iterations: number iterations for running the Sinkhorn Algorithm in the model for optimal
partial assignment of detected feature matches
    - default (int): 20
- match_threshold: confidence threshold (we choose 0.2) to retain some matches from the soft assignment
stage
    - default (float): 0.2
- descriptor_dim: the dimensions for the estimated desciptor generated from the detector used
    - default (int): 256
- setting: the string to determine if the images are "indoor" or "outdoor"
    - default (str): indoor

Function Call Parameters:
- features (list[Points2D]): list of features detected per scene estimated from the feature detection module
""" # TODO: Fill in details for the matcher. Be precise as we want the agent to know when exactly to use this
        
        self.example = f"""
Initialization: 
# Determine the detector that was used previously and initialize module with said detector

# Feature Matcher Module initialized with the SIFT detector
feature_matcher = FeatureMatchSuperGluePair(detector='sift') # Initialized with detector 'sift' for proper matching

# Feature Matcher Module intialized with the SuperPoint detector
feature = FeatureMatchSuperGluePair(detector='SuperPoint') # Initialized with detector 'orb' for proper matching

Example Usage in Script:  
features = feature_detector() # Call Feature Detector Module on image frames

tracked_features = feature_matcher(features=features) # Features used from Feature Detector Module are input to feature module
"""

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

        # if detector.lower() == "sift":
        #     self.descriptor_dim = 128
        # else: # SuperPoint
        #     self.descriptor_dim = descriptor_dim

        config_settings = {
            'weights': setting,
            'sinkhorn_iterations': sinkhorn_iterations,
            'match_threshold': match_threshold,
            'descriptor_dim': descriptor_dim,
        }

        self.matcher = SuperGlue(config=config_settings).eval().to(device)

        # self.ep_check = EpipoleChecker(pxl_min=25)

    # def __call__(self, features: list[Points2D]) -> PointsMatched:
        
    #     matched_points = self.match_full(features) 

    #     return matched_points
    
    def match_full(self, features: list[Points2D]) -> PointsMatched:
        torch.set_grad_enabled(False)

        img_size = features[0].image_size
        img_scale = features[0].reshape_scale
        matched_points = PointsMatched(pairwise_matches=[], 
                                       image_size=img_size,
                                       image_scale=img_scale)

        for scene in tqdm(range(0, len(features) - 1), desc="Detecting Correspondences"):
            pt1 = features[scene]
            pt2 = features[scene + 1]

            feats0 = {"keypoints0": torch.from_numpy(pt1.points2D).unsqueeze(0).cuda(),
                      "descriptors0": torch.from_numpy(pt1.descriptors).unsqueeze(0).cuda().permute(0, 2, 1),
                      "scores0": torch.from_numpy(pt1.scores.T).cuda(),
                      "image_size0": torch.from_numpy(pt1.image_size).unsqueeze(0).cuda()}
            
            feats1 = {"keypoints1": torch.from_numpy(pt2.points2D).unsqueeze(0).cuda(),
                      "descriptors1": torch.from_numpy(pt2.descriptors).unsqueeze(0).cuda().permute(0, 2, 1),
                      "scores1": torch.from_numpy(pt2.scores.T).cuda(),
                      "image_size1": torch.from_numpy(pt2.image_size).unsqueeze(0).cuda()}

            # print(torch.from_numpy(pt2.points2D).unsqueeze(0).cuda().shape)
            idx1, idx2 = self.matcher_parser({**feats0, **feats1})

            # print(idx1.shape)
            new_pt1 = Points2D(**pt1.splice_2D_points(idx1))
            new_pt2 = Points2D(**pt2.splice_2D_points(idx2))

            inlier_pts1, inlier_pts2, _ = self.outlier_reject(new_pt1, new_pt2)

            # inlier_pts1, inlier_pts2 = self.ep_check(inlier_pts1, inlier_pts2)

            matched_points.set_matching_pair(np.hstack((inlier_pts1.points2D, inlier_pts2.points2D)))

        return matched_points

    # def outlier_reject(self, pts1: Points2D, pts2: Points2D) -> tuple[Points2D, Points2D]:
    #     # print(pts1.points2D.shape)
    #     # print(pts2.points2D.shape)
    #     _, mask = cv2.findFundamentalMat(pts1.points2D, pts2.points2D, cv2.FM_RANSAC, ransacReprojThreshold=4.0)

    #     # Could update points2D to inlier points with Mask
    #     inlier_pts1 = Points2D(**pts1.set_inliers(mask))
    #     inlier_pts2 = Points2D(**pts2.set_inliers(mask))

    #     return inlier_pts1, inlier_pts2

    def matcher_parser(self, feature_pair: dict = {}) -> tuple[list, list]:        
        # matches = self.matcher.knnMatch(desc1, desc2, k=2)

        pred = self.matcher(feature_pair)
        matches = pred["matches0"].detach().cpu().numpy()

        valid_idx1 = matches > -1
        valid_idx2 = matches[valid_idx1]
        # print(valid_idx1)
        # print(valid_idx2)
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
                 RANSAC: bool = True,
                 RANSAC_threshold: float = 3.0,
                 RANSAC_conf: float = 0.99):
        
        SUPPORTED_FEATURES = ["superpoint", "sp", "sift"]

        if detector.lower() not in SUPPORTED_FEATURES:
            message = 'Error: detector is not supported. Use one of ' + str(self.FORMATS) + ' instead to use this Feature Matcher.'
            raise Exception(message)

        super().__init__(detector=detector.lower(), 
                         cam_data=cam_data,
                         RANSAC_conf=RANSAC_conf,
                         RANSAC=RANSAC,
                         RANSAC_threshold=RANSAC_threshold)
        

        self.module_name = "FeatureMatchLightGluePair"
        self.description = f"""
Detects point correspondance between two sequential frames at once to detect matching 
features across a set of images. The feature matching algorithm used is the LightGlue deep
learning model trained as a feature matcher. Unless specified directly, assume the features 
are detected using the SuperPoint deep learning feature detector algorithm and initialize 
through the detector parameter. Use this matching module when features need to be matched 
in images where the scene contains low-lit enviornment or illumination changes occur in 
the scene (Outdoors for example) with faster run time being REQUIRED, or when specified directly.

Other supported detectors are: SIFT and SuperPoint

Initialization Parameters:

- detector (str): Name of Feature Detector that was used to estimate the features provided.

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

Other supported detectors are: SIFT and SuperPoint

Function Call Parameters:

- features list[Points2D]: list of features detected per scene estimated from the feature detection module
""" # TODO: Fill in details for the matcher. Be precise as we want the agent to know when exactly to use this
        
        self.example = f"""
Initialization: 
# Determine the detector that was used previously and initialize module with said detector

# Feature Matcher Module initialized with the SIFT detector
feature_matcher = FeatureMatchLightGluePair(detector='sift') # Initialized with detector 'sift' for proper matching

# Feature Matcher Module intialized with the SuperPoint detector
feature = FeatureMatchLightGluePair(detector='SuperPoint') # Initialized with detector 'orb' for proper matching


Example Usage in Script:  
features = feature_detector() # Call Feature Detector Module on image frames

tracked_features = feature_matcher(features=features) # Features used from Feature Detector Module are input to feature module
"""
        self.matcher = LightGlue(features=self.detector, 
                                 n_layers = n_layers,
                                 flash = flash, 
                                 mp = mp, 
                                 depth_confidence = depth_confidence,
                                 width_confidence = width_confidence, 
                                 filter_threshold = filter_threshold).eval().cuda()

        # self.ep_check = EpipoleChecker(pxl_min=10)
        # self.normalize = Normalization(K=self.K, dist=self.dist)


    # def __call__(self, features: list[Points2D]) -> PointsMatched:
        
    #     matched_points = self.match_full(features) 

    #     return matched_points
    
    def match_full(self, features: list[Points2D]) -> PointsMatched:
        torch.set_grad_enabled(False)

        img_size = features[0].image_size
        img_scale = features[0].reshape_scale
        matched_points = PointsMatched(pairwise_matches=[], 
                                       image_size=img_size,
                                       image_scale=img_scale)

        for scene in tqdm(range(0, len(features) - 1), desc="Detecting Correspondences"):
            pt1 = features[scene]
            pt2 = features[scene + 1]

            if self.detector == 'sift':
                feats0 = {"keypoints": torch.from_numpy(pt1.points2D).unsqueeze(0).cuda(),
                        "descriptors": torch.from_numpy(pt1.descriptors).unsqueeze(0).cuda(),
                        "scales": torch.from_numpy(pt1.scale).cuda(),
                        "oris": torch.from_numpy(pt1.orientation).cuda(),
                        "image_size": torch.from_numpy(pt1.image_size).unsqueeze(0).cuda()}
                
                feats1 = {"keypoints": torch.from_numpy(pt2.points2D).unsqueeze(0).cuda(),
                        "descriptors": torch.from_numpy(pt2.descriptors).unsqueeze(0).cuda(),
                        "scales": torch.from_numpy(pt2.scale).cuda(),
                        "oris": torch.from_numpy(pt2.orientation).cuda(),
                        "image_size": torch.from_numpy(pt2.image_size).unsqueeze(0).cuda()}
            else:
                feats0 = {"keypoints": torch.from_numpy(pt1.points2D).unsqueeze(0).cuda(),
                        "descriptors": torch.from_numpy(pt1.descriptors).unsqueeze(0).cuda(),
                        "image_size": torch.from_numpy(pt1.image_size).unsqueeze(0).cuda()}
                
                feats1 = {"keypoints": torch.from_numpy(pt2.points2D).unsqueeze(0).cuda(),
                        "descriptors": torch.from_numpy(pt2.descriptors).unsqueeze(0).cuda(),
                        "image_size": torch.from_numpy(pt2.image_size).unsqueeze(0).cuda()}

            idx1, idx2 = self.matcher_parser({"image0": feats0, "image1": feats1})

            new_pt1 = Points2D(**pt1.splice_2D_points(idx1))
            new_pt2 = Points2D(**pt2.splice_2D_points(idx2))

            inlier_pts1, inlier_pts2, F_mat = self.outlier_reject(new_pt1, new_pt2)
            # inlier_pts1, inlier_pts2 = self.ep_check(inlier_pts1, inlier_pts2, F_mat)
            #inlier_pts1, inlier_pts2 = self.ep_check(new_pt1, new_pt2)

            matched_points.set_matching_pair(np.hstack((inlier_pts1.points2D, inlier_pts2.points2D)))

        return matched_points

    # def outlier_reject(self, pts1: Points2D, pts2: Points2D) -> tuple[Points2D, Points2D]:
    #     pts1_norm = self.normalize(pts1)
    #     pts2_norm = self.normalize(pts2)

    #     # print(pts1_norm)
    #     F_mat, mask = cv2.findFundamentalMat(pts1_norm, 
    #                                          pts2_norm, 
    #                                          cv2.FM_RANSAC, 
    #                                          ransacReprojThreshold=1.5)

    #     # Could update points2D to inlier points with Mask
    #     inlier_pts1 = Points2D(**pts1.set_inliers(mask))
    #     inlier_pts2 = Points2D(**pts2.set_inliers(mask))

    #     return inlier_pts1, inlier_pts2, F_mat

    def matcher_parser(self, feature_pair: dict = {}) -> tuple[list, list]:        
        # matches = self.matcher.knnMatch(desc1, desc2, k=2)

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
                 RANSAC: bool = True,
                 RANSAC_threshold: float = 3.0,
                 RANSAC_conf: float = 0.99,
                 lowes_thresh: float = 0.75):

        super().__init__(detector=detector.lower(), 
                         cam_data=cam_data,
                         RANSAC_conf=RANSAC_conf,
                         RANSAC=RANSAC,
                         RANSAC_threshold=RANSAC_threshold)

        self.module_name = "FeatureMatchFlannPair"
        self.description = f"""
Detects point correspondance between two sequential frames at once to detect matching 
features across a set of images. The feature matching algorithm used is the Flann feature 
detector. Use this feature matcher for when a Nearest Neighbor algorithm is called for 
and speed is a necessary requirement. Unless specified directly, assume the features 
are detected using the SIFT algorithm and initialize through the detector parameter. 
Other supported detectors are: SIFT, ORB, SuperPoint, and FAST.

SuperPoint and Sift share the same parameters, whereas ORB and FAST share the same parameters.

Use this module in the case a faster feature matcher is needed for feature pair correspondences, 
in which this module utilizes a nearest neighbor method for fast matching.

Initialization Parameters: 
- detector: String representing the name of the feature detector used for the features provided.
    - Default (str): SIFT
    
Function Call Parameters:
- features list[Points2D]: list of features detected per scene estimated from the feature detection module

"""
        self.example = f"""
Initialization: 
# Determine the detector that was used previously and initialize module with said detector

# Feature Matcher Module initialized with the SIFT detector
feature_matcher = FeatureMatchFlannPair(detector='sift') # Initialized with detector 'sift' for proper matching

# Feature Matcher Module initialized with the SuperPoint detector
feature_matcher = FeatureMatchFlannPair(detector='superpoint') # Initialized with detector 'sift' for proper matching

# Feature Matcher Module initialized with the ORB detector 
feature = FeatureMatchFlannPair(detector='orb') # Initialized with detector 'orb' for proper matching

# Feature Matcher Module intialized with the FAST detector
feature = FeatureMatchFlannPair(detector='fast') # Initialized with detector 'orb' for proper matching

Example Usage in Script:  
features = feature_detector() # Call Feature Detector Module on image frames

detected_features = feature_matcher(features=features) # Features used from Feature Detector Module are input to feature module
"""

        if self.detector ==  self.DETECTORS[:2]:
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
        # self.ransac = RANSAC
        # self.ransac_threshold = RANSAC_threshold
        # self.ransac_conf = RANSAC_conf

        self.matcher = cv2.FlannBasedMatcher(index_params,
                                             search_params)

        # self.ep_check = EpipoleChecker(pxl_min=25)
        # self.normalize = Normalization(K=self.K, dist=self.dist) # Move to Base Class

    # def __call__(self, features: list[Points2D]) -> PointsMatched:
        
    #     matched_points = self.match_full(features) 

    #     return matched_points
    
    def match_full(self, features: list[Points2D]) -> PointsMatched:
        img_size = features[0].image_size
        img_scale = features[0].reshape_scale
        matched_points = PointsMatched(pairwise_matches=[], 
                                       image_size=img_size,
                                       image_scale=img_scale)

        for scene in tqdm(range(0, len(features) - 1), desc="Detecting Correspondences"):
            pt1 = features[scene]
            pt2 = features[scene + 1]

            idx1, idx2 = self.matcher_parser(pt1.descriptors, pt2.descriptors)

            new_pt1 = Points2D(**pt1.splice_2D_points(idx1))
            new_pt2 = Points2D(**pt2.splice_2D_points(idx2))

            inlier_pts1, inlier_pts2, F = self.outlier_reject(new_pt1, new_pt2)
            # inlier_pts1, inlier_pts2 = self.ep_check(inlier_pts1, inlier_pts2, F, self.normalize)

            matched_points.set_matching_pair(np.hstack((inlier_pts1.points2D, inlier_pts2.points2D)))
            # matched_points.set_matching_pair(np.hstack((inlier_pts1, inlier_pts2)))

        return matched_points

    # def outlier_reject(self, pts1: Points2D, pts2: Points2D) -> tuple[Points2D, Points2D]: # Move to Base Class
        
    #     pts1_norm = self.normalize(pts1)
    #     pts2_norm = self.normalize(pts2)

    #     # F, mask = cv2.findFundamentalMat(pts1.points2D, pts2.points2D, cv2.FM_LMEDS)
    #     # F, mask = cv2.findFundamentalMat(pts1_norm, pts2_norm, cv2.FM_LMEDS)
    #     if self.ransac:
    #         F, mask = cv2.findFundamentalMat(pts1_norm, pts2_norm, cv2.FM_RANSAC, 
    #                                          ransacReprojThreshold=self.ransac_threshold, 
    #                                          confidence=self.ransac_conf)
    #     else:
    #         F, mask = cv2.findFundamentalMat(pts1_norm, pts2_norm, cv2.FM_LMEDS)

    #     # Could update points2D to inlier points with Mask
    #     inlier_pts1 = Points2D(**pts1.set_inliers(mask))
    #     inlier_pts2 = Points2D(**pts2.set_inliers(mask))

    #     return inlier_pts1, inlier_pts2, F

    def matcher_parser(self, desc1: np.ndarray, desc2: np.ndarray) -> tuple[list, list]:        
        # print(desc1.shape)
        # print(desc1.dtype)
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
                 
        # if self.detector == self.DETECTORS[0] or self.detector == self.DETECTORS[2]:
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
                 RANSAC: bool = True,
                 RANSAC_threshold: float = 3.0,
                 RANSAC_conf: float = 0.99,
                 GMS: bool = False):

        super().__init__(detector=detector.lower(), 
                         cam_data=cam_data,
                         RANSAC_threshold=RANSAC_threshold,
                         RANSAC=RANSAC,
                         RANSAC_conf=RANSAC_conf)

        self.module_name = "FeatureMatchBFPair"
        self.description = f"""
Detects point correspondance between two sequential frames at a time to detect matching 
features across a set of images. The feature matching algorithm used is the Brute-Force 
feature detector. Unless specified directly, assume the features are detected using the SIFT 
algorithm and initialize through the detector parameter. 
Other supported detectors are: SIFT, ORB, SuperPoint, and FAST. 

Use this Feature Matching Module when fast, real time, feature matching needs to be 
conducted. This is less accurate than Flann and the ML models, but is much faster.

Initalization Parameters:
- detector: String representing the name of the feature detector used for the features provided.
    - Default (str): SIFT
    
Function Call Parameters:
- features list[Points2D]: list of features detected per scene estimated from the feature detection module
"""
        self.example = f"""
Initialization: 
# Determine the detector that was used previously and initialize module with said detector

# Feature Matcher Module initialized with the SIFT detector
feature_matcher = FeatureMatchBFPair(detector='sift') # Initialized with detector 'sift' for proper matching

# Feature Matcher Module initialized with the ORB detector 
feature = FeatureMatchBFPair(detector='orb') # Initialized with detector 'orb' for proper matching

# Feature Matcher Module intialized with the FAST detector
feature = FeatureMatchBFPair(detector='fast') # Initialized with detector 'orb' for proper matching

Example Usage in Script:  
features = feature_detector() # Call Feature Detector Module on image frames

tracked_features = feature_matcher(features=features) # Features used from Feature Detector Module are input to feature module
"""

        if self.detector ==  self.DETECTORS[0]:
            norm_type = cv2.NORM_L2
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            norm_type = cv2.NORM_HAMMING

        self.cross_check = cross_check
        self.gms = GMS
        self.k = k
        # self.ransac = RANSAC
        # self.ransac_threshold = RANSAC_threshold
        # self.ransac_conf = RANSAC_conf

        self.matcher = cv2.BFMatcher(normType=norm_type, 
                                     crossCheck=self.cross_check)
        # self.normalize = Normalization(K=self.K, dist=self.dist)

        # self.ep_check = EpipoleChecker(pxl_min=25)
        
    # def __call__(self, features: list[Points2D]) -> PointsMatched:
        
    #     matched_points = self.match_full(features) 
   
    #     return matched_points
    
    def match_full(self, features: list[Points2D]) -> PointsMatched:
        img_size = features[0].image_size
        img_scale = features[0].reshape_scale
        matched_points = PointsMatched(pairwise_matches=[], 
                                       image_size=img_size,
                                       image_scale=img_scale)

        for scene in tqdm(range(0, len(features) - 1), desc="Detecting Correspondences"):
            pt1 = features[scene]
            pt2 = features[scene + 1]

            idx1, idx2 = self.matcher_parser(pt1.descriptors, pt2.descriptors)

            new_pt1 = Points2D(**pt1.splice_2D_points(idx1))
            new_pt2 = Points2D(**pt2.splice_2D_points(idx2))

            inlier_pts1, inlier_pts2, _ = self.outlier_reject(new_pt1, new_pt2)

            matched_points.set_matching_pair(np.hstack((inlier_pts1.points2D, inlier_pts2.points2D)))

        return matched_points

    # def outlier_reject(self, pts1: Points2D, pts2: Points2D) -> tuple[Points2D, Points2D]:
    #     pts1_norm = self.normalize(pts1)
    #     pts2_norm = self.normalize(pts2)
        
    #     # _, mask = cv2.findFundamentalMat(pts1.points2D, pts2.points2D, cv2.FM_LMEDS)
    #     if self.ransac:
    #         _, mask = cv2.findFundamentalMat(pts1_norm, pts2_norm, cv2.FM_RANSAC, 
    #                                          ransacReprojThreshold=self.ransac_threshold, 
    #                                          confidence=self.ransac_conf)
    #     else:
    #         _, mask = cv2.findFundamentalMat(pts1_norm, pts2_norm, cv2.FM_LMEDS)

    #     # Could update points2D to inlier points with Mask
    #     inlier_pts1 = Points2D(**pts1.set_inliers(mask))
    #     inlier_pts2 = Points2D(**pts2.set_inliers(mask))

    #     return inlier_pts1, inlier_pts2

    def matcher_parser(self, desc1: np.ndarray, desc2: np.ndarray) -> tuple[list, list]:
        # img_shape = (image_size[0], image_size[1]) # Convert from HxW to WxH (OpenCV Convention)


        if self.cross_check:
            matches = self.matcher.match(desc1,desc2)
        else:
            matches = self.matcher.knnMatch(desc1, desc2, k=self.k)
                
        if not self.cross_check:
            # Conduct Lowe's Test Here
            good = []
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good.append(m)
        # if not self.cross_check:
            pts1_idx = [good[i].queryIdx for i in range(len(good))]
            pts2_idx = [good[i].trainIdx for i in range(len(good))]
        else:
            # if self.gms: # Specifically ORB detector
            #     matches = matchGMS(img_shape, img_shape, )
             
            pts1_idx = [matches[i].queryIdx for i in range(len(matches))]
            pts2_idx = [matches[i].trainIdx for i in range(len(matches))]

        return pts1_idx, pts2_idx