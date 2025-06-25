import cv2
import numpy as np
import glob
from tqdm import tqdm
from .DataTypes.datatype import Points2D, PointsMatched
from baseclass import FeatureMatching


class FeatureMatchFlannTracking(FeatureMatching):
    def __init__(self, detector:str = 'sift'):
        self.module_name = "FeatureMatchFlannTracking"

        self.description = ""

        self.example = ""


        super().__init__(detector.lower())


        if self.detector ==  self.DETECTORS[0]:
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        else: # Fast and Orb
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm = FLANN_INDEX_LSH,
                                table_number = 6, # 12
                                key_size = 12,     # 20
                                multi_probe_level = 1) #2
        
        search_params = dict(checks=50)   # or pass empty dictionary

        self.matcher = cv2.FlannBasedMatcher(index_params,search_params)

        self.track_map = {}
        self.next_track_id = 0
        self.observations = []

        
    def __call__(self, features: list[Points2D]) -> PointsMatched:
        
        matched_points = self.match_full(features) # TODO: Edit how PointsMatched is Filled

        return matched_points
    
    def match_full(self, features: list[Points2D]) -> PointsMatched:
        tracked_features = PointsMatched()
        for scene in tqdm(range(0, len(features) - 1)):
            pt1 = features[scene]
            pt2 = features[scene + 1]

            matches, idx1, idx2 = self.matcher_parser(pt1.descriptors, pt2.descriptors) # Match and Lowe's Ratio Test

            new_pt1 = Points2D(**pt1.splice_2D_points(idx1))
            new_pt2 = Points2D(**pt2.splice_2D_points(idx2))

            # Outlier Rejection Here
            matches_inlier, inlier_pts1, inlier_pts2 = self.outlier_reject(matches, new_pt1, new_pt2)

            # Feature Tracking algorithm here
            self.tracking_points(scene, inlier_pts1, inlier_pts2, matches_inlier)

            # matched_points.append([new_pt1, new_pt2])

        tracked_features.set_matched_matrix(self.observations)
        tracked_features.track_map = self.track_map
        tracked_features.point_count = self.next_track_id - 1
        
        return tracked_features

    def outlier_reject(self, matches:list, pts1: Points2D, pts2: Points2D) -> tuple[Points2D, Points2D]:
        _, mask = cv2.findFundamentalMat(pts1.points2D, pts2.points2D, cv2.FM_LMEDS)

        # Could update points2D to inlier points with Mask
        inlier_pts1 = Points2D(**pts1.set_inliers(mask))
        inlier_pts2 = Points2D(**pts2.set_inliers(mask))

        matches_np = np.array(matches)

        print(inlier_pts1.points2D.shape)
        matches_inlier = matches_np[mask.ravel()==1].tolist()
        print(len(matches_inlier))
        return matches_inlier, inlier_pts1, inlier_pts2

    def matcher_parser(self, desc1: np.ndarray, desc2: np.ndarray) -> tuple[list, list]:        
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
                 
        if self.detector == self.DETECTORS[0]:
            # Conduct Lowe's Test Here
            good_matches = []
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good_matches.append(m)

            pts1_idx = [good_matches[i].queryIdx for i in range(len(good_matches))]
            pts2_idx = [good_matches[i].trainIdx for i in range(len(good_matches))]

            return good_matches, pts1_idx, pts2_idx
        else:

            pts1_idx = [matches[i].queryIdx for i in range(len(matches))]
            pts2_idx = [matches[i].trainIdx for i in range(len(matches))]

            return matches, pts1_idx, pts2_idx

    def tracking_points(self, frame_id: float, pts1: Points2D, pts2: Points2D, matches: list) -> None:        
        for i in range(pts1.points2D.shape[0]):
            pt1_id = pts1.points2D[i, :].tobytes()
            pt2_id = pts2.points2D[i, :].tobytes()

            key1 = (frame_id, pt1_id)
            key2 = (frame_id + 1, pt2_id)

            if key2 in self.track_map:
                continue

            if key1 in self.track_map:
                track_id = self.track_map[key1]
            else:
                track_id = self.next_track_id
                self.next_track_id += 1
                pt1 = pts1.points2D[i, :]
                self.observations.append([float(track_id), frame_id, pt1[0], pt1[1]])
                self.track_map[key1] = track_id
        
            pt2 = pts2.points2D[i, :] 
            self.observations.append([float(track_id), frame_id + 1, pt2[0], pt2[1]])
            self.track_map[key2] = track_id


class FeatureMatchFlannPair(FeatureMatching):
    def __init__(self, detector:str = 'sift'):
        self.module_name = "FeatureMatchFlannPair"

        self.description = ""

        self.example = ""


        super().__init__(detector.lower())


        if self.detector ==  self.DETECTORS[0]:
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        else: # Fast and Orb
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm = FLANN_INDEX_LSH,
                                table_number = 6, # 12
                                key_size = 12,     # 20
                                multi_probe_level = 1) #2
        
        search_params = dict(checks=50)   # or pass empty dictionary

        self.matcher = cv2.FlannBasedMatcher(index_params,search_params)

        
    def __call__(self, features: list[Points2D]) -> PointsMatched:
        
        matched_points = self.match_full(features) # TODO: Edit how PointsMatched is Filled

        return matched_points
    
    def match_full(self, features: list[Points2D]) -> list[list[Points2D, Points2D]]:
        matched_points = []

        for scene in tqdm(range(0, len(features) - 1)):
            pt1 = features[scene]
            pt2 = features[scene + 1]

            idx1, idx2 = self.matcher_parser(pt1.descriptors, pt2.descriptors)

            new_pt1 = Points2D(**pt1.splice_2D_points(idx1))
            new_pt2 = Points2D(**pt2.splice_2D_points(idx2))

            matched_points.append([new_pt1, new_pt2])

        return matched_points


    def matcher_parser(self, desc1: np.ndarray, desc2: np.ndarray) -> tuple[list, list]:        
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
                 
        if self.detector == self.DETECTORS[0]:
            # Conduct Lowe's Test Here
            good = []
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good.append(m)

        pts1_idx = [good[i].queryIdx for i in range(len(good))]
        pts2_idx = [good[i].trainIdx for i in range(len(good))]

        return pts1_idx, pts2_idx
    
class FeatureMatchBFPair(FeatureMatching):
    def __init__(self, detector:str = 'sift'):
        self.module_name = "FeatureMatchBFPair"

        self.description = ""

        self.example = ""


        super().__init__(detector.lower())

        if self.detector ==  self.DETECTORS[0]:
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
    def __call__(self, features: list[Points2D]) -> PointsMatched:
        
        matched_points = self.match_full(features) # Convert to PointsMatched Properly
   
        return matched_points
    
    def match_full(self, features: list[Points2D]) -> list[list[Points2D, Points2D]]:
        matched_points = []

        for scene in tqdm(range(0, len(features) - 1)):
            pt1 = features[scene]
            pt2 = features[scene + 1]

            idx1, idx2 = self.matcher_parser(pt1.descriptors, pt2.descriptors)

            new_pt1 = Points2D(**pt1.splice_2D_points(idx1))
            new_pt2 = Points2D(**pt2.splice_2D_points(idx2))

            matched_points.append([new_pt1, new_pt2]) # TODO: Edit how PointsMatched is Filled

        return matched_points


    def matcher_parser(self, desc1: np.ndarray, desc2: np.ndarray) -> tuple[list, list]:
        knn = False
      
        if self.detector == self.DETECTORS[0]:
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
            knn = True
        else:
            matches = self.matcher.match(desc1,desc2)
                
        if self.detector == self.DETECTORS[0]:
            # Conduct Lowe's Test Here
            good = []
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good.append(m)
        
        if knn:
            pts1_idx = [good[i].queryIdx for i in range(len(good))]
            pts2_idx = [good[i].trainIdx for i in range(len(good))]
        else:
            pts1_idx = [matches[i].queryIdx for i in range(len(matches))]
            pts2_idx = [matches[i].trainIdx for i in range(len(matches))]

        return pts1_idx, pts2_idx