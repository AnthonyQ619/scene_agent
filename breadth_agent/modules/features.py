import cv2
import numpy as np
import glob
from DataTypes.datatype import Points2D, Calibration

LISTED_TYPES = ['corner', 'sift', 'orb']
MATCHERS = ['flann', 'bruteforce', 'bf']


class FeatureDetection():
    def __init__(self, image_path = None):
        self.image_path = sorted(glob.glob(image_path + "\\*"))
        self.listed_types = ['corner', 'sift', 'orb']

    def detect_features(self, type = None, model_name = None):
        detected_features = []

        if type not in LISTED_TYPES:
            print("Error in Feature Selection. Use one of: corner, sift, or orb")
            
            return 
        
        if type.lower() == LISTED_TYPES[0]:
            for img in self.image_path:
                im_gray = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY)
                corners = cv2.cornerHarris(im_gray, 2,3,0.04)
                detected_features.append(Points2D(corners))
            return detected_features
        elif type.lower() == LISTED_TYPES[1]:
            detector = cv2.SIFT_create()
        elif type.lower() == LISTED_TYPES[2]:
            detector = cv2.ORB_create()
        
        for img in self.image_path:
            im_gray = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY)

            kp, des = detector.detectAndCompute(im_gray, None)
            pts = np.array([kp[i].pt for i in range(len(kp))])

            detected_features.append(Points2D(points2D = pts, descriptors=des))
        
        return detected_features
    

class FeatureMatching():
    def __init__(self, matcher = None, type = 'partial', detector = 'sift'):
        if matcher is None:
            self.matcher_type = "FLANN"
        elif matcher not in MATCHERS:
            print("Error in Matcher Selection. Use one of: BruteForce(BF) or FLANN")
            return
        
        self.detector = detector.lower()
        self.matcher_type = matcher

        if self.matcher_type.lower() == MATCHERS[0]:
            if self.detector == LISTED_TYPES[1]:
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            else:
                FLANN_INDEX_LSH = 6
                index_params = dict(algorithm = FLANN_INDEX_LSH,
                                    table_number = 6, # 12
                                    key_size = 12,     # 20
                                    multi_probe_level = 1) #2
            
            search_params = dict(checks=50)   # or pass empty dictionary
 
            self.matcher = cv2.FlannBasedMatcher(index_params,search_params)

        elif self.matcher_type.lower() in MATCHERS[1:]: 
            if self.detector == LISTED_TYPES[1]:
                self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            else:
                self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
    def __call__(self, features: list[Points2D], type = 'full') -> list:
        
        if type.lower() == 'full':
            matched_points = self.match_full(features)
        elif type.lower() in ['pair', 'pairwise', 'partial']:
            matched_points = self.match_pair(features[0], features[1])

        return matched_points
    
    def match_full(self, features: list[Points2D]) -> list[list[Points2D, Points2D]]:
        matched_points = []

        for scene in range(0, len(features) - 1):
            pt1 = features[scene]
            pt2 = features[scene + 1]

            idx1, idx2 = self.matcher_parser(pt1.descriptors, pt2.descriptors)

            pt1.update_2D_points_index(idx1)
            pt2.update_2D_points_index(idx2)

            matched_points.append([pt1, pt2])

        return matched_points


    def match_pair(self, pts1: Points2D, pts2: Points2D) -> list[Points2D, Points2D]:
        idx1, idx2 = self.matcher_parser(pts1.descriptors, pts2.descriptors)

        pts1.update_2D_points_index(idx1)
        pts2.update_2D_points_index(idx2)

        return pts1, pts2

    def matcher_parser(self, desc1: np.ndarray, desc2: np.ndarray) -> tuple[list, list]:
        knn = False
        if self.matcher_type.lower() == "flann":
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
            knn = True
        elif self.matcher_type.lower() in MATCHERS[1:]:
            if self.detector == LISTED_TYPES[1]:
                matches = self.matcher.knnMatch(desc1, desc2, k=2)
                knn = True
            else:
                matches = self.matcher.match(desc1,desc2)
                
        if self.detector == LISTED_TYPES[1]:
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

        

            
