import numpy as np
import cv2
from tqdm import tqdm
from .DataTypes.datatype import Points2D, Points3D, Calibration, Scene 


class OutlierRejection():
    def __init__(self, calibration: Calibration | None = None):
        if calibration is not None:
            self.K = calibration.K1
        self.OPTIONS = ['essential', 'fundamental', 'homography', 'projective']
        self.FORMATS = ['full', 'partial', 'pair']
    
    def __call__(self, option: str, format: str, pts: list[list[Points2D]]) -> list[list[Points2D]]:
        if option.lower() not in self.OPTIONS:
            message = 'Error: no such option exist. Use on of ' + str(self.OPTIONS)
            raise Exception(message)
        if format.lower() not in self.FORMATS:
            message = 'Error: no such format exist. Use on of ' + str(self.FORMATS)
            raise Exception(message)

        if format.lower() == self.FORMATS[0]:
            new_feature_matches = []

            if option.lower() == self.OPTIONS[0]:
                for i in tqdm(range(len(pts))):
                    scene = pts[i]
                    pts1, pts2 = scene[0], scene[1]
                    mask = self.find_essential(pts1, pts2)

                    new_pt1 = Points2D(**pts1.set_inliers(mask))
                    new_pt2 = Points2D(**pts2.set_inliers(mask))
                    new_feature_matches.append([new_pt1, new_pt2])

            elif option.lower() == self.OPTIONS[1]:
                for i in tqdm(range(len(pts))):
                    scene = pts[i]
                    pts1, pts2 = scene[0], scene[1]
                    mask = self.find_fundamental(pts1, pts2)

                    new_pt1 = Points2D(**pts1.set_inliers(mask))
                    new_pt2 = Points2D(**pts2.set_inliers(mask))
                    new_feature_matches.append([new_pt1, new_pt2])
                
            elif option.lower() in self.OPTIONS[2:]:
                for i in tqdm(range(len(pts))):
                    scene = pts[i]
                    pts1, pts2 = scene[0], scene[1]
                    mask = self.find_projective(pts1, pts2)
                    
                    new_pt1 = Points2D(**pts1.set_inliers(mask))
                    new_pt2 = Points2D(**pts2.set_inliers(mask))
                    new_feature_matches.append([new_pt1, new_pt2])
                
            return new_feature_matches
        elif format.lower() in self.FORMATS[1:]:
            pts1, pts2 = pts[0][0], pts[0][1]
            if option.lower() == self.OPTIONS[0]:
                mask = self.find_essential(pts1, pts2)
                new_pt1 = Points2D(**pts1.set_inliers(mask))
                new_pt2 = Points2D(**pts2.set_inliers(mask))
            elif option.lower() == self.OPTIONS[1]:
                mask = self.find_fundamental(pts1, pts2)
                new_pt1 = Points2D(**pts1.set_inliers(mask))
                new_pt2 = Points2D(**pts2.set_inliers(mask))
            elif option.lower() in self.OPTIONS[2:]:
                mask = self.find_projective(pts1, pts2)
                new_pt1 = Points2D(**pts1.set_inliers(mask))
                new_pt2 = Points2D(**pts2.set_inliers(mask))

            return [[new_pt1, new_pt2]]

    def find_fundamental(self, pts1: Points2D, pts2: Points2D) -> np.ndarray:
        _, mask = cv2.findFundamentalMat(pts1.points2D, pts2.points2D, cv2.FM_LMEDS)

        # Could update points2D to inlier points with Mask

        return mask
    
    def find_essential(self, pts1: Points2D, pts2: Points2D) -> np.ndarray:
        _, mask = cv2.findEssentialMat(pts1.points2D, pts2.points2D, self.K, method = cv2.RANSAC)

        return mask
    
    def find_projective(self, pts1: Points2D, pts2: Points2D) -> np.ndarray:
        MIN_MATCH_COUNT = 10

        if pts1.points2D.shape[0] > MIN_MATCH_COUNT:
            _, mask = cv2.findHomography(pts1.points2D, pts2.points2D, cv2.RANSAC, 5.0)
            # Could update points2D to inlier points with Mask

            return mask
        else:
            message = "Error: Not enough points for computation. Requires: " + str(MIN_MATCH_COUNT)
            raise Exception(message)