# from modules.utilities.utilities import CalibrationReader
from modules.features import FeatureDetectionSIFT, FeatureDetectionSP, FeatureDetectionORB, FeatureDetectionFAST
from modules.featurematching import (FeatureMatchFlannTracking, 
                                     FeatureMatchBFTracking,
                                     FeatureMatchFlannPair,
                                     FeatureMatchBFPair,
                                     FeatureMatchLoftrPair,
                                     FeatureMatchLightGlueTracking, 
                                     FeatureMatchSuperGlueTracking, 
                                     FeatureMatchLightGluePair, 
                                     FeatureMatchSuperGluePair,
                                     FeatureMatchRoMAPair)
from modules.visualize import VisualizeScene
from modules.cameramanager import CameraDataManager
import glob
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Construct Modules with Initialized Arguments
# image_path = "C:\\Users\\Anthony\\Documents\\Projects\datasets\\Structure-from-Motion\\sfm_dataset"
# calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\Structure-from-Motion\\calibration_new.npz"
# image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\scan6_illumination_change"
# calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\calibration_DTU_new.npz"
# image_path = "D:\\aquir\\Documents\\Datasets\\CO3Dv2_DATASET\\apple\\189_20393_38136\\images"
# calibration_path = "D:\\aquir\\Documents\\Datasets\\CO3Dv2_DATASET\\apple\\calibration_new_189_20393_38136.npz"
image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Tanks_and_Temples\\Family"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Tanks_and_Temples\\calibration_new_2048.npz"

camera_data = CameraDataManager(image_path=image_path,
                                calibration_path=calibration_path).get_camera_data()
# Feature Module Initialization
# calibration_data = CalibrationReader(calibration_path).get_calibration()
feature_detector = FeatureDetectionSIFT(cam_data=camera_data, 
                                        max_keypoints=15000,
                                        contrast_threshold=0.009,
                                        edge_threshold=10)
# feature_detector = FeatureDetectionORB(cam_data=camera_data, 
#                                         max_keypoints=20000,
#                                         fast_threshold=20,
#                                         # set_nms=True,
#                                         # set_nms_allowed_points=12000,
#                                         # set_nms_tolerance = 0.2
#                                         )

# feature_detector = FeatureDetectionSP(cam_data=camera_data, 
#                                         max_keypoints=5000)
# feature_matcher = FeatureMatchLightGluePair(cam_data=camera_data,
#                                             detector="superpoint")
# feature_matcher = FeatureMatchSuperGluePair(cam_data=camera_data,
#                                             detector='superpoint', 
#                                             setting='indoor')
# feature_tracker = FeatureMatchFlannTracking(cam_data=camera_data, 
#                                             detector="sift",
#                                             RANSAC_threshold=0.02,
#                                             lowes_thresh=0.75)
feature_tracker = FeatureMatchBFTracking(cam_data=camera_data,
                                         detector="sift",
                                         RANSAC_threshold=0.02,
                                         lowes_thresh=0.75)

feature_matcher = FeatureMatchBFPair(detector="sift", 
                                     cam_data=camera_data,
                                     cross_check=False,
                                     RANSAC_threshold=0.02,
                                     lowes_thresh=0.70)

# feature_matcher = FeatureMatchFlannPair(detector="sift", 
#                                      cam_data=camera_data,
#                                      RANSAC_threshold=0.03,
#                                      lowes_thresh=0.68)
# feature_matcher = FeatureMatchRoMAPair(img_path=image_path, setting="outdoor")
# feature_tracker = FeatureMatchSuperGlueTracking(cam_data=camera_data,
#                                                 detector='superpoint', 
#                                                 setting='indoor',
#                                                 RANSAC_threshold=0.01)

# Solution Pipeline 
detected_features = feature_detector()
# print(detected_features)

# matched_features = feature_matcher(detected_features)
# matched_features = feature_matcher()

# tracked_features = feature_tracker(detected_features)

tracks = True
if tracks: 
    tracked_features = feature_tracker(detected_features)
    image_path = sorted(glob.glob(image_path + "\\*"))

    j = 0
    for i in range(1000):
        if (tracked_features.access_point3D(i).shape[0] > 4):
            if j < 3: 
                j += 1
                continue
            print(f"Point {i}:", tracked_features.access_point3D(i))
            print()
            fig,ax = plt.subplots(1, tracked_features.access_point3D(i).shape[0], figsize=(30, 15))

            for j in range(tracked_features.access_point3D(i).shape[0]):
                img = cv2.resize(cv2.imread(image_path[j]), tracked_features.image_size, interpolation=cv2.INTER_AREA)
                #img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax[j].imshow(img)
                xx = tracked_features.access_point3D(i)[j, 1]
                yy = tracked_features.access_point3D(i)[j, 2]
                circ = Circle((xx,yy),20)
                ax[j].add_patch(circ)
            
            plt.tight_layout()
            plt.show()

            break

else: 
    pair = 4

    matched_features = feature_matcher(detected_features)

    def plot_matches(img1, img2, feat1, feat2):
    
        # Stack images horizontally
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        h_max = max(h1, h2)
        canvas = np.zeros((h_max, w1 + w2, 3), dtype=np.uint8)
        
        # Make sure both are color for plotting
        def to_color(im):
            return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) if im.ndim == 2 else im
        canvas[:h1, :w1] = to_color(img1)
        canvas[:h2, w1:w1+w2] = to_color(img2)
        
        # Draw circles and connecting lines
        for i in range(feat1.shape[0]):
            x1, y1 = feat1[i, :]
            x2, y2 = feat2[i, :]
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2) + w1, int(y2))
            #print(pt1)
            cv2.circle(canvas, pt1, 5, (0,255,0), -1)
            cv2.circle(canvas, pt2, 5, (0,255,0), -1)
            cv2.line(canvas, pt1, pt2, (255,0,0), 1)
        
        plt.figure(figsize=(10,5))
        plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    image_path = sorted(glob.glob(image_path + "\\*"))

    pt1, pt2 = matched_features.access_matching_pair(pair)

    img1 = cv2.resize(cv2.imread(image_path[pair]), matched_features.image_size, interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(cv2.imread(image_path[pair+1]), matched_features.image_size, interpolation=cv2.INTER_AREA)

    plot_matches(img1, img2, pt1, pt2)
