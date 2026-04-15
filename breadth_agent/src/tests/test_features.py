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
from matplotlib.collections import LineCollection

def plot_features(img_path: str, pts:np.ndarray, image_size):
    fig, ax = plt.subplots(figsize=(8, 8))
    img = cv2.resize(cv2.imread(img_path), image_size, interpolation=cv2.INTER_AREA)
    #img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)

    
    # Draw keypoints
    ax.scatter(pts[:, 0], pts[:, 1], c='r', s=5)
    # ax.scatter(pts2_offset[:, 0], pts2_offset[:, 1], c=point_color, s=5)
    
    plt.tight_layout()
    plt.axis('off')
    # plt.show()
    plt.savefig(os.path.dirname(os.path.abspath(__file__)) + '/feature_images/detected_features.png')

# Construct Modules with Initialized Arguments
# image_path = "C:\\Users\\Anthony\\Documents\\Projects\datasets\\Structure-from-Motion\\sfm_dataset"
# calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\Structure-from-Motion\\calibration_new.npz"
# image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\scan6_illumination_change"
# calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\calibration_DTU_new.npz"
image_path = "/home/anthonyq/datasets/DTU/DTU/scan22"
calibration_path = "/home/anthonyq/datasets/DTU/DTU/calibration_DTU_new.npz"
# image_path = "/home/anthonyq/datasets/tanks_and_temples/Francis"
# calibration_path = "/home/anthonyq/datasets/tanks_and_temples/calibration_new_1920.npz"

camera_data = CameraDataManager(image_path=image_path,
                                max_images = 5,
                                calibration_path=calibration_path).get_camera_data()
# Feature Module Initialization
# calibration_data = CalibrationReader(calibration_path).get_calibration()
# feature_detector = FeatureDetectionSIFT(cam_data=camera_data,
#                                         max_keypoints=12000)
# feature_detector = FeatureDetectionORB(cam_data=camera_data, 
#                                         max_keypoints=20000,
#                                         fast_threshold=20,
#                                         # set_nms=True,
#                                         # set_nms_allowed_points=12000,
#                                         # set_nms_tolerance = 0.2
#                                         )

feature_detector = FeatureDetectionSP(cam_data=camera_data, 
                                        max_keypoints=5000)
# feature_matcher = FeatureMatchLightGluePair(cam_data=camera_data,
#                                             detector="superpoint")
# feature_matcher = FeatureMatchSuperGluePair(cam_data=camera_data,
#                                             detector='superpoint', 
#                                             setting='indoor')
feature_tracker = FeatureMatchLightGlueTracking(cam_data=camera_data, 
                                                detector="sift",
                                                RANSAC_threshold=0.015,
                                                RANSAC_conf=0.999)
# feature_tracker = FeatureMatchFlannTracking(cam_data=camera_data, 
#                                             detector="sift",
#                                             lowes_thresh=0.70,
#                                             RANSAC_threshold=0.015,
#                                             RANSAC_conf=0.999)
# feature_tracker = FeatureMatchBFTracking(cam_data=camera_data,
#                                          detector="sift",
#                                          k=2,
#                                          cross_check=False,
#                                          lowes_thresh=0.70,
#                                          RANSAC_threshold=0.015,
#                                          RANSAC_conf=0.999)

feature_matcher = FeatureMatchLightGluePair(cam_data=camera_data,
                                            detector="superpoint",
                                            RANSAC_threshold=0.02,
                                            RANSAC_conf=0.999)

# feature_matcher = FeatureMatchSuperGluePair(cam_data=camera_data,
#                                             detector="superpoint",
#                                             RANSAC_threshold=0.02,
#                                             RANSAC_conf=0.999)
# feature_matcher = FeatureMatchBFPair(detector="sift", 
#                                      cam_data=camera_data,
#                                      cross_check=False,
#                                      RANSAC_threshold=0.05,
#                                      lowes_thresh=0.750)

# feature_matcher = FeatureMatchFlannPair(cam_data=camera_data,
#                                         detector="sift",
#                                         k=2,
#                                         lowes_thresh=0.78,
#                                         RANSAC=True,
#                                         RANSAC_threshold=0.02,
#                                         RANSAC_conf=0.999)
# feature_matcher = FeatureMatchRoMAPair(img_path=image_path, setting="outdoor")
# feature_tracker = FeatureMatchSuperGlueTracking(cam_data=camera_data,
#                                                 detector='superpoint', 
#                                                 setting='indoor',
#                                                 RANSAC_threshold=0.01)

# Solution Pipeline 
detected_features = feature_detector()
# print(detected_features)


tracks = False
vis = True
server = True
pair = True
# matched_features = feature_matcher(detected_features)

# Read Images
image_path = sorted(glob.glob(image_path + "/*"))

plot_features(image_path[0], detected_features[0].points2D, detected_features[0].image_size)

if tracks and vis: 
    tracked_features = feature_tracker(detected_features)
    # image_path = sorted(glob.glob(image_path + "/*"))

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
                circ = Circle((xx,yy),80, color='r')
                ax[j].add_patch(circ)
                ax[j].axis('off')
            plt.tight_layout()
            # plt.axis('off')
            # plt.show()
            plt.savefig(os.path.dirname(os.path.abspath(__file__)) + '/feature_images/tracked_feature.png')

            break

elif vis: 
    pair = 0

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
    
    def plot_matches_server(img1, img2, pts1, pts2, max_lines=None, line_color='green', 
                 point_color='red', linewidth=0.5, alpha=0.7):
        """
        Visualize matching feature points between two images.

        Args:
            img1, img2: HxW or HxWx3 numpy arrays
            pts1, pts2: Nx2 arrays of (x, y) coordinates
            max_lines: optional int to subsample matches for clarity
            line_color: color of match lines
            point_color: color of keypoints
        """

        assert pts1.shape == pts2.shape, "Point arrays must match"
        
        # Optionally subsample matches (important for large N)
        if max_lines is not None and pts1.shape[0] > max_lines:
            idx = np.random.choice(pts1.shape[0], max_lines, replace=False)
            pts1 = pts1[idx]
            pts2 = pts2[idx]

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # Create combined canvas
        H = max(h1, h2)
        W = w1 + w2
        canvas = np.zeros((H, W, 3), dtype=img1.dtype)

        canvas[:h1, :w1] = img1
        canvas[:h2, w1:w1+w2] = img2

        # Offset pts2 x-coordinates
        pts2_offset = pts2.copy()
        pts2_offset[:, 0] += w1

        # Create line segments
        lines = np.stack([pts1, pts2_offset], axis=1)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(canvas)
        ax.axis('off')

        # Draw lines efficiently
        lc = LineCollection(lines, colors=line_color, linewidths=linewidth, alpha=alpha)
        ax.add_collection(lc)

        # Draw keypoints
        ax.scatter(pts1[:, 0], pts1[:, 1], c=point_color, s=5)
        ax.scatter(pts2_offset[:, 0], pts2_offset[:, 1], c=point_color, s=5)

        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.dirname(os.path.abspath(__file__)) + '/feature_images/my_figure.png')

    pt1, pt2 = matched_features.access_matching_pair(pair)

    img1 = cv2.cvtColor(cv2.resize(cv2.imread(image_path[pair]), matched_features.image_size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.resize(cv2.imread(image_path[pair+1]), matched_features.image_size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)

    if server:
        plot_matches_server(img1, img2, pt1, pt2)
    else:
        plot_matches(img1, img2, pt1, pt2)
