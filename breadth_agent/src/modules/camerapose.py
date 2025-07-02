import numpy as np
import cv2
from baseclass import CameraPoseEstimatorClass
from tqdm import tqdm
from .DataTypes.datatype import Points2D, Calibration, Points3D, CameraPose


class CamPoseEstimatorEssentialToPnP(CameraPoseEstimatorClass):
    def __init__(self, calibration: Calibration, image_path: str, detector: str):
        super().__init__(calibration, image_path)

        self.module_name = "CamPoseEstimatorEssentialToPnP"
        self.description = f"""
Estimates the camera pose for each frame in a set of images for a monocular camera. The
process of this module is to estimate the essential matrix for the first pair of images 
of the image set, recover the camera pose that's up-to-scale, then use the PnP algorithm
to estimate the rest of the camera's pose for each following images so each trajectory is 
in scale to the first pose estimation. Use this module for Monocular cameras that are 
calibrated for a given image set.
"""

        self.example = f"""
Initialization: 
image_path = ... # Path to set of images

calibration_data = CalibrationReader(calibration_path).get_calibration() # Used to get Calibration Data

CamPoseEstimatorEssentialToPnP(calibration=calibration_data, image_path=image_path, detector="sift") # Detector is determined by feature module initialization for feature detector

Function call:  
features = feature_detector() # Call Feature Detector Module on image frames

pose_estimator(features=features) # Features used from Feature Detector Module
"""

        # Assume Basic Flann Matcher -TODO: REMOVE THIS PORTION OF THE CODE - TAKE IN PAIRED FEATURE MATCHING DATA
        self.detector = detector.lower()
        if self.detector ==  "sift":
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

        

    def __call__(self, features: list[Points2D] | None = None) -> CameraPose:
         
        # Get First set of camera poses (Initial and 2nd Camera)
        pts1, pts2 = self.match_pairs(features[0], features[1])

        cam_poses = self.estimate_first_pair(pts1, pts2) # First two poses defined here
        self.cam_poses = cam_poses

        cloud = self.two_view_triangulation(cam_poses.camera_pose[0], cam_poses.camera_pose[1], pts1, pts2)

        for i in tqdm(range(len(features) - 2)):

            if i > 0:
                cloud = self.two_view_triangulation(pose1, pose2, pts1, pts2)

            pts3_t = features[i+2]
            pts2_3, pts3 = self.match_pairs(features[i + 1], pts3_t)

            index, pts2_3_com, pts3_com, pts2_3_new, pts3_new = self.three_view_tracking(pts2.points2D, pts2_3.points2D, pts3.points2D)

            new_pose = self.estimate_pose_pnp(cloud[index], pts2_3_com, pts3_com, self.cam_poses.camera_pose[-1])

            pose1 = self.cam_poses.camera_pose[-1]
            pose2 = new_pose
            pts1 = pts2_3
            pts2 = pts3

            self.cam_poses.camera_pose.append(new_pose)

        return self.cam_poses
    
    def match_pairs(self, pts1: Points2D, pts2: Points2D) -> tuple[Points2D, Points2D]:  
        idx1, idx2 = self.matcher_parser(pts1.descriptors, pts2.descriptors)

        # print(len(idx1), len(idx2))
        new_pt1 = Points2D(**pts1.splice_2D_points(idx1))
        new_pt2 = Points2D(**pts2.splice_2D_points(idx2))

        _, mask = cv2.findFundamentalMat(new_pt1.points2D, new_pt2.points2D, cv2.FM_LMEDS)

        # Could update points2D to inlier points with Mask
        inlier_pts1 = Points2D(**new_pt1.set_inliers(mask))
        inlier_pts2 = Points2D(**new_pt2.set_inliers(mask))


        return inlier_pts1, inlier_pts2

    def matcher_parser(self, desc1: np.ndarray, desc2: np.ndarray) -> tuple[list, list]:
        knn = False
      
        if self.detector == "sift":
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
            knn = True
        else:
            matches = self.matcher.match(desc1,desc2)
                
        if self.detector == "sift":
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
    
    def three_view_tracking(self, pts2: np.ndarray, pts2_3: np.ndarray, pts3: np.ndarray):
        #pts2 is the set of keypoints obtained from image(n-1) and image(n)
        #pts2_3 and pts3 are the set of keypoints obtained from image(n) and image(n+1)
        
        # Finding Commmon Points
        index1=[]
        index2=[]
        for i in range(pts2.shape[0]):
            if (pts2[i,:] == pts2_3).any():
                index1.append(i)

            idx2_3 = np.where(pts2_3 == pts2[i,:])[0]

            if idx2_3.size != 0:
                index2.append(idx2_3[0])
        
        #Finding New Points
        pts3_new=[]
        pts2_3_new=[]
        
        for i in range(pts3.shape[0]):
            if i not in index2:
                pts3_new.append(list(pts3[i,:]))
                pts2_3_new.append(list(pts2_3[i,:]))
        
        index1=np.array(index1)
        index2=np.array(index2)
        pts2_3_common=pts2_3[index2]
        pts3_common=pts3[index2]
                
        return index1,pts2_3_common,pts3_common,np.array(pts2_3_new),np.array(pts3_new)


    def estimate_first_pair(self, pts1: Points2D, pts2: Points2D) -> CameraPose:
        cam_poses = CameraPose()

        initial_pose = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
        cam_poses.camera_pose.append(initial_pose)

        E, mask = cv2.findEssentialMat(pts1.points2D, pts2.points2D, self.K1)

        _, R, T, _ = cv2.recoverPose(points1 = pts2.points2D, points2=pts1.points2D, cameraMatrix=self.K1, E = E)

        new_pose = np.hstack((R, T.reshape(3, 1)))
        cam_poses.camera_pose.append(new_pose)

        return cam_poses

    def estimate_pose_pnp(self, point_cloud: np.ndarray, pts1: np.ndarray, pts2: np.ndarray, prev_pose: np.ndarray) -> np.ndarray:
        _,rot,trans,_=cv2.solvePnPRansac(point_cloud, pts2, self.K1, self.dist1, cv2.SOLVEPNP_ITERATIVE)
        
        rot,_=cv2.Rodrigues(rot)

        new_pose = np.hstack((rot, trans))

        # new_pose = np.empty((3,4))
        # new_pose[:3,:3] = rot @ self.cam_poses.camera_pose[-1][:3,:3]
        # new_pose[:3,3]  = self.cam_poses.camera_pose[-1][:3, 3] + self.cam_poses.camera_pose[-1][:3, :3] @ trans

        # self.cam_poses.camera_pose.append(new_pose)

        return new_pose
        

    def two_view_triangulation(self, pose_1: np.ndarray, pose_2: np.ndarray, pts1: Points2D, pts2: Points2D) -> np.ndarray:
        proj_1 = self.K1 @ pose_1
        proj_2 = self.K1 @ pose_2


        cloud = cv2.triangulatePoints(proj_1, proj_2, pts1.points2D.T, pts2.points2D.T)
        cloud/=cloud[3]

        cloud=cv2.convertPointsFromHomogeneous(cloud.T)
        
        return cloud