import numpy as np
import cv2
from baseclass import CameraPoseEstimatorClass
from tqdm import tqdm
from .DataTypes.datatype import Points2D, Calibration, Points3D, CameraPose


class CamPoseEstimatorEssentialToPnP(CameraPoseEstimatorClass):
    def __init__(self, calibration: Calibration, image_path: str):
        super().__init__(calibration, image_path)

        self.module_name = "CamPoseEstimatorEssentialToPnP"
        self.description = ""
        self.example = ""

        # Assume Basic Flann Matcher
        

    def __call__(self, features: list[Points2D] | None = None):
         
        pass
    
    

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

    def two_view_triangulation(self, pose_1: np.ndarray, pose_2: np.ndarray, pts1: Points2D, pts2: Points2D) -> np.ndarray:
        proj_1 = self.K1 @ pose_1
        proj_2 = self.K1 @ pose_2

        cloud = cv2.triangulatePoints(proj_1,proj_2, pts1.points2D, pts2.points2D)
        cloud/=cloud[3]
        
        return cloud