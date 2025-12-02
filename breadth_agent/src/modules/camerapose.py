import numpy as np
import cv2
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms as TF

from modules.baseclass import CameraPoseEstimatorClass
from modules.DataTypes.datatype import Points2D, Calibration, Points3D, CameraPose, PointsMatched, CameraData
from modules.models.sfm_models.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from modules.models.sfm_models.vggt.utils.geometry import unproject_depth_map_to_point_map
from modules.models.sfm_models.vggt.models.vggt import VGGT
from modules.models.sfm_models.vggt.utils.load_fn import load_and_preprocess_images

import glob
##########################################################################################################
############################################### ML MODULES ###############################################

class CamPoseEstimatorVGGTModel(CameraPoseEstimatorClass):
    def __init__(self, 
                 cam_data: CameraData):
        
        super().__init__(cam_data = cam_data)

        self.module_name = "CamPoseEstimatorVGGTModel"
        self.description = f"""
Estimates the camera pose for each frame in a set of images for a monocular camera. The 
process of this module is to estimate the camera pose utilizing the Visual Geometry 
Grounded Transformer (VGGT) Model, a feed-forward neural network that directly infers 
all key 3D attributes of a scene, including camera parameters, point maps, depth maps, 
and 3D point tracks, from one, a few, or hundreds of its views. However, this module
only utilizes the pose estimation feature with intrinsic estimation. This module can estimate
the camera poses from just images alone, without features needing to be detected prior.

Utilize this module in cases where images do not have extreme overlap, scale is needed for a 
monocular camera setup, or runtime is not extremely necessary for computation.

Initialization Parameters:
- cam_data: Data container to hold images and calibration data, read from the CameraDataManager.

Function Call Parameters:
- None

Module Input:
    None
    
Module Output: 
    CameraPose:
        camera_pose: list[np.ndarray]   [3 x 4] (np.float) Camera pose for each corresponding frame. Each pose is 3x4 (R, T)
        rotations: list[np.ndarray]     [3 x 3] (np.float) Rotation matrices for each corresponding frame (Derived from camera_pose)
        translations: list[np.ndarray]  [3 x 1] (np.float) Translation matrices for each corresponding frame (Derived from camera_pose)
"""

        self.example = f"""
Initialization: 
pose_estimator = CamPoseEstimatorVGGTModel(cam_data = camera_data)

Function call:  
pose_estimator() # No Features used with this module
"""

        # Initialize Model
        WEIGHT_MODULE = str(os.path.dirname(__file__)) + "\\models\\sfm_models\\vggt\\weights\\model.pt"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        self.model = VGGT().to(device)
        self.model.load_state_dict(torch.load(WEIGHT_MODULE, weights_only=True))

        # Load Images in correct format for VGGT inference
        to_tensor = TF.ToTensor()
        tensor_img_list = []
        for ind in range(len(self.image_list)):
            tensor_img_list.append(to_tensor(self.image_list[ind]))

        self.images = torch.stack(tensor_img_list).to(device) 

        self.img_shape = self.image_list[0].shape[:2] # Images 

    # def __call__(self, features: list[Points2D] | None = None) -> CameraPose:
    def _estimate_camera_poses(self,
                               camera_poses: CameraPose, 
                               feature_pairs: PointsMatched) -> CameraPose:
        
        assert self.img_shape[0] == self.img_shape[1], "Input images must be square size, or Height must equal Width. Must reshape images to a square size, such as (1024, 1024)"
        # return super()._estimate_camera_poses(camera_poses, feature_pairs)
        # cam_poses = CameraPose()

        # VGGT Fixed Resolution to 518 for Inference
        images = F.interpolate(self.images, size=(518, 518), mode="bilinear", align_corners=False)
        new_scale = self.img_shape[0] / 518 # Get change of scale from old shape to new smaller shape

        # images = self.images
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=self.dtype):
                images = images[None]  # add batch dimension
                aggregated_tokens_list, ps_idx = self.model.aggregator(images)

            img_shape = images.shape
            # Predict Cameras
            pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
            # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            
            intrinsic_np = intrinsic.squeeze(0).detach().cpu().numpy()
            extrinsic_np = extrinsic.squeeze(0).detach().cpu().numpy()

            for i in range(extrinsic_np.shape[0]):
                camera_poses.camera_pose.append(extrinsic_np[i, :, :])
                camera_poses.rotations.append(extrinsic_np[i, :, :3])
                camera_poses.translations.append(extrinsic_np[i, :, 3:])
            
            # Store Intrinsics -> Reset camerapose to multi_cam approach
            intrins = []
            dists = []   # Assume Camera image were undistorted for now

            intrinsic_np[:, :2, :] *=  new_scale
            print(new_scale)
            for i in range(intrinsic_np.shape[0]):
                intrins.append(intrinsic_np[i, :, :])
                dists.append(np.zeros((1,5)))

        self.cam_data.apply_new_calibration(intrins, dists)

        # print("Image Shape", img_shape)
        # print(camera_poses.camera_pose)
        torch.cuda.empty_cache() #Empty GPU cache
        return camera_poses
                
###########################################################################################################
############################################ CLASSICAL MODULES ############################################

class CamPoseEstimatorEssentialToPnP(CameraPoseEstimatorClass):
    def __init__(self, 
                 cam_data: CameraData,
                 iteration_count: int = 200,
                 reprojection_error: float = 3.0,
                 confidence: float = 0.99):
        super().__init__(cam_data = cam_data)

        self.module_name = "CamPoseEstimatorEssentialToPnP"
        self.description = f"""
Estimates the camera pose for each frame in a set of images for a monocular camera. The
process of this module is to estimate the essential matrix for the first pair of images 
of the image set, recover the camera pose that's up-to-scale, then use the PnP algorithm
to estimate the rest of the camera's pose for each following images so each trajectory is 
in scale to the first pose estimation. Use this module for Monocular cameras that are 
calibrated for a given image set.

Initialization Parameters:
- cam_data: Data container to hold images and calibration data, read from the CameraDataManager.
- iteration_count: Number of iterations to run the Levenberg-Marquardt algorithm for Pose Estimation with PnP
    - Default (int): 200,
- reprojection_error: Inlier threshold value used by the RANSAC procedure. The parameter value is the maximum allowed distance between the observed and computed point projections to consider it an inlier.
    - Default (float): 3.0
- confidence: The probability that the algorithm produces a useful result. 
    - Default (float): 0.99

Function Call Parameters:
- features_pairs (PointsMatched): Data Type containing the detected feature correspondences of image pairs
estimated from the feature matcher modules.

Module Input:
    PointsMatched (Matched Features across image pairs)
        pairwise_matches: list[np.ndarray]  [N x 4] -> [x1, y1, x2, y2]. Data Structure to store Pairwise feature matches.
        multi_view: bool                    Determine if Pairwise/Multi-View Feature Matching (Should be False for Pairwise in this function)
        image_size: np.ndarray              [1 x 2] [np.int64] (Simply Image Shape: (W, H))
        image_scale: list[float]            [W_scale, H_scale] if image is resized
    
Module Output: 
    CameraPose:
        camera_pose: list[np.ndarray]   [3 x 4] (np.float) Camera pose for each corresponding frame. Each pose is 3x4 (R, T)
        rotations: list[np.ndarray]     [3 x 3] (np.float) Rotation matrices for each corresponding frame (Derived from camera_pose)
        translations: list[np.ndarray]  [3 x 1] (np.float) Translation matrices for each corresponding frame (Derived from camera_pose)
"""

        self.example = f"""
Initialization: 
CamPoseEstimatorEssentialToPnP(cam_data = camera_data, calibration=calibration_data) 

Function call:  
features = feature_detector() # Call Feature Detector Module on image frames

feature_pairs = feature_matcher(features) # Call Feature Matcher Module on detected features

camera_pose = pose_estimator(feature_pairs=feature_pairs) # Features used from Feature Detector Module
"""
        self.reproj_error = reprojection_error
        self.iteration_ct = iteration_count
        self.confidence = confidence
        

    def _estimate_camera_poses(self,
                               camera_poses: CameraPose,
                               feature_pairs: PointsMatched) -> CameraPose:
        assert(feature_pairs.multi_view == False), "Features passed must be two view correspondences. Ensure to invoke Feature Matching Two View tools prior to this call."
       
        # Get First set of camera poses (Initial and 2nd Camera)
        pts1, pts2 = feature_pairs.access_matching_pair(0)

        # cam_poses = self.estimate_first_pair(pts1, pts2) # First two poses defined here
        # self.cam_poses = cam_poses
        self.estimate_first_pair(pts1, pts2, camera_poses) # First two poses defined here

        cloud = self.two_view_triangulation(camera_poses.camera_pose[0], camera_poses.camera_pose[1], pts1, pts2)

        # for i in tqdm(range(len(features) - 2)):
        for i in tqdm(range(1, len(feature_pairs.pairwise_matches)), 
                      desc='Estimating Camera Poses'):
            
            if i > 1:
                cloud = self.two_view_triangulation(pose1, pose2, pts1, pts2)

            # pts3_t = features[i+2]
            # pts2_3, pts3 = self.match_pairs(features[i + 1], pts3_t)
            pts2_3, pts3 = feature_pairs.access_matching_pair(i)

            index, pts2_3_com, pts3_com, pts2_3_new, pts3_new = self.three_view_tracking(pts2, pts2_3, pts3)

            # print("Prev PAIR", pts2_3_com.shape)
            # print("Current POINTS", pts3_com.shape)
            new_pose = self.estimate_pose_pnp(cloud[index], pts2_3_com, pts3_com, camera_poses.camera_pose[-1])

            pose1 = camera_poses.camera_pose[-1]
            pose2 = new_pose
            pts1 = pts2_3
            pts2 = pts3

            camera_poses.camera_pose.append(new_pose)

        return camera_poses

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


    def estimate_first_pair(self, pts1: np.ndarray, pts2: np.ndarray, camera_poses: CameraPose) -> CameraPose: #pts1: Points2D, pts2: Points2D) -> CameraPose:
        # cam_poses = CameraPose()

        initial_pose = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]], dtype=float)
        camera_poses.camera_pose.append(initial_pose)

        # E, mask = cv2.findEssentialMat(pts1.points2D, pts2.points2D, self.K1)

        # _, R, T, _ = cv2.recoverPose(points1 = pts2.points2D, points2=pts1.points2D, cameraMatrix=self.K1, E = E)

        E, mask = cv2.findEssentialMat(pts1, pts2, self.K_mat, method=cv2.RANSAC, prob=0.999, threshold=0.3)

        _, R, T, _ = cv2.recoverPose(points1 = pts2, points2=pts1, cameraMatrix=self.K_mat, E = E)

        new_pose = np.hstack((R, T.reshape(3, 1)))
        camera_poses.camera_pose.append(new_pose)

        # return cam_poses

    def estimate_pose_pnp(self, point_cloud: np.ndarray, pts1: np.ndarray, pts2: np.ndarray, prev_pose: np.ndarray) -> np.ndarray:
        #cv2.solvePnPRansac(point_cloud, pts2, self.K1, self.dist1, cv2.SOLVEPNP_ITERATIVE)
        _,rot,trans,_= cv2.solvePnPRansac(objectPoints=point_cloud, 
                                          imagePoints=pts2, 
                                          cameraMatrix=self.K_mat, 
                                          distCoeffs=self.dist, 
                                          useExtrinsicGuess=False,
                                          reprojectionError= self.reproj_error,
                                          iterationsCount= self.iteration_ct,
                                          confidence=self.confidence,
                                          flags=cv2.SOLVEPNP_ITERATIVE)
        
        rot,_=cv2.Rodrigues(rot)

        new_pose = np.hstack((rot, trans))

        # new_pose = np.empty((3,4))
        # new_pose[:3,:3] = rot @ self.cam_poses.camera_pose[-1][:3,:3]
        # new_pose[:3,3]  = self.cam_poses.camera_pose[-1][:3, 3] + self.cam_poses.camera_pose[-1][:3, :3] @ trans

        # self.cam_poses.camera_pose.append(new_pose)

        return new_pose
        

    def two_view_triangulation(self, pose_1: np.ndarray, pose_2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:

        # Normalize Points
        pt1 = cv2.undistortPoints(pts1.T, self.K_mat, self.dist)
        pt2 = cv2.undistortPoints(pts2.T, self.K_mat, self.dist)
        
        P1mtx = np.eye(3) @ pose_1
        P2mtx = np.eye(3) @ pose_2

        # cloud = cv2.triangulatePoints(proj_1, proj_2, pts1.T, pts2.T)
        cloud = cv2.triangulatePoints(P1mtx, P2mtx, pt1, pt2)
        cloud/=cloud[3]

        cloud=cv2.convertPointsFromHomogeneous(cloud.T)
        
        return cloud