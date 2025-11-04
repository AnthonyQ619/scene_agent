import numpy as np
import cv2
import os
from tqdm import tqdm
import torch

from baseclass import CameraPoseEstimatorClass
from .DataTypes.datatype import Points2D, Calibration, Points3D, CameraPose, PointsMatched, CameraData
from models.sfm_models.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from models.sfm_models.vggt.utils.geometry import unproject_depth_map_to_point_map
from models.sfm_models.vggt.models.vggt import VGGT
from models.sfm_models.vggt.utils.load_fn import load_and_preprocess_images

##########################################################################################################
############################################### ML MODULES ###############################################

class CamPoseEstimatorVGGTModel(CameraPoseEstimatorClass):
    def __init__(self, image_path: str, calibration: Calibration | None = None):
        super().__init__(image_path=image_path, calibration=calibration)

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
- image_path (str): the image path in which the images are stored and to utilize for scene building
- calibration: Data type that stores the camera's calibration data initialized from the calibration 
reader module
    - Default (Calibration): None (assume's no calibration data, and will estimate in model instead)

Function Call Parameters:
- None
"""

        self.example = f"""
Initialization: 
image_path = ... # Path to set of images
calibration_data = CalibrationReader(calibration_path).get_calibration() # Used to get Calibration Data

# With calibration provided
pose_estimator = CamPoseEstimatorVGGTModel(image_path = image_path, calibration = calibration_data)

# Without calibration provided
pose_estimator = CamPoseEstimatorVGGTModel(image_path = image_path)

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

        self.images = load_and_preprocess_images(self.image_path).to(device)

    def __call__(self, features: list[Points2D] | None = None) -> CameraPose:
        cam_poses = CameraPose()

        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=self.dtype):
                self.images = self.images[None]  # add batch dimension
                aggregated_tokens_list, ps_idx = self.model.aggregator(self.images)

            img_shape = self.images.shape
            # Predict Cameras
            pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
            # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, self.images.shape[-2:])
            
            intrinsic_np = intrinsic.squeeze(0).detach().cpu().numpy()
            extrinsic_np = extrinsic.squeeze(0).detach().cpu().numpy()

            for i in range(extrinsic_np.shape[0]):
                cam_poses.camera_pose.append(extrinsic_np[i, :, :])
                cam_poses.rotations.append(extrinsic_np[i, :, :3])
                cam_poses.translations.append(extrinsic_np[i, :, 3:])
            
            # Store Intrinsics -> Reset camerapose to multi_cam approach
            intrins = []
            dists = []  # Assume Camera image were undistorted for now

            for i in range(intrinsic_np.shape[0]):
                intrins.append(intrinsic_np[i, :, :])
                dists.append(np.zeros((1,5)))

            self.calibration.setup_multi_cam(intrins, dists)

        print("Image Shape", img_shape)
        return cam_poses
                

###########################################################################################################
############################################ CLASSICAL MODULES ############################################

class CamPoseEstimatorEssentialToPnP(CameraPoseEstimatorClass):
    def __init__(self, cam_data: CameraData,
                 iteration_count: int = 200,
                 reprojection_error: float = 3.0,
                 confidence: float = 0.99):
        super().__init__(camera_data = cam_data)

        self.module_name = "CamPoseEstimatorEssentialToPnP"
        self.description = f"""
Estimates the camera pose for each frame in a set of images for a monocular camera. The
process of this module is to estimate the essential matrix for the first pair of images 
of the image set, recover the camera pose that's up-to-scale, then use the PnP algorithm
to estimate the rest of the camera's pose for each following images so each trajectory is 
in scale to the first pose estimation. Use this module for Monocular cameras that are 
calibrated for a given image set.

Initialization Parameters:
- calibration (Calibration): Data type that stores the camera's calibration data initialized from the 
calibration reader module
- image_path (str): The image path in which the images are stored and to utilize for scene building

Function Call Parameters:
- features_pairs (PointsMatched): Data Type containing the detected feature correspondences of image pairs
estimated from the feature matcher modules.
"""

        self.example = f"""
Initialization: 
image_path = ... # Path to set of images
calibration_data = CalibrationReader(calibration_path).get_calibration() # Used to get Calibration Data

CamPoseEstimatorEssentialToPnP(image_path=image_path, calibration=calibration_data) 

Function call:  
features = feature_detector() # Call Feature Detector Module on image frames

feature_pairs = feature_matcher(features) # Call Feature Matcher Module on detected features

pose_estimator(feature_pairs=feature_pairs) # Features used from Feature Detector Module
"""
        self.reproj_error = reprojection_error
        self.iteration_ct = iteration_count
        self.confidence = confidence
        

    def __call__(self, features_pairs: PointsMatched | None = None) -> CameraPose:
        assert(features_pairs.multi_view == False), "Features passed must be two view correspondences. Ensure to invoke Feature Matching Two View tools prior to this call."
        # Update Calibration if necessary for image resizing/shaping from feature detectors
        # self.calibration.update_cal_img_shape(features_pairs.image_scale)
        # print(features_pairs.image_scale)
        # self._setup_calibration(features_pairs.image_scale) # Update/setup calibration info

        # Get First set of camera poses (Initial and 2nd Camera)
        pts1, pts2 = features_pairs.access_matching_pair(0)

        cam_poses = self.estimate_first_pair(pts1, pts2) # First two poses defined here
        self.cam_poses = cam_poses

        cloud = self.two_view_triangulation(cam_poses.camera_pose[0], cam_poses.camera_pose[1], pts1, pts2)

        # for i in tqdm(range(len(features) - 2)):
        for i in tqdm(range(1, len(features_pairs.pairwise_matches))):
            
            if i > 1:
                cloud = self.two_view_triangulation(pose1, pose2, pts1, pts2)

            # pts3_t = features[i+2]
            # pts2_3, pts3 = self.match_pairs(features[i + 1], pts3_t)
            pts2_3, pts3 = features_pairs.access_matching_pair(i)

            index, pts2_3_com, pts3_com, pts2_3_new, pts3_new = self.three_view_tracking(pts2, pts2_3, pts3)

            # print("Prev PAIR", pts2_3_com.shape)
            # print("Current POINTS", pts3_com.shape)
            new_pose = self.estimate_pose_pnp(cloud[index], pts2_3_com, pts3_com, self.cam_poses.camera_pose[-1])

            pose1 = self.cam_poses.camera_pose[-1]
            pose2 = new_pose
            pts1 = pts2_3
            pts2 = pts3

            self.cam_poses.camera_pose.append(new_pose)

        return self.cam_poses
    
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


    def estimate_first_pair(self, pts1: np.ndarray, pts2: np.ndarray) -> CameraPose: #pts1: Points2D, pts2: Points2D) -> CameraPose:
        cam_poses = CameraPose()

        initial_pose = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
        cam_poses.camera_pose.append(initial_pose)

        # E, mask = cv2.findEssentialMat(pts1.points2D, pts2.points2D, self.K1)

        # _, R, T, _ = cv2.recoverPose(points1 = pts2.points2D, points2=pts1.points2D, cameraMatrix=self.K1, E = E)

        E, mask = cv2.findEssentialMat(pts1, pts2, self.K_mat, method=cv2.RANSAC, prob=0.999, threshold=0.3)

        _, R, T, _ = cv2.recoverPose(points1 = pts2, points2=pts1, cameraMatrix=self.K_mat, E = E)

        new_pose = np.hstack((R, T.reshape(3, 1)))
        cam_poses.camera_pose.append(new_pose)

        return cam_poses

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
        # proj_1 = self.K1 @ pose_1
        # proj_2 = self.K1 @ pose_2

        #print(pts1.shape)
        pt1 = cv2.undistortPoints(pts1.T, self.K_mat, self.dist)
        pt2 = cv2.undistortPoints(pts2.T, self.K_mat, self.dist)
        
        P1mtx = np.eye(3) @ pose_1
        P2mtx = np.eye(3) @ pose_2


        # cloud = cv2.triangulatePoints(proj_1, proj_2, pts1.T, pts2.T)
        cloud = cv2.triangulatePoints(P1mtx, P2mtx, pt1, pt2)
        cloud/=cloud[3]

        cloud=cv2.convertPointsFromHomogeneous(cloud.T)
        
        return cloud