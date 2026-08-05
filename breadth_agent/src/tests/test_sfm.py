from sfmcore.features import FeatureDetectionSP
from sfmcore.featurematching import FeatureMatchSuperGluePair
# from modules.camerapose import CamPoseEstimatorEssentialToPnP
from sfmcore.featuretracking import FeatureTrackFromPairsUnionFind
from sfmcore.optimization import BundleAdjustmentOptimizerLocal
from sfmcore.scenereconstruction import Sparse3DReconstructionIncremental, SparseSceneEstimationCOLMAPGlobal, Sparse3DReconstructionMapAnything
from sfmcore.visualize import VisualizeScene, visualize_camera_poses_plotly, visualize_3d_points
import os
from sfmcore.baseclass import SfMScene

ID = "1"
log_dir = "/home/anthonyq/projects/scene_agent/breadth_agent/results/co3d/apple_110_13051_23361_vggt_random_10"
gpu_num = "2"


# Construct Modules with Initialized Arguments
image_path = "/home/anthonyq/datasets/DTU/scan15"
calibration_path = "/home/anthonyq/datasets/DTU/calibration_DTU_new.npz"

## NEW SFM PIPELINE
# Step 1: Read in Calibration/Image Data
reconstructed_scene = SfMScene(ID,
                                image_path = image_path, 
                                log_dir=log_dir,
                                gpu_num=gpu_num,
                                max_images = 15,
                                calibration_path = calibration_path)

# Step 2: Detect Features
reconstructed_scene.FeatureDetectionSIFT(
    max_keypoints=10000,
    contrast_threshold=0.009,
    edge_threshold=20,
)
from sfmcore.camerapose import CamPoseEstimatorEssentialToPnP
# Step 3: Detect Feature Pairs
reconstructed_scene.FeatureMatchFlannPair(detector="sift",
                                        k=2,
                                        lowes_thresh=0.78,
                                        RANSAC_homography=False,
                                        RANSAC_threshold=1.0,
                                        RANSAC_conf=0.999)

# FeatureMatchLightGluePair(
#     detector='sift',   
#     RANSAC_threshold=0.02,
#     RANSAC_conf=0.999
# )

# Step 4: Detect/Estimate Camera Poses
reconstructed_scene.CamPoseEstimatorEssentialToPnP(
    iteration_count=150,
    reprojection_error = 3.0,
    optimizer = ("BundleAdjustmentOptimizerLocal", {
        "max_num_iterations": 25,
        "robust_loss": True
    }),
)

# Step 5: Detect Feature Tracks
reconstructed_scene.FeatureTrackFromPairsUnionFind()

# Step 6: Estimate Sparse Reconstruction
# reconstructed_scene.SparseSceneEstimationCOLMAPGlobal(
#     min_track_len = 3,
#     min_num_matches = 30,
#     max_epipolar_error = 1.0,
#     min_tri_angle_deg = 1.0,
#     max_angular_reproj_error_deg  = 1.0,
#     max_normalized_reproj_error  = 0.01,
#     ba_num_iterations = 3
# )
reconstructed_scene.Sparse3DReconstructionIncremental(
    min_observe=3,
    min_angle=1.5,
    # multi_view=True,
    max_reproj_error=1.5,
    reproj_threshold=1.0,
    max_filter_iterations=5
)

# Step 7: Run Optimization
reconstructed_scene.BundleAdjustmentOptimizerGlobal(
    max_num_iterations=130,
)

visualize_3d_points(reconstructed_scene.optimized_scene.points3D.points3D)
# Step 7: Run Optimization
# reconstructed_scene.BundleAdjustmentOptimizerGlobal(
#     max_num_iterations=200,
# )