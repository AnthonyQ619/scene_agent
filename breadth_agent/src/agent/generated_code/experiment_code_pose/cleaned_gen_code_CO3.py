
# STEP 1: Read in Camera Data
from modules.cameramanager import CameraDataManager

image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Experiments\\co3dv2\\hydrant\\411_56064_108483\\images"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Experiments\\co3dv2\\hydrant\\calibration_new_411_56064_108483.npz"

CDM = CameraDataManager(image_path=image_path,
                        calibration_path=calibration_path)

camera_data = CDM.get_camera_data()

# STEP 2: Detect Features (SIFT)
from modules.features import FeatureDetectionSIFT

feature_detector = FeatureDetectionSIFT(cam_data=camera_data,
                                        max_keypoints=12000,
                                        n_octave_layers=3,
                                        contrast_threshold=0.01,
                                        edge_threshold=12,
                                        sigma=1.6)

features = feature_detector()

# STEP 3: Detect Pairwise Feature Matches (BF Pair) - relax RANSAC threshold
from modules.featurematching import FeatureMatchBFPair

feature_matcher = FeatureMatchBFPair(cam_data=camera_data,
                                     detector='sift',
                                     k=2,
                                     cross_check=False,
                                     RANSAC=True,
                                     RANSAC_threshold=0.5,  # relaxed for normalized coords
                                     RANSAC_conf=0.99,
                                     lowes_thresh=0.70)

matched_features = feature_matcher(features=features)

# STEP 4: Estimate Camera Poses (Essential -> PnP) with Local BA
from modules.optimization import BundleAdjustmentOptimizerLocal
from modules.camerapose import CamPoseEstimatorEssentialToPnP

optimizer_local = BundleAdjustmentOptimizerLocal(cam_data=camera_data,
                                                 refine_focal_length=False,
                                                 refine_principal_point=False,
                                                 refine_extra_params=False,
                                                 max_num_iterations=40,
                                                 use_gpu=True,
                                                 robust_loss=True,
                                                 window_size=8,
                                                 min_track_len=3)

pose_estimator = CamPoseEstimatorEssentialToPnP(cam_data=camera_data,
                                                iteration_count=300,
                                                reprojection_error=3.0,
                                                confidence=0.995,
                                                optimizer=optimizer_local)

cam_poses = pose_estimator(feature_pairs=matched_features)

# STEP 5: Track Features Across Multiple Views (BF Tracking) - relax RANSAC threshold
from modules.featurematching import FeatureMatchBFTracking

feature_tracker = FeatureMatchBFTracking(cam_data=camera_data,
                                         detector='sift',
                                         k=2,
                                         cross_check=False,
                                         RANSAC_threshold=0.5,  # relaxed for normalized coords
                                         RANSAC_conf=0.99,
                                         lowes_thresh=0.75)

tracked_features = feature_tracker(features=features)

# STEP 6: Minimal Sparse 3D Reconstruction (support for BA)
from modules.scenereconstruction import Sparse3DReconstructionMono

# Relax thresholds per guidance: allow shorter tracks and smaller baselines
sparse_reconstruction_multi = Sparse3DReconstructionMono(cam_data=camera_data,
                                                         view='multi',
                                                         min_observe=3,
                                                         min_angle=1.0,
                                                         reproj_error=3.0)

# Fallback two-view reconstructor (in case tracking did not set multi_view or produced too few tracks)
sparse_reconstruction_two = Sparse3DReconstructionMono(cam_data=camera_data,
                                                       view='two',
                                                       min_observe=3,
                                                       min_angle=1.0,
                                                       reproj_error=3.0)

# Prefer multi-view if available, else fall back to two-view with pairwise matches
if hasattr(tracked_features, "multi_view") and bool(tracked_features.multi_view):
    sparse_scene = sparse_reconstruction_multi(tracked_features=tracked_features, cam_poses=cam_poses)
    # If no points were triangulated, try two-view using pairwise matches
    if (not hasattr(sparse_scene, "points3D")) or (getattr(sparse_scene.points3D, "points3D", None) is None) or (getattr(sparse_scene.points3D, "points3D", []).__len__() == 0):
        sparse_scene = sparse_reconstruction_two(matched_features, cam_poses)
else:
    sparse_scene = sparse_reconstruction_two(matched_features, cam_poses)

# STEP 7: Global Bundle Adjustment (fixed intrinsics)
from modules.optimization import BundleAdjustmentOptimizerGlobal

optimizer_global = BundleAdjustmentOptimizerGlobal(cam_data=camera_data,
                                                   refine_focal_length=False,
                                                   refine_principal_point=False,
                                                   refine_extra_params=False,
                                                   max_num_iterations=200,
                                                   use_gpu=True,
                                                   robust_loss=True)

optimal_scene = optimizer_global(sparse_scene)

# STEP 8: Deliver Pose-Only Output
camera_poses_optimized = optimal_scene.cam_poses  # List of 3x4 [R|t] per frame
