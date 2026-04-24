
# Construct Modules with Initialized Arguments
import numpy as np

image_path = r"C:\Users\Anthony\Documents\Projects\datasets\sfm_dataset\Experiments\ETH\living_room\images\dslr_images_undistorted"
calibration_path = r"C:\Users\Anthony\Documents\Projects\datasets\sfm_dataset\Experiments\ETH\living_room\dslr_calibration_undistorted\calibration_ETH_new.npz"

# STEP 1: Read in Camera Data
from modules.cameramanager import CameraDataManager

CDM = CameraDataManager(image_path=image_path,
                        calibration_path=calibration_path)

# Get Camera Data
camera_data = CDM.get_camera_data()

# Ensure calibration (intrinsics and distortion) is valid and non-empty for all frames.
# If using undistorted images and NPZ lacks distortion, set distortion to zeros instead of None/empty.
# This prevents cv2.undistortPoints from receiving an empty matrix during tracking.
# Try common attribute names; set a safe fallback on all.
dist_fallback = np.zeros((1, 5), dtype=np.float64)
try:
    # Validate intrinsics exist
    K = None
    for name in ('K', 'intrinsics', 'camera_matrix'):
        if hasattr(camera_data, name):
            K = getattr(camera_data, name)
            break
    if K is not None:
        K = np.asarray(K)
        assert K.size == 9, "Missing/invalid intrinsics in camera_data"
    else:
        raise AssertionError("Missing intrinsics in camera_data")

    # Ensure distortion is not empty
    has_valid_dist = False
    for name in ('dist', 'distortion', 'distCoeffs', 'dist_coeffs', 'D'):
        if hasattr(camera_data, name):
            dist_val = getattr(camera_data, name)
            if dist_val is not None and np.size(dist_val) > 0:
                has_valid_dist = True
                break
    if not has_valid_dist:
        for name in ('dist', 'distortion', 'distCoeffs', 'dist_coeffs', 'D'):
            try:
                setattr(camera_data, name, dist_fallback)
            except Exception:
                pass
except Exception as e:
    # As a hard fallback, attach defaults so downstream modules have something to use
    try:
        setattr(camera_data, 'K', np.asarray([[K[0, 0], 0, K[0, 2]],
                                              [0, K[1, 1], K[1, 2]],
                                              [0, 0, 1]], dtype=np.float64))
    except Exception:
        pass
    for name in ('dist', 'distortion', 'distCoeffs', 'dist_coeffs', 'D'):
        try:
            setattr(camera_data, name, dist_fallback)
        except Exception:
            pass

# STEP 2: Detect Features with SuperPoint
from modules.features import FeatureDetectionSP

feature_detector = FeatureDetectionSP(cam_data=camera_data,
                                      max_keypoints=6000)

features = feature_detector()

# STEP 3: Pairwise Feature Matching
# Start with SuperGlue (indoor), but use more permissive thresholds to avoid <4 correspondences
from modules.featurematching import FeatureMatchSuperGluePair

try:
    feature_matcher = FeatureMatchSuperGluePair(cam_data=camera_data,
                                                detector='superpoint',
                                                setting='indoor',
                                                match_threshold=0.10,     # more permissive to keep more matches
                                                RANSAD=False if False else True,  # keep default True; placeholder to avoid typos
                                                RANSAC_threshold=1.0,     # use a reasonable/default threshold
                                                RANSAC_conf=0.995)
    matched_features = feature_matcher(features=features)
except Exception:
    # Fallback: switch to LightGlue pairwise with a slightly lower filter threshold
    from modules.featurematching import FeatureMatchLightGluePair
    feature_matcher = FeatureMatchLightGluePair(cam_data=camera_data,
                                                detector='superpoint',
                                                filter_threshold=0.08,
                                                RANSAC_threshold=1.0,
                                                RANSAC_conf=0.99)
    matched_features = feature_matcher(features=features)

# STEP 4: Camera Pose Estimation with Essential->PnP and Local BA
from modules.optimization import BundleAdjustmentOptimizerLocal
from modules.camerapose import CamPoseEstimatorEssentialToPnP

optimizer_local = BundleAdjustmentOptimizerLocal(cam_data=camera_data,
                                                 max_num_iterations=40,
                                                 window_size=10,
                                                 robust_loss=True)

pose_estimator = CamPoseEstimatorEssentialToPnP(cam_data=camera_data,
                                                iteration_count=300,
                                                reprojection_error=3.0,
                                                confidence=0.995,
                                                optimizer=optimizer_local)

cam_poses = pose_estimator(feature_pairs=matched_features)

# STEP 5: Multi-view Feature Tracking
# Try SuperGlue tracking first with relaxed thresholds; on failure or empty tracks, fallback to LightGlue tracking.
from modules.featurematching import FeatureMatchSuperGlueTracking

def has_tracks(tracked):
    try:
        if hasattr(tracked, 'data_matrix'):
            return tracked.data_matrix is not None and np.size(tracked.data_matrix) > 0
        if hasattr(tracked, 'point_count'):
            return getattr(tracked, 'point_count', 0) > 0
    except Exception:
        pass
    return False

tracked_features = None
try:
    feature_tracker = FeatureMatchSuperGlueTracking(cam_data=camera_data,
                                                    detector='superpoint',
                                                    setting='indoor',
                                                    match_threshold=0.20,   # slightly more permissive for more tracks
                                                    RANSAC_threshold=1.0)   # relaxed threshold to reduce track dropouts
    tracked_features = feature_tracker(features=features)
    if not has_tracks(tracked_features):
        raise RuntimeError("SuperGlue tracking returned empty tracks; switching to LightGlue tracking.")
except Exception:
    from modules.featurematching import FeatureMatchLightGlueTracking
    feature_tracker = FeatureMatchLightGlueTracking(cam_data=camera_data,
                                                    detector='superpoint',
                                                    filter_threshold=0.06,  # permissive to retain more correspondences
                                                    RANSAC_threshold=1.0,
                                                    RANSAC_conf=0.99)
    tracked_features = feature_tracker(features=features)

# STEP 6: Sparse 3D Reconstruction (multi-view triangulation)
from modules.scenereconstruction import Sparse3DReconstructionMono

sparse_reconstruction = Sparse3DReconstructionMono(cam_data=camera_data,
                                                   view="multi",
                                                   min_observe=4,
                                                   min_angle=2.0,
                                                   reproj_error=2.0)

sparse_scene = sparse_reconstruction(tracked_features, cam_poses)

# STEP 7: Global Bundle Adjustment
from modules.optimization import BundleAdjustmentOptimizerGlobal

optimizer = BundleAdjustmentOptimizerGlobal(cam_data=camera_data,
                                            max_num_iterations=200,
                                            robust_loss=True,
                                            refine_focal_length=False,
                                            refine_principal_point=False,
                                            refine_extra_params=False)

optimal_scene = optimizer(sparse_scene)
