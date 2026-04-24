
"""
Structure-from-Motion (SfM) for Outdoor Statue Scene using Monocular Images

Problem Statement and Approach:
Given a small set of monocular images (3 frames) of an outdoor statue captured under consistent lighting and mild viewpoint changes,
reconstruct the sparse 3D geometry and estimate camera poses using a robust SfM pipeline.

Scene Description and Module Choices:
- Outdoor scene, textured statue and pedestal, consistent illumination, monocular camera, small baseline changes.
- Use robust, learned feature matcher (LoFTR) for pairwise correspondences due to potential viewpoint/illumination robustness and texture richness.
- Use SuperPoint-based detection and LightGlue tracking for multi-view feature tracks across all images.
- Use Essential matrix-based initialization and PnP registration for camera pose estimation (CamPoseEstimatorEssentialToPnP).
- Use Sparse3DReconstructionMono to triangulate a sparse point cloud from tracks and refined poses (likely includes bundle adjustment internally).

Sub-Problems and Solutions:
1) Feature Detection:
   - Solution: FeatureDetectionSP to detect SuperPoint features in all images.

2) Feature Matching (Pairwise):
   - Solution: FeatureMatchLoftrPair for robust pairwise correspondences to bootstrap pose estimation.
   - Fallback: FeatureMatchRoMAPair if LoFTR returns insufficient matches.

3) Geometric Verification:
   - Solution: Implicitly handled within CamPoseEstimatorEssentialToPnP via Essential matrix estimation with RANSAC.

4) Camera Pose Estimation:
   - Solution: CamPoseEstimatorEssentialToPnP (E-matrix for relative pose + PnP to register subsequent views).

5) Multi-View Feature Tracking:
   - Solution: FeatureMatchLightGlueTracking(detector="superpoint") to obtain feature tracks across all frames.

6) Sparse 3D Reconstruction:
   - Solution: Sparse3DReconstructionMono to triangulate and refine 3D points and camera poses (bundle adjustment).

7) Optional Dense Reconstruction and Surface Reconstruction:
   - Not implemented here due to API scope; can be added later with MVS and meshing (e.g., Poisson) modules if available.

Note:
- Visualization intentionally omitted per instructions (do not use Visualizer module).
- Adjust image_path and calibration_path to your dataset locations.
"""

from modules.utilities.utilities import CalibrationReader
from modules.features import FeatureDetectionSP
from modules.featurematching import FeatureMatchLightGlueTracking, FeatureMatchLoftrPair, FeatureMatchRoMAPair
from modules.camerapose import CamPoseEstimatorEssentialToPnP
from modules.scenereconstruction import Sparse3DReconstructionMono


def run_pairwise_matching(image_path):
    """
    Try LoFTR pairwise matching; if insufficient, fallback to RoMA.
    Returns:
        feature_pairs: pairwise correspondences suitable for pose estimation.
    """
    # Primary matcher: LoFTR (robust for moderate viewpoint changes)
    loftr = FeatureMatchLoftrPair(img_path=image_path)
    feature_pairs = loftr()

    # Heuristic check for fallback (API-dependent; here we check None/emptiness if available)
    need_fallback = False
    try:
        need_fallback = (feature_pairs is None) or (hasattr(feature_pairs, "__len__") and len(feature_pairs) == 0)
    except Exception:
        # If length check fails (unknown structure), proceed without fallback
        need_fallback = False

    if need_fallback:
        roma = FeatureMatchRoMAPair(img_path=image_path)
        feature_pairs = roma()

    return feature_pairs


def estimate_camera_poses(calibration_data, image_path, feature_pairs):
    """
    Estimate camera poses using essential matrix + PnP from pairwise matches.
    Returns:
        cam_poses: estimated camera extrinsics for all images
    """
    pose_estimator = CamPoseEstimatorEssentialToPnP(calibration=calibration_data, image_path=image_path)
    cam_poses = pose_estimator(features_pairs=feature_pairs)
    return cam_poses


def detect_and_track_features(image_path):
    """
    Detect SuperPoint features and track them across all frames using LightGlue.
    Returns:
        tracked_features: multi-view feature tracks
    """
    feature_detector = FeatureDetectionSP(image_path=image_path)
    features = feature_detector()

    feature_tracker = FeatureMatchLightGlueTracking(detector="superpoint")
    tracked_features = feature_tracker(features=features)
    return tracked_features


def reconstruct_sparse(calibration_data, image_path, tracked_features, cam_poses, mode="multi"):
    """
    Triangulate and refine a sparse 3D reconstruction.
    Args:
        mode: "multi" for multi-view tracks, or "two" for pairwise reconstruction.
    Returns:
        sparse_scene: reconstructed sparse 3D structure and refined poses
    """
    sparse_reconstruction = Sparse3DReconstructionMono(calibration=calibration_data, image_path=image_path)
    if mode == "multi":
        sparse_scene = sparse_reconstruction(tracked_features, cam_poses, view="multi")
    else:
        # For completeness if pairwise-only pipeline is needed:
        sparse_scene = sparse_reconstruction(tracked_features, cam_poses, view="two")
    return sparse_scene


def main():
    # User: set these to your dataset paths
    image_path = r"C:\Users\Anthony\Documents\Projects\datasets\Structure-from-Motion\sfm_dataset"
    calibration_path = r"C:\Users\Anthony\Documents\Projects\datasets\Structure-from-Motion\calibration.npz"

    # 1) Load intrinsics
    calibration_data = CalibrationReader(calibration_path).get_calibration()

    # 2) Pairwise feature matching for initial geometry
    feature_pairs = run_pairwise_matching(image_path=image_path)

    # 3) Camera pose estimation from pairwise correspondences
    cam_poses = estimate_camera_poses(calibration_data=calibration_data, image_path=image_path, feature_pairs=feature_pairs)

    # 4) Multi-view feature detection and tracking
    tracked_features = detect_and_track_features(image_path=image_path)

    # 5) Sparse 3D reconstruction with bundle adjustment (if available in module)
    sparse_scene = reconstruct_sparse(
        calibration_data=calibration_data,
        image_path=image_path,
        tracked_features=tracked_features,
        cam_poses=cam_poses,
        mode="multi",
    )

    # 6) Output artifacts are returned as Python objects; persist or visualize as needed externally.
    # For example:
    # - Save camera poses, sparse points, descriptors, etc., using your own I/O utilities.
    # - Visualization intentionally omitted.

    # Minimal reporting (types) to confirm pipeline execution
    try:
        print("Camera poses type:", type(cam_poses))
    except Exception:
        pass
    try:
        print("Tracked features type:", type(tracked_features))
    except Exception:
        pass
    try:
        print("Sparse scene type:", type(sparse_scene))
    except Exception:
        pass


if __name__ == "__main__":
    main()
