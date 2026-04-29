from pathlib import Path
import torch
import torch.nn.functional as nnF
from torchvision.io import read_image, write_png
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.utils import flow_to_image
import numpy as np
import cv2
import glob
from modules.utilities import image_builder, resize_dataset, clean_dir

device = "cuda" if torch.cuda.is_available() else "cpu"

weights = Raft_Large_Weights.DEFAULT
transforms = weights.transforms()

model = raft_large(weights=weights, progress=True).to(device).eval()

def load_rgb(path: str) -> torch.Tensor:
    img = read_image(path)[:3]  # C,H,W, uint8
    return img

def pad_to_multiple_of_8(img: torch.Tensor):
    _, h, w = img.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    padded = nnF.pad(img, (0, pad_w, 0, pad_h))
    return padded, h, w

def estimate_flow(img1_path: str, img2_path: str):
    img1 = load_rgb(img1_path)
    img2 = load_rgb(img2_path)

    img1, h, w = pad_to_multiple_of_8(img1)
    img2, _, _ = pad_to_multiple_of_8(img2)

    img1 = img1[None]  # 1,C,H,W
    img2 = img2[None]

    img1, img2 = transforms(img1, img2)

    with torch.no_grad():
        flow_predictions = model(img1.to(device), img2.to(device))
        flow = flow_predictions[-1][0]  # 2,H,W, final RAFT iteration

    flow = flow[:, :h, :w].permute(1, 2, 0).cpu().numpy()  # H,W,2
    return flow

def flow_to_matches(flow, step=8, min_flow=0.5, max_flow=300.0):
    """
    Convert dense optical flow into sparse point correspondences.

    Args:
        flow: H x W x 2 RAFT optical flow
        step: sample every N pixels
        min_flow: ignore tiny motion
        max_flow: ignore extreme likely-bad motion

    Returns:
        pts1: N x 2 points in image 1
        pts2: N x 2 corresponding points in image 2
    """
    H, W = flow.shape[:2]

    ys, xs = np.mgrid[0:H:step, 0:W:step]
    xs = xs.reshape(-1)
    ys = ys.reshape(-1)

    sampled_flow = flow[ys, xs]
    dx = sampled_flow[:, 0]
    dy = sampled_flow[:, 1]

    x2 = xs + dx
    y2 = ys + dy

    flow_mag = np.sqrt(dx ** 2 + dy ** 2)

    valid = (
        (x2 >= 0) & (x2 < W) &
        (y2 >= 0) & (y2 < H) &
        (flow_mag >= min_flow) &
        (flow_mag <= max_flow)
    )

    pts1 = np.stack([xs[valid], ys[valid]], axis=1).astype(np.float32)
    pts2 = np.stack([x2[valid], y2[valid]], axis=1).astype(np.float32)

    return pts1, pts2


def estimate_motion_from_flow(flow, K):
    pts1, pts2 = flow_to_matches(flow, step=8)

    E, mask = cv2.findEssentialMat(
        pts1,
        pts2,
        K,
        method=cv2.USAC_MAGSAC,
        prob=0.999,
        threshold=1.0
    )

    if E is None:
        raise RuntimeError("Essential matrix estimation failed.")

    inliers, R, t, pose_mask = cv2.recoverPose(
        E,
        pts1,
        pts2,
        K,
        mask=mask
    )

    return {
        "R": R,
        "t_direction": t.reshape(3),
        "num_matches": len(pts1),
        "num_inliers": int(inliers),
        "inlier_ratio": float(inliers / max(len(pts1), 1)),
        "pts1": pts1,
        "pts2": pts2,
        "essential_mask": mask,
        "pose_mask": pose_mask,
    }

def flow_magnitude_stats(flow):
    mag = np.linalg.norm(flow, axis=2)

    return {
        "mean_flow": float(np.mean(mag)),
        "median_flow": float(np.median(mag)),
        "p75_flow": float(np.percentile(mag, 75)),
        "p90_flow": float(np.percentile(mag, 90)),
    }

def read_calibration(cal_file_path: str):
        data = np.load(cal_file_path)
        data.allow_pickle = True 
        full_cal_data = dict(data)

        # Keys are
        # - k_mats: (N, 3, 3) -> N = num of cameras
        # - dists: (N, 1, 5) -> N = num of cameras 
        # - baseline_ext: None or (3, 4) -> the baseline of stereo camera

        K_mats = full_cal_data['k_mats']
        dists = full_cal_data['dists']
        baseline_ext = full_cal_data['baseline_ext']

        intrinsics = []
        distortions = []

        for i in range(K_mats.shape[0]):
            intrinsics.append(K_mats[i])
            distortions.append(dists[i])

        print(type(baseline_ext))
        if baseline_ext != None:
            assert len(intrinsics) == 2, "Camera is not stereo, or not enough information stored in calibration file. Revisit calibration information"
            extrinsic = baseline_ext
        else:
            extrinsic = None

        return intrinsics, distortions, extrinsic

def read_camera_flow(image_paths, intrinsics):
    important_fields = {
    "overall_motion_magnitude_score": "How much apparent image motion exists overall.",
    "high_motion_tail_score": "How severe the larger motion regions are.",
    "motion_variability_score": "How consistent or inconsistent the motion is across pairs.",
    "low_baseline_risk_score": "Whether motion/parallax may be too small for good reconstruction.",
    "large_motion_risk_score": "Whether motion may be too large for reliable matching.",
    "rotation_change_deg_median": "How much the camera orientation typically changes.",
    "large_rotation_risk_score": "Whether many pairs have difficult rotation changes.",
    }
    # if calib_path is not None:
    #     intrinsics, distortion, _ = read_calibration(calib_path)
    # else: 
    #     intrinsics, distortion = None, None
    
    # image_dir = Path(image_paths)
    # images = sorted(image_dir.glob("*.png"))
    images = sorted(glob.glob(image_paths + "/*"))

    # print(images)
    flows = []
    for i in range(len(images) - 1):
        flow = estimate_flow(str(images[i]), str(images[i + 1])) 
        flows.append(flow)


    # First Metric
    mean_flow = []
    median_flow = []
    p75_flow = []
    p90_flow = []
    H, W = cv2.imread(images[0]).shape[:2]
    # diag = np.sqrt(H**2 + W**2)
    for flow in flows:
        results = flow_magnitude_stats(flow)
        
        mean_flow.append(results["mean_flow"])
        median_flow.append(results["median_flow"])
        p75_flow.append(results["p75_flow"])
        p90_flow.append(results["p90_flow"])
    
    flow_scores = summarize_flow_pair_stats(mean_flow, median_flow, p75_flow, p90_flow, image_shape=(H, W))
    # if intrinsics is None:
    #     results = summarize_flow_pair_stats(mean_flow, median_flow, p75_flow, p90_flow)

    #     return results

    context = f"""
Overall: Higher Score means larger camera motion/baseline across all metrics.

Overall Motion Magnitude Score: {flow_scores["overall_motion_magnitude_score"]}
    - {important_fields["overall_motion_magnitude_score"]}
    - Large score > 0.09
    - Low means weak apparent motion/parallax; large means stronger camera or scene displacement.
High Motion Tail Score: {flow_scores["high_motion_tail_score"]}
    - {important_fields["high_motion_tail_score"]}
    - Large Score > 0.13
    - Low means even high-motion regions are mild; large means some regions have strong 
      displacement that may challenge matching.
Motion Variabbility Score: {flow_scores["motion_variability_score"]}
    - {important_fields["motion_variability_score"]}
    - Large Score > 0.035
    - Low means pair-to-pair motion is consistent; large means some image pairs move much 
      more than others.
Low Baseline Risk Score: {flow_scores["low_baseline_risk_score"]}
    - {important_fields["low_baseline_risk_score"]}
    - Large Score > 0.35
    - Low means few pairs have weak motion; large means many pairs may lack enough parallax 
    - for good triangulation.
Large Motion Risk Score: {flow_scores["large_motion_risk_score"]}
    - {important_fields["large_motion_risk_score"]}
    - Large Score > 0.45
    - Low means few pairs have excessive motion; large means many pairs may be difficult for 
      feature matching/tracking.
    - Note: If all other scores are within bounds, can ignore this score as it is sensitive to
      large motion that is still within bounds for feature matching/tracking.
    """
    # print(intrinsics)
    if intrinsics is not None:
        Rs = []
        for flow in flows:
            result = estimate_motion_from_flow(flow, intrinsics)#[0])
            R = result["R"]
            # t_dir = result["t_direction"]
            Rs.append(rotation_angle_deg(R))
        r_scores = summarize_rotation_scores(Rs)
        rotation_context = f"""
Rotation Change in Degree: {r_scores["rotation_change_deg_median"]}
    - {important_fields["rotation_change_deg_median"]}
    - Large Score > 18.0
    - Low means little camera orientation change; large means typical image pairs 
      have significant viewpoint rotation.
Large Rotation Risk Score: {r_scores["large_rotation_risk_score"]}
    - {important_fields["large_rotation_risk_score"]}
    - Large Score > 0.33
    - Low means few pairs exceed the large-rotation threshold; large means many
      pairs may be difficult for standard feature matching.
        """

        context += rotation_context

    return context
        # return mean_flow, median_flow, p75_flow, p90_flow, Rs
    

def rotation_angle_deg(R):
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))

def summarize_rotation_scores(R_angles):
    if R_angles is None or len(R_angles) == 0:
        return {}
    angles = np.asarray(R_angles, dtype=np.float32)
    # angles = np.asarray([rotation_angle_deg(R) for R in Rs], dtype=np.float32)

    return {
        "rotation_change_deg_median": float(np.median(angles)),
        "rotation_change_deg_p75": float(np.percentile(angles, 75)),
        "rotation_change_deg_p90": float(np.percentile(angles, 90)),
        "rotation_variability_deg": float(np.percentile(angles, 75) - np.percentile(angles, 25)),
        "large_rotation_risk_score": float(np.mean(angles > 20.0)),
    }

def summarize_flow_pair_stats(
    mean_flow,
    median_flow,
    p75_flow,
    p90_flow,
    image_shape=None,
    low_motion_thresh=0.005,
    high_motion_thresh=0.08,
):
    """
    Summarizes pairwise optical-flow magnitude stats into dataset-level signals.

    Parameters
    ----------
    mean_flow, median_flow, p75_flow, p90_flow:
        Lists of scalar flow magnitudes, one value per image pair.

    image_shape:
        Optional (H, W). If given, flow magnitudes are normalized by image diagonal.

    low_motion_thresh:
        Normalized p75 threshold below which pair may have weak baseline/parallax.

    high_motion_thresh:
        Normalized p90 threshold above which pair may have large apparent motion.

    Returns
    -------
    Dictionary of compact dataset-level motion scores.
    """

    mean_flow = np.asarray(mean_flow, dtype=np.float32)
    median_flow = np.asarray(median_flow, dtype=np.float32)
    p75_flow = np.asarray(p75_flow, dtype=np.float32)
    p90_flow = np.asarray(p90_flow, dtype=np.float32)

    if image_shape is not None:
        H, W = image_shape[:2]
        diag = np.sqrt(H**2 + W**2)

        mean_flow = mean_flow / diag
        median_flow = median_flow / diag
        p75_flow = p75_flow / diag
        p90_flow = p90_flow / diag

    def iqr(x):
        return float(np.percentile(x, 75) - np.percentile(x, 25))

    scores = {
        # Main camera/image motion signal
        "overall_motion_magnitude_score": float(np.median(p75_flow)),

        # Typical apparent motion
        "typical_motion_score": float(np.median(median_flow)),

        # Strong-motion tail, useful for detecting fast camera motion or large baselines
        "high_motion_tail_score": float(np.median(p90_flow)),

        # How inconsistent motion is across pairs
        "motion_variability_score": iqr(p75_flow),

        # Near-worst-case pair motion
        "max_pair_motion_score": float(np.percentile(p90_flow, 90)),

        # Risk that adjacent pairs have too little baseline/parallax
        "low_baseline_risk_score": float(np.mean(p75_flow < low_motion_thresh)),

        # Risk that image pairs are too far apart / too much apparent motion
        "large_motion_risk_score": float(np.mean(p90_flow > high_motion_thresh)),

        # Optional raw averages
        "mean_pair_mean_flow": float(np.mean(mean_flow)),
        "mean_pair_median_flow": float(np.mean(median_flow)),
        "mean_pair_p75_flow": float(np.mean(p75_flow)),
        "mean_pair_p90_flow": float(np.mean(p90_flow)),
    }

    return scores


# img_path = "/home/anthonyq/datasets/tanks_and_temples/Lighthouse"
# cal_path = "/home/anthonyq/datasets/tanks_and_temples/calibration_new_2048.npz"

# image_dir = Path(img_path)
# images = sorted(image_dir.glob("*.jpg"))[:40]

# # result = read_camera_flow(img_path, cal_path)
# resized_dir, resized_img_list, K = resize_dataset(image_path=images,
#                                                        max_size=640,
#                                                        calib_path=cal_path)
# result = read_camera_flow(resized_dir, K)
# clean_dir(resized_dir)

# print(result)