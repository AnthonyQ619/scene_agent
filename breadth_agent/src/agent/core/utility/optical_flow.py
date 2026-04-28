from pathlib import Path
import torch
import torch.nn.functional as nnF
from torchvision.io import read_image, write_png
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.utils import flow_to_image
import numpy as np
import cv2

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

def read_camera_flow(image_paths, calib_path):
    if calib_path is not None:
        intrinsics, distortion, _ = read_calibration(calib_path)
    else: 
        intrinsics, distortion = None, None
    
    image_dir = Path(image_paths)
    images = sorted(image_dir.glob("*.png"))[:10]

    print(images)
    flows = []
    for i in range(len(images) - 1):
        flow = estimate_flow(str(images[i]), str(images[i + 1])) 
        flows.append(flow)


    # First Metric
    mean_flow = []
    median_flow = []
    p75_flow = []
    p90_flow = []

    for flow in flows:
        results = flow_magnitude_stats(flow)

        mean_flow.append(results["mean_flow"])
        median_flow.append(results["median_flow"])
        p75_flow.append(results["p75_flow"])
        p90_flow.append(results["p90_flow"])
    
    if intrinsics is None:
        return mean_flow, median_flow, p75_flow, p90_flow, None
    else:
        Rs = []
        for flow in flows:
            result = estimate_motion_from_flow(flow, intrinsics[0])
            R = result["R"]
            # t_dir = result["t_direction"]
            Rs.append(R)
        return mean_flow, median_flow, p75_flow, p90_flow, Rs