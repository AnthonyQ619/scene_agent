import argparse
import json
import os
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def read_point_cloud(path: str) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        raise ValueError(f"Could not read point cloud or point cloud is empty: {path}")
    return pcd


def clean_point_cloud(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    pts = np.asarray(pcd.points)
    mask = np.isfinite(pts).all(axis=1)

    cleaned = o3d.geometry.PointCloud()
    cleaned.points = o3d.utility.Vector3dVector(pts[mask])

    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        cleaned.colors = o3d.utility.Vector3dVector(colors[mask])

    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        cleaned.normals = o3d.utility.Vector3dVector(normals[mask])

    return cleaned


def pcd_to_numpy(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    return np.asarray(pcd.points, dtype=np.float64)


def numpy_to_pcd(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return pcd


def bbox_diag(points: np.ndarray) -> float:
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    points_h = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=points.dtype)],
        axis=1,
    )
    transformed = (T @ points_h.T).T[:, :3]
    return transformed


def estimate_initial_scale_translation(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """
    Initial transform:
        dst ~= s * src + t

    No rotation is estimated here.
    """
    src_center = src_pts.mean(axis=0)
    dst_center = dst_pts.mean(axis=0)

    src_diag = bbox_diag(src_pts)
    dst_diag = bbox_diag(dst_pts)

    if src_diag < 1e-12:
        raise ValueError("Source point cloud has near-zero bounding-box diagonal.")

    scale = dst_diag / src_diag

    T = np.eye(4)
    T[:3, :3] = np.eye(3) * scale
    T[:3, 3] = dst_center - scale * src_center

    return T


def umeyama_sim3(
    src: np.ndarray,
    dst: np.ndarray,
    estimate_scale: bool = True,
    allow_reflection: bool = False,
) -> np.ndarray:
    """
    Estimate Sim(3) transform T mapping src -> dst:

        dst ~= s * R @ src + t

    Returns 4x4 matrix.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)

    if src.shape != dst.shape:
        raise ValueError(f"src and dst must have same shape, got {src.shape} and {dst.shape}")

    if src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("src and dst must be Nx3 arrays")

    n = src.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 correspondences for Sim(3) estimation")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / n

    U, singular_values, Vt = np.linalg.svd(cov)

    S = np.eye(3)

    det = np.linalg.det(U @ Vt)
    if det < 0 and not allow_reflection:
        S[-1, -1] = -1.0

    R = U @ S @ Vt

    if estimate_scale:
        var_src = np.mean(np.sum(src_c ** 2, axis=1))
        scale = np.trace(np.diag(singular_values) @ S) / max(var_src, 1e-12)
    else:
        scale = 1.0

    t = mu_dst - scale * R @ mu_src

    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t

    return T


def preprocess_for_fpfh(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    if len(pcd_down.points) < 20:
        raise ValueError(
            f"Downsampled cloud has too few points: {len(pcd_down.points)}. "
            f"Use a smaller voxel size."
        )

    normal_radius = voxel_size * 2.0
    feature_radius = voxel_size * 5.0

    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=30,
        )
    )

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=feature_radius,
            max_nn=100,
        ),
    )

    return pcd_down, fpfh


def run_fpfh_global_registration(
    src_pcd: o3d.geometry.PointCloud,
    dst_pcd: o3d.geometry.PointCloud,
    voxel_size: float,
) -> np.ndarray:
    """
    Rigid global registration after source has already been roughly scaled/translated.
    This estimates rotation + translation, not scale.
    """
    src_down, src_fpfh = preprocess_for_fpfh(src_pcd, voxel_size)
    dst_down, dst_fpfh = preprocess_for_fpfh(dst_pcd, voxel_size)

    distance_threshold = voxel_size * 1.5

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down,
        dst_down,
        src_fpfh,
        dst_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iteration=100000,
            confidence=0.999,
        ),
    )

    print("[FPFH RANSAC]")
    print(f"  fitness:    {result.fitness:.6f}")
    print(f"  inlier_rmse:{result.inlier_rmse:.6f}")

    return result.transformation


def trimmed_sim3_icp(
    src_pts_original: np.ndarray,
    dst_pts: np.ndarray,
    init_T: np.ndarray,
    num_iters: int = 30,
    trim_fraction: float = 0.7,
    max_corr_dist: float | None = None,
    estimate_scale: bool = True,
    min_corr: int = 100,
) -> tuple[np.ndarray, dict]:
    """
    Iterative closest point with Sim(3) updates.

    At each iteration:
        1. Transform source using current T.
        2. Find nearest GT point.
        3. Keep the closest trim_fraction correspondences.
        4. Estimate incremental Sim(3) update from transformed source to target.
        5. Compose update.

    This keeps the transform as one global similarity transform.
    """
    if not (0.0 < trim_fraction <= 1.0):
        raise ValueError("trim_fraction must be in (0, 1].")

    tree = cKDTree(dst_pts)
    T = init_T.copy()

    history = []

    prev_rmse = None

    for i in range(num_iters):
        src_trans = transform_points(src_pts_original, T)

        dists, nn_idx = tree.query(src_trans, k=1, workers=-1)

        if max_corr_dist is not None:
            valid = dists < max_corr_dist
        else:
            valid = np.ones_like(dists, dtype=bool)

        valid_indices = np.flatnonzero(valid)

        if valid_indices.size < min_corr:
            print(
                f"[Sim3-ICP] Iter {i:02d}: too few correspondences "
                f"({valid_indices.size}), stopping."
            )
            break

        valid_dists = dists[valid_indices]

        keep_count = max(min_corr, int(np.floor(trim_fraction * valid_indices.size)))
        keep_count = min(keep_count, valid_indices.size)

        sorted_local = np.argsort(valid_dists)
        keep_indices = valid_indices[sorted_local[:keep_count]]

        src_corr = src_trans[keep_indices]
        dst_corr = dst_pts[nn_idx[keep_indices]]

        delta_T = umeyama_sim3(
            src_corr,
            dst_corr,
            estimate_scale=estimate_scale,
            allow_reflection=False,
        )

        T = delta_T @ T

        kept_dists = dists[keep_indices]
        rmse = float(np.sqrt(np.mean(kept_dists ** 2)))
        mean = float(np.mean(kept_dists))
        median = float(np.median(kept_dists))
        p90 = float(np.percentile(kept_dists, 90))

        history.append(
            {
                "iteration": i,
                "num_corr": int(keep_indices.size),
                "rmse": rmse,
                "mean": mean,
                "median": median,
                "p90": p90,
            }
        )

        print(
            f"[Sim3-ICP] Iter {i:02d}: "
            f"corr={keep_indices.size}, "
            f"rmse={rmse:.6f}, "
            f"mean={mean:.6f}, "
            f"median={median:.6f}, "
            f"p90={p90:.6f}"
        )

        if prev_rmse is not None:
            improvement = abs(prev_rmse - rmse)
            if improvement < 1e-7:
                print(f"[Sim3-ICP] Converged at iter {i:02d}.")
                break

        prev_rmse = rmse

    diagnostics = {
        "num_iterations_run": len(history),
        "history": history,
    }

    return T, diagnostics


def decompose_sim3(T: np.ndarray) -> dict:
    A = T[:3, :3]
    scale = float(np.cbrt(np.linalg.det(A)))

    if abs(scale) < 1e-12:
        R = A
    else:
        R = A / scale

    return {
        "scale": scale,
        "rotation_det": float(np.linalg.det(R)),
        "translation": T[:3, 3].tolist(),
        "matrix": T.tolist(),
    }


def copy_colors_and_normals(
    original: o3d.geometry.PointCloud,
    transformed_points: np.ndarray,
) -> o3d.geometry.PointCloud:
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(transformed_points)

    if original.has_colors():
        out.colors = original.colors

    if original.has_normals():
        # Normals should only receive rotation, not scale/translation.
        # For evaluation, normals are usually irrelevant, so preserving original normals
        # is not critical.
        out.normals = original.normals

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Align an arbitrary-frame predicted dense PLY to a DTU GT PLY using a single Sim(3) transform."
    )

    parser.add_argument("--pred_ply", required=True, help="Predicted dense PLY from SfMAgent / pycolmap.")
    parser.add_argument("--gt_ply", required=True, help="DTU ground-truth PLY, e.g. stl001_total.ply.")
    parser.add_argument("--out_ply", required=True, help="Output aligned PLY to feed into DTU MATLAB eval.")

    parser.add_argument(
        "--voxel_size",
        type=float,
        default=None,
        help=(
            "Voxel size for registration. If omitted, uses GT bbox diagonal / 250. "
            "Use DTU units. Try 0.5 or 1.0 if DTU is in mm-like scale."
        ),
    )

    parser.add_argument(
        "--alignment_points",
        type=int,
        default=200000,
        help="Maximum number of points used for Sim3-ICP alignment.",
    )

    parser.add_argument(
        "--icp_iters",
        type=int,
        default=30,
        help="Number of trimmed Sim3-ICP iterations.",
    )

    parser.add_argument(
        "--trim_fraction",
        type=float,
        default=0.7,
        help="Fraction of closest correspondences kept during trimmed Sim3-ICP.",
    )

    parser.add_argument(
        "--max_corr_dist",
        type=float,
        default=None,
        help="Optional max correspondence distance during Sim3-ICP. If omitted, no hard cutoff is used.",
    )

    parser.add_argument(
        "--no_fpfh",
        action="store_true",
        help="Disable FPFH global registration. Use only scale/centroid init + Sim3-ICP.",
    )

    parser.add_argument(
        "--save_transform",
        default=None,
        help="Optional path to save final 4x4 transform matrix as txt.",
    )

    parser.add_argument(
        "--save_diagnostics",
        default=None,
        help="Optional path to save diagnostics JSON.",
    )

    parser.add_argument(
        "--scan_id",
        type=int,
        default=None,
        help="Optional DTU scan id. If provided with --dtu_points_dir, also saves scanXXX.ply there.",
    )

    parser.add_argument(
        "--dtu_points_dir",
        default=None,
        help="Optional DTU result Points directory. Used with --scan_id.",
    )

    args = parser.parse_args()

    pred = clean_point_cloud(read_point_cloud(args.pred_ply))
    gt = clean_point_cloud(read_point_cloud(args.gt_ply))

    pred_pts_full = pcd_to_numpy(pred)
    gt_pts_full = pcd_to_numpy(gt)

    print(f"Loaded predicted cloud: {args.pred_ply}")
    print(f"  points: {len(pred_pts_full)}")
    print(f"Loaded GT cloud: {args.gt_ply}")
    print(f"  points: {len(gt_pts_full)}")

    if args.voxel_size is None:
        args.voxel_size = bbox_diag(gt_pts_full) / 250.0

    print(f"Using voxel_size: {args.voxel_size:.6f}")

    # Downsample for registration.
    pred_down = pred.voxel_down_sample(args.voxel_size)
    gt_down = gt.voxel_down_sample(args.voxel_size)

    pred_pts_down = pcd_to_numpy(pred_down)
    gt_pts_down = pcd_to_numpy(gt_down)

    print(f"Downsampled predicted points: {len(pred_pts_down)}")
    print(f"Downsampled GT points: {len(gt_pts_down)}")

    # Limit alignment point count for speed.
    if len(pred_pts_down) > args.alignment_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pred_pts_down), size=args.alignment_points, replace=False)
        pred_pts_align = pred_pts_down[idx]
    else:
        pred_pts_align = pred_pts_down

    if len(gt_pts_down) > args.alignment_points:
        rng = np.random.default_rng(123)
        idx = rng.choice(len(gt_pts_down), size=args.alignment_points, replace=False)
        gt_pts_align = gt_pts_down[idx]
    else:
        gt_pts_align = gt_pts_down

    # Initial scale + translation.
    T_init = estimate_initial_scale_translation(pred_pts_align, gt_pts_align)
    pred_init_pts = transform_points(pred_pts_down, T_init)
    pred_init_pcd = numpy_to_pcd(pred_init_pts)

    T = T_init.copy()

    # Optional FPFH global rotation/translation initialization.
    if not args.no_fpfh:
        try:
            T_fpfh = run_fpfh_global_registration(
                pred_init_pcd,
                gt_down,
                args.voxel_size,
            )
            T = T_fpfh @ T
        except Exception as exc:
            print(f"[Warning] FPFH global registration failed: {exc}")
            print("[Warning] Continuing with scale+centroid initialization only.")

    # Sim3-ICP refinement.
    T_final, icp_diag = trimmed_sim3_icp(
        src_pts_original=pred_pts_align,
        dst_pts=gt_pts_align,
        init_T=T,
        num_iters=args.icp_iters,
        trim_fraction=args.trim_fraction,
        max_corr_dist=args.max_corr_dist,
        estimate_scale=True,
        min_corr=100,
    )

    final_info = decompose_sim3(T_final)

    print("\nFinal Sim(3):")
    print(f"  scale:        {final_info['scale']:.9f}")
    print(f"  rotation det: {final_info['rotation_det']:.9f}")
    print(f"  translation:  {final_info['translation']}")

    # Apply final transform to original full-res prediction.
    pred_pts_aligned = transform_points(pred_pts_full, T_final)
    pred_aligned = copy_colors_and_normals(pred, pred_pts_aligned)

    out_path = Path(args.out_ply)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_path), pred_aligned)
    print(f"\nSaved aligned prediction:")
    print(f"  {out_path}")

    if args.save_transform is not None:
        transform_path = Path(args.save_transform)
        transform_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(transform_path, T_final)
        print(f"Saved transform:")
        print(f"  {transform_path}")

    diagnostics = {
        "pred_ply": args.pred_ply,
        "gt_ply": args.gt_ply,
        "out_ply": str(out_path),
        "voxel_size": args.voxel_size,
        "trim_fraction": args.trim_fraction,
        "max_corr_dist": args.max_corr_dist,
        "sim3": final_info,
        "icp": icp_diag,
        "note": (
            "This aligns the predicted point cloud to DTU GT coordinates using a single global Sim(3) transform. "
            "The output PLY can be passed to the DTU MATLAB evaluation script. "
            "Official DTU masking/region filtering should still be handled by the MATLAB evaluator."
        ),
    }

    if args.save_diagnostics is not None:
        diag_path = Path(args.save_diagnostics)
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        with open(diag_path, "w") as f:
            json.dump(diagnostics, f, indent=2)
        print(f"Saved diagnostics:")
        print(f"  {diag_path}")

    if args.scan_id is not None and args.dtu_points_dir is not None:
        points_dir = Path(args.dtu_points_dir)
        points_dir.mkdir(parents=True, exist_ok=True)

        dtu_name = f"scan{args.scan_id:03d}.ply"
        dtu_out = points_dir / dtu_name

        o3d.io.write_point_cloud(str(dtu_out), pred_aligned)

        print(f"Saved DTU-style prediction:")
        print(f"  {dtu_out}")


if __name__ == "__main__":
    main()