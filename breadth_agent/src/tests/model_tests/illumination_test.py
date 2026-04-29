from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence
import cv2
import numpy as np
import glob

# -----------------------------
# Label helpers
# -----------------------------

def label_score(score: float, low: float = 0.12, high: float = 0.28) -> str:
    """
    Generic LOW / MEDIUM / HIGH label for normalized change scores.
    Tune thresholds for your datasets.
    """
    if score < low:
        return "LOW"
    elif score < high:
        return "MEDIUM"
    return "HIGH"


def safe_norm_hist(hist: np.ndarray) -> np.ndarray:
    hist = hist.astype(np.float32)
    s = hist.sum()
    if s <= 1e-8:
        return hist
    return hist / s


# -----------------------------
# Per-image metrics
# -----------------------------

@dataclass
class ImageIlluminationStats:
    path: str

    mean_luminance: float
    median_luminance: float
    std_luminance: float

    shadow_clip_ratio: float
    highlight_clip_ratio: float

    mean_a: float
    mean_b: float
    colorfulness: float


def compute_image_stats(image_path: str | Path) -> tuple[ImageIlluminationStats, dict]:
    """
    Computes per-image illumination/color stats and returns histograms
    used for pairwise comparison.

    Notes:
    - OpenCV reads images as BGR.
    - LAB L channel is used as luminance.
    - LAB A/B channels are used for color shift.
    """

    image_path = Path(image_path)
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    L = lab[:, :, 0].astype(np.float32)  # 0 to 255
    A = lab[:, :, 1].astype(np.float32)  # centered around 128
    B = lab[:, :, 2].astype(np.float32)  # centered around 128

    # Normalize luminance stats to [0, 1]
    mean_lum = float(np.mean(L) / 255.0)
    median_lum = float(np.median(L) / 255.0)
    std_lum = float(np.std(L) / 255.0)

    # Clipping ratios
    shadow_clip_ratio = float(np.mean(L <= 5))
    highlight_clip_ratio = float(np.mean(L >= 250))

    # Mean color channels, centered around 0 and normalized roughly to [-1, 1]
    mean_a = float((np.mean(A) - 128.0) / 127.0)
    mean_b = float((np.mean(B) - 128.0) / 127.0)

    # Simple colorfulness proxy from LAB chroma
    chroma = np.sqrt((A - 128.0) ** 2 + (B - 128.0) ** 2)
    colorfulness = float(np.mean(chroma) / 181.0)  # 181 ~= sqrt(127^2 + 127^2)

    # Histograms for pairwise distribution shift
    lum_hist = cv2.calcHist(
        [lab],
        channels=[0],
        mask=None,
        histSize=[64],
        ranges=[0, 256],
    )
    lum_hist = safe_norm_hist(lum_hist)

    ab_hist = cv2.calcHist(
        [lab],
        channels=[1, 2],
        mask=None,
        histSize=[32, 32],
        ranges=[0, 256, 0, 256],
    )
    ab_hist = safe_norm_hist(ab_hist)

    stats = ImageIlluminationStats(
        path=str(image_path),
        mean_luminance=mean_lum,
        median_luminance=median_lum,
        std_luminance=std_lum,
        shadow_clip_ratio=shadow_clip_ratio,
        highlight_clip_ratio=highlight_clip_ratio,
        mean_a=mean_a,
        mean_b=mean_b,
        colorfulness=colorfulness,
    )

    aux = {
        "lum_hist": lum_hist,
        "ab_hist": ab_hist,
    }

    return stats, aux


# -----------------------------
# Pairwise change metrics
# -----------------------------

@dataclass
class PairIlluminationChange:
    pair: tuple[int, int]

    delta_mean_luminance: float
    delta_median_luminance: float
    delta_luminance_std: float

    luminance_hist_distance: float
    color_hist_distance: float
    white_balance_shift: float

    delta_shadow_clip_ratio: float
    delta_highlight_clip_ratio: float

    illumination_change_score: float
    color_shift_score: float
    exposure_shift_score: float
    combined_change_score: float

    illumination_label: str
    color_shift_label: str
    exposure_shift_label: str
    combined_label: str


def compute_pair_change(
    i: int,
    j: int,
    stats: list[ImageIlluminationStats],
    aux: list[dict],
) -> PairIlluminationChange:
    s1 = stats[i]
    s2 = stats[j]

    # Basic luminance deltas
    delta_mean_lum = abs(s1.mean_luminance - s2.mean_luminance)
    delta_median_lum = abs(s1.median_luminance - s2.median_luminance)
    delta_std_lum = abs(s1.std_luminance - s2.std_luminance)

    # Histogram distances are already roughly [0, 1] for Bhattacharyya
    lum_hist_dist = float(
        cv2.compareHist(aux[i]["lum_hist"], aux[j]["lum_hist"], cv2.HISTCMP_BHATTACHARYYA)
    )

    color_hist_dist = float(
        cv2.compareHist(aux[i]["ab_hist"], aux[j]["ab_hist"], cv2.HISTCMP_BHATTACHARYYA)
    )

    # White balance / color cast shift from mean LAB a/b
    wb_vec_1 = np.array([s1.mean_a, s1.mean_b], dtype=np.float32)
    wb_vec_2 = np.array([s2.mean_a, s2.mean_b], dtype=np.float32)

    # Max possible distance in normalized AB space is about sqrt(8), but sqrt(2)
    # is a practical scale for most real images.
    white_balance_shift = float(
        np.linalg.norm(wb_vec_1 - wb_vec_2) / np.sqrt(2.0)
    )
    white_balance_shift = float(np.clip(white_balance_shift, 0.0, 1.0))

    delta_shadow_clip = abs(s1.shadow_clip_ratio - s2.shadow_clip_ratio)
    delta_highlight_clip = abs(s1.highlight_clip_ratio - s2.highlight_clip_ratio)

    # Weighted scores.
    # Tune these weights depending on whether you care more about raw brightness,
    # luminance distribution, color cast, or clipping.
    illumination_change_score = float(
        0.35 * delta_mean_lum
        + 0.25 * delta_median_lum
        + 0.15 * delta_std_lum
        + 0.25 * lum_hist_dist
    )

    color_shift_score = float(
        0.65 * color_hist_dist
        + 0.35 * white_balance_shift
    )

    exposure_shift_score = float(
        0.45 * delta_mean_lum
        + 0.25 * delta_shadow_clip
        + 0.25 * delta_highlight_clip
        + 0.05 * delta_std_lum
    )

    combined_change_score = float(
        0.45 * illumination_change_score
        + 0.35 * color_shift_score
        + 0.20 * exposure_shift_score
    )

    return PairIlluminationChange(
        pair=(i, j),

        delta_mean_luminance=float(delta_mean_lum),
        delta_median_luminance=float(delta_median_lum),
        delta_luminance_std=float(delta_std_lum),

        luminance_hist_distance=float(lum_hist_dist),
        color_hist_distance=float(color_hist_dist),
        white_balance_shift=float(white_balance_shift),

        delta_shadow_clip_ratio=float(delta_shadow_clip),
        delta_highlight_clip_ratio=float(delta_highlight_clip),

        illumination_change_score=illumination_change_score,
        color_shift_score=color_shift_score,
        exposure_shift_score=exposure_shift_score,
        combined_change_score=combined_change_score,

        illumination_label=label_score(illumination_change_score),
        color_shift_label=label_score(color_shift_score),
        exposure_shift_label=label_score(exposure_shift_score),
        combined_label=label_score(combined_change_score),
    )


# -----------------------------
# Dataset-level summary
# -----------------------------

def summarize_scores(values: Sequence[float]) -> dict:
    values = np.asarray(values, dtype=np.float32)

    if len(values) == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "max": 0.0,
        }

    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def compute_dataset_illumination_summary(
    image_paths: Sequence[str | Path],
    pairs: Sequence[tuple[int, int]] | None = None,
    worst_k: int = 5,
) -> dict:
    """
    Computes dataset-level illumination/color summary.

    By default, this compares consecutive image pairs:
        (0, 1), (1, 2), ..., (N-2, N-1)

    You can also pass custom pairs, for example retrieved/matched image pairs.
    """

    image_paths = list(image_paths)

    if len(image_paths) < 2:
        raise ValueError("Need at least two images to compute illumination change.")

    image_stats: list[ImageIlluminationStats] = []
    image_aux: list[dict] = []

    for path in image_paths:
        stats, aux = compute_image_stats(path)
        image_stats.append(stats)
        image_aux.append(aux)

    if pairs is None:
        pairs = [(i, i + 1) for i in range(len(image_paths) - 1)]

    pair_changes = [
        compute_pair_change(i, j, image_stats, image_aux)
        for i, j in pairs
    ]

    illum_scores = [p.illumination_change_score for p in pair_changes]
    color_scores = [p.color_shift_score for p in pair_changes]
    exposure_scores = [p.exposure_shift_score for p in pair_changes]
    combined_scores = [p.combined_change_score for p in pair_changes]

    # Ratios of problematic pairs
    high_illum_ratio = float(np.mean([p.illumination_label == "HIGH" for p in pair_changes]))
    high_color_ratio = float(np.mean([p.color_shift_label == "HIGH" for p in pair_changes]))
    high_exposure_ratio = float(np.mean([p.exposure_shift_label == "HIGH" for p in pair_changes]))
    high_combined_ratio = float(np.mean([p.combined_label == "HIGH" for p in pair_changes]))

    # Dataset-level labels use p75 so a few bad pairs do not dominate,
    # but recurring instability still shows up.
    dataset_illum_score = summarize_scores(illum_scores)["p75"]
    dataset_color_score = summarize_scores(color_scores)["p75"]
    dataset_exposure_score = summarize_scores(exposure_scores)["p75"]
    dataset_combined_score = summarize_scores(combined_scores)["p75"]

    worst_pairs = sorted(
        pair_changes,
        key=lambda x: x.combined_change_score,
        reverse=True,
    )[:worst_k]

    summary = {
        "num_images": len(image_paths),
        "num_pairs_evaluated": len(pair_changes),

        "dataset_labels": {
            "illumination_change": label_score(dataset_illum_score),
            "color_shift": label_score(dataset_color_score),
            "exposure_shift": label_score(dataset_exposure_score),
            "combined_illumination_color_change": label_score(dataset_combined_score),
        },

        "dataset_scores": {
            "illumination_change_score_p75": float(dataset_illum_score),
            "color_shift_score_p75": float(dataset_color_score),
            "exposure_shift_score_p75": float(dataset_exposure_score),
            "combined_change_score_p75": float(dataset_combined_score),
        },

        "score_statistics": {
            "illumination": summarize_scores(illum_scores),
            "color_shift": summarize_scores(color_scores),
            "exposure_shift": summarize_scores(exposure_scores),
            "combined": summarize_scores(combined_scores),
        },

        "problem_pair_ratios": {
            "high_illumination_change_ratio": high_illum_ratio,
            "high_color_shift_ratio": high_color_ratio,
            "high_exposure_shift_ratio": high_exposure_ratio,
            "high_combined_change_ratio": high_combined_ratio,
        },

        "worst_pairs": [asdict(p) for p in worst_pairs],

        "image_stats": [asdict(s) for s in image_stats],

        "agent_interpretation": make_agent_interpretation(
            illumination_label=label_score(dataset_illum_score),
            color_label=label_score(dataset_color_score),
            exposure_label=label_score(dataset_exposure_score),
            combined_label=label_score(dataset_combined_score),
            high_combined_ratio=high_combined_ratio,
        ),
    }

    return summary


def make_agent_interpretation(
    illumination_label: str,
    color_label: str,
    exposure_label: str,
    combined_label: str,
    high_combined_ratio: float,
) -> dict:
    """
    Converts numeric labels into useful LLM-agent-facing guidance.
    """

    expected_failure_modes = []
    recommended_actions = []

    if illumination_label == "HIGH":
        expected_failure_modes.append("Local descriptors may become unstable across views.")
        recommended_actions.append("Prefer illumination-robust descriptors or learned matchers.")
        recommended_actions.append("Consider CLAHE, histogram normalization, or RootSIFT.")

    if color_label == "HIGH":
        expected_failure_modes.append("Color/white-balance shift may reduce descriptor consistency.")
        recommended_actions.append("Use grayscale-based detection/description or color normalization.")

    if exposure_label == "HIGH":
        expected_failure_modes.append("Over/underexposed regions may have weak or missing features.")
        recommended_actions.append("Reject severely clipped frames or reduce their matching priority.")

    if combined_label == "HIGH":
        recommended_actions.append("Avoid relying only on classical sparse matching if inlier ratios are low.")
        recommended_actions.append("Consider SuperPoint+LightGlue, DISK+LightGlue, LoFTR, or RoMa.")

    if high_combined_ratio > 0.35:
        expected_failure_modes.append("Illumination/color instability affects a large fraction of pairs.")
        recommended_actions.append("Mark dataset as high-risk for feature-based SfM.")

    if not expected_failure_modes:
        expected_failure_modes.append("Illumination/color changes are unlikely to be the primary failure source.")

    if not recommended_actions:
        recommended_actions.append("Standard feature detection and matching should be reasonable from an illumination perspective.")

    return {
        "expected_failure_modes": expected_failure_modes,
        "recommended_actions": recommended_actions,
    }

# Test script
path = "..."
image_paths = sorted(glob.glob(path + "/*"))

summary = compute_dataset_illumination_summary(image_paths)

print(summary["dataset_labels"])
print(summary["dataset_scores"])
print(summary["problem_pair_ratios"])
print(summary["agent_interpretation"])