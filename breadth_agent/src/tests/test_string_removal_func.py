import ast


sample_code = """
# Construct Modules with Initialized Arguments
image_path ="/home/anthonyq/datasets/DTU/DTU/scan14" #"C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\scan14_normal_lighting"
calibration_path = "/home/anthonyq/datasets/DTU/DTU/calibration_DTU_new.npz" #"C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\calibration_DTU_new.npz"

from modules.features import FeatureDetectionSIFT
from modules.featurematching import FeatureMatchFlannPair, FeatureMatchBFTracking
from modules.camerapose import CamPoseEstimatorEssentialToPnP
from modules.optimization import BundleAdjustmentOptimizerLocal
from modules.scenereconstruction import Sparse3DReconstructionMono, Dense3DReconstructionMono
from modules.optimization import BundleAdjustmentOptimizerGlobal
from modules.baseclass import SfMScene

# Step 1: Read in Calibration/Image Data
reconstructed_scene = SfMScene(id=1,
                                image_path = image_path, 
                                max_images = 20,
                                calibration_path = calibration_path)

# Step 2: Detect Features
reconstructed_scene.FeatureDetectionSIFT(
    max_keypoints=15000,
    contrast_threshold=0.009,
    edge_threshold=20
)

# Step 3: Detect Feature Pairs
reconstructed_scene.FeatureMatchFlannPair(
    detector="sift",
    lowes_thresh=0.8,
    RANSAC_homography=False,
    RANSAC_threshold=2.0,
)

# Step 4: Detect/Estimate Camera Poses
reconstructed_scene.CamPoseEstimatorEssentialToPnP(
    reprojection_error=3.0,
    iteration_count=200,
    confidence=0.995
)

# Step 5: Detect Feature Tracks
reconstructed_scene.FeatureMatchBFTracking(
    detector="sift",
    RANSAC_threshold=1.0,
    lowes_thresh=0.65
)

# Step 6: Estimate Sparse Reconstruction
reconstructed_scene.Sparse3DReconstructionMono(
    min_observe=4,
    min_angle=2.0,
    multi_view=True
)

# Step 7: Run Optimization
reconstructed_scene.BundleAdjustmentOptimizerGlobal(
    max_num_iterations=200
)

# STEP 8: Run Rense Reconstruction Algorithm
reconstructed_scene.Dense3DReconstructionMono(
    reproj_error=3.0,
    min_triangulation_angle=1.0,
    num_samples=15,
    num_iterations=3
)
"""

def remove_module_call(script: str, module_name: str) -> str:
    """
    Removes calls like:
        reconstructed_scene.<module_name>(...)
    from a Python script string.

    Example:
        remove_module_call(script, "Dense3DReconstructionMono")
    """
    tree = ast.parse(script)

    lines = script.splitlines()

    ranges_to_remove = []

    for node in ast.walk(tree):
        # Looking for standalone expressions:
        # reconstructed_scene.Dense3DReconstructionMono(...)
        if not isinstance(node, ast.Expr):
            continue

        call = node.value
        if not isinstance(call, ast.Call):
            continue

        func = call.func
        if not isinstance(func, ast.Attribute):
            continue

        if func.attr != module_name:
            continue

        # Optional: make sure it is called from reconstructed_scene
        if isinstance(func.value, ast.Name) and func.value.id == "reconstructed_scene":
            start = node.lineno
            end = node.end_lineno
            ranges_to_remove.append((start, end))

    if not ranges_to_remove:
        return script

    # Remove the last matching module call
    start, end = ranges_to_remove[-1]

    new_lines = [
        line
        for i, line in enumerate(lines, start=1)
        if not (start <= i <= end)
    ]

    # Clean trailing blank lines
    return "\n".join(new_lines).rstrip() + "\n"

def remove_keyword_from_first_call(
    script: str,
    call_name: str,
    keyword_name: str,
) -> str:
    """
    Removes one keyword argument from the first call to `call_name`.

    Example:
        remove_keyword_from_first_call(
            script,
            call_name="SfMScene",
            keyword_name="max_images"
        )

    Removes:
        max_images=20

    while preserving the rest of the script, including comments.
    """

    tree = ast.parse(script)

    # Convert line/column offsets to absolute string offsets
    line_starts = []
    offset = 0
    for line in script.splitlines(keepends=True):
        line_starts.append(offset)
        offset += len(line)

    def abs_offset(lineno: int, col: int) -> int:
        return line_starts[lineno - 1] + col

    target_keyword = None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        # Match SfMScene(...)
        if isinstance(node.func, ast.Name) and node.func.id == call_name:
            for kw in node.keywords:
                if kw.arg == keyword_name:
                    target_keyword = kw
                    break

        if target_keyword is not None:
            break

    if target_keyword is None:
        return script

    start = abs_offset(target_keyword.lineno, target_keyword.col_offset)
    end = abs_offset(target_keyword.end_lineno, target_keyword.end_col_offset)

    # Extend removal to include a following comma, if present
    i = end
    while i < len(script) and script[i].isspace():
        i += 1

    if i < len(script) and script[i] == ",":
        end = i + 1
    else:
        # Otherwise remove the preceding comma
        j = start - 1
        while j >= 0 and script[j].isspace():
            j -= 1

        if j >= 0 and script[j] == ",":
            start = j

    return script[:start] + script[end:]

# Test dense removal
script_without_dense = remove_module_call(
    sample_code,
    module_name="Dense3DReconstructionMono"
)

generated_script_without_max = remove_keyword_from_first_call(
    sample_code,
    call_name="SfMScene",
    keyword_name="max_images",
)

print(generated_script_without_max)