from autosfm import AutoSFM
from core.logger import Logger

api_directory = "/home/anthonyq/projects/scene_agent/breadth_agent/src/agent/agent_details/tool_context"
instruction_path = "/home/anthonyq/projects/scene_agent/breadth_agent/src/agent/agent_details/agent_instructions/prompt_enh_examples.txt"
logger = Logger(desc="DTU_scan20", log_dir="/home/anthonyq/projects/scene_agent/breadth_agent/results")
autosfm = AutoSFM(model_name="gpt-5", 
                api_directory=api_directory,#'/work/scene_agent/breadth_agent/src/agent/agent_details/tool_context', 
                instruction_path=instruction_path,#'/work/scene_agent/breadth_agent/src/agent/agent_details/agent_instructions/prompt_enh_examples.txt', 
                reasoning_effort="medium",
                logger=logger)

# from core.generator import Generator
# model_name="gpt-5"
# api_directory = '/work/scene_agent/breadth_agent/src/agent/agent_details/tool_context'
# reasoning_effort="medium"
# generator = Generator(model= model_name, api_directory=api_directory, reasoning_effort=reasoning_effort)

# Prompt
# Fail Test
# image_path = "/home/anthonyq/datasets/ETH/ETH/living_room/images/dslr_images_undistorted"#r"/work/dataset/DTU/scan10/images"
# calibration_path = "/home/anthonyq/datasets/ETH/ETH/office/dslr_calibration_undistorted/calibration_ETH_new.npz"#r"/work/dataset/DTU/calibration_DTU_new.npz"
# Successful Test
image_path = "/home/anthonyq/datasets/DTU/DTU/scan20/images"
calibration_path = "/home/anthonyq/datasets/DTU/DTU/calibration_DTU_new.npz"
reconstruction_type = "Sparse Reconstruction"
gpu_mem = "12gb"
temp_prompt = {'images':image_path,
'calibration':calibration_path,
'recon_type':reconstruction_type,
'gpu_mem':gpu_mem}

results = autosfm.run(temp_prompt)
breakpoint()

# from core.promptenhancer import PromptEnhancerLLM
# instruction_path = '/work/scene_agent/breadth_agent/src/agent/agent_details/agent_instructions/prompt_enh_examples.txt'
# enhancer = PromptEnhancerLLM(model= model_name, reasoning_effort=reasoning_effort, instruction_path=instruction_path)

# new_query = enhancer(temp_prompt)
# new_prompt = f"""
# {new_query["statement"]}
# {new_query["reconstruction"]}
# {new_query["calibration"]}
# {new_query["memory"]}
# Use Image Path in Code: {temp_prompt['images']}
# Use Calibration Path in Code: {temp_prompt['calibration']}
# """
# plan = generator(new_prompt, temp_prompt['images'])
# breakpoint()

# process = """

# GOAL RECONSTRUCTION: Sparse Reconstruction (calibrated SfM with accurate poses and a globally consistent sparse point cloud)

# Scene description and rationale:
# - The images depict a high-textured cardboard box with strong edges, corners, and pen strokes under bright indoor lighting on a largely white background. Surfaces are rigid and richly textured, ideal for classical keypoint detectors and robust geometric reasoning.
# - Camera moves incrementally around the object with substantial overlap; intrinsics are known from DTU calibration.
# - Hardware has 12 GB VRAM; avoid heavy transformer pose estimators. Use classical SIFT + robust matching/tracking, PnP-based pose estimation, and bundle adjustment.

# STEP 1: Initialize the scene through SfMScene and read in camera data
# - Initialize the scene as SfMScene(...)
# - Set the image path to the DTU sequence:
#   - image_path = /work/dataset/DTU/scan10/images
# - Limit images to a manageable subset for consistent overlap and runtime:
#   - max_images = 20
# - Provide the known calibration for a calibrated monocular workflow:
#   - calibration_path = /work/dataset/DTU/calibration_DTU_new.npz
# - Reasoning: We have known intrinsics and incremental motion; this fits a classical calibrated SfM pipeline.

# STEP 2: Detect Features
# - Use FeatureDetectionSIFT
#   - max_keypoints = 12000
#     - Reasoning: The box has many edges and textures; increasing keypoints yields dense, reliable correspondences across views.
#   - contrast_threshold = 0.01 (default 0.04)
#     - Reasoning: Lower threshold to admit more features on textured paint/pen strokes.
#   - edge_threshold = 12 (default 10)
#     - Reasoning: Slightly higher to capture additional edge-like features along box seams and corners.
#   - sigma = 1.6 (default)
# - Why SIFT: High texture, good lighting, and rigid geometry favor SIFT for accurate, repeatable keypoints and descriptors.

# STEP 3: Detect Pairwise Feature Matches for initial pose bootstrapping
# - Use FeatureMatchBFPair with SIFT descriptors for robust, exhaustive matching
#   - detector = "sift"
#   - k = 2
#   - lowes_thresh = 0.76
#     - Reasoning: Slightly stricter than default to reduce ambiguous matches while retaining enough inliers for PnP.
#   - RANSAC = True
#   - RANSAC_threshold = 0.02 (normalized coords)
#     - Reasoning: Tight epipolar inlier threshold given good calibration and clean features.
#   - RANSAC_conf = 0.999
#   - RANSAC_homography = False
#     - Reasoning: Structure is not dominantly planar at the image scale; fundamental model is appropriate.
# - Why BF: Maximizes match accuracy, which is crucial for stable PnP and minimal outlier contamination.

# STEP 4: Estimate camera poses (extrinsics) using matched pairs
# - Use CamPoseEstimatorEssentialToPnP
#   - iteration_count = 220
#     - Reasoning: Sufficient LM iterations for well-conditioned correspondences.
#   - reprojection_error = 3.0 (pixels)
#   - confidence = 0.995
#   - ba_per_frame = 4 (default)
#   - optimizer = ("BundleAdjustmentOptimizerLocal", {
#       "max_num_iterations": 25,
#       "robust_loss": True,
#       "use_gpu": False,
#       "window_size": 8,
#       "min_track_len": 3
#     })
#     - Reasoning: Even with SIFT, small local BA windows help dampen drift from any residual mismatches and stabilize scale/trajectory early; CPU BA suffices.
# - Why EssentialToPnP: Calibrated monocular with incremental motion; this yields accurate, in-scale poses without heavy models.

# STEP 5: Track features across multiple images for multi-view triangulation
# - Use FeatureMatchBFTracking with SIFT for accurate long tracks
#   - detector = "sift"
#   - k = 2
#   - cross_check = True
#     - Reasoning: Symmetric consistency improves track purity for triangulation.
#   - lowes_thresh = 0.72
#     - Reasoning: Slightly tighter threshold to prioritize clean, long tracks.
#   - RANSAC_threshold = 0.015 (normalized coords)
#     - Reasoning: Tight inlier gate to maintain high-quality tracks in this textured, well-lit scene.
#   - RANSAC_conf = 0.999
# - Why BFTracking: Emphasizes accuracy and track consistency over speed, producing long, clean multi-view tracks ideal for precise sparse reconstruction.

# STEP 6: Sparse 3D reconstruction from multi-view
# - Use Sparse3DReconstructionMono
#   - view = True (multi_view = True; using tracked features)
#   - min_observe = 4
#     - Reasoning: Strong texture and long tracks allow a stricter minimum, improving triangulation stability.
#   - min_angle = 2.5 degrees
#     - Reasoning: Enforce a moderate triangulation angle for accurate depth while retaining sufficient points.
#   - reproj_error = 2.0 (pixels)
#     - Reasoning: Tight reprojection filtering leverages good features/poses to keep only geometrically consistent 3D points.
# - Output: Initial sparse point cloud, camera poses, and observations.

# STEP 7: Global bundle adjustment for final consistent sparse model
# - Use BundleAdjustmentOptimizerGlobal
#   - refine_focal_length = False
#   - refine_principal_point = False
#   - refine_extra_params = False
#     - Reasoning: Camera intrinsics are known and pre-calibrated; fix them to prevent bias.
#   - max_num_iterations = 180
#   - use_gpu = False
#   - robust_loss = True
# - Reasoning: Final global BA jointly refines camera poses and 3D structure to minimize reprojection error and remove residual inconsistencies; no need for GPU at this problem size.

# Notes and final remarks:
# - Dense reconstruction is not required; stop after global BA on the sparse model.
# - The chosen classical pipeline (SIFT + BF matching/tracking + EssentialToPnP + BA) is memory efficient for 12 GB VRAM, leverages known intrinsics, and fits the high-texture, well-lit, incremental-view DTU scene for accurate poses and a consistent sparse point cloud.
# """

# from core.compiler import Compiler
# script_dir = '/work/scene_agent/breadth_agent/src/tests'
# api_directory = '/work/scene_agent/breadth_agent/src/agent/agent_details/tool_context'
# model_name="gpt-5"
# reasoning_effort="medium"

# compiler = Compiler(script_dir=script_dir, model = model_name, api_directory=api_directory, reasoning_effort=reasoning_effort)

# program = compiler(process)
# breakpoint()