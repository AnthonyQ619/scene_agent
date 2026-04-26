# from code_refinement_agent import CodeRefinementLLM
from core.imageanalyzer import ImageAnalyzer
from core.generator import Generator
from core.compiler import Compiler
from core.debugger import Debugger
from core.promptenhancer import PromptEnhancerLLM
from sceneprogllm import SceneProgTemplate, LLM
# from src.agent.code_refinement_agent import CodeRefine
import time
import os
import subprocess
import sys
import glob

class Executor():
    def __init__(self, 
                 script_dir: str, 
                 output_file: str):
        if os.path.isdir(script_dir):
            self.dir = script_dir   # Path for temp directory
        else:
            Exception("No Directory or Python File exists")

        self.log_file = os.path.join(self.dir, output_file)


    def __call__(self, script_code: str):

        script_path = os.path.join(self.dir, "temp_gen_code.py")
        if not os.path.isfile(script_path):
            Exception("No Directory or Python File exists")
        
        # Create Code
        temp_file = open(script_path, 'w')
        temp_file.write(script_code)
        temp_file.close()


        # Change current working directory
        os.chdir(self.dir)
        current_env = os.environ.copy()

        with open(self.log_file, "w") as f:
            print("🚀 Running script...")
            result = subprocess.run(
                [sys.executable, "temp_gen_code.py"],
                env=current_env,
                stdout=f,      # send standard output to file
                stderr=f       # send error output to file
            )

            f.close()

        log_file = open(self.log_file, 'r')

        output = log_file.read().strip()
        
        log_file.close()
        self.cleanup(self.log_file, script_path)

        if result.returncode == 0:
            return output, True
        else:
            return output, False
    
    def cleanup(self, log_file_path: str, script_path: str):
        if os.path.exists(log_file_path):
            os.remove(log_file_path)
        if os.path.exists(script_path):
            os.remove(script_path)

class SceneReconAgent:
    def __init__(self,
                 api_directory: str,
                 script_dir: str, 
                 output_file: str,
                 instruction_path: str | None = None,
                 k_images: int = 5,
                 model_name="gpt-5",    
                 reasoning_effort="medium",
                 num_procedures=5,
                 num_samples=10):
        
        self.template = SceneProgTemplate(
"""
<cr>
You are an expert in Python scripting for the provided Custom API above for Structure from Motion solutions. I want you to debug the following code by referring to the provided API information.
I want you to especially pay attention to the Module Names and parameter names being assigned to the modules called. The module names and parameters should be exactly the same as the one in the documentation.
An example of a valid output is:
```python
# STEP 2: Estimate Features per Image
from modules.features import FeatureDetectionSIFT
# Feature Module Initialization
feature_detector = FeatureDetectionSIFT(cam_data=camera_data, 
                                        max_keypoints=19000,
                                        contrast_threshold=0.005,
                                        edge_threshold=10)

# Detect Features for all Images
features = feature_detector()

# STEP 3: Match Features Per Image 
from modules.featurematching import FeatureMatchFlannPair
# Pairwise Feature Matching Module Initialization
feature_matcher = FeatureMatchFlannPair(detector='sift', 
                                        cam_data=camera_data,
                                        RANSAC_threshold=0.3,
                                        lowes_thresh=0.75)

# Detect Image Pair Correspondences for Pose Estimation
feature_pairs = feature_matcher(features=features)
# Code continued...
```
Key things to ensure: 
1. For import feature tracking modules, from modules.featurematching import ... instead of setting it like from modules.featuretracking import ... (this is a common mistake).
2. For setting local bundle adjustment in pose estimation, do it like optimizer_local = BundleAdjustmentOptimizerLocal(..) to pose_estimator = CamPoseEstimatorEssentialToPnP(..., optimizer=optimizer_local) as this is a special case.
3. For constructing dense reconstruction, the function call dense_reconstruction = Dense3DReconstructionMono(...) does not take scene data due to sparse reconstruction creating a colmap workspace, so this module uses that workspace
4. For constructing dense reconstruction with VGGT, skip sparse reconstruction, such as CamPoseEstimatorVGGTModel(...) to Dense3DReconstructionVGGT(...) as this is a special case.
</cr>

<tr>
You are an expert in Python scripting for the provided Custom API above for Structure from Motion solutions. I want you to debug the following code by referring to the provided API information.
You will be given a python script that uses the the Custom Python API to create a reconstruction pipeline with the traceback containing errors obtained by running the script.
Your task is to fix the code. An example of a valid output (Specifically step 2 and 3, other steps are excluded in this example):
```python
# STEP 2: Estimate Features per Image
from modules.features import FeatureDetectionSIFT
# Feature Module Initialization
feature_detector = FeatureDetectionSIFT(cam_data=camera_data, 
                                        max_keypoints=19000,
                                        contrast_threshold=0.005,
                                        edge_threshold=10)

# Detect Features for all Images
features = feature_detector()

# STEP 3: Match Features Per Image 
from modules.featurematching import FeatureMatchFlannPair
# Pairwise Feature Matching Module Initialization
feature_matcher = FeatureMatchFlannPair(detector='sift', 
                                        cam_data=camera_data,
                                        RANSAC_threshold=0.3,
                                        lowes_thresh=0.75)

# Detect Image Pair Correspondences for Pose Estimation
feature_pairs = feature_matcher(features=features)
# Code continued...
```
Key things to ensure: 
1. For import feature tracking modules, from modules.featurematching import ... instead of setting it like from modules.featuretracking import ... (this is a common mistake).
2. For setting local bundle adjustment in pose estimation, do it like optimizer_local = BundleAdjustmentOptimizerLocal(..) to pose_estimator = CamPoseEstimatorEssentialToPnP(..., optimizer=optimizer_local) as this is a special case.
3. For constructing dense reconstruction, the function call dense_reconstruction = Dense3DReconstructionMono(...) does not take scene data due to sparse reconstruction creating a colmap workspace, so this module uses that workspace
4. For constructing dense reconstruction with VGGT, skip sparse reconstruction, such as CamPoseEstimatorVGGTModel(...) to Dense3DReconstructionVGGT(...) as this is a special case.
5. Most importantly, do not write your own code to bypass errors. Resolve any bugs by either fixing incorrect parameters, trying different modules, or fixing syntax errors. Do not generate any python code outside of the API usage.
</tr>
""",
)
        # For Debugging
        self.script_dir = script_dir

        # Read API Desc
        # Gather API Information
        if api_directory is None:
            self.api_files = []
        else:
            self.api_files = sorted(glob.glob(api_directory + "\\*"))

        api_desc = ""
        for file in self.api_files:
            context = open(file, 'r')
            examples = context.read()
            context.close()
            api_desc += examples + "\n"

        # For Clean up on Script
        system_desc = f"""
You are an expert in generating Structure from Motion solution pipelines in Computer Vision, specifically python scripts using a custom API.
Given python code that reconstructs the scene using the provided custom python API, you are set to make a few changes on the code to clean up
the code generation process for the final code.

Given the API description
=========================
{api_desc}
=========================
You are set to make two changes, and do not edit anything else. 

First, remove max image count in the Camera Data Manager initialization like so:
# What is provided
CDM = CameraDataManager(image_path=image_path,
                        max_images = 20
                        calibration_path=calibration_path)

# What is needed
CDM = CameraDataManager(image_path=image_path,
                        calibration_path=calibration_path)

If the dense reconstruction module is provided (Dense3DReconstructionVGGT or Dense3DReconstructionMono),
swap the build_dense function with the provided __call__ function:

# WHAT IS PROVIDED
from modules.scenereconstruction import Dense3DReconstructionMono

# Conduct Dense Reconstruction Module
dense_reconstruction = Dense3DReconstructionMono(cam_data=camera_data,
                                                  reproj_error=3.0,
                                                  min_triangulation_angle=1.0,
                                                  num_samples=15,
                                                  num_iterations=3)

# Estimate sparse 3D scene from tracked features and camera poses
dense_scene = dense_reconstruction.build_dense(sparse_scene=optimal_scene)

# WHAT IS NEEDED
from modules.scenereconstruction import Dense3DReconstructionMono

# Conduct Dense Reconstruction Module
dense_reconstruction = Dense3DReconstructionMono(cam_data=camera_data,
                                                  reproj_error=3.0,
                                                  min_triangulation_angle=1.0,
                                                  num_samples=15,
                                                  num_iterations=3)

# Estimate sparse 3D scene from tracked features and camera poses
dense_scene = dense_reconstruction(sparse_scene=optimal_scene)

The goal is to output the same python code that is provided but only make the above changes if they exist.
"""

        self.exec = Executor(script_dir=script_dir, output_file=output_file)

        #self.analyzer = ImageAnalyzer(model = model_name, k_images= k_images, reasoning_effort=reasoning_effort)
        self.enhancer = PromptEnhancerLLM(model= model_name, reasoning_effort=reasoning_effort, instruction_path=instruction_path)
        self.generator = Generator(model= model_name, api_directory=api_directory, reasoning_effort=reasoning_effort)
        self.compiler = Compiler(script_dir=script_dir, model = model_name, api_directory=api_directory, reasoning_effort=reasoning_effort)
        self.debugger = Debugger(executor=self.exec, template=self.template, model=model_name, reasoning_effort=reasoning_effort, api_path=api_directory)
        self.cleaner = LLM(system_desc = system_desc,
                            response_format = "code",
                            model_name = model_name,
                            reasoning_effort = reasoning_effort,
                            temperature = 0.8)
        
    def __call__(self, prompt):
        log_file = open(self.script_dir + "/time_log.txt", 'w')

#         t0 = time.time()
#         result = self.analyzer(prompt['images'])
#         t1 = time.time()
#         log_file.write("ANALYZER LLM (Reading Images to Text):\n" + str(t1-t0) + "\n")

#         new_prompt = f"""
# Scene Description: {result}
# Camera Calibration Path: {prompt['calibration']}
# Reconstruction Type: {prompt['recon_type']}
# Hardware Memory: {prompt['gpu_mem']}
# """
#         t0 = time.time()
#         new_query = self.enhancer(new_prompt)
#         t1 = time.time()
#         log_file.write("ENHANCER LLM (PROMPT GENERATRION):\n" + str(t1-t0) + "\n")

#         new_prompt = f"""
# {new_query["statement"]}
# {new_query["reconstruction"]}
# {new_query["calibration"]}
# {new_query["memory"]}
# {new_query["scene_summary"]}
# Use Image Path in Code: {prompt['images']}
# Use Calibration Path in Code: {prompt['calibration']}
# """   
        # This is to guide the problem in a more formal prompt

        # PLANNER #
        pre_prompt = f"""
# Camera Calibration Path: {prompt['calibration']}
# Reconstruction Type: {prompt['recon_type']}
# Hardware Memory: {prompt['gpu_mem']}
# """
        t0 = time.time()
        new_query = self.enhancer(pre_prompt)
        t1 = time.time()
        log_file.write("ENHANCER LLM (PROMPT GENERATRION):\n" + str(t1-t0) + "\n")

        new_prompt = f"""
{new_query["statement"]}
{new_query["reconstruction"]}
{new_query["calibration"]}
{new_query["memory"]}
Use Image Path in Code: {prompt['images']}
Use Calibration Path in Code: {prompt['calibration']}
"""
        print(new_prompt)
        print({prompt['images']})
        t0 = time.time()
        procedure = self.generator(new_prompt, prompt['images'])
        t1 = time.time()
        log_file.write("GENERATOR LLM (PROCEDURE GENERATRION):\n" + str(t1-t0) + "\n")

        # Save the Procedure Generation
        temp_file = open(self.script_dir + "/gen_procedure.txt", 'w', encoding="utf-8")

        temp_file.write(procedure)

        temp_file.close()

        print(procedure)
        # PLANNER #

        
        # PROGRAM SYNTHESIS #
        t0 = time.time()
        code = self.compiler(procedure)
        t1 = time.time()
        log_file.write("COMPILER LLM (CODE GENERATRION):\n" + str(t1-t0) + "\n")

        # Save the initial Code
        temp_file = open(self.script_dir + "/gen_code.py", 'w', encoding="utf-8")

        temp_file.write(code)

        temp_file.close()
        """
        t0 = time.time()
        refined_code = self.debugger(code)
        t1 = time.time()
        log_file.write("DEBUGGER LLM:\n" + str(t1-t0) + "\n")
        # PROGRAM SYNTHESIS #

        # Save the refined Code
        temp_file = open(self.script_dir + "\\refined_gen_code.py", 'w', encoding="utf-8")

        temp_file.write(refined_code)

        temp_file.close()


        # Clean Code to ignore Dense Reconstruction !
        t0 = time.time()
        cleaned_code = self.cleaner(refined_code)
        t1 = time.time()
        log_file.write("CLEANER LLM:\n" + str(t1-t0) + "\n")

        # Save the refined Code
        temp_file = open(self.script_dir + "\\cleaned_gen_code.py", 'w', encoding="utf-8")

        temp_file.write(cleaned_code)

        temp_file.close()
        # Cleaned Code

        # OPTIMIZER GOES HERE

        log_file.close()

        # return refined_code"""
        return True



# Initial Params
api_path = '/home/anthonyq/projects/scene_agent/breadth_agent/src/agent/agent_details/tool_context'
log_file = "sfm_log.txt"
script_dir_testing = '/home/anthonyq/projects/scene_agent/breadth_agent/src/agent/generated_code'
instruction_path = '/home/anthonyq/projects/scene_agent/breadth_agent/src/agent/agent_details/agent_instructions/prompt_enh_examples.txt'

# Prompt
image_path = r"/home/anthonyq/datasets/DTU/DTU/scan10/images"
calibration_path = r"/home/anthonyq/datasets/DTU/DTU/calibration_DTU_new.npz"
reconstruction_type = "Sparse Reconstruction"
gpu_mem = "12gb"
temp_prompt = {'images':image_path, 
                'calibration':calibration_path, 
                'recon_type':reconstruction_type, 
                'gpu_mem':gpu_mem}      

sa = SceneReconAgent(api_directory=api_path,
                     script_dir=script_dir_testing,
                     output_file=log_file,
                     instruction_path=instruction_path)

result = sa(temp_prompt)

print(result)