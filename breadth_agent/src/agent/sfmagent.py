from code_refinement_agent import CodeRefinementLLM
from core.imageanalyzer import ImageAnalyzer
from core.generator import Generator
from core.compiler import Compiler
from core.debugger import Debugger
from core.promptenhancer import PromptEnhancerLLM
from sceneprogllm import SceneProgTemplate
# from src.agent.code_refinement_agent import CodeRefine
import time
import os
import subprocess
import sys

class Executor():
    def __init__(self, 
                 script_dir: str, 
                 output_file: str):
        if os.path.isdir(script_dir):
            self.dir = script_dir   # Path for temp directory
        else:
            Exception("No Directory or Python File exists")

        self.log_file = os.path.join(self.dir, output_file)


    def __call__(self, script_file: str, script_code: str):
        script_path = os.path.join(self.dir, script_file)
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
                [sys.executable, script_file],
                env=current_env,
                stdout=f,      # send standard output to file
                stderr=f       # send error output to file
            )

            f.close()

        log_file = open(self.log_file, 'r')

        output = log_file.read().strip()
        
        log_file.close()
        self.cleanup(self.log_file)

        if result.returncode == 0:
            return output, True
        else:
            return output, False
    
    def cleanup(self, log_file_path: str):
        if os.path.exists(log_file_path):
            os.remove(log_file_path)

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
</tr>
""",
)

        self.exec = Executor(script_dir=script_dir, output_file=output_file)

        self.analyzer = ImageAnalyzer(model = model_name, k_images= k_images, reasoning_effort=reasoning_effort)
        self.enhancer = PromptEnhancerLLM(model= model_name, reasoning_effort=reasoning_effort, instruction_path=instruction_path)
        self.generator = Generator(model= model_name, api_directory=api_directory, reasoning_effort=reasoning_effort)
        self.compiler = Compiler(script_dir=script_dir, model = model_name, api_directory=api_directory, reasoning_effort=reasoning_effort)
        self.debugger = Debugger(executor=self.exec, template=self.template, model=model_name, reasoning_effort=reasoning_effort, api_path=api_directory)

    def __call__(self, prompt):

        result = self.analyzer(prompt['images'])

        

        # procedure = self.generator(result)

        # code = self.compiler(procedure)

        # print(code)



# Initial Params
api_path = 'C:\\Users\\Anthony\\Documents\\Projects\\scene_agent\\breadth_agent\\src\\agent\\agent_details\\tool_context'
log_file = "sfm_log.txt"
script_dir_testing = 'C:\\Users\\Anthony\\Documents\\Projects\\scene_agent\\breadth_agent\\src\\agent\\generated_code'
instruction_path = 'C:\\Users\\Anthony\\Documents\\Projects\\scene_agent\\breadth_agent\\src\\agent\\agent_details\\agent_instructions\\prompt_enh_examples.txt'

# Prompt
image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\scan118_normal_lighting"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\calibration_DTU_new.npz"
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

sa(temp_prompt)