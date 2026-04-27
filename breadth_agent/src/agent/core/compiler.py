import os
import glob
from pathlib import Path
from sceneprogllm import LLM
from concurrent.futures import ThreadPoolExecutor

class Compiler:
    def __init__(self,
                 script_dir: str, 
                 model: str | None = None, 
                 instruction_path: str | None = None,
                 api_directory: str | None = None,
                 temperature: float = 0.8,
                 reasoning_effort: str ="medium"
                 ):
        self.CWD = str(Path(__file__).resolve().parents[1])

        import uuid
        self.id = uuid.uuid4()
        self.exec = Executor(self.id)

         # Further Written Instructions
        if instruction_path is None:
            examples = ""
        else:
            context = open(instruction_path, 'r')
            examples = context.read()
            context.close()

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

        # Gather in-context examples 
        ## Gather Path to context
        full_context_file_s_p_dir = os.path.join(self.CWD, 'agent_details', 'script_context', 'process_to_scripts.txt')
        
        ## Read Context File
        context_s_p_file = open(full_context_file_s_p_dir, 'r')
        context_s_to_p = context_s_p_file.read()

        ## Split Context
        context_s_to_p = context_s_to_p.split("==$#$==")

        context_str = ""
        ## Build In-Context Example Paragraph:
        # print("CONTEXT SPLIT:", len(context_s_to_p))
        for i in range(len(context_s_to_p) - 1):
            s_and_p = context_s_to_p[i].split("==#$#==")
            scene = s_and_p[0]
            procedure = s_and_p[1]

            context_str += f"Example {i + 1}:\n" + scene + "\n" + procedure + "\n"

        # Fixed System Description for the API

        system_desc = f"""
You are an expert in generating Structure from Motion solution pipelines in Computer Vision, specifically python scripts using a custom API.
Given a step by step process to construct an optimal SfM pipeline in plain english, you are supposed to write its corresponding python script 
using the provided API to create a reconstructed 3D scene/pose estimation.

Your code should strictly do what is mentioned in the process and nothing more or less. To help you write the code, you will be provided with a 
the procedure will provide what tools to invoke and the parameters to use in each module called.
As an example consider the following process and the corresponding code transformation:

{context_str}

Please write a Structure-From-Motion pipeline for the computer vision problem you see that contains the reconstruction type of the problem and 
a provided procedure for reconstruction, which will inform you on what modules or tools to use from the provided api, and textual description of 
the SfM pipeline to develop. For each step in the provided procedure, utilize the correct sub-module tool provided from the API documentation. 
Only use the provided API code library, do not use any outside software. Ensure to use the code from the API documentation. Do not 
write any text, ONLY PYTHON CODE SIMILAR TO THE PROVIDED EXAMPLES IN FORMAT.

The goal is to design the corresponding python code to the provided step-by-step procedure that lays out the  reasoning for each module chosen that 
best fits the scene given. Note that you don't have to import anything, SfMScene along with any other packages are already imported for you!
Lastly, there is a unique ID for each program, just include the ID variable, it is defined along with the imports so don't worry about assigning it, use it already as a parameter for the SfMScene constructor.
"""

        self.imported_modules = f"""
from modules.features import (FeatureDetectionSIFT, FeatureDetectionSP, FeatureDetectionORB)
from modules.featurematching import (FeatureMatchFlannTracking, FeatureMatchBFPair, FeatureMatchFlannPair, FeatureMatchBFTracking, FeatureMatchLightGlueTracking, FeatureMatchSuperGlueTracking, FeatureMatchLightGluePair, FeatureMatchSuperGluePair)
from modules.camerapose import (CamPoseEstimatorEssentialToPnP, CamPoseEstimatorVGGTModel)
from modules.scenereconstruction import (Sparse3DReconstructionMono, Sparse3DReconstructionVGGT, Dense3DReconstructionVGGT, Dense3DReconstructionMono)
from modules.optimization import (BundleAdjustmentOptimizerLocal, BundleAdjustmentOptimizerGlobal)
from modules.baseclass import SfMScene
ID = "{self.id}"
"""

        full_desc = api_desc + "\n" + system_desc

        self.compiler = LLM(system_desc = full_desc,
                            response_format = "code",
                            model_name = model,
                            reasoning_effort = reasoning_effort,
                            temperature = temperature)

    def run_with_self_refine(self, query):
        program = self.compiler(query)
        prompt = f"""
You have generated the following code for the given process:
Process:
{query}
Program:
{program}
I want you to verify if the code that you generated adheres to the syntax and guidelines of the API. If it does, return the same code as your final answer. If it does not, please fix the code and return the corrected code as your final answer.
Key things to ensure:
1. For setting local bundle adjustment in pose estimation, apply the parameter like this to the pose estimation module: reconstructed_scene.CamPoseEstimatorEssentialToPnP(..., optimizer = ("BundleAdjustmentOptimizerLocal", 
        'max_num_iterations': 20,
        'window_size': 8,
        'robust_loss': True,
        'use_gpu': False
    ) as this is a special case for the inclusion of the parameter.
2. For constructing dense reconstruction with VGGT explicitly, skip sparse reconstruction from the VGGT module, and follow the pipeline of CamPoseEstimatorVGGTModel(...) to Dense3DReconstructionVGGT(...) as this is a special case.
3. Most importantly, do not write your own code to bypass any errors. Resolve any errors by either fixing incorrect parameters, or fixing syntax errors. Do not generate any python code outside of the API usage.
"""
        refined_program = self.compiler(prompt)
        return refined_program

    def choose_program(self, programs):
        
        num_failures = 0
        failed_programs = []
        for p in programs:
            output, success = self.exec(p)
            if success: 
                with open("/work/tmp/metric_" + str(self.id) + ".txt", "r") as f:
                    metric = f.read().strip()
                return p, metric
            else:
                num_failures += 1
                failed_programs.append((p, output))

        from random import choice
        p, output = choice(failed_programs)
        return p, output


    def __call__(self, query, num_samples=5):
        with ThreadPoolExecutor(max_workers=min(20, num_samples)) as executor:
            futures = [
                executor.submit(
                    generate_program_worker,
                    self.run_with_self_refine,
                    query,
                    self.imported_modules
                )
                for _ in range(num_samples)
            ]
            sampled_programs = [f.result() for f in futures]

        # filter out any empty responses due to errors
        sampled_programs = [sp for sp in sampled_programs if sp.strip() != ""]
        return self.choose_program(sampled_programs)

def generate_program_worker(runner, process, imports):
    return imports + "\n" + runner(process)


class Executor:
    def __init__(self, id):
        
        self.id = id
        import os
        os.makedirs("/work/tmp", exist_ok=True)
        self.log_file = "/work/tmp/log_file_" + str(id) + ".txt"


    def __call__(self, program: str):

        # Create Code
        temp_file = open("/work/tmp/tmp_prog_" + str(self.id) + ".py", 'w')
        temp_file.write(program)
        temp_file.close()


        # Change current working directory
        current_env = os.environ.copy()
        import sys
        import subprocess
        with open(self.log_file, "w") as f:
            print("🚀 Running script...")
            result = subprocess.run(
                [sys.executable, "/work/tmp/tmp_prog_" + str(self.id) + ".py"],
                env=current_env,
                stdout=f,      # send standard output to file
                stderr=f       # send error output to file
            )

            f.close()

        log_file = open(self.log_file, 'r')

        output = log_file.read().strip()
        
        log_file.close()
        self.cleanup(self.log_file, "/work/tmp/tmp_prog_" + str(self.id) + ".py")

        if result.returncode == 0:
            return output, True
        else:
            return output, False
    
    def cleanup(self, log_file_path: str, script_path: str):
        if os.path.exists(log_file_path):
            os.remove(log_file_path)
        if os.path.exists(script_path):
            os.remove(script_path)