from sceneprogllm import LLM
import glob
import os
from pathlib import Path

class Generator:
    def __init__(self,
                 model: str | None = None, 
                 instruction_path: str | None = None,
                 api_directory: str | None = None,
                 temperature: float = 0.8,
                 reasoning_effort: str ="medium"
                 ):
        self.CWD = str(Path(__file__).resolve().parents[1])

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
        full_context_file_s_p_dir = os.path.join(self.CWD, 'agent_details', 'script_context', 'scene_to_process.txt')
        
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
You are an expert in generating Structure from Motion solution pipelines in Computer Vision. Given a description of a scene regarding
scene texture, scene lighting, shadows on objects of interest, whether camera is calibrated, type of reconstruction (Pose, Sparse, or Dense),
or harware constraint, you are supposed to devise, or edit, a step by step procedure in plain english on what is needed to properly reconstruct the
scene given the image content utilizing knowledge of the given API library geared towards SfM reconstruction.

Please write a Structure-From-Motion pipeline for the computer vision problem you see that contain an example description of the scene to inform you 
of the environment, which will inform you on what modules or tools to use from the provided api, and textual description of the SfM pipeline to 
develop in the form of an example process. For each step in the solution statement, utilize the correct sub-module tool provided from the API 
documentation that best fits the scene description we are encountering. Only use the provided API code library, do not use any outside software. 
Ensure to use the code from the API documentation. Do NOT write any code, instead layout the plan of what tools to use at each step by stating 
the sub-modules name to use, and parameters to input. The following are a few examples of input prompts and reference procedures to generate.

{context_str}

The goal is to, from the description of the scene, design a step-by-step procedure that lays out the reasoning for each module chosen that best fits 
the scene given, and for each step in the provided process, utilize the correct sub-module tool provided from the API documentation. Only use the 
provided API code library, do not use any outside software. Ensure to use the code from the API documentation. Do not write any code, ONLY WRITTEN TEXT
SIMILAR TO THE PROVIDED EXAMPLES IN THE EXACT FORMAT.
"""
        # print(context_str)

        full_desc = api_desc + "\n" + system_desc

        self.generator = LLM(system_desc = full_desc,
                            response_format = "text",
                            model_name = model,
                            reasoning_effort = reasoning_effort,
                            temperature = temperature)
        
#         prompter_desc = """
# You are an expert in generating Structure from Motion solution pipelines in Computer Vision. Given a description of a scene regarding
# scene texture, scene lighting, shadows on objects of interest, whether camera is calibrated, type of reconstruction (Pose, Sparse, or Dense),
# or harware constraint, you are supposed to devise a problem statement that summarizes the scene, and
# how we want to build a 3D reconstruction from that description.
        
# DO NOT LIST AN ENTIRE SOLUTION PIPELINE. 
# Structure the description as a problem statement that designates the computer vision problem to solve, 
# description of the scene, and how we want to solve said problem (Sparse, Dense, Visual Odometry, etc.) with any possible constraints
# in mind (GPU memory capabilities, etc.).

# For the description of the scene, focus on the different textures in the scene described from the prompt, whether lighting
# or shadows affect objects visibility and texture, outdoor or indoor, scene rotates around object or continuing on a path, etc.

# In the prompt, you will be given a potential path to calibration, reconstruction type, and gpu memory. Determine
# if a calibration path is provided, if the image set is calibrated, type of reconstruction we need to do (Dense or Sparse),
# and whether the GPU provided is large enough to utilize models like VGGT when necessary or if the approach should be 
# more memory conservative.
# """
#         sys_desc = api_desc + "\n" + prompter_desc

#         self.prompter_llm = LLM(system_desc = sys_desc,
#                             response_format = "text",
#                             model_name = model,
#                             reasoning_effort = reasoning_effort,
#                             temperature = temperature)

    def __call__(self, query):
        # full_query = self.prompter_llm(query) + '\n' + image_analysis

        output = self.generator(query)

        return output
        