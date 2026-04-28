import glob
import os
import cv2
from pathlib import Path
from sceneprogllm import LLM
from modules.utilities import image_builder, resize_dataset, clean_dir
from utility.optical_flow import read_camera_flow
from concurrent.futures import ThreadPoolExecutor

class Generator:
    def __init__(self,
                 model: str | None = None,
                 api_directory: str | None = None,
                 temperature: float = 0.8,
                    reasoning_effort: str ="medium"
                    ):
        self.CWD = str(Path(__file__).resolve().parents[1])
        examples = ""
        self.plan = None

        # Gather API Information
        if api_directory is None:
            self.api_files = []
        else:
            self.api_files = sorted(glob.glob(api_directory + "/*"))

        api_desc = ""
        for file in self.api_files:
            context = open(file, 'r')
            examples = context.read()
            context.close()
            api_desc += examples + "\n"

        # Gather in-context examples 
        ## Gather Path to context
        full_context_file_proc_dir = os.path.join(self.CWD, 'agent_details', 'script_context', 'process.txt')
        
        ## Read Context File
        context_proc_file = open(full_context_file_proc_dir, 'r')
        self.context_str = context_proc_file.read()
        context_proc_file.close()

        # TODO 
#         self.plan_evaluator = LLM(
#             system_desc="""
# You are an expert Structure from Motion workflow plans.
# Given the image of the scene and reconstruction guidelines we wish to follow (prompt) and several step-by-step node procedures, your task is to:
# 	1.	Analyze each SfM plan in detail — what modules are called, parameters used, and if we invoke the correct sub-modules for the reconstruction type.
# 	2.	Compare how accurately each sub-module that is invoked best fits the given scene we wish to reconstruct.
# 	3.	Choose the best procedure and explain why — citing specific SfM planning logic, module selection, and why parameterization of tools make sense.

# Judge using these key aspects:
# 	•	Choice of sub-modules accurately coincide with the image of the scene and best use-cases.
# 	•	Sub-module selection fits within the system constraints of the user prompt.
# 	•	Reconstruction type is followed precisely, and our last sub-module invoked directly represents the prompt (Pose, Sparse, or Dense).

# You are to provide me the index (starting from 1) of the best plan amongst those provided to you, so that I can pick it easily.
# Your response should be a single integer indicating the best plan index, without any additional commentary.
# """,
#             response_format="json",
#             response_params={"best_plan_index":"int"},
#             model_name=model,
#             reasoning_effort=reasoning_effort
#         )

        # Gather Metric Result Context for Optimization
        ## Gather Path to context
        full_context_file_proc_dir = os.path.join(self.CWD, 'agent_details', 'optimize_context', 'metric_context.txt')
        
        ## Read Context File
        context_metric_file = open(full_context_file_proc_dir, 'r')
        self.metric_prompt = context_metric_file.read()
        context_metric_file.close()

        # Fixed System Description for the API
        system_desc = f"""
You are an expert in generating Structure from Motion solution pipelines in Computer Vision. Given a subset of images of the given scene, whether camera 
is calibrated, type of reconstruction (Pose, Sparse, or Dense), or harware constraint, you are supposed to devise, or edit, a step by step procedure in 
plain english on what is needed to properly reconstruct the scene given the image content utilizing knowledge of the given API library geared towards 
SfM reconstruction.

Please write a Structure-From-Motion pipeline for the computer vision problem you see that contain an example description of the scene to inform you 
of the environment, which will inform you on what modules or tools to use from the provided api, and textual description of the SfM pipeline to 
develop in the form of an example process. For each step in the solution statement, utilize the correct sub-module tool provided from the API 
documentation that best fits the scene description we are encountering. Only use the provided API code library, do not use any outside software. 
Ensure to use the code from the API documentation. Do NOT write any code, instead layout the plan of what tools to use at each step by stating 
the sub-modules name to use, parameters to input, and reasoning for the parameters to use. The following is the API description of each module
and sub-module that can be use for each step in the procedure to generate

{api_desc}

The goal is to, from the description statement guidlines and image sample of the scene, design a step-by-step procedure that lays out the reasoning for 
each module chosen that best fits the scene given, and for each step in the provided process, utilize the correct sub-module tool provided from the API 
documentation. Only use the provided API code library, do not use any outside software. Ensure to use the code from the API documentation. Do not write 
any code, ONLY WRITTEN TEXT SIMILAR TO THE PROVIDED CONTEXT EXAMPLES THE GIVEN PROMPT AND IN THE EXACT FORMAT.
"""

        self.generator = LLM(system_desc = system_desc,
                            response_format = "text",
                            model_name = model,
                            reasoning_effort = reasoning_effort,
                            temperature = temperature)
        
        # Do we also need to give the LLM knowledge of the API? Or do we call generator first??
        # Build Planner LLM 
        self.plan_evaluator = LLM(
            system_desc="""
You are an expert Structure from Motion workflow plans.
Given the image of the scene and reconstruction guidelines we wish to follow (prompt) and several step-by-step node procedures, your task is to:
	1.	Analyze each SfM plan in detail — what modules are called, parameters used, and if we invoke the correct sub-modules for the reconstruction type.
	2.	Compare how accurately each sub-module that is invoked best fits the given scene we wish to reconstruct.
	3.	Choose the best procedure and explain why — citing specific SfM planning logic, module selection, and why parameterization of tools make sense.

Judge using these key aspects:
	•	Choice of sub-modules accurately coincide with the image of the scene and best use-cases.
	•	Sub-module selection fits within the system constraints of the user prompt.
	•	Reconstruction type is followed precisely, and our last sub-module invoked directly represents the prompt (Pose, Sparse, or Dense).

You are to provide me the index (starting from 1) of the best plan amongst those provided to you, so that I can pick it easily.
Your response should be a single integer indicating the best plan index, without any additional commentary.
""",
            response_format="json",
            response_params={"best_plan_index":"int"},
            model_name=model,
            reasoning_effort=reasoning_effort
        )

         # Build In-Context image examples here
        img_path = os.path.join(self.CWD, 'agent_details', 'image_context') # MAKE THIS MORE PATH ORIENTED
        self.image_paths = sorted(glob.glob(img_path + "/*"))
        self.new_query_img_path = None

    def get_multiple_plans(self, query, query_video_path, num_plans=5):
        with ThreadPoolExecutor(max_workers=min(20, num_plans)) as executor:
            futures = [
                executor.submit(self.__call__, query, query_video_path)
                for _ in range(num_plans)
            ]
            plans = [future.result() for future in futures]
        return plans

    def enhance_prompt_for_selection(self, query, query_video_path):
        # User query is of this format
        # f"""
        # {new_query["statement"]}
        # {new_query["reconstruction"]}
        # {new_query["calibration"]}
        # {new_query["memory"]}
        # Use Image Path in Code: {prompt['images']}
        # Use Calibration Path in Code: {prompt['calibration']}
        # """
        new_prompt = f"""
The following are a few examples of reference procedures to generate with corresponding images:
{self.context_str}

Each procedure example is titled "Procedure:Num", and each corresponding image is titled "image_context(Num).png",
where each matching "Num" value between procedure and image title is the corresponding image set and generated procedure.
In short, the first 8 images provided correspond to the first 8 procedure examples in respective order. The final image 
is the given scene from the user to generate a plan for reconstruction. 

The followiing information is provided from the user query to guide your chosen sub-modules for each step of the generated plan/procedure.
{query}

You are now given the following plans/procedures of a Structure from Motion workflow generated from the above query and final image as the
scene to reconstruct\n:
"""
        return new_prompt

    def select_plan(self, plans, query, query_video_path):

        enhanced_prompt = self.enhance_prompt_for_selection(query, query_video_path)

        for idx, pl in enumerate(plans):
            enhanced_prompt += f"\nPlan {idx+1}:\n{pl}\n"
        
        enhanced_prompt += "\nSelect the plan that most best fits a plausible structure from motion pipeline for the given scene and query."
        response = self.plan_evaluator(enhanced_prompt, image_paths=self.image_paths)["best_plan_index"]
        best_index = int(response) - 1  # Convert to 0-based index
        best_plan = plans[best_index]
        return best_plan
    
    def forward(self, query, query_video_path, feedback=None): 
        enhanced_prompt = self.enhance_prompt(query, query_video_path)

        if feedback is not None and self.plan is not None:
            enhanced_prompt += f"""
Last time, you have created the following plan for the user query:
{self.plan}
Upon execution of the plan, the system has provided you with the following feedback containing various metrics on the quality of the plan:
{feedback}
Consider the following ideas on how to improve the plan based on the received metrics: 
{self.metric_prompt}
Please create a new plan that incorporates the feedback and improvement ideas to better fit the user query and scene. 
Your Output:
""" 

        plans = self.get_multiple_plans(enhanced_prompt, query_video_path, num_plans=5)
        output = self.select_plan(plans, query, query_video_path)
        # output = self.generator(enhanced_prompt, image_paths=self.image_paths)
        return output

    def __call__(self, query, query_video_path, feedback=None):
        plan = self.forward(query, query_video_path, feedback)
        self.plan = plan
        return plan
    
    def enhance_prompt(self, query, query_video_path):
        enhanced_prompt = f"""
The following are a few examples of reference procedures to generate with corresponding images:
{self.context_str}

Each procedure example is titled "Procedure:Num", and each corresponding image is titled "image_context(Num).png",
where each matching "Num" value between procedure and image title is the corresponding image set and generated procedure.
In short, the first 8 images provided correspond to the first 8 procedure examples in respective order. The final image 
is the given scene from the user to generate a procedure for. 

The followiing information is provided to guide your chosen sub-modules for each step of the generated procedure.
{query}
"""
        #TODO: Include enhanced prompt for motion cues here.
        mean_flow, median_flow, p75_flow, p90_flow, Rs = read_camera_flow(query_video_path, query["calibration"])

        enhanced_prompt_motion = f"""
TODO: Fill
"""

        dataset_path = sorted(glob.glob(query_video_path + "/*  "))

        # This is so we don't repeatedly add the same image to the self.image_paths
        if self.new_query_img_path is None: 
            new_img = image_builder(image_path=query_video_path, 
                                    max_size=350, 
                                    k=5) 
            
            # Save image and input new path to image list containing context imaages
            PATH = os.path.dirname(os.path.realpath(__file__))
            new_img_path = os.path.join(PATH, "temp_images")

            if not os.path.exists(new_img_path):
                os.makedirs(new_img_path)
            
            cv2.imwrite(new_img_path + f"/query_img.png", new_img)

            self.new_query_img_path = new_img_path + f"/query_img.png"
            self.image_paths.append(new_img_path + f"/query_img.png")
        
        return enhanced_prompt
    


def generate_plans(planner, query, query_video_path):
    plan = planner(query, query_video_path)
    return plan