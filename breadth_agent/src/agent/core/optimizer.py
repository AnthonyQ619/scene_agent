from sceneprogllm import LLM
import glob
import os
import cv2
from pathlib import Path
from modules.utilities import image_builder
import glob
from compiler import compiler
from debugger import debugger

class Generator:
    def __init__(self,
                 executer,
                 template,
                 model: str | None = None, 
                 instruction_path: str | None = None,
                 api_directory: str | None = None,
                 script_dir: str, 
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
        full_context_file_proc_dir = os.path.join(self.CWD, 'agent_details', 'optimize_context', 'optimal_desc.txt')
        
        ## Read Context File
        context_proc_file = open(full_context_file_proc_dir, 'r')
        self.context_str = context_proc_file.read()

        system_desc = f"""
You are an expert in generating Structure from Motion solution pipelines in Computer Vision. Given a subset of images of the given scene, whether camera 
type of reconstruction (Pose, Sparse, or Dense), current SfM coded pipeline, and resulting metrics log, you are supposed to devise a step by step plan in 
plain english on what is needed to properly optimize the given SfM workflow depending on the image content and provided metrics primarily utilizing 
knowledge of the given API library geared towards SfM reconstruction.

Please write the optimal set of changes in plain english for the provided Structure-From-Motion pipeline in python code for the computer vision problem you see 
that contain an image of the scene, corresponding metrics that designate the performance at each step of the pipeline, and the type of reconstruction
to focus on. For each step in the proposal statement of edits to make, utilize the correct sub-module tool provided from the API documentation that would boost 
performance from the resulting metrics and that best fits the described scene. Only use the provided API code library to suggest changes, do not use any outside 
software.  Ensure to use the code from the API documentation. Do NOT write any code, instead layout the plan of what tools to change or parameters to edit for the 
chosen tools at each step by stating the sub-modules name to use (IF CHANGING SUB-MODULES), parameters to edit and what value, and reasoning for the chosen parameters 
to edit. The following is the API description of each module and sub-module that can be use for each step in the procedure to generate

{api_desc}

The goal is to, from the problem statement guidlines, provided image sample of the scene, and the SfM coded pipeline performance at each stage, design a step-by-step 
proposal that lays out the reasoning for each edit at each step chosen that best fits the scene given, and for each step in the provided proposal, provide the edits in plain
english on what parameters to tune and the new value, or what sub-module to replace at the current step given the performance, using primarily the parameters and 
tools provided from the API documentation. Only use the provided API code library, do not use any outside software. Ensure to use the code from the API documentation. 
Do not write any code, ONLY WRITTEN TEXT SIMILAR TO THE PROVIDED CONTEXT EXAMPLES THE GIVEN IN THE PROMPT AND IN THE EXACT FORMAT.
"""
        self.proposal = LLM(system_desc = system_desc,
                                response_format = "text",
                                model_name = model,
                                reasoning_effort = reasoning_effort,
                                temperature = temperature)

        self.compiler = Compiler(script_dir=script_dir, model = model_name, api_directory=api_directory, reasoning_effort=reasoning_effort)
        self.debugger = Debugger(executor=executer, template=template, model=model_name, reasoning_effort=reasoning_effort, api_path=api_directory)

    def __call__(self, query, query_img, k:int = 5):
        formal_query = 
        f"""
        {query}

        Given the following proposals, make a suggestion different than the ones provided below:
        """
        for j in range(5):
            proposals = []

            for i in range(k):
                result = self.proposal(formal_query, image_paths=query_img)
                formal_query = formal_query + f"\nProposal {i}:\n{result}"
                proposals.append(result)
            
            generated_codes = []
            for i in range(len(proposals)):
                code = self.compiler(proposals[i]) # Need to edit compiler here!!
                refined_code = self.debugger(code)
                generated_codes.append(refined_code)

            best_code = Evaluater(generated_codes, metric_paths) # Was LLM but testing outside of DTU changing to a more fixed choice
            formal_query = f"""
            {query["reconstruction"]}
            {best_code}

            Given the following proposals, make a suggestion different than the ones provided below:
            """