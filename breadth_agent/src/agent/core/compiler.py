from sceneprogllm import LLM
import glob
import os
from pathlib import Path


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
best fits the scene given.
"""
        # print(context_str)

        full_desc = api_desc + "\n" + system_desc

        self.compiler = LLM(system_desc = full_desc,
                            response_format = "code",
                            model_name = model,
                            reasoning_effort = reasoning_effort,
                            temperature = temperature)
        

    def __call__(self, query):
        

        output = self.compiler(query)


        return output