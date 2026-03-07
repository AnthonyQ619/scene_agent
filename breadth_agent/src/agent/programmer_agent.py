from openai import OpenAI
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage

import glob
import os 
import numpy as np
from pydantic import BaseModel, Field

class ProgramManagerLLM():
    def __init__(self,
                 model: str | None = None, 
                 instruction_path: str | None = None,
                 api_directory: str | None = None,
                 temperature: float = 0.8):
        
        if instruction_path is None:
            examples = ""
        else:
            context = open(instruction_path, 'r')
            examples = context.read()
            context.close()

        default_desc = """
You are an assisstant with focus on computer vision problems. Please write a Structure-From-Motion pipeline for the computer vision problem 
you see that contain an example description of the scene to inform you of the environment, which will inform you on what modules or tools to use from
the provided api, and textual description of the SfM pipeline to develop in the form of an example process. For each step in the solution statement, utilize the 
correct sub-module tool provided from the API documentation that best fits the scene description we are encountering. Only use the provided API
code library, do not use any outside software. Ensure to use the code from the API documentation. Do NOT write any code, instead layout
the plan of what tools to use at each step by stating the sub-modules name to use, and parameters to input.
"""

        if api_directory is None:
            self.api_files = []
        else:
            self.api_files = sorted(glob.glob(api_directory + "\\*"))

        api_desc = ""
        api_key=os.getenv("OPENAI_API_KEY")

        for file in self.api_files:
            context = open(file, 'r')
            examples = context.read()
            context.close()
            api_desc += examples + "\n"

        if model is None:
            self.model_name = "gpt-5-mini"
        else:
            self.model_name = model

        self.sys_msg = SystemMessage(default_desc + examples + "\n" + api_desc)

        # Structured Output
        class ProgramPlan(BaseModel):
            """Description of Scene with important details"""
            feature_step: str = Field(..., description="Description of what Feature Detector to use from the provided Detection Modules from API Description")
            matcher_step: str = Field(..., description="Description of what Feature Matcher (Pairwise) to use from the provided Feature Matching Modules from API Description")
            pose_step: str = Field(..., description="Description of what Camera Pose Estimation Tool to use from the provided  Pose Estimation Modules from API Description")
            tracking_step: str = Field(...,  description="Description of what Feature Tracker to use from the provided Feature Tracking Modules from API Description")
            scene_est_step: str = Field(..., description="Description of what Scene Estimation Algorithm to use from the provided Scene Estimation Modules from API Description")
            optim_step: str = Field(..., description="Description of what algorithm to use from the provided Parameters in the Optimization Module from API Description")

        self.model = init_chat_model(model=self.model_name,
                                     api_key=api_key,
                                     temperature=temperature)
        
        self.model = self.model.with_structured_output(ProgramPlan)

    def __call__(self, query):
        message = HumanMessage(content=[
            {"type": "text", "text": query}])
        
        output = self.model.invoke([self.sys_msg, message])

        return output
    

class ProgrammerSnippetLLM():
    def __init__(self,
                 system_desc: str,
                 model: str | None = None, 
                 api_file: str | None = None,
                 temperature: float = 0.8):
        self.code_desc = """
You are an assisstant with focus on computer vision problems. Please write code for the computer vision problem you see that 
contain the description of the scene to inform you what parameters to use from the provided api, and proposed initial module 
to use for the first step of code to generate. The code must be written in python utilizing the python API provided. Do not
generate any extra code.
"""
        # Set up LLM specific description that corresponds the correct module to generate code from
        self.sys_desc = system_desc

        # Read in the API description of specific Module
        context = open(api_file, 'r')
        self.api_examples = context.read()
        context.close()

        if model is None:
            self.model_name = "gpt-5-mini"
        else:
            self.model_name = model

        api_key=os.getenv("OPENAI_API_KEY")

        self.sys_msg = SystemMessage(self.code_desc + self.sys_desc + self.api_examples + "Output executable Python Code only, no text output otherwise")

        self.model = init_chat_model(model=self.model_name,
                                     api_key=api_key,
                                     temperature=temperature)
        
    def __call__(self, query):
        message = HumanMessage(content=[
            {"type": "text", "text": query}])
        
        output = self.model.invoke([self.sys_msg, message])

        return output
    

class ProgramProcessorLLM():
    def __init__(self,
                 model: str | None = None, 
                 instruction_path: str | None = None,
                 api_directory: str | None = None,
                 temperature: float = 0.8):
        
        if instruction_path is None:
            examples = ""
        else:
            context = open(instruction_path, 'r')
            example = context.read()
            context.close()

        default_desc = """
You are an assisstant with focus on computer vision problems. Please write a Structure-From-Motion pipeline for the computer vision problem 
you see that contain an example description of the scene to inform you of the environment, which will inform you on what modules or tools to use from
the provided api, and textual description of the SfM pipeline to develop in the form of an example process. For each step in the solution statement, utilize the 
correct sub-module tool provided from the API documentation that best fits the scene description we are encountering. Only use the provided API
code library, do not use any outside software. Ensure to use the code from the API documentation. Do NOT write any code, instead layout
the plan of what tools to use at each step by stating the sub-modules name to use, and parameters to input.

"""

        if api_directory is None:
            self.api_files = []
        else:
            self.api_files = sorted(glob.glob(api_directory + "\\*"))

        api_desc = ""
        api_key=os.getenv("OPENAI_API_KEY")

        for file in self.api_files:
            context = open(file, 'r')
            examples = context.read()
            context.close()
            api_desc += examples + "\n"

        if model is None:
            self.model_name = "gpt-5-mini"
        else:
            self.model_name = model

        self.sys_msg = SystemMessage(default_desc + example + "\n" + api_desc)

        # Structured Output
        class ProcessPlan(BaseModel):
            """Description of Scene with important details"""
            gen_process: str = Field(..., description="Following the provided Process examples, construct a process for the reconstruction goal using the API documentation provided")
            
        self.model = init_chat_model(model=self.model_name,
                                     api_key=api_key,
                                     temperature=temperature)
        
        self.model = self.model.with_structured_output(ProcessPlan)

    def __call__(self, query):
        message = HumanMessage(content=[
            {"type": "text", "text": query}])
        
        output = self.model.invoke([self.sys_msg, message])

        return output
    
class ProgramCompilerLLM():
    def __init__(self,
                 model: str | None = None, 
                 instruction_path: str | None = None,
                 api_directory: str | None = None,
                 temperature: float = 0.8):
        
        if instruction_path is None:
            examples = ""
        else:
            context = open(instruction_path, 'r')
            example = context.read()
            context.close()

        default_desc = """
You are an assisstant with focus on computer vision problems. Please write a Structure-From-Motion pipeline for the computer vision problem 
you see that contain an example description of the scene to inform you of the environment and a direct textual description of the SfM pipeline to develop in 
the form of an example process, which will inform you of what tools to use in the initial generation of the code. For each step in the 
provided process, utilize the correct sub-module tool provided from the API documentation. Only use the provided API code library, do not use any outside 
software. Ensure to use the code from the API documentation. Do not write any text, ONLY PYTHON CODE SIMILAR TO THE PROVIDED EXAMPLES IN FORMAT.


"""

        if api_directory is None:
            self.api_files = []
        else:
            self.api_files = sorted(glob.glob(api_directory + "\\*"))

        api_desc = ""
        api_key=os.getenv("OPENAI_API_KEY")

        for file in self.api_files:
            context = open(file, 'r')
            examples = context.read()
            context.close()
            api_desc += examples + "\n"

        if model is None:
            self.model_name = "gpt-5-mini"
        else:
            self.model_name = model

        self.sys_msg = SystemMessage(default_desc + example + "\n" + api_desc)

        # Structured Output
        class CodePlan(BaseModel):
            """Description of Scene with important details"""
            gen_code: str = Field(..., description="Following the provided Process examples, construct a process for the reconstruction goal using the API documentation provided")
            
        self.model = init_chat_model(model=self.model_name,
                                     api_key=api_key,
                                     temperature=temperature)
        
        self.model = self.model.with_structured_output(CodePlan)

    def __call__(self, query):
        message = HumanMessage(content=[
            {"type": "text", "text": query}])
        
        output = self.model.invoke([self.sys_msg, message])

        return output

class ProgrammerAgent():
    def __init__(self,
                 model: str | None = None):
        
        self.CWD = os.path.dirname(os.path.realpath(__file__))
        # Setup the ProgrammerSnippets -> This allows for better optimization and parameter tuning
        full_context_file_sp_dir = os.path.join(self.CWD, 'agent_details', 'script_context', 'scene_to_process.txt')
        full_context_file_p_to_c_dir = os.path.join(self.CWD, 'agent_details', 'script_context', 'process_to_scripts.txt')
        context_s_p = open(full_context_file_sp_dir, 'r')
        self.context_s_to_p = context_s_p.read()

        context_p_c = open(full_context_file_p_to_c_dir, 'r')
        self.context_p_to_c = context_p_c.read()
        # Read in instruction set for each code snippet 
        api_dir = os.path.join(self.CWD, 'agent_details', 'tool_context')
        programmer_descriptions_path_process = os.path.join(self.CWD, 'agent_details', 'agent_instructions', 'scene_to_process_ex.txt')
        programmer_descriptions_path_code = os.path.join(self.CWD, 'agent_details', 'agent_instructions', 'process_to_code_example.txt')
        # programmer_descriptions_proc_f = open(programmer_descriptions_path_process, 'r')
        # programmer_descriptions_code_f = open(programmer_descriptions_path_code, 'r')

        # program_desc_snippets = programmer_descriptions.split("==%==")
        self.processor_llm = ProgramProcessorLLM(model = "gpt-5-2025-08-07",
                                                 instruction_path=programmer_descriptions_path_process,
                                                 api_directory=api_dir)
        
        self.compiler_llm = ProgramCompilerLLM(model = "gpt-5-2025-08-07",
                                                 instruction_path=programmer_descriptions_path_code,
                                                 api_directory=api_dir)
    
    def __call__(self, query):
        
        query_w_context = self.context_s_to_p + "\n" + query 

        response = self.processor_llm(query=query_w_context)
        new_query = response.gen_process
        print(new_query)

        query_w_context = self.context_p_to_c + "\n" + new_query

        response = self.compiler_llm(query=new_query)

        code = response.gen_code
        print(code)

        programmer_save_path = os.path.join(self.CWD, 'generated_code', 'gen_code.py')

        temp_file = open(programmer_save_path, 'w')
        temp_file.write(code)

        return code
        # Set up Each programmer with the set description and path to API tool
        