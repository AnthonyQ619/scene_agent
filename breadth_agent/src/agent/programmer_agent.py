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
you see that contain a problem statement of the approach to use, description of the scene to inform you what modules or tools to use from
the provided api, and a list of sub-problems with solutions and description to follow. For each step in the solution statement, utilize the 
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
    

class ProgrammerAgent():
    def __init__(self,
                 model: str | None = None):
        
        CWD = os.path.dirname(os.path.realpath(__file__))
        # Setup the ProgrammerSnippets -> This allows for better optimization and parameter tuning

        # Read in instruction set for each code snippet 
        programmer_descriptions_path = os.path.join(CWD, 'agent_details', 'agent_instructions', 'programmer_snippet_desc.txt')
        programmer_descriptions = open(programmer_descriptions_path, 'r')

        program_desc_snippets = programmer_descriptions.split("==%==")

        # Set up Each programmer with the set description and path to API tool
        