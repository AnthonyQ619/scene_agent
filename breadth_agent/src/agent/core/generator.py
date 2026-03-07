from sceneprogllm import LLM
import glob
import os
import cv2
from pathlib import Path
from modules.utilities import image_builder
import glob

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
        full_context_file_proc_dir = os.path.join(self.CWD, 'agent_details', 'script_context', 'process.txt')
        
        ## Read Context File
        context_proc_file = open(full_context_file_proc_dir, 'r')
        self.context_str = context_proc_file.read()

        # ## Split Context
        # context_s_to_p = context_s_to_p.split("==$#$==")

        # context_str = ""
        # ## Build In-Context Example Paragraph:
        # # print("CONTEXT SPLIT:", len(context_s_to_p))
        # for i in range(len(context_s_to_p) - 1):
        #     s_and_p = context_s_to_p[i].split("==#$#==")
        #     scene = s_and_p[0]
        #     procedure = s_and_p[1]

        #     context_str += f"Example {i + 1}:\n" + scene + "\n" + procedure + "\n"

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
        
# Please write a Structure-From-Motion pipeline for the computer vision problem you see that contain an example description of the scene to inform you 
# of the environment, which will inform you on what modules or tools to use from the provided api, and textual description of the SfM pipeline to 
# develop in the form of an example process. For each step in the solution statement, utilize the correct sub-module tool provided from the API 
# documentation that best fits the scene description we are encountering. Only use the provided API code library, do not use any outside software. 
# Ensure to use the code from the API documentation. Do NOT write any code, instead layout the plan of what tools to use at each step by stating 
# the sub-modules name to use, and parameters to input. The following are a few examples of input prompts and reference procedures to generate.

# {context_str}

# The goal is to, from the description of the scene, design a step-by-step procedure that lays out the reasoning for each module chosen that best fits 
# the scene given, and for each step in the provided process, utilize the correct sub-module tool provided from the API documentation. Only use the 
# provided API code library, do not use any outside software. Ensure to use the code from the API documentation. Do not write any code, ONLY WRITTEN TEXT
# SIMILAR TO THE PROVIDED EXAMPLES IN THE EXACT FORMAT.

        # print(context_str)

        # full_desc = api_desc + "\n" + system_desc

        self.generator = LLM(system_desc = system_desc,
                            response_format = "text",
                            model_name = model,
                            reasoning_effort = reasoning_effort,
                            temperature = temperature)
        
        # Build In-Context image examples here
        img_path = os.path.join(self.CWD, 'agent_details', 'image_context')
        self.image_paths = sorted(glob.glob(img_path + "\\*"))

    def __call__(self, query, query_imgs):
        # full_query = self.prompter_llm(query) + '\n' + image_analysis
        improved_prompt = f"""
The following are a few examples of reference procedures to generate with corresponding images:
{self.context_str}

Each procedure example is titled "Procedure:Num", and each corresponding image is titled "image_context(Num).png",
where each matching "Num" value between procedure and image title is the corresponding image set and generated procedure.
In short, the first 8 images provided correspond to the first 8 procedure examples in respective order. The final image 
is the given scene from the user to generate a procedure for. 

The followiing information is provided to guide your chosen sub-modules for each step of the generated procedure.
{query}
"""


        # image_path = sorted(glob.glob(query_imgs + "\\*"))
        # Use equivalent params from context builder images
        new_img = image_builder(image_path=query_imgs, 
                                max_size=350, 
                                k=5)
        # Save image and input new path to image list containing context imaages
        PATH = os.path.dirname(os.path.realpath(__file__))
        new_img_path = os.path.join(PATH, "temp_images")

        if not os.path.exists(new_img_path):
            os.makedirs(new_img_path)
        
        cv2.imwrite(new_img_path + f"\\query_img.png", new_img)

        self.image_paths.append(new_img_path + f"\\query_img.png")
        print(self.image_paths)
        output = self.generator(improved_prompt, image_paths=self.image_paths)

        return output
        