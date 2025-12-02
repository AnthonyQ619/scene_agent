from openai import OpenAI
import numpy as np
# from sceneprogllm import LLM
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage
import base64
import glob
import os 
import re
from pydantic import BaseModel, Field


class DescriptionEncoder():
    def __init__(self, model: str = "text-embedding-3-small",
                    dimensions: int = 256,
                    encoding_format: str ="float"
                 ):
        
        self.model = model
        self.dimensions = dimensions
        self.encoding_format = encoding_format

        self.client = OpenAI()

    def encode_desc(self, query: str):
        clean_query = query.lstrip().rstrip().replace("\n", "")

        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=clean_query,
            dimensions=256,
            encoding_format="float"
            )
        
        return np.array(response.data[0].embedding)
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0  # Handle cases where one or both vectors are zero vectors

        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)

        return cosine_similarity
    
    def compare_encoding(self, query_embed: np.ndarray, db_embed: list[np.ndarray], k_choices: int = 2):
        score_list = []
        for vec in db_embed:
            score = self.cosine_similarity(query_embed, vec)
            score_list.append((score, vec))

        sorted_scores = sorted(score_list)
        print(sorted_scores)
        chosen_vecs = [desc[1] for desc in sorted_scores[:k_choices]]

        return chosen_vecs
        

    def context_db_reader(self, file_path: str, file_path_context: str):
        file_to_read = open(file_path, 'r')
        file_lines = file_to_read.readlines()
        desc_list = []

        embed_vec_list = []
        embed_vec_to_desc = {}
        desc_to_script = {}
        
        for i in range(0, len(file_lines), 2):
            embed_vec = file_lines[i]
            embed_vec = np.array(list(map(float, embed_vec.replace('[', '').replace(']', '').split(','))))
            desc = file_lines[i+1]
            desc_list.append(desc)

            embed_vec_list.append(embed_vec)
            embed_vec_to_desc[embed_vec.tobytes()] = desc

        context_scripts = open(file_path_context, 'r')
        content = context_scripts.read()

        split_db = content.split("===$&$===")

        for i in range(1, len(split_db) - 1):
            context_script = split_db[i].split("# ==#$#==")
            #descs = context_script[0].lstrip().rstrip().replace('"""','').replace("\n", "")
            descs = desc_list[i-1]
            script = context_script[1]

            desc_to_script[descs] = script

        return embed_vec_list, embed_vec_to_desc, desc_to_script

class ImageAnalysisLLM():
    def __init__(self, 
                 model: str | None = None, 
                 system_desc: str | None = None,
                 k_images: int = 4,
                 temperature: float = 0.8,
                 ):
        self.k = k_images

        api_key=os.getenv("OPENAI_API_KEY")

        default_desc = f"""
        You are an agent set to evaluate the {self.k} incoming images. For your response, 
        you need to:

        1. Determine if the images are outdoors or indoors. 
        2. Determine if there are illumination changes across the {self.k} images evaluated. 
        3. Determine if the image contains low textured regions, or high textured regions.
        4. Describe key features of the scene, specifically if the data rotates around an object,
        contains an object of interest in the scene at all times, if the object is highly textured or
        contains diffuse lighting, etc.
        5. Across the {self.k} images, determine if movement includes any extreme 
        viewpoint changes, or if the camera gradually moves across the scene in increments.
        """
        
        # Hard code model for now
        self.model_name = "gpt-5-2025-08-07"

        if system_desc is None:
            self.sys_msg = SystemMessage(default_desc)
        else:
            self.sys_msg = SystemMessage(system_desc)

        # Structured Output
        class SceneDescription(BaseModel):
            """Description of Scene with important details"""
            setting: str = Field(..., description="Description of if the scene is outdoor or indoors")
            lighting: str = Field(..., description=f"Description of if there are illumination changes across the {self.k} images evaluated.")
            textured: str = Field(..., description="Description of if the images contain objects of high or low textured regions")
            general_desc:str = Field(..., 
                                     description="Description of key features about the scene, such as camera moving around object, buildings, if object is highly textured, etc.")
            view_changes: str = Field(..., description=f"Desctiption of if camera movement is extreme, or small, across the the {self.k} images")

        self.model = init_chat_model(model=self.model_name,
                                     api_key=api_key,
                                     temperature=temperature)
        
        self.model = self.model.with_structured_output(SceneDescription)
        # self.image_analysis_model = LLM(name = 'llm_image_evaluator', 
        #                                 system_desc = self.sys_desc, 
        #                                 response_format = response_format,
        #                                 use_cache = use_cache,
        #                                 model_name = self.model,
        #                                 temperature = temperature)
    def parse_prompt(self, query):
        if os.name == "nt":
            path_pattern = r'((?:[A-Za-z]:\\|\\\\)[^\\\r\n]+(?:\\[^\\\r\n]+)*)'
        elif os.name == 'posix':
            path_pattern = r'((?:/[^/\s]+)+(?:/?)?)'

        found_paths = re.findall(path_pattern, query)

        image_paths = sorted(glob.glob(found_paths[0] + "\\*"))[:self.k]

        return image_paths
    
    def __call__(self, query):
        
        image_paths = self.parse_prompt(query)

        
        # Helper to encode image as base64 for LangChain
        def encode_image_as_base64(path: str) -> str:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        # Create message with text and multiple images
        message = HumanMessage(content=[
            {"type": "text", "text": "Analyze the set of input images."},
            *[
                {
                    "type": "image",
                    "base64": encode_image_as_base64(path),
                    "mime_type": f"image/{path.split('.')[-1]}"
                }
                for path in image_paths
            ]
        ])

        # output = self.image_analysis_model("Analyze the set of input images.", image_paths=image_paths)
        output = self.model.invoke([self.sys_msg, message])

        # header = "Scene Description: \n"

        return output

class PromptEnhancerLLM():
    def __init__(self,
                 model: str | None = None,
                 system_desc: str | None = None,
                 instruction_path: str | None = None,
                 temperature: float = 0.8,
                 ):
        
        default_desc = f"""
You are an agent instructed to breakdown the scene information that we want to build a 3D reconstruction
from.

The goal is to, from the description of the scene, design a problem statement that summarizes the scene, and
how we want to build a 3D reconstruction from that description. Structure the description as a problem statement 
that designates the computer vision problem to solve, description of the scene, and how we want to solve said
problem, i.e what algorithms we want to use.

Avoid coding anything, or suggesting any libraries, such as OpenCV. The goal is to keep
everything in a text description and as general as possible. Aim for a textbook description level.
Then, discuss the problem solution. 

Determine what steps are required to solve the specific computer vision problem such as 
feature detection, feature matching, feature tracking, camera pose estimation, intrinsic estimation,
3d point cloud reconstruction, dense reconsturction with multi-view stereo, etc.
"""
        
        # Hard code model for now
        self.model_name = "gpt-5-2025-08-07"

        if system_desc is None:
            self.sys_desc = default_desc
        else:
            self.sys_desc = system_desc

        if instruction_path is None:
            examples = ""
        else:
            context = open(instruction_path, 'r')
            examples = context.read()
            context.close()

        api_key=os.getenv("OPENAI_API_KEY")

        self.sys_msg = SystemMessage(self.sys_desc + examples)

        # Build LLM

        # self.prompt_enhancer =  LLM("llm_prompt_analyzer", 
        #                             system_desc = self.full_desc,
        #                             response_format="json",
        #                             json_keys=["statement:str", "solution:str"],
        #                             use_cache = use_cache,
        #                             model_name = self.model,
        #                             temperature = temperature)

        # Structured Output
        class ProblemStatement(BaseModel):
            """Description of Scene with important details"""
            statement: str = Field(..., description="Description of the problem statement constructed")
            solution: str = Field(..., description=f"Description of the general solution to take for 3D reconstruction via Structure from Motion")
            scene_summary: str = Field(..., description="Summary of the scene description of which we plan to reconstruct. Don't mention image count as description was only based on a subset of images")
            
        self.model = init_chat_model(model=self.model_name,
                                     api_key=api_key,
                                     temperature=temperature)
        
        self.model = self.model.with_structured_output(ProblemStatement)

    def __call__(self, query):
        # response = self.prompt_enhancer(query)
        message = HumanMessage(content=[
            {"type": "text", "text": query}])
        
        output = self.model.invoke([self.sys_msg, message])

        return output


class PromptEnhancmentAgent():
    def __init__(self,
                 embedding_database_path: str, 
                 full_length_path: str, 
                 model: str | None = None, 
                 image_model_desc: str | None = None, 
                 prompt_model_desc: str | None = None,
                 instruction_path: str | None = None,
                 k_images: int = 3,
                 k_choices: int = 3,
                 temperature: float = 0.8):
        
        if model is None:
            self.image_analysis =   ImageAnalysisLLM(system_desc=image_model_desc,
                                                     k_images=k_images,
                                                     temperature=temperature)
            self.prompt_analysis = PromptEnhancerLLM(system_desc=prompt_model_desc,
                                                     instruction_path=instruction_path,
                                                     temperature=temperature)
        else:
            self.image_analysis = ImageAnalysisLLM(model=model, 
                                                   system_desc=image_model_desc,
                                                   k_images=k_images,
                                                   temperature=temperature)
            self.prompt_analysis = PromptEnhancerLLM(model=model,
                                                     system_desc=prompt_model_desc,
                                                     instruction_path=instruction_path,
                                                     temperature=temperature)
        
        self.encoder_helper = DescriptionEncoder()

        # self.k_choices = k_choices
        
        # embed_vec_list, embed_vec_to_desc, desc_to_script = self.encoder_helper.context_db_reader(embedding_database_path, 
        #                                                                                      full_length_path)
        
        # self.embed_vec_list = embed_vec_list
        # self.embed_db = embed_vec_to_desc
        # self.context_db = desc_to_script
            
    def __call__(self, query):
        print(query)
        image_analysis_step = self.image_analysis(query=query)

        scene_description = f"""
Scene Setting: {image_analysis_step.setting}
Scene Lighting Effects: {image_analysis_step.lighting}
Scene Object Textures: {image_analysis_step.textured}
Scene Key Features: {image_analysis_step.general_desc}
Scene Camera Movement: {image_analysis_step.view_changes}
"""

        # return scene_description
        prompt_analysis_step = self.prompt_analysis(query=scene_description)

        general_prompt = f"""
Scene Summary: {prompt_analysis_step.scene_summary}
Problem Statement: {prompt_analysis_step.statement}
Problem Solution: {prompt_analysis_step.solution}
"""
        return scene_description, general_prompt 
    
        # enhanced_prompt =   (prompt_analysis_step['statement'] + "\n\n"
        #                     + image_analysis_step + "\n\n"
        #                     + "Problem Solution:\n"
        #                     + str(prompt_analysis_step['solution'])
        # )# image_analysis_step + "\n" + prompt_analysis_step

        # return enhanced_prompt, prompt_analysis_step['statement']
        