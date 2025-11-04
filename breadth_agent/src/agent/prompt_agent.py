from openai import OpenAI
from sceneprogllm import LLM
import glob
import os 
import re

class ImageAnalysisLLM():
    def __init__(self, 
                 model: str | None = None, 
                 system_desc: str | None = None,
                 k_images: int = 3,
                 response_format: str = "text",
                 use_cache: bool = False,
                 temperature: float = 0.8,
                 ):
        self.k = k_images
        default_desc = f"""
        You are an agent set to evaluate the {self.k} incoming images. For your response, 
        you need to:

        1. Determine if the images are outdoors or indoors. 
        2. Determine if there are illumination changes across the {self.k} images evaluated. 
        3. Determine if the image contains low textured regions, or high textured regions.
        4. Determine if the images provided belong to a stereo camera or monocular
        camera. (Check image path if left/right are included in image names) 
        5. Across the {self.k} images, determine if movement includes any extreme 
        viewpoint changes, or if the camera gradually moves across the scene in increments.
        """
        
        # Hard code model for now
        self.model = "gpt-4o-mini"

        if system_desc is None:
            self.sys_desc = default_desc
        else:
            self.sys_desc = system_desc

        self.image_analysis_model = LLM(name = 'llm_image_evaluator', 
                                        system_desc = self.sys_desc, 
                                        response_format = response_format,
                                        use_cache = use_cache,
                                        model_name = self.model,
                                        temperature = temperature)
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

        output = self.image_analysis_model("Analyze the set of input images.", image_paths=image_paths)

        header = "Scene Description: \n"

        return header + output

class PromptEnhancerLLM():
    def __init__(self,
                 model: str | None = None,
                 system_desc: str | None = None,
                 instruction_path: str | None = None,
                 use_cache: bool = False,
                 temperature: float = 0.8,
                 ):
        
        default_desc = f"""
You are an agent instructed to breakdown prompts that discuss a problem that 
is related to computer vision into a high-level, detailed, solution.

The goal is to understand what is the general computer vision problem to complete at hand,
state what that problem is, then discuss what are the necessary steps to solve that problem
in the most detailed manner, but keep things brief when possible. 

Avoid coding anything, or suggesting any libraries, such as OpenCV. The goal is to keep
everything in a text description and as general as possible. Aim for a textbook description level.

With this in mind, first identify what type of computer vision problem the prompt wants to solve.
This will be the problem statement, in which you state a brief description of the scene and what 
computer vision algorithm will be deployed.

Then, discuss the problem solution. 
Determine what steps are required to solve the specific computer vision problem such as 
feature detection, feature matching, feature tracking, camera pose estimation, intrinsic estimation,
3d point cloud reconstruction, dense reconsturction with multi-view stereo, etc.
"""
        
        # Hard code model for now
        self.model = "gpt-4o-mini"

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

        self.full_desc = self.sys_desc + examples

        # Build LLM

        self.prompt_enhancer =  LLM("llm_prompt_analyzer", 
                                    system_desc = self.full_desc,
                                    response_format="json",
                                    json_keys=["statement:str", "solution:str"],
                                    use_cache = use_cache,
                                    model_name = self.model,
                                    temperature = temperature)

    def __call__(self, query):
        response = self.prompt_enhancer(query)

        return response


class PromptEnhancmentAgent():
    def __init__(self, model: str | None = None, 
                 image_model_desc: str | None = None, 
                 prompt_model_desc: str | None = None,
                 instruction_path: str | None = None,
                 k_images: int = 3,
                 response_format: str = "text",
                 use_cache: bool = False,
                 temperature: float = 0.8):
        
        if model is None:
            self.image_analysis =   ImageAnalysisLLM(system_desc=image_model_desc,
                                                     k_images=k_images,
                                                     response_format=response_format,
                                                     use_cache=use_cache,
                                                     temperature=temperature)
            self.prompt_analysis = PromptEnhancerLLM(system_desc=prompt_model_desc,
                                                     instruction_path=instruction_path,
                                                     use_cache=use_cache,
                                                     temperature=temperature)
        else:
            self.image_analysis = ImageAnalysisLLM(model=model, 
                                                   system_desc=image_model_desc,
                                                   k_images=k_images,
                                                   response_format=response_format,
                                                   use_cache=use_cache,
                                                   temperature=temperature)
            self.prompt_analysis = PromptEnhancerLLM(model=model,
                                                     system_desc=prompt_model_desc,
                                                     instruction_path=instruction_path,
                                                     use_cache=use_cache,
                                                     temperature=temperature)
            
    def __call__(self, query):
        print(query)
        image_analysis_step = self.image_analysis(query=query)

        prompt_analysis_step = self.prompt_analysis(query=query + "\n" + image_analysis_step)


        enhanced_prompt =   (prompt_analysis_step['statement'] + "\n\n"
                            + image_analysis_step + "\n\n"
                            + "Problem Solution:\n"
                            + str(prompt_analysis_step['solution'])
        )# image_analysis_step + "\n" + prompt_analysis_step

        return enhanced_prompt, prompt_analysis_step['statement']
        