from sceneprogllm import LLM

class PromptEnhancerLLM():
    def __init__(self,
                 model: str = "gpt-5-nano", 
                 instruction_path: str | None = None,
                 temperature: float = 0.8,
                 reasoning_effort: str ="medium"
                 ):
        
        system_desc = f"""
You are an expert in generating Structure from Motion solution pipelines in Computer Vision. Given a description of a scene regarding
scene texture, scene lighting, shadows on objects of interest, whether camera is calibrated, type of reconstruction (Pose, Sparse, or Dense),
or harware constraint, you are supposed to devise a problem statement that summarizes the scene, and
how we want to build a 3D reconstruction from that description.
        
DO NOT LIST AN ENTIRE SOLUTION PIPELINE. 
Structure the description as a problem statement that designates the computer vision problem to solve, 
whether data is calibrated, and how we want to solve said problem (Sparse, Dense, Visual Odometry, etc.) with any possible constraints
in mind (GPU memory capabilities, etc.).

In the prompt, you will be given a potential path to calibration, reconstruction type, and gpu memory. Determine
if a calibration path is provided, if the image set is calibrated, type of reconstruction we need to do (Dense or Sparse),
and whether the GPU provided is large enough to utilize models like VGGT when necessary or if the approach should be 
more memory conservative.
"""
# REMOVED PORTION
# For the description of the scene, focus on the different textures in the scene described from the prompt, whether lighting
# or shadows affect objects visibility and texture, outdoor or indoor, scene rotates around object or continuing on a path, etc.
# REMOVED PORTION

# Avoid coding anything, or suggesting any libraries, such as OpenCV. The goal is to keep
# everything in a text description and as general as possible. Aim for a textbook description level.
# Then, discuss the problem solution. 

# Determine what steps are required to solve the specific computer vision problem such as 
# feature detection, feature matching, feature tracking, camera pose estimation, intrinsic estimation,
# 3d point cloud reconstruction, dense reconsturction with multi-view stereo, etc.


        if instruction_path is None:
            add_instruction = ""
        else:
            context = open(instruction_path, 'r')
            add_instruction = context.read()
            context.close()

        footer = """
You are to provide your answer in the form of a json with the keys stated below, and what the content should be filled with in each:
statement: Description of the problem statement constructed. DO NOT LIST A SOLUTION PIPELINE. Just general description of what the SfM problem we are solving
reconstruction: Description of whether we should reconstruct the scene in sparse or dense format, or if we are focusing on pose estimation
calibration: Description of if the image data set provided is calibrated or not
memory: Description of whether we have enough memory for VGGT/MapAnything/MONSt3R to be used in specific cases, or we should use the classical approach of feature detection based solution
"""
# scene_summary: Summary of the scene description of which we plan to reconstruct. Utilize the provided information. Don't mention image count as description was only based on a subset of images
        # Build LLM
        sys_desc = system_desc + "\n" + add_instruction + "\n" + footer

        self.prompt_enhancer =  LLM(system_desc = sys_desc,
                                    response_format="json",
                                    response_params={"statement":"str", 
                                               "reconstruction":"str", 
                                               "calibration":"str",
                                               "memory":"str"},
                                            #    "scene_summary":"str"},
                                    model_name = model,
                                    temperature = temperature,
                                    reasoning_effort = reasoning_effort)


    def __call__(self, query):
        output = self.prompt_enhancer(query)

        return output