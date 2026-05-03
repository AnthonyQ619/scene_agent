from autosfm import AutoSFM
from core.logger import Logger


api_directory = "/home/anthonyq/projects/scene_agent/breadth_agent/src/agent/agent_details/tool_context"
instruction_path = "/home/anthonyq/projects/scene_agent/breadth_agent/src/agent/agent_details/agent_instructions/prompt_enh_examples.txt"

image_path = "/home/anthonyq/datasets/co3d_v2/hydrant/167_18184_34441/images"
calibration_path = "/home/anthonyq/datasets/co3d_v2/hydrant/calibration_new_167_18184_34441.npz"

logger_file = "co3d_hydrant_log"
log_dir = "/home/anthonyq/projects/scene_agent/breadth_agent/results/co3d/hydrant"
logger = Logger(desc=logger_file, log_dir=log_dir)

gpu_num = "6"

autosfm = AutoSFM(model_name="gpt-5", 
                api_directory=api_directory,#'/work/scene_agent/breadth_agent/src/agent/agent_details/tool_context', 
                instruction_path=instruction_path,#'/work/scene_agent/breadth_agent/src/agent/agent_details/agent_instructions/prompt_enh_examples.txt', 
                reasoning_effort="medium",
                logger=logger,
                gpu_num=gpu_num)

# Prompt
reconstruction_type = "Sparse Reconstruction"
gpu_mem = "48gb"
temp_prompt = {'images':image_path,
'calibration':calibration_path,
'recon_type':reconstruction_type,
'gpu_mem':gpu_mem}

results = autosfm.run(temp_prompt)

print(results[0])
print('\n', results[1])
print('\n', results[2])