from autosfm import AutoSFM
from core.logger import Logger


api_directory = "/home/anthonyq/projects/scene_agent/breadth_agent/src/agent/agent_details/tool_context"
instruction_path = "/home/anthonyq/projects/scene_agent/breadth_agent/src/agent/agent_details/agent_instructions/prompt_enh_examples.txt"

image_path = "/home/anthonyq/datasets/DTU/DTU/scan14"
calibration_path = "/home/anthonyq/datasets/DTU/DTU/calibration_DTU_new.npz"

logger_file = "scan14_log"
log_dir = "/home/anthonyq/projects/scene_agent/breadth_agent/results/DTU/scan14"
logger = Logger(desc=logger_file, log_dir=log_dir)


autosfm = AutoSFM(model_name="gpt-5", 
                api_directory=api_directory,#'/work/scene_agent/breadth_agent/src/agent/agent_details/tool_context', 
                instruction_path=instruction_path,#'/work/scene_agent/breadth_agent/src/agent/agent_details/agent_instructions/prompt_enh_examples.txt', 
                reasoning_effort="medium",
                logger=logger)

# Prompt
reconstruction_type = "Camera Pose Reconstruction"
gpu_mem = "48gb"
temp_prompt = {'images':image_path,
'calibration':calibration_path,
'recon_type':reconstruction_type,
'gpu_mem':gpu_mem}

results = autosfm.run(temp_prompt)

print(results[0])
print('\n', results[1])
print('\n', results[2])