from autosfm import AutoSFM
from core.logger import Logger


api_directory = "/home/anthonyq/projects/scene_agent/breadth_agent/src/agent/agent_details/tool_context"
instruction_path = "/home/anthonyq/projects/scene_agent/breadth_agent/src/agent/agent_details/agent_instructions/prompt_enh_examples.txt"
gpu_num = "2" # Change accordingly!

# Prompt Info
gpu_mem = "48gb"
reconstruction_type = "Sparse Reconstruction"

#1920 Calibration Images for all of them!
TT_images = ["barn_1_40", "barn_186_225", "barn_371_410"]


for i in range(len(TT_images)):
    image_path = f"/home/anthonyq/datasets/tanks_and_temples/{TT_images[i]}"
    calibration_path = "/home/anthonyq/datasets/tanks_and_temples/calibration_new_1920.npz"

    logger_file = f"TT_{TT_images[i]}_log"
    log_dir = f"/home/anthonyq/projects/scene_agent/breadth_agent/results/TT/{TT_images[i]}"
    logger = Logger(desc=logger_file, log_dir=log_dir)
    autosfm = AutoSFM(model_name="gpt-5", 
                    api_directory=api_directory,#'/work/scene_agent/breadth_agent/src/agent/agent_details/tool_context', 
                    instruction_path=instruction_path,#'/work/scene_agent/breadth_agent/src/agent/agent_details/agent_instructions/prompt_enh_examples.txt', 
                    reasoning_effort="medium",
                    logger=logger,
                    gpu_num=gpu_num)

    # Prompt
    temp_prompt = {'images':image_path,
    'calibration':calibration_path,
    'recon_type':reconstruction_type,
    'gpu_mem':gpu_mem}

    results = autosfm.run(temp_prompt)
    
    print(f"CODE from Scene TT_{TT_images[i]}:")
    print('\n', results[1])
    print("\n")
    print(f'Results from Scene {TT_images[i]} in Metrics:\n', results[2])