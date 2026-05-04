from autosfm import AutoSFM
from core.logger import Logger


api_directory = "/home/anthonyq/projects/scene_agent/breadth_agent/src/agent/agent_details/tool_context"
instruction_path = "/home/anthonyq/projects/scene_agent/breadth_agent/src/agent/agent_details/agent_instructions/prompt_enh_examples.txt"
gpu_num = "3"

# Prompt Info
gpu_mem = "48gb"
reconstruction_type = "Sparse Reconstruction"

ETH_images = ["relief", "relief_2", "terrace", "terrains"]


for i in range(len(ETH_images)):
    image_path = f"/home/anthonyq/datasets/ETH/ETH/{ETH_images[i]}/images/dslr_images_undistorted"
    calibration_path = f"/home/anthonyq/datasets/ETH/ETH/{ETH_images[i]}/dslr_calibration_undistorted/calibration_ETH_new.npz"

    logger_file = f"eth_{ETH_images[i]}_log"
    log_dir = f"/home/anthonyq/projects/scene_agent/breadth_agent/results/ETH/eth_{ETH_images[i]}"
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
    
    print(f"CODE from Scene eth_{ETH_images[i]}:")
    print('\n', results[1])
    print("\n")
    print(f'Results from scene {ETH_images[i]} in Metrics:\n', results[2])