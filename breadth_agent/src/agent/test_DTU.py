from autosfm import AutoSFM
from core.logger import Logger


api_directory = "/home/anthonyq/projects/scene_agent/breadth_agent/src/agent/agent_details/tool_context"
instruction_path = "/home/anthonyq/projects/scene_agent/breadth_agent/src/agent/agent_details/agent_instructions/prompt_enh_examples.txt"
gpu_num = "2"
scans = [4]#[1, 4, 9, 10]

for i in range(len(scans)):
    image_path = f"/home/anthonyq/datasets/DTU/scan{scans[i]}"
    calibration_path = f"/home/anthonyq/datasets/DTU/calibration_DTU_new.npz"

    logger_file = f"scan{scans[i]}_log"
    log_dir = f"/home/anthonyq/projects/scene_agent/breadth_agent/results/DTU/scan{scans[i]}"
    logger = Logger(desc=logger_file, log_dir=log_dir)

    autosfm = AutoSFM(model_name="gpt-5", 
                    api_directory=api_directory,#'/work/scene_agent/breadth_agent/src/agent/agent_details/tool_context', 
                    instruction_path=instruction_path,#'/work/scene_agent/breadth_agent/src/agent/agent_details/agent_instructions/prompt_enh_examples.txt', 
                    reasoning_effort="medium",
                    logger=logger,
                    gpu_num=gpu_num)

    # Prompt
    reconstruction_type = "Dense Reconstruction"
    gpu_mem = "48gb"
    temp_prompt = {'images':image_path,
    'calibration':calibration_path,
    'recon_type':reconstruction_type,
    'gpu_mem':gpu_mem}

    results = autosfm.run(temp_prompt)

    # print(results[0])
    # print('\n', results[1])
    print("\n")
    print(f'Results from scane{scans[i]} in Metrics:\n', results[2])