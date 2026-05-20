from autosfm import AutoSFM
from core.logger import Logger


api_directory = "/home/anthonyq/projects/scene_agent/breadth_agent/src/agent/agent_details/tool_context"
instruction_path = "/home/anthonyq/projects/scene_agent/breadth_agent/src/agent/agent_details/agent_instructions/prompt_enh_examples.txt"

gpu_num = "6"
reconstruction_type = "Camera Pose Reconstruction"

img_postfix = "vggt_random_10" # Swap to Sequential String when ready
# img_postfix = "middle_sequential_10"
co3d_images = ["mouse/107_12753_23606", "mouse/377_43416_86289", 
               "orange/374_42196_84367", "orange/385_45386_90752", 
               "plant/247_26441_50907", "plant/374_42005_84358"]

for i in range(len(co3d_images)):
    img_seq = co3d_images[i]
    c, seq = img_seq.split('/')
    image_path = f"/home/anthonyq/datasets/co3d_v2/{img_seq}/{img_postfix}"
    calibration_path = f"/home/anthonyq/datasets/co3d_v2/{c}/calibration_new_{seq}.npz"

    # Setup Logger
    logger_file = f"co3d_{c}_{seq}_log"
    log_dir = f"/home/anthonyq/projects/scene_agent/breadth_agent/results/co3d/{c}_{seq}_{img_postfix}"
    logger = Logger(desc=logger_file, log_dir=log_dir)

    # Setup agent
    autosfm = AutoSFM(model_name="gpt-5", 
                    api_directory=api_directory,#'/work/scene_agent/breadth_agent/src/agent/agent_details/tool_context', 
                    instruction_path=instruction_path,#'/work/scene_agent/breadth_agent/src/agent/agent_details/agent_instructions/prompt_enh_examples.txt', 
                    reasoning_effort="medium",
                    logger=logger,
                    gpu_num=gpu_num)

    # Prompt
    gpu_mem = "48gb"
    temp_prompt = {'images':image_path,
    'calibration':calibration_path,
    'recon_type':reconstruction_type,
    'gpu_mem':gpu_mem}

    results = autosfm.run(temp_prompt)

    print(f"CODE from Scene co3dv2_{co3d_images[i]}:")
    print('\n', results[1])
    print("\n")
    print(f'Results from Scene co3dv2_{co3d_images[i]} in Metrics:\n', results[2])