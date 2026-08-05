import os
import torch
import numpy as np
import gzip
import json
import random
import logging
import warnings
import glob
from pathlib import Path
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt_ba import run_vggt_with_ba

# Log Folder
log_folder = "/home/anthonyq/projects/scene_agent/breadth_agent/results/vggt_sparse_results"
gpu_num = "7"

# Load Model/Images
def load_model(device, model_path):
    """
    Load the VGGT model.

    Args:
        device: Device to load the model on
        model_path: Path to the model checkpoint

    Returns:
        Loaded VGGT model
    """
    print("Initializing and loading VGGT model...")
    model = VGGT()
    # _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    # model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    print(f"USING {model_path}")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to(device)
    return model

WEIGHT_MODULE = "/home/anthonyq/projects/scene_agent/breadth_agent/src/modules/models/sfm_models/vggt/weights/model.pt"

# Co3Dv2 Test Run!
# img_postfix = "vggt_random_10" # Swap to Sequential String when ready
img_postfix = "middle_sequential_10"
co3d_images = ["apple/110_13051_23361", "apple/189_20393_38136",
               "ball/123_14363_28981", "ball/375_42693_85518",
               "bench/415_57112_110099", "bench/415_57121_110109",
               "book/119_13962_28926", "book/247_26469_51778", 
               "bowl/69_5465_12831", "bowl/70_5792_13401", 
               "broccoli/372_41112_81867", "broccoli/412_56288_108844",
               "cake/374_42274_84517", "cake/403_53094_103680", 
               "donut/391_47032_93657", "donut/403_52964_103416", 
               "hydrant/167_18184_34441", "hydrant/411_56064_108483"]
d_set = "co3d"
failed_runs = []

# Setup device and data type
device = f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Load model
model = load_model(device, model_path=WEIGHT_MODULE)

with torch.no_grad():
    errors = []
    with open(f"{log_folder}/{d_set}/mean_result.txt", "w") as f:
        for i in range(len(co3d_images)):
            img_seq = co3d_images[i]
            c, seq = img_seq.split('/')
            image_path = f"/home/anthonyq/datasets/co3d_v2/{img_seq}/{img_postfix}"
            
            outpath = f"{log_folder}/{d_set}/{img_postfix}/{c}/{seq}"
            os.makedirs(f"{log_folder}/{d_set}/{img_postfix}/{c}", exist_ok=True)
            log_file = f"{img_seq}_{img_postfix}_pose_log"
            image_path_list = glob.glob(os.path.join(image_path, "*"))

            images = load_and_preprocess_images(image_path_list).to(device)
            _, error = run_vggt_with_ba(model, images, f"{log_folder}/{d_set}/{img_postfix}", log_file, dtype=dtype)

            # _, error = demo_fn(gpu_num, image_path, f"{log_folder}/{d_set}/{img_postfix}", log_file)
            print(f"{img_seq}_{img_postfix} Error: {error}")
            errors.append(error)
            f.write(f"{img_seq}_{img_postfix} Error: {error}\n")

        print("Reprojection Mean:", np.mean(errors))
        # with open(f"{log_folder}/{d_set}/mean_result.txt", "w") as f:
        f.write(f"Mean reprojection value: {np.mean(errors)}\n")