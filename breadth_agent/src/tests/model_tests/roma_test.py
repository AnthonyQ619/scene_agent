from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from romatch.utils.utils import tensor_to_pil
from romatch import roma_indoor

device = "cuda:5" if torch.cuda.is_available() else "cpu"

if device == "cuda:5":
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
else:
    dtype = torch.float32

roma_model = roma_indoor(device=device)

im1_path = "/home/anthonyq/datasets/DTU/DTU/scan10/images/clean_001_4_r5000.png"
im2_path = "/home/anthonyq/datasets/DTU/DTU/scan10/images/clean_002_4_r5000.png"
H, W = roma_model.get_output_resolution()
W_A, H_A = Image.open(im1_path).size
W_B, H_B = Image.open(im2_path).size

# Match
warp, certainty = roma_model.match(im1_path, im2_path, device=device)

matches, certainty = roma_model.sample(warp, certainty)
kpts1, kpts2 = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)

kpt1_np, kpt2_np = kpts1.cpu().numpy(), kpts2.cpu().numpy()

print(kpt1_np)
print(certainty)