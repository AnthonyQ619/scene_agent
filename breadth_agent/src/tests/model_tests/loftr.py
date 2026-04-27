import cv2
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

fname1 = "/home/anthonyq/datasets/DTU/DTU/scan20/images/clean_001_4_r5000.png"
fname2 = "/home/anthonyq/datasets/DTU/DTU/scan20/images/clean_002_4_r5000.png"

img1 = K.image_to_tensor(np.array(Image.open(fname1).convert("RGB"))).float()[None, ...] / 255.0
img2 = K.image_to_tensor(np.array(Image.open(fname2).convert("RGB"))).float()[None, ...] / 255.0

img1 = K.geometry.resize(img1, (600, 375), antialias=True)
img2 = K.geometry.resize(img2, (600, 375), antialias=True)


matcher = KF.LoFTR(pretrained="indoor")

input_dict = {
    "image0": K.color.rgb_to_grayscale(img1),  # LofTR works on grayscale images only
    "image1": K.color.rgb_to_grayscale(img2),
}

with torch.inference_mode():
    correspondences = matcher(input_dict)

for k, v in correspondences.items():
    print(k)

mkpts0 = correspondences["keypoints0"].cpu().numpy()
mkpts1 = correspondences["keypoints1"].cpu().numpy()
Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)

inliers = inliers > 0
print(mkpts0.shape)
print(inliers.sum())