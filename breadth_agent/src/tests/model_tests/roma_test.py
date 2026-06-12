from romatch import roma_outdoor
from PIL import Image
import numpy as np
import cv2
device = "cuda:7"

roma_model = roma_outdoor(device=device)

imA_path = "/home/anthonyq/datasets/DTU/scan4/clean_001_max.png"
imB_path = "/home/anthonyq/datasets/DTU/scan4/clean_002_max.png"
img1 = Image.open(imA_path)
img2 = Image.open(imB_path)
W_A, H_A = img1.size
W_B, H_B = (W_A, H_A)
# Match
warp, certainty = roma_model.match(img1, img2, device=device)
# Sample matches for estimation
matches, certainty = roma_model.sample(warp, certainty)
# Convert to pixel coordinates (RoMa produces matches in [-1,1]x[-1,1])
kptsA, kptsB = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
# Find a fundamental matrix (or anything else of interest)
print(kptsA.cpu().numpy().shape)
F, mask = cv2.findFundamentalMat(
    kptsA.cpu().numpy(), kptsB.cpu().numpy(), ransacReprojThreshold=0.2, method=cv2.USAC_MAGSAC, confidence=0.999999, maxIters=10000
)
