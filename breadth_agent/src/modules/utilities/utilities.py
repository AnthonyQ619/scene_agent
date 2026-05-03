import cv2
import numpy as np
import sys
import re
sys.path.insert(0, "C:\\Users\\Anthony\\Documents\\Projects\\scene_agent\\breadth_agent\\src\\modules")
from modules.DataTypes.datatype import Calibration
import glob
from pathlib import Path
import os

# function to convert calibration data into a readable numpy file

def convertToNPZ(path_name:str, file_name: str | None = None, np_data: list[np.ndarray] | None = None):
    ordered_keys = ['K1', 'Dist', 'K2', 'Dist2', 'Extrinsics']
    calibration = {}
    if file_name is not None:
        # File should store data line by line
        # K1: k11, k12, k13, k21, k22, ...
        # Dist: d1, d2, d3, d4, d5
        # K2: etc..
        # Dist2: d1, d2, ..
        # Extrinsics: R11, R12, ... , T1, T2, T3
        data = open(file_name)
        lines = data.readlines()

        for i in range(len(lines)):
            line = lines[i]
            
            cal_data = line.strip().split(',')
            cal_data[0] = cal_data[0].split(":")[1].lstrip()
            cal_data = np.array(cal_data, dtype=np.float32).reshape((3,3))
            calibration[ordered_keys[i]] = cal_data
            
            print(cal_data)

        print(calibration)
        calibrationer = Calibration(**calibration)

        print(calibrationer.K1)
        np.savez(path_name + "\\calibration.npz", **calibration)
    elif np_data is not None:
        for i in range(len(np_data)):
            calibration[ordered_keys[i]] = np_data[i]

        np.savez(path_name + "\\calibration.npz", **calibration)


class CalibrationReader:
    def __init__(self, file_path: str):
        #self.calibration = Calibration() # Think about given a subset to a full set of calibration data, how we can fill the calibration data like so by passing in certain arguments
        self.calibration = self.read_npz(file_path)

    def read_npz(self, file_path) -> Calibration:
        data = np.load(file_path)
        full_data = dict(data)
        calibration = Calibration(**full_data)

        return calibration
    
    def get_calibration(self) -> Calibration:
        return self.calibration

# Image Parsing Module
def image_builder(image_path: str, max_size: int, k: int = 5):
    # Internal Helper Functions
    def dataset_parser(ds_length: int, img_total: int) -> list[int]:
        index_skip = ds_length // img_total
        indices = [i*index_skip for i in range(img_total)]

        return indices

    def read_image(image_path: str, 
                max_size: int,
                interpolation=cv2.INTER_AREA):
            
            img = cv2.imread(image_path, cv2.IMREAD_COLOR) #cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")

            h, w = img.shape[:2]
            max_dim = max(h, w)

            # No resize needed
            if max_dim <= max_size:
                return img, 1.0

            scale = max_size / max_dim
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))

            resized = cv2.resize(
                img,
                (new_w, new_h),
                interpolation=interpolation
            )   

            return resized

    def build_images(image_path: list, max_size: int) -> np.ndarray:
        temp_img = None
        curr_img = None

        for img in image_path:
            temp_img = read_image(img, max_size=max_size)

            if curr_img is None:
                curr_img = temp_img
            else:
                curr_img = np.hstack((curr_img, temp_img))

        return curr_img

    all_images = str(Path(image_path) / "*")
    img_set = sorted(glob.glob(all_images))#(image_path + "\\*"))
    chosen_indices = dataset_parser(len(img_set), k)
    full_img_path = [img_set[i] for i in chosen_indices]

    new_img = build_images(full_img_path, max_size=max_size)

    return new_img
def read_image(image_path: str, 
                max_size: int,
                interpolation=cv2.INTER_AREA):
            
    img = cv2.imread(image_path, cv2.IMREAD_COLOR) #cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w = img.shape[:2]
    max_dim = max(h, w)

    # No resize needed
    if max_dim <= max_size:
        return img, 1.0

    scale = max_size / max_dim
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=interpolation
    )   
    scale = (new_w / w, new_h / h)

    return resized, scale

def resize_dataset(image_path: list, max_size: int, calib_path: str | None = None, script_id:str="") -> str:
    temp_directory = Path(__file__).resolve().parents[3] / "results" / f"resized_dataset_{script_id}"
    if not os.path.exists(str(temp_directory)):
            os.makedirs(str(temp_directory))

    for i in range(len(image_path)):
        temp_img, scale = read_image(image_path[i], max_size=max_size)
        cv2.imwrite(str(temp_directory / f"image{i}.png"), temp_img)

    if calib_path is not None:
        width_scale, height_scale = scale
        data = np.load(calib_path)
        data.allow_pickle = True 
        full_cal_data = dict(data)

        # Keys are
        # - k_mats: (N, 3, 3) -> N = num of cameras
        # - dists: (N, 1, 5) -> N = num of cameras 
        # - baseline_ext: None or (3, 4) -> the baseline of stereo camera

        intrinsics = full_cal_data['k_mats'][0]
    
        intrinsics[0,0] = width_scale * intrinsics[0,0]   # fx x width_scale = fx'
        intrinsics[1,1] = height_scale * intrinsics[1,1]  # fy x height_scale = fy'
        intrinsics[0,2] = width_scale * intrinsics[0,2]   # cx x width_scale = cx'
        intrinsics[1,2] = height_scale * intrinsics[1,2]  # cy x height_scale = cy'
    else:
        intrinsics = None

    return str(temp_directory), sorted(glob.glob(str(temp_directory / "*"))), intrinsics


def clean_dir(directory: str) -> None:
    import shutil

    # Recursively deletes the directory and all subdirectories/files
    shutil.rmtree(directory)

# if __name__=='__main__':
#     # text_path = 'C:\\Users\\Anthony\\Documents\\Projects\\datasets\\Structure-from-Motion\\calibration.txt'
#     path_name = 'C:\\Users\\Anthony\\Documents\\Projects\\datasets\\Structure-from-Motion\\calibration.npz'
#     # convertToNPZ(path_name=path_name, file_name=text_path)

#     calibration = CalibrationReader(path_name).get_calibration()

#     print(calibration.K1)






