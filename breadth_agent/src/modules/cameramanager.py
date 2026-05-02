import numpy as np
import cv2
from typing import List, Optional, Tuple, Union
from PIL import Image, ImageOps
# from models.sfm_models.vggt.utils.load_fn import load_and_preprocess_images, load_and_preprocess_images_square
from modules.DataTypes.datatype import (CameraData,
                                 Points2D, 
                                 Calibration, 
                                 Points3D, 
                                 CameraPose, 
                                 Scene, 
                                 PointsMatched,
                                 BundleAdjustmentData)
import glob
import random
from tqdm import tqdm
import os
import shutil
from pathlib import Path
# from baseclass import ImageProcessorClass

def uniform_image_subset(image_files, num_samples):
    if num_samples <= 0:
        return []

    if len(image_files) <= num_samples:
        return image_files

    indices = np.linspace(
        0,
        len(image_files) - 1,
        num_samples,
        dtype=int
    )

    return [image_files[i] for i in indices]

class CameraDataManager():
    def __init__(self, 
                 image_path: str,
                 max_images: int | None = None,
                 calibration_path: str | None = None,
                 target_resolution: Tuple[int, int] | None = None,
                 colmap_workspace: str | None = None):

        file_names = sorted(p.name for p in Path(image_path).glob("*"))
        if max_images is None:
            if os.name == 'nt':
                image_files = sorted(glob.glob(image_path + "\\*"))#[:5]
            else:
                image_files = sorted(glob.glob(image_path + "/*"))
        else:
            if os.name == 'nt':
                image_files = sorted(glob.glob(image_path + "\\*"))#[:max_images]
            else:
                image_files = sorted(glob.glob(image_path + "/*"))
            if len(image_files) > max_images:
                # image_files = uniform_image_subset(image_files, num_samples=max_images)
                image_files = image_files[:max_images]

        print(image_files)
        # self.directory_path = Path(__file__).resolve().parents[2]
        # self.directory_path = str(self.directory_path / "results" / "workspace") 
        self.directory_path = colmap_workspace
        if os.path.exists(self.directory_path):
            # Delete the directory and all its contents
            shutil.rmtree(self.directory_path)

        # Recreate an empty directory
        os.makedirs(self.directory_path)
        if os.name == 'nt':
            self.image_dir = self.directory_path + "\\images"
        else:
            self.image_dir = self.directory_path + "/images"
        # Recreate an empty image directory
        os.makedirs(self.image_dir)

        #self.camera_data = CameraData()    
        if calibration_path is None:
            calibrated = False
            intrinsics = None
            distortions = None
            extrinsic = None
        else:
            intrinsics, distortions, extrinsic = self._read_calibration(cal_file_path=calibration_path)
            calibrated = True

        image_list, image_scale, image_shape_old, image_shape_new = self._read_images(image_path=image_files,
                                                                                      max_size=1600,
                                                                                      target_resolution=target_resolution,
                                                                                      is_cal=calibrated)
        
        # intrinsics, distortions, extrinsic = self._read_calibration(cal_file_path=calibration_path)
        stereo = False
        multi_camera = False

        if extrinsic is not None:
            stereo = True
        if intrinsics is not None and len(intrinsics) > 2:
            multi_camera = True

        self.cam_data = CameraData(image_names=file_names,
                                   image_list=image_list,
                                   image_shape_old=image_shape_old,
                                   image_shape_new=image_shape_new,
                                   image_scale=image_scale,
                                   intrinsics=intrinsics,
                                   distortions=distortions,
                                   extrinsic=extrinsic,
                                   stereo=stereo,
                                   multi_cam=multi_camera)
        
        # Update Intrinsics if needed
        self.cam_data.update_calibration(img_scale=image_scale)

    def _read_calibration(self, 
                          cal_file_path: str) -> tuple[List[np.ndarray],
                                                       List[np.ndarray]]:
        data = np.load(cal_file_path)
        data.allow_pickle = True 
        full_cal_data = dict(data)

        # Keys are
        # - k_mats: (N, 3, 3) -> N = num of cameras
        # - dists: (N, 1, 5) -> N = num of cameras 
        # - baseline_ext: None or (3, 4) -> the baseline of stereo camera

        K_mats = full_cal_data['k_mats']
        dists = full_cal_data['dists']
        baseline_ext = full_cal_data['baseline_ext']

        intrinsics = []
        distortions = []

        for i in range(K_mats.shape[0]):
            intrinsics.append(K_mats[i])
            distortions.append(dists[i])

        print(type(baseline_ext))
        if baseline_ext != None:
            assert len(intrinsics) == 2, "Camera is not stereo, or not enough information stored in calibration file. Revisit calibration information"
            extrinsic = baseline_ext
        else:
            extrinsic = None

        return intrinsics, distortions, extrinsic
 
    def _read_images(self, 
                     image_path: str,
                     max_size: int,
                     target_resolution: Tuple[int, int] | None = None,
                     is_cal: bool = False) -> tuple[List[np.ndarray],
                                                        Tuple[float, float],
                                                        Tuple[int, int]]:
        
        image_list = []

        # Only used for models in which intrinsics can be estimated.
        # This is due to all ML models taking in square images as input
        if not is_cal:
            if target_resolution is None:
                target_res = 1024 # Default to size 1024 pixel height x width
            else:
                target_res = target_resolution[0] # Target first shape value in resolution (Should be squared anyways)
            
            for i in tqdm(range(len(image_path)), desc="Reading Images"):
                img = image_path[i]

                # Read Image and attain meta data
                image = Image.open(img)
                image = ImageOps.exif_transpose(image)
                width, height = image.size
                image = image.convert("RGB") # Confirm image is in RGB

                # Make the image square by padding the shorter dimension
                max_dim = max(width, height)

                # Calculate padding
                left = (max_dim - width) // 2
                top = (max_dim - height) // 2

                # Calculate scale factor for resizing
                scale = target_res / max_dim

                square_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
                square_img.paste(image, (left, top))

                # Resize to target size
                square_img = square_img.resize((target_res, target_res), Image.Resampling.BICUBIC)
                square_img_np = np.asarray(square_img)

                image_list.append(square_img_np)
                
                # Write Image to Colmap Workspace
                if os.name == 'nt':
                    file_name = f"{self.image_dir}\\{i:06d}.png"
                else:
                    file_name = f"{self.image_dir}/{i:06d}.png"
                square_img.save(file_name)
                # square_img.save(f"{self.image_dir}\\{i:06d}.png")

            image_scale = (scale, scale)
            image_shape_old = (width, height)
            image_shape_new = (target_res, target_res)
        else:
            img = image_path[0]
            image = Image.open(img)
            image = ImageOps.exif_transpose(image)
            w, h = image.size
            image = image.convert("RGB") # Confirm image is in RGB

            # Get largest dim of image
            max_dim = max(h, w)
 
            if target_resolution is not None:
                h_new, w_new = target_resolution #TODO: ENSURE EVERYTHING IS (W, H), including RESHAPE PARAM
                scale = (w_new / w, h_new / h)
                target_res = (w_new, h_new)
            # elif max_dim <= max_size: # No resize needed
            #     scale = (1.0, 1.0)
            #     target_res = (w, h)
            # else:
            #     scale_pt = max_size / max_dim
            #     new_w = int(round(w * scale_pt))
            #     new_h = int(round(h * scale_pt))
            #     scale = (scale_pt, scale_pt)
            #     target_res = (new_w, new_h)
            elif h > 1800 or w > 1800:
                if h > w:
                    h_new, w_new = (1600, 1200)
                elif w > h: 
                    h_new, w_new = (1200, 1600)
                elif w == h:
                    h_new, w_new = (1024, 1024)
                    
                scale = (w_new / w, h_new / h)
                target_res = (w_new, h_new)
            else:
                scale = (1.0, 1.0)
                target_res = (w, h)

           
            image_scale = scale
            image_shape_old = (w, h)
            image_shape_new = target_res

            # Read Images
            for i in tqdm(range(len(image_path)), desc="Reading Images"):
                img = image_path[i]
                image = Image.open(img)
                image = ImageOps.exif_transpose(image)
                width, height = image.size
                image = image.convert("RGB") # Confirm image is in RGB

                image = image.resize(target_res, Image.Resampling.BICUBIC)
                image_np = np.asarray(image)

                image_list.append(image_np)

                # Write Image to Colmap Workspace
                if os.name == 'nt':
                    file_name = f"{self.image_dir}\\{i:06d}.png"
                else:
                    file_name = f"{self.image_dir}/{i:06d}.png"
                image.save(file_name)

        return image_list, image_scale, image_shape_old, image_shape_new
            
    def get_camera_data(self) -> CameraData:
        return self.cam_data



