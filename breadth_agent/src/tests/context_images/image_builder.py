import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from modules.utilities import image_builder

# def dataset_parser(ds_length: int, img_total: int) -> list[int]:
#     index_skip = ds_length // img_total
#     indices = [i*index_skip for i in range(img_total)]

#     return indices


# def read_image(image_path: str, 
#                max_size: int,
#                interpolation=cv2.INTER_AREA):
        
#         img = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
#         if img is None:
#             raise ValueError(f"Could not read image: {image_path}")

#         h, w = img.shape[:2]
#         max_dim = max(h, w)

#         # No resize needed
#         if max_dim <= max_size:
#             return img, 1.0

#         scale = max_size / max_dim
#         new_w = int(round(w * scale))
#         new_h = int(round(h * scale))

#         resized = cv2.resize(
#             img,
#             (new_w, new_h),
#             interpolation=interpolation
#         )   

#         return resized

# def build_images(image_path: list, max_size: int) -> np.ndarray:
#     temp_img = None
#     curr_img = None

#     for img in image_path:
#         temp_img = read_image(img, max_size=max_size)

#         if curr_img is None:
#             curr_img = temp_img
#         else:
#             curr_img = np.hstack((curr_img, temp_img))

#     return curr_img

# def image_builder(image_path: str, max_size: int, k: int = 5):
#     # if skip_frames:
#     #     imgs = sorted(glob.glob(image_path + "\\*"))
#     #     indices = [i*skip_num for i in range(k)]
#     #     print(indices)
#     #     full_img_path = [imgs[i] for i in indices]
#     # else: 
#     #     full_img_path = sorted(glob.glob(image_path + "\\*"))[:k]
    
#     img_set = sorted(glob.glob(image_path + "\\*"))
#     chosen_indices = dataset_parser(len(img_set), k)

#     full_img_path = [img_set[i] for i in chosen_indices]

#     new_img = build_images(full_img_path, max_size=max_size)

#     return new_img

def main():
    image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\ETH\\statue\\images\\dslr_images_undistorted"
    img_to_save = r"C:\Users\Anthony\Documents\Projects\scene_agent\breadth_agent\src\tests\context_images"
    i = 8

    # Use equivalent parameters to 
    new_img = image_builder(image_path=image_path, 
                            max_size=350, 
                            k=5)
    
    # Show Image to User
    imgplot = plt.imshow(new_img)
    plt.show()
    
    # Save Image for Context
    cv2.imwrite(img_to_save + f"\\image_context{i}.png", cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))
    

if __name__ == "__main__":
    main()