from sceneprogllm import LLM
import cv2
import os
import glob
import numpy as np


class ImageAnalyzer:
    def __init__(self, 
                 model: str = "gpt-5-nano", 
                 k_images: int = 4,
                 temperature: float = 0.8,
                 reasoning_effort: str ="medium"
                 ):
        
        self.k = k_images

        system_desc = f"""
        You are an expert set to evaluate the {self.k} incoming images. Given an image prompt (which can range from images rotating outside in of an object
        to a video walking along a path in an indoor or outdoor scene), you are supposed to describe every facet of the scene in plain english. For your response, 
        you need to describe the following characteristics of the scene. Specifically, be descriptive, but brief. Maximumg of 75 words per category:

        1. Determine if the images are of scenes outdoors or indoors environments. 
        2. Determine if there are consistent illumination changes across the {self.k} images evaluated in the scene or objects of interest.
        3. Determine if there a shadows across the objects of interest that limit texture visibility, and whether its low, moderate, or high impact.
        4. Determine if the image contains low, moderate, or high textured regions in objects you deem as closer to the images (High texture = many 
            edges and corner a detector like SIFT can detect).
        5. Describe key features of the scene, specifically [For this category, ensure each sub-category is limited to 50 words]
            4.1 if the scene rotates around an object or following along some path,
            4.2 contains an object of interest in the scene at all times, or no consistent object at all in the scene, 
            4.3 material of the object exhibits specular or diffuse lighting effects, and how that effects object texture
            4.4 Background of the scene exhibits any information that can be reconstructed, or depth disparity of object is extreme to background.
        6. Across the {self.k} images, determine if movement includes any extreme viewpoint changes, or if the camera gradually moves across the scene in increments.
        """
        
        self.analyzer = LLM(system_desc = system_desc,
                            response_format = "text",
                            model_name = model,
                            reasoning_effort = reasoning_effort,
                            temperature = temperature)

    def resize_images(self,
                      image_path: list[str],
                      max_size: int,
                      interpolation=cv2.INTER_AREA):
        PATH = os.path.dirname(os.path.realpath(__file__))
        new_img_path = os.path.join(PATH, "temp_images")

        if not os.path.exists(new_img_path):
            os.makedirs(new_img_path)

        print(image_path)
        for i in range(len(image_path)):
            img = cv2.imread(image_path[i], cv2.IMREAD_COLOR)
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

            cv2.imwrite(new_img_path + f"\\image_{i}.png", resized)

        return new_img_path

    
    def __call__(self, query_imgs):

        image_paths = sorted(glob.glob(query_imgs + "\\*"))[:self.k]
        print("IMAGE PATH:", image_paths)

        image_paths = sorted(glob.glob(self.resize_images(image_paths, max_size=256) + "\\*"))

        output = self.analyzer("Analyze the set of input images.", image_paths=image_paths)

        return output