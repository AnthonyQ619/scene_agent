from agents.code_generator import ChatGPT
import agents.code_generator
import os


open_ai_key = "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"


def script_extraction_helper_python(agent_response):
    code_response = agent_response.model_dump()['output'][0]['content'][0]['text']

    script_start = code_response.find('```python') # Find beginning of code in LaTeX response
    script_start += len("```python")
    script_end = code_response[script_start:].find('```') # Find the end of the python code

    script_end += script_start # Ensure Script End index is aligned with original text size

    script = code_response[script_start:script_end]

    return script

def script_extraction_helper_python_o3(agent_response):
    script_start = agent_response.find('#!/usr/bin/env python3') # Find beginning of code in LaTeX response
    script_start += len("#!/usr/bin/env python3")
    script_end = agent_response[script_start:].find('---') # Find the end of the python code

    script_end += script_start # Ensure Script End index is aligned with original text size

    script = agent_response[script_start:script_end-3]

    return script

header_sub = "You are an assisstant with focus on computer vision problems. Please breakdown the computer vision problems you see into sub-problems with solutions, with the combination of the sub-problem solutions solves the original problem."
closer_sub = """
Example: Given a video, write a python script to conduct object detection per frame, and output all the frames in which a 'Bike' exists in.

Response:

sub_problems = {
  'Parse Video for Inidividual Frames': 'In this step, we will read in the video provided in the file, utilize a video reader function to parse the video into image frames, and output each video frame as an image',

  'Use Object Detection to Detect 'Bike' in Image': 'In this step, load a pre-trained object detection model to identify known objects in images. Then, use the model to detect the object 'bike' in images provided to the model',

  'Keep Track of Images with Detected Object': 'In this step, we verify the output of the object detection model to see if it contains the object of interest, if so, we save the frame number, if not we continue to the next frame of the video'
}

Example: Given a pair of images, write a python script to find corresponding features in the image.

Response:

sub_problems = {
    'Load Images': 'In this step, we will read the two input images using an image processing library to prepare them for feature detection.',

    'Convert Images to Grayscale': 'In this step, we will convert both images to grayscale to simplify the feature detection process, as color information is not necessary for this task.',

    'Detect Keypoints and Descriptors': 'In this step, we will use a feature detection algorithm (e.g., SIFT, ORB) to identify keypoints in both images and compute their corresponding descriptors.',

    'Match Features Between Images': 'In this step, we will use a feature matching technique (e.g., FLANN or BFMatcher) to find correspondences between the descriptors of the two images.',

    'Filter Matches': 'In this step, we will apply a filtering technique (e.g., ratio test or distance threshold) to retain only the best matches and remove outliers.',

    'Outlier Rejection': 'In this step, we will apply and outlier rejection technique (e.g., RANSAC) to optimize correct point correspondences between images.',

    'Visualize Correspondences': 'In this step, we will draw the matched keypoints on both images to visualize the correspondences, helping to confirm the accuracy of the matches.'
}


Just write the sub-problems and their solutions. Remember to include outlier rejection for feature detection, such as RANSAC, or optimization algorithms in 3D settings. Do Not Write the Python Code!
"""

header_code = "You are an assisstant with focus on computer vision problems. Please write code for the computer vision problems you see that are now sub-problems with solutions and description, and solve the combination of the sub-problems into one solution."
closer_code = """
Example:

sub_problems = {
  'Parse Video for Inidividual Frames': 'In this step, we will read in the video provided in the file, utilize a video reader function to parse the video into image frames, and output each video frame as an image',

  'Use Object Detection to Detect 'Bike' in Image': 'In this step, load a pre-trained object detection model to identify known objects in images. Then, use the model to detect the object 'bike' in images provided to the model',

  'Keep Track of Images with Detected Object': 'In this step, we verify the output of the object detection model to see if it contains the object of interest, if so, we save the frame number, if not we continue to the next frame of the video'
}

Response:
import cv2
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import numpy as np
import os

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

# Define COCO class labels (Faster R-CNN is trained on COCO dataset)
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe',
    'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet',
    'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

BIKE_CLASS = 'bicycle'
BIKE_INDEX = COCO_CLASSES.index(BIKE_CLASS)
CONFIDENCE_THRESHOLD = 0.5  # Adjust confidence threshold as needed

# Define transformation
transform = T.Compose([
    T.ToPILImage(),
    T.ToTensor()
])

def detect_bike(frame):
    # Detects a bicycle and draws bounding boxes.
    img = transform(frame).unsqueeze(0).to(device)  # Add batch dimension and move to GPU
    with torch.no_grad():
        predictions = model(img)

    labels = predictions[0]['labels'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    boxes = predictions[0]['boxes'].cpu().numpy()

    for label, score, box in zip(labels, scores, boxes):
        if label == BIKE_INDEX and score > CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            return True, frame
    return False, frame

def process_video(video_path, output_folder, output_video):
    # Process video, save detected frames, and reconstruct video.
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    detected_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        detected, frame_with_box = detect_bike(frame)
        if detected:
            output_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(output_path, frame_with_box)
            detected_frames.append(frame_with_box)
            print(f"Saved: {output_path}")

    cap.release()
    print(f"Processing complete. {len(detected_frames)} frames saved.")

    # Reconstruct video
    if detected_frames:
        height, width, _ = detected_frames[0].shape
        out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 5, (width, height))
        for frame in detected_frames:
            out.write(frame)
        out.release()
        print(f"Video saved: {output_video}")

# Example usage
video_path = "input_video.mp4"
output_folder = "output_frames"
output_video = "output_video.mp4"
process_video(video_path, output_folder, output_video)

Just write the code according to the description of the sub-problems. The prompts given will be in a dictionary format of key and value. Where key is the sub-problem title, and the value is the description of how to solve it.
"""

def main():
    model_sub_problem = ChatGPT(open_ai_key, "gpt-4o-mini-2024-07-18",
                            system_header=header_sub,
                            system_closer=closer_sub,
                            k_ref=3)

    model_code_generator = ChatGPT(open_ai_key, "gpt-4o-mini-2024-07-18",
                            system_header=header_code,
                            system_closer=closer_code)

    prompt = """
Given a video of a camera moving across a scene, write a python script to construct a mosaic from the frames of the video.
"""

    response = model_sub_problem.generate_code(prompt)

    code_response = response.model_dump()['output'][0]['content'][0]['text']
    print(code_response)
    file_path_sol = "temp_code_mosaic_v2.txt"

    with open(file_path_sol, "w") as file:
        file.write(code_response)

    response = model_code_generator.generate_code(code_response)

    code_response = response.model_dump()['output'][0]['content'][0]['text']
    
    file_path = "temp_code_mosaic_v2.py"

    with open(file_path, "w") as file:
        file.write(code_response)

main()