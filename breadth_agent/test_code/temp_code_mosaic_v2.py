# Here's a complete solution that addresses the sub-problems outlined in your dictionary. The code will read a video, process each frame, detect keypoints, match features, estimate homography, warp frames, and finally blend them to create a mosaic.

# ```python
import cv2
import numpy as np
import os

### Manually Included (Didn't have video data - only images)
def load_images_from_folder(folder_path):
    images = []
    for filename in sorted(os.listdir(folder_path)):
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is not None:
            images.append(img)
    return images
############################################################

def parse_video(video_path):
    """Parse Video into Individual Frames."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def preprocess_frames(frames, size=(640, 480)):
    """Preprocess Frames."""
    processed_frames = []
    for frame in frames:
        resized_frame = cv2.resize(frame, size)
        normalized_frame = cv2.normalize(resized_frame, None, 0, 255, cv2.NORM_MINMAX)
        processed_frames.append(normalized_frame)
    return processed_frames

def detect_keypoints_and_descriptors(frames):
    """Detect Keypoints and Descriptors in Frames."""
    orb = cv2.ORB_create()
    keypoints_descriptors = []
    for frame in frames:
        keypoints, descriptors = orb.detectAndCompute(frame, None)
        keypoints_descriptors.append((keypoints, descriptors))
    return keypoints_descriptors

def match_features(kd1, kd2):
    """Match Features Between Consecutive Frames."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(kd1[1], kd2[1])
    return matches

def filter_matches(matches):
    """Filter Matches."""
    # Sort matches based on distance
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def estimate_homography(kp1, kp2, matches):
    """Estimate Homography."""
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    return homography

def warp_frames(frames, homographies):
    """Warp Frames to a Common Perspective."""
    warped_frames = []
    for i, frame in enumerate(frames):
        if i == 0:
            warped_frames.append(frame)
        else:
            h = homographies[i-1]
            warped_frame = cv2.warpPerspective(frame, h, (frame.shape[1], frame.shape[0]))
            warped_frames.append(warped_frame)
    return warped_frames

def blend_frames(warped_frames):
    """Blend Frames to Construct Mosaic."""
    mosaic = warped_frames[0]
    for i in range(1, len(warped_frames)):
        mosaic = cv2.addWeighted(mosaic, 0.5, warped_frames[i], 0.5, 0)
    return mosaic

def post_process_mosaic(mosaic):
    """Post-process Mosaic."""
    # Example: Crop and sharpen the mosaic
    cropped_mosaic = mosaic[50:-50, 50:-50]  # Simple crop
    sharpened_mosaic = cv2.GaussianBlur(cropped_mosaic, (0, 0), 3)
    sharpened_mosaic = cv2.addWeighted(cropped_mosaic, 1.5, sharpened_mosaic, -0.5, 0)
    return sharpened_mosaic

def create_mosaic(video_path, output_path):
    frames = load_images_from_folder(video_path) #parse_video(video_path)
    processed_frames = preprocess_frames(frames)
    keypoints_descriptors = detect_keypoints_and_descriptors(processed_frames)

    homographies = []
    for i in range(len(keypoints_descriptors) - 1):
        matches = match_features(keypoints_descriptors[i], keypoints_descriptors[i + 1])
        filtered_matches = filter_matches(matches)
        homography = estimate_homography(keypoints_descriptors[i][0], keypoints_descriptors[i + 1][0], filtered_matches)
        homographies.append(homography)

    warped_frames = warp_frames(processed_frames, homographies)
    mosaic = blend_frames(warped_frames)
    final_mosaic = post_process_mosaic(mosaic)

    cv2.imwrite(output_path, final_mosaic)
    print(f"Mosaic saved to {output_path}")

# Example usage
video_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\prompt_review\\image_data\\mosaic-data"
output_path = "mosaic_output_3.jpg"
create_mosaic(video_path, output_path)
# ```

# ### Explanation of the Code:
# 1. **Parse Video into Individual Frames**: Reads the video and extracts each frame.
# 2. **Preprocess Frames**: Resizes and normalizes the frames for consistency.
# 3. **Detect Keypoints and Descriptors**: Uses ORB to find keypoints and compute descriptors for each frame.
# 4. **Match Features Between Consecutive Frames**: Matches features between consecutive frames using BFMatcher.
# 5. **Filter Matches**: Sorts matches based on distance to retain the best ones.
# 6. **Estimate Homography**: Calculates the homography matrix using RANSAC to relate the matched keypoints.
# 7. **Warp Frames to a Common Perspective**: Applies the homography to warp each frame.
# 8. **Blend Frames to Construct Mosaic**: Blends the warped frames to create a mosaic.
# 9. **Post-process Mosaic**: Applies final adjustments to enhance the visual quality of the mosaic.

# This code provides a complete pipeline for creating a mosaic from a video, addressing each sub-problem in the process.