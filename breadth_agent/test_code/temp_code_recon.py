# Here's a complete code implementation that addresses the sub-problems outlined in your dictionary. This code will load images, perform feature detection, match features, estimate camera poses, reject outliers, triangulate 3D points, perform bundle adjustment, and visualize the results.

# ```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import normalize

# Load Images and Calibration Parameters
def load_images_and_calibration(image_paths, calibration_params):
    images = [cv2.imread(img_path) for img_path in image_paths]
    K = calibration_params['intrinsic']  # Camera intrinsic matrix
    return images, K

# Feature Detection and Description
def detect_and_describe_features(images):
    orb = cv2.ORB_create()
    keypoints = []
    descriptors = []
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        keypoints.append(kp)
        descriptors.append(des)
    return keypoints, descriptors

# Match Features Between Images
def match_features(descriptors):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = []
    for i in range(len(descriptors) - 1):
        match = bf.match(descriptors[i], descriptors[i + 1])
        matches.append(match)
    return matches

# Estimate Relative Camera Poses
def estimate_camera_poses(matches, keypoints, K):
    poses = []
    for i, match in enumerate(matches):
        src_pts = np.float32([keypoints[i][m.queryIdx].pt for m in match]).reshape(-1, 2)
        dst_pts = np.float32([keypoints[i + 1][m.trainIdx].pt for m in match]).reshape(-1, 2)
        E, mask = cv2.findEssentialMat(src_pts, dst_pts, K)
        _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts, K)
        poses.append((R, t))
    return poses

# Outlier Rejection with RANSAC
def ransac_outlier_rejection(matches, keypoints, K):
    inliers = []
    for i, match in enumerate(matches):
        src_pts = np.float32([keypoints[i][m.queryIdx].pt for m in match]).reshape(-1, 2)
        dst_pts = np.float32([keypoints[i + 1][m.trainIdx].pt for m in match]).reshape(-1, 2)
        E, mask = cv2.findEssentialMat(src_pts, dst_pts, K, method=cv2.RANSAC, threshold=1.0)
        inliers.append(mask.ravel().tolist())
    return inliers

# Triangulate 3D Points
def triangulate_points(poses, inliers, keypoints, K):
    points_3D = []
    for i, (R, t) in enumerate(poses):
        if i == 0:
            continue
        src_pts = np.float32([keypoints[i - 1][m.queryIdx].pt for m in matches[i - 1] if inliers[i - 1][m.queryIdx]]).reshape(-1, 2)
        dst_pts = np.float32([keypoints[i][m.trainIdx].pt for m in matches[i - 1] if inliers[i - 1][m.queryIdx]]).reshape(-1, 2)
        points_4D = cv2.triangulatePoints(np.hstack((np.eye(3), np.zeros((3, 1)))), np.hstack((R, t)), src_pts.T, dst_pts.T)
        points_3D.append(points_4D[:3] / points_4D[3])  # Convert to 3D points
    return np.hstack(points_3D)

# Bundle Adjustment (simplified)
def bundle_adjustment(points_3D, poses):
    # This is a placeholder for a more complex optimization routine
    return points_3D  # In a real implementation, you would optimize here

# Visualize Sparse 3D Points
def visualize_3D_points(points_3D):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3D[0], points_3D[1], points_3D[2])
    plt.show()

# Example usage
image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']  # Replace with your image paths
calibration_params = {
    'intrinsic': np.array([[2393.952166119461, -3.410605131648481e-13, 932.3821770809047], 
                           [0, 2398.118540286656, 628.2649953288065], 
                           [0, 0, 1]])  # Replace with actual values
}

images, K = load_images_and_calibration(image_paths, calibration_params)
keypoints, descriptors = detect_and_describe_features(images)
matches = match_features(descriptors)
poses = estimate_camera_poses(matches, keypoints, K)
inliers = ransac_outlier_rejection(matches, keypoints, K)
points_3D = triangulate_points(poses, inliers, keypoints, K)
points_3D = bundle_adjustment(points_3D, poses)
visualize_3D_points(points_3D)
# ```

# ### Description of Each Step:
# 1. **Load Images and Calibration Parameters**: Reads images and camera calibration parameters.
# 2. **Feature Detection and Description**: Uses ORB to detect keypoints and compute descriptors.
# 3. **Match Features Between Images**: Matches features between consecutive images using BFMatcher.
# 4. **Estimate Relative Camera Poses**: Estimates camera poses using essential matrix decomposition.
# 5. **Outlier Rejection with RANSAC**: Filters out outliers from matches using RANSAC.
# 6. **Triangulate 3D Points**: Triangulates 3D points from inlier correspondences.
# 7. **Bundle Adjustment**: Placeholder for optimizing 3D points and camera poses.
# 8. **Visualize Sparse 3D Points**: Visualizes the resulting 3D points in a 3D space.

# ### Note:
# - Replace `fx`, `fy`, `cx`, and `cy` in the intrinsic matrix with actual camera parameters.
# - The bundle adjustment step is simplified and would typically require a more complex optimization routine.