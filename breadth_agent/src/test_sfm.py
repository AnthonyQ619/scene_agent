from modules.geometry import GeometryComposition, GeometryProcessing, GeometrySceneEstimate, CameraPoseEstimator
from modules.features import FeatureDetection, FeatureMatching
from modules.utilities.utilities import CalibrationReader
# Visualizer
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\datasets\\Structure-from-Motion\\sfm_dataset"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\Structure-from-Motion\\calibration.npz"

calibration_data = CalibrationReader(calibration_path).get_calibration()
feature_detector = FeatureDetection(image_path)
feature_matcher = FeatureMatching("flann", type = 'partial', detector="sift")

geom_processing = GeometryProcessing(calibration=calibration_data)
pose_estimator = CameraPoseEstimator(calibration_data)
geom_composition = GeometryComposition(image_path)
scene_estimator = GeometrySceneEstimate(calibration_data)


detected_features = feature_detector()

matched_features = feature_matcher(detected_features) # Maybe include sparse vs. dense feature detection here

for i in range(len(matched_features)):
    # print(matched_features[i])
    print(str(matched_features[i][0].points2D.shape) + " & " + str(matched_features[i][1].points2D.shape))

essential_matrices = geom_processing("essential", "full", matched_features)

camera_poses = pose_estimator(option="essential", points=matched_features, e_mat=essential_matrices) # Currently outputs no list[np.ndarray] (No custom typing) -> Originally had CameraPose Datatype, but doesn't work with homograhpy being included in compose module

composed_cam_poses = geom_composition("pose", camera_poses)

scene = scene_estimator(points=matched_features, camera_poses=composed_cam_poses)

print(scene.points3D.points3D.shape)

points = scene.points3D.points3D

# Create the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='blue')

# Set axis labels
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# Show the plot
plt.show()