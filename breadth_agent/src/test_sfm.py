from modules.geometry import GeometryComposition, GeometryProcessing, GeometrySceneEstimate, CameraPoseEstimator
from modules.features import FeatureDetection, FeatureMatching
from modules.utilities.utilities import CalibrationReader
from modules.optimization import OutlierRejection
from modules.visualize import VisualizeScene
# Visualizer
# import open3d as o3d
# import open3d.visualization.gui as gui
# import open3d.visualization.rendering as rendering
# import numpy as np
# import matplotlib.pyplot as plt

# def temp_function_outlier_removal(points):
#     center_points = np.zeros((points.shape[0], 3))

#     diff = np.linalg.norm(center_points - points, axis=1)
#     diff[diff <= 50.0] = 1
#     diff[diff > 50.0] = 0

#     new_points = points[diff == 1]

#     return new_points



# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\datasets\\Structure-from-Motion\\sfm_dataset"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\Structure-from-Motion\\calibration.npz"

# Feature Module Initialization
calibration_data = CalibrationReader(calibration_path).get_calibration()
feature_detector = FeatureDetection(image_path, type = "sift")
feature_matcher = FeatureMatching("flann", type = 'full', detector="sift")
outlier_rejector = OutlierRejection(calibration=calibration_data)

# Geometry Module Initialization
geom_processing = GeometryProcessing(calibration=calibration_data)
pose_estimator = CameraPoseEstimator(calibration_data)
geom_composition = GeometryComposition(image_path)
scene_estimator = GeometrySceneEstimate(calibration_data)

# Visualization Module Initialization
visualizer = VisualizeScene()


# Solution Pipeline 
detected_features = feature_detector()

matched_features = feature_matcher(detected_features) # Maybe include sparse vs. dense feature detection here

inlier_features = outlier_rejector(option="fundamental", format="full", pts=matched_features) # -> check if there's diveristy, if not bunch it together with feature matcher

essential_matrices = geom_processing("essential", "full", inlier_features) 

camera_poses = pose_estimator(option="essential", points=inlier_features, e_mat=essential_matrices) # Currently outputs no list[np.ndarray] (No custom typing) -> Originally had CameraPose Datatype, but doesn't work with homograhpy being included in compose module
# Utilize Pose Estimator 

composed_cam_poses = geom_composition("pose", camera_poses)

scene = scene_estimator(points=inlier_features, camera_poses=composed_cam_poses)

visualizer(scene)


# new_point_cloud = []
# for i in range(len(composed_cam_poses)):
#     new_point_cloud.append(composed_cam_poses[i][:,3:])

# new_point_cloud = np.array(new_points).squeeze()
# print(new_point_cloud.shape)
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)

# gui.Application.instance.initialize()

# window = gui.Application.instance.create_window("Mesh-Viewer", 1024, 750)

# scene = gui.SceneWidget()
# scene.scene = rendering.Open3DScene(window.renderer)

# window.add_child(scene)

# matGT = rendering.MaterialRecord()
# matGT.shader = 'defaultUnlit'
# matGT.point_size = 7.0
# matGT.base_color = np.ndarray(shape=(4,1), buffer=np.array([0.0, 0.0, 1.0, 1.0]), dtype=float)

# scene.scene.add_geometry("mesh_name2", pcd, matGT)
# scene.scene.add_geometry("mesh_name3", o3d.geometry.TriangleMesh.create_coordinate_frame(), rendering.MaterialRecord())

# bounds = pcd.get_axis_aligned_bounding_box()
# scene.setup_camera(60, bounds, bounds.get_center())

# gui.Application.instance.run()  # Run until user closes window
