from .DataTypes.datatype import Scene, CameraPose, CameraData
from .baseclass import VisualizeClass
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import matplotlib.pyplot as plt
import cv2
import pytransform3d.transformations as pt
import pytransform3d.camera as pc
import pytransform3d.visualizer as pv


class VisualizeScene(VisualizeClass):
    def __init__(self):
        self.module_name = "VisualizeScene"
        self.description = ""
        self.example = ""

        # Current Visualizer only handles discrete types of data
        self.FORMATS = ["point cloud", "Mesh"]

        # Set up GUI for visualization
        gui.Application.instance.initialize()

        self.window = gui.Application.instance.create_window("Mesh-Viewer", 1024, 750)
        self.scene = gui.SceneWidget()
        

    def __call__(self, data: Scene | np.ndarray, store:bool = False, path:str | None = None, format: str = "point cloud") -> None:
        data_np = data.points3D.points3D
        # data_np = data

        # TESTING
        # max_magnitude_row_index = np.argmax(np.linalg.norm(data_np, axis=1))
        # row_with_largest_magnitude = data_np[max_magnitude_row_index, :]
        # mask = np.isin(element = data.bal_data.observations[:, 1], test_elements=np.array([max_magnitude_row_index]))
        # desired_array = data.bal_data.observations[mask]
        # # data.bal_data.
        # print("OUTLIER POINT", row_with_largest_magnitude)
        # print("INLIER POINT", data_np[1, :])
        # print("2D Points (Normalized)", desired_array)
        # print("Point Index", max_magnitude_row_index)

        # DONE TESTING - TODO: REMOVE
        if format.lower() == self.FORMATS[0]:
            colors = data.points3D.color
            print(colors)
            pcd, bounds, mat = self.visualize_point_cloud(data_np, colors)

        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.scene)
        
        self.scene.scene.add_geometry("mesh_name2", pcd, mat)
        self.scene.scene.add_geometry("mesh_name3", o3d.geometry.TriangleMesh.create_coordinate_frame(), rendering.MaterialRecord())

        self.scene.setup_camera(60, bounds, bounds.get_center())

        gui.Application.instance.run()  # Run until user closes window

    def visualize_point_cloud(self, data: np.ndarray, color: np.ndarray | None) -> tuple:
        mat = rendering.MaterialRecord()
        mat.shader = 'defaultUnlit'
        mat.point_size = 7.0
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data)
        print("COLOR", color)
        if color is None:
            print("HERE")
            # Set a baseline color
            mat.base_color = np.ndarray(shape=(4,1), buffer=np.array([0.0, 0.0, 1.0, 1.0]), dtype=float)
        else:
            # Set color
            pcd.colors = o3d.utility.Vector3dVector(color)

        bounds = pcd.get_axis_aligned_bounding_box()
        
        return pcd, bounds, mat
        
    def visualize_pose(self, cam_poses: CameraPose):
        # import open3d as o3d
        # import open3d.visualization.gui as gui
        # import open3d.visualization.rendering as rendering
        # import numpy as np

        new_point_cloud = []
        for i in range(len(cam_poses.camera_pose)):
            new_point_cloud.append(cam_poses.camera_pose[i][:,3:])

        new_point_cloud = np.array(new_point_cloud).squeeze()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(new_point_cloud)

        gui.Application.instance.initialize()

        window = gui.Application.instance.create_window("Mesh-Viewer", 1024, 750)

        scene = gui.SceneWidget()
        scene.scene = rendering.Open3DScene(window.renderer)

        window.add_child(scene)

        matGT = rendering.MaterialRecord()
        matGT.shader = 'defaultUnlit'
        matGT.point_size = 7.0
        matGT.base_color = np.ndarray(shape=(4,1), buffer=np.array([0.0, 0.0, 1.0, 1.0]), dtype=float)

        scene.scene.add_geometry("mesh_name2", pcd, matGT)
        scene.scene.add_geometry("mesh_name3", o3d.geometry.TriangleMesh.create_coordinate_frame(), rendering.MaterialRecord())

        bounds = pcd.get_axis_aligned_bounding_box()
        scene.setup_camera(60, bounds, bounds.get_center())

        gui.Application.instance.run()  # Run until user closes window


class VisualizePose(VisualizeClass):
    def __init__(self):
        self.module_name = "VisualizePose"
        self.description = ""
        self.example = ""
        

    def __call__(self, pose_data: CameraPose | np.ndarray, camera_data: CameraData | None = None) -> None:
        if camera_data is None:
            sensor_size = np.array([0.036, 0.024])
            intrinsic_matrix = np.array(
                [
                    [0.05, 0, sensor_size[0] / 2.0],
                    [0, 0.05, sensor_size[1] / 2.0],
                    [0, 0, 1],
                ]
            )
        else:
            intrinsic_matrix = camera_data.get_K()
            if isinstance(intrinsic_matrix, list):
                intrinsic_matrix = intrinsic_matrix[0]

        if isinstance(pose_data, CameraPose):
            camera_poses = pose_data.camera_pose
        else:
            camera_poses = pose_data

        virtual_image_distance = 1

        transformation_matrices = np.empty((len(camera_poses), 4, 4))
        for i, camera_pose in enumerate(camera_poses):
            R = camera_pose[:, :3]
            p = camera_pose[:, 3:].ravel()
            # print(p.shape)
            transformation_matrices[i] = pt.transform_from(R=R, p=p)

        fig = pv.figure()
        for pose in transformation_matrices:
            fig.plot_transform(A2B=pose, s=0.2)
            fig.plot_camera(
                # ax,
                cam2world=pose,
                M=intrinsic_matrix,
                sensor_size=sensor_size,
                virtual_image_distance=virtual_image_distance,
            )
        # plt.show()
        fig.show()
