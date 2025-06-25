from baseclass import VisualizeClass
from .DataTypes.datatype import Scene
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import matplotlib.pyplot as plt
import cv2

class VisualizeScene(VisualizeClass):
    def __init__(self):
        self.module_name = "VisualizeScene"
        self.description = ""
        self.example = ""

        # Current Visualizer only handles discrete types of data
        self.FORMATS = ["point cloud", "Mesh"]

        # Set up GUI for visualization
        gui.Application.instance.initialize()

        window = gui.Application.instance.create_window("Mesh-Viewer", 1024, 750)
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(window.renderer)
        window.add_child(self.scene)

    def __call__(self, data: Scene | np.ndarray, store:bool = False, path:str | None = None, format: str = "point cloud") -> None:
        if format.lower() == self.FORMATS[0]:
            pcd, bounds, mat = self.visualize_point_cloud(data)

        self.scene.scene.add_geometry("mesh_name2", pcd, mat)
        self.scene.scene.add_geometry("mesh_name3", o3d.geometry.TriangleMesh.create_coordinate_frame(), rendering.MaterialRecord())

        self.scene.setup_camera(60, bounds, bounds.get_center())

        gui.Application.instance.run()  # Run until user closes window

    def visualize_point_cloud(self, data: Scene) -> tuple:
        mat = rendering.MaterialRecord()
        mat.shader = 'defaultUnlit'
        mat.point_size = 7.0
        mat.base_color = np.ndarray(shape=(4,1), buffer=np.array([0.0, 0.0, 1.0, 1.0]), dtype=float)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data.points3D.points3D)

        bounds = pcd.get_axis_aligned_bounding_box()
        
        return pcd, bounds, mat
        
