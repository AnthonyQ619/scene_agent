from sfmcore.DataTypes.sceneDT import Scene
from sfmcore.DataTypes.cameraDT import CameraData
from sfmcore.DataTypes.cameraposeDT import CameraPose
from .baseclass import VisualizeClass
import matplotlib as plt
plt.use('Agg')
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import matplotlib.pyplot as plt
import cv2
import pytransform3d.transformations as pt
import pytransform3d.camera as pc
import pytransform3d.visualizer as pv
import os
import plotly.graph_objects as go
from pathlib import Path


class VisualizeScene(VisualizeClass):
    def __init__(self, server:bool = False, img_path:str = "output.png"):
        self.module_name = "VisualizeScene"
        self.description = ""
        self.example = ""

        # Current Visualizer only handles discrete types of data
        self.FORMATS = ["point cloud", "Mesh"]
        
    def __call__(
        self,
        points3d,
        output_path="recon_points.html",
        frustum_scale=0.2,
        direction_scale=0.35,
        image_aspect=4 / 3,
        show_forward=True,
        max_points=100_000,
        title="Camera Poses",
    ):
        output_path = Path(output_path)
        fig = go.Figure()

        points3d = np.asarray(points3d, dtype=np.float64)

        if points3d.ndim != 2 or points3d.shape[1] != 3:
            raise ValueError(
                f"Expected points3d with shape [M, 3], got {points3d.shape}"
            )

        valid = np.isfinite(points3d).all(axis=1)
        points3d = points3d[valid]

        if len(points3d) > max_points:
            idx = np.random.choice(len(points3d), max_points, replace=False)
            points3d = points3d[idx]

        fig.add_trace(
            go.Scatter3d(
                x=points3d[:, 0],
                y=points3d[:, 1],
                z=points3d[:, 2],
                mode="markers",
                name="Sparse 3D points",
                marker=dict(size=1, opacity=0.6),
            )
        )

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="data",
                zaxis=dict(autorange="reversed")
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            legend=dict(itemsizing="constant"),
        )

        fig.write_html(str(output_path), include_plotlyjs="cdn")
        print(f"Saved interactive visualization to: {output_path.resolve()}")

        return fig


class VisualizePose(VisualizeClass):
    def __init__(self):
        self.module_name = "VisualizePose"
        self.description = ""
        self.example = ""
    
    @staticmethod
    def invert_pose(T):
        """
        Invert a 4x4 SE(3) pose matrix.

        If T is T_cw, returns T_wc.
        If T is T_wc, returns T_cw.
        """
        T = np.asarray(T, dtype=np.float64)
        R = T[:3, :3]
        t = T[:3, 3]

        T_inv = np.eye(4, dtype=np.float64)
        T_inv[:3, :3] = R.T
        T_inv[:3, 3] = -R.T @ t
        return T_inv

    @staticmethod
    def pose_to_camera_center(T_wc):
        """
        Extract camera center from camera-to-world pose.
        """
        return T_wc[:3, 3]

    @staticmethod
    def make_camera_frustum_points(
        T_wc,
        scale=0.2,
        image_aspect=4 / 3,
        fov_scale=0.75,
    ):
        """
        Create 5 frustum points in world coordinates:
            0: camera center
            1-4: image plane corners

        Assumes a simple canonical camera looking along +Z in camera coordinates.

        Parameters
        ----------
        T_wc : np.ndarray
            4x4 camera-to-world pose.
        scale : float
            Overall frustum size.
        image_aspect : float
            Width / height ratio.
        fov_scale : float
            Controls image plane size relative to depth.
        """
        w = scale * image_aspect * fov_scale
        h = scale * fov_scale
        z = scale

        points_cam = np.array(
            [
                [0.0, 0.0, 0.0],   # camera center
                [-w, -h, z],       # bottom-left
                [w, -h, z],        # bottom-right
                [w, h, z],         # top-right
                [-w, h, z],        # top-left
            ],
            dtype=np.float64,
        )

        points_cam_h = np.concatenate(
            [points_cam, np.ones((points_cam.shape[0], 1))],
            axis=1,
        )

        points_world = (T_wc @ points_cam_h.T).T[:, :3]
        return points_world

    @staticmethod
    def add_camera_frustum_trace(
        fig,
        frustum_points,
        name="camera",
        showlegend=False,
        line_width=3,
    ):
        """
        Add a wireframe camera frustum to a Plotly figure.
        """
        c = frustum_points[0]
        p1, p2, p3, p4 = frustum_points[1:5]

        # Lines:
        # center to corners + image plane rectangle
        lines = [
            (c, p1),
            (c, p2),
            (c, p3),
            (c, p4),
            (p1, p2),
            (p2, p3),
            (p3, p4),
            (p4, p1),
        ]

        x, y, z = [], [], []
        for a, b in lines:
            x.extend([a[0], b[0], None])
            y.extend([a[1], b[1], None])
            z.extend([a[2], b[2], None])

        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                name=name,
                line=dict(width=line_width),
                showlegend=showlegend,
            )
        )

    @staticmethod
    def add_camera_forward_trace(
        fig,
        T_wc,
        scale=0.35,
        name="camera direction",
        showlegend=False,
    ):
        """
        Add a line showing the camera forward direction.

        This assumes camera forward is +Z in camera coordinates.
        If your convention uses -Z, change forward_cam to [0, 0, -1].
        """
        center = T_wc[:3, 3]
        forward_cam = np.array([0.0, 0.0, 1.0])
        forward_world = T_wc[:3, :3] @ forward_cam
        endpoint = center + scale * forward_world

        fig.add_trace(
            go.Scatter3d(
                x=[center[0], endpoint[0]],
                y=[center[1], endpoint[1]],
                z=[center[2], endpoint[2]],
                mode="lines",
                name=name,
                line=dict(width=5),
                showlegend=showlegend,
            )
        )

    @staticmethod
    def prep_w2c_poses(poses: CameraPose):
        w2c_poses = poses.camera_pose
        new_poses = []

        for pose in w2c_poses:
            new_pose = np.vstack((pose, np.asarray([[0,0,0,1]], dtype=np.float64)))
            new_poses.append(new_pose)

        return np.asarray(new_poses, dtype=np.float64)

    def __call__(
        self,
        poses,
        points3d=None,
        pose_type="T_wc",
        image_names=None,
        output_path="camera_poses.html",
        frustum_scale=0.2,
        direction_scale=0.35,
        image_aspect=4 / 3,
        show_forward=True,
        max_points=100_000,
        title="Camera Poses",
    ):
        """
        Visualize camera poses and optional sparse 3D points using Plotly.

        Parameters
        ----------
        poses : list[np.ndarray] or np.ndarray
            Camera poses. Shape can be:
                - list of 4x4 arrays
                - array of shape [N, 4, 4]

        points3d : np.ndarray, optional
            Sparse 3D points with shape [M, 3].

        pose_type : str
            Either:
                - "T_wc": camera-to-world pose
                - "T_cw": world-to-camera pose

            For OpenCV-style extrinsics [R|t] that map X_world -> X_cam,
            use pose_type="T_cw".

        image_names : list[str], optional
            Optional labels for each camera.

        output_path : str
            Path to save the interactive HTML file.

        frustum_scale : float
            Size of camera frustums.

        direction_scale : float
            Length of camera forward direction ray.

        image_aspect : float
            Width / height ratio for frustum shape.

        show_forward : bool
            Whether to draw camera viewing direction rays.

        max_points : int
            Maximum number of 3D points to plot.

        title : str
            Plot title.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            The Plotly figure object.
        """
        output_path = Path(output_path)
        
        # print(poses.shape)
        # poses = np.asarray(poses, dtype=np.float64)
        poses = self.prep_w2c_poses(poses)
        print(poses.shape)
        

        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError(
                f"Expected poses with shape [N, 4, 4], got {poses.shape}"
            )

        if pose_type not in {"T_wc", "T_cw"}:
            raise ValueError("pose_type must be either 'T_wc' or 'T_cw'")

        if image_names is not None and len(image_names) != len(poses):
            raise ValueError("image_names must have the same length as poses")

        fig = go.Figure()

        camera_centers = []

        for i, T in enumerate(poses):
            if pose_type == "T_cw":
                T_wc = self.invert_pose(T)
            else:
                T_wc = T

            center = self.pose_to_camera_center(T_wc)
            camera_centers.append(center)

            label = image_names[i] if image_names is not None else f"cam_{i:04d}"

            frustum_points = self.make_camera_frustum_points(
                T_wc,
                scale=frustum_scale,
                image_aspect=image_aspect,
            )

            self.add_camera_frustum_trace(
                fig,
                frustum_points,
                name=label,
                showlegend=False,
            )

            if show_forward:
                self.add_camera_forward_trace(
                    fig,
                    T_wc,
                    scale=direction_scale,
                    name=f"{label}_forward",
                    showlegend=False,
                )

        camera_centers = np.asarray(camera_centers)

        # Add camera centers as points with labels.
        fig.add_trace(
            go.Scatter3d(
                x=camera_centers[:, 0],
                y=camera_centers[:, 1],
                z=camera_centers[:, 2],
                mode="markers+text",
                text=image_names if image_names is not None else [str(i) for i in range(len(poses))],
                textposition="top center",
                name="Camera centers",
                marker=dict(size=4),
            )
        )

        # Add trajectory line connecting camera centers.
        fig.add_trace(
            go.Scatter3d(
                x=camera_centers[:, 0],
                y=camera_centers[:, 1],
                z=camera_centers[:, 2],
                mode="lines",
                name="Camera trajectory",
                line=dict(width=4),
            )
        )

        # Optional sparse points.
        if points3d is not None:
            points3d = np.asarray(points3d, dtype=np.float64)

            if points3d.ndim != 2 or points3d.shape[1] != 3:
                raise ValueError(
                    f"Expected points3d with shape [M, 3], got {points3d.shape}"
                )

            valid = np.isfinite(points3d).all(axis=1)
            points3d = points3d[valid]

            if len(points3d) > max_points:
                idx = np.random.choice(len(points3d), max_points, replace=False)
                points3d = points3d[idx]

            fig.add_trace(
                go.Scatter3d(
                    x=points3d[:, 0],
                    y=points3d[:, 1],
                    z=points3d[:, 2],
                    mode="markers",
                    name="Sparse 3D points",
                    marker=dict(size=1, opacity=0.6),
                )
            )

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="data",
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            legend=dict(itemsizing="constant"),
        )

        fig.write_html(str(output_path), include_plotlyjs="cdn")
        print(f"Saved interactive visualization to: {output_path.resolve()}")

        return fig


# def visualize_3d_points(
#     points3d,
#     output_path="recon_points.html",
#     frustum_scale=0.2,
#     direction_scale=0.35,
#     image_aspect=4 / 3,
#     show_forward=True,
#     max_points=100_000,
#     title="Camera Poses",
# ):
#     output_path = Path(output_path)
#     fig = go.Figure()

#     points3d = np.asarray(points3d, dtype=np.float64)

#     if points3d.ndim != 2 or points3d.shape[1] != 3:
#         raise ValueError(
#             f"Expected points3d with shape [M, 3], got {points3d.shape}"
#         )

#     valid = np.isfinite(points3d).all(axis=1)
#     points3d = points3d[valid]

#     if len(points3d) > max_points:
#         idx = np.random.choice(len(points3d), max_points, replace=False)
#         points3d = points3d[idx]

#     fig.add_trace(
#         go.Scatter3d(
#             x=points3d[:, 0],
#             y=points3d[:, 1],
#             z=points3d[:, 2],
#             mode="markers",
#             name="Sparse 3D points",
#             marker=dict(size=1, opacity=0.6),
#         )
#     )

#     fig.update_layout(
#         title=title,
#         scene=dict(
#             xaxis_title="X",
#             yaxis_title="Y",
#             zaxis_title="Z",
#             aspectmode="data",
#             zaxis=dict(autorange="reversed")
#         ),
#         margin=dict(l=0, r=0, b=0, t=40),
#         legend=dict(itemsizing="constant"),
#     )

#     fig.write_html(str(output_path), include_plotlyjs="cdn")
#     print(f"Saved interactive visualization to: {output_path.resolve()}")

#     return fig

# def visualize_camera_poses_plotly(
#     poses,
#     points3d=None,
#     pose_type="T_wc",
#     image_names=None,
#     output_path="camera_poses.html",
#     frustum_scale=0.2,
#     direction_scale=0.35,
#     image_aspect=4 / 3,
#     show_forward=True,
#     max_points=100_000,
#     title="Camera Poses",
# ):
#     """
#     Visualize camera poses and optional sparse 3D points using Plotly.

#     Parameters
#     ----------
#     poses : list[np.ndarray] or np.ndarray
#         Camera poses. Shape can be:
#             - list of 4x4 arrays
#             - array of shape [N, 4, 4]

#     points3d : np.ndarray, optional
#         Sparse 3D points with shape [M, 3].

#     pose_type : str
#         Either:
#             - "T_wc": camera-to-world pose
#             - "T_cw": world-to-camera pose

#         For OpenCV-style extrinsics [R|t] that map X_world -> X_cam,
#         use pose_type="T_cw".

#     image_names : list[str], optional
#         Optional labels for each camera.

#     output_path : str
#         Path to save the interactive HTML file.

#     frustum_scale : float
#         Size of camera frustums.

#     direction_scale : float
#         Length of camera forward direction ray.

#     image_aspect : float
#         Width / height ratio for frustum shape.

#     show_forward : bool
#         Whether to draw camera viewing direction rays.

#     max_points : int
#         Maximum number of 3D points to plot.

#     title : str
#         Plot title.

#     Returns
#     -------
#     fig : plotly.graph_objects.Figure
#         The Plotly figure object.
#     """
#     output_path = Path(output_path)
    
#     # print(poses.shape)
#     # poses = np.asarray(poses, dtype=np.float64)
#     poses = prep_w2c_poses(poses)
#     print(poses.shape)
    

#     if poses.ndim != 3 or poses.shape[1:] != (4, 4):
#         raise ValueError(
#             f"Expected poses with shape [N, 4, 4], got {poses.shape}"
#         )

#     if pose_type not in {"T_wc", "T_cw"}:
#         raise ValueError("pose_type must be either 'T_wc' or 'T_cw'")

#     if image_names is not None and len(image_names) != len(poses):
#         raise ValueError("image_names must have the same length as poses")

#     fig = go.Figure()

#     camera_centers = []

#     for i, T in enumerate(poses):
#         if pose_type == "T_cw":
#             T_wc = invert_pose(T)
#         else:
#             T_wc = T

#         center = pose_to_camera_center(T_wc)
#         camera_centers.append(center)

#         label = image_names[i] if image_names is not None else f"cam_{i:04d}"

#         frustum_points = make_camera_frustum_points(
#             T_wc,
#             scale=frustum_scale,
#             image_aspect=image_aspect,
#         )

#         add_camera_frustum_trace(
#             fig,
#             frustum_points,
#             name=label,
#             showlegend=False,
#         )

#         if show_forward:
#             add_camera_forward_trace(
#                 fig,
#                 T_wc,
#                 scale=direction_scale,
#                 name=f"{label}_forward",
#                 showlegend=False,
#             )

#     camera_centers = np.asarray(camera_centers)

#     # Add camera centers as points with labels.
#     fig.add_trace(
#         go.Scatter3d(
#             x=camera_centers[:, 0],
#             y=camera_centers[:, 1],
#             z=camera_centers[:, 2],
#             mode="markers+text",
#             text=image_names if image_names is not None else [str(i) for i in range(len(poses))],
#             textposition="top center",
#             name="Camera centers",
#             marker=dict(size=4),
#         )
#     )

#     # Add trajectory line connecting camera centers.
#     fig.add_trace(
#         go.Scatter3d(
#             x=camera_centers[:, 0],
#             y=camera_centers[:, 1],
#             z=camera_centers[:, 2],
#             mode="lines",
#             name="Camera trajectory",
#             line=dict(width=4),
#         )
#     )

#     # Optional sparse points.
#     if points3d is not None:
#         points3d = np.asarray(points3d, dtype=np.float64)

#         if points3d.ndim != 2 or points3d.shape[1] != 3:
#             raise ValueError(
#                 f"Expected points3d with shape [M, 3], got {points3d.shape}"
#             )

#         valid = np.isfinite(points3d).all(axis=1)
#         points3d = points3d[valid]

#         if len(points3d) > max_points:
#             idx = np.random.choice(len(points3d), max_points, replace=False)
#             points3d = points3d[idx]

#         fig.add_trace(
#             go.Scatter3d(
#                 x=points3d[:, 0],
#                 y=points3d[:, 1],
#                 z=points3d[:, 2],
#                 mode="markers",
#                 name="Sparse 3D points",
#                 marker=dict(size=1, opacity=0.6),
#             )
#         )

#     fig.update_layout(
#         title=title,
#         scene=dict(
#             xaxis_title="X",
#             yaxis_title="Y",
#             zaxis_title="Z",
#             aspectmode="data",
#         ),
#         margin=dict(l=0, r=0, b=0, t=40),
#         legend=dict(itemsizing="constant"),
#     )

#     fig.write_html(str(output_path), include_plotlyjs="cdn")
#     print(f"Saved interactive visualization to: {output_path.resolve()}")

#     return fig