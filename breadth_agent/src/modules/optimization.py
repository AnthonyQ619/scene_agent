import logging
import os
import pathlib
import random
import time
from typing import Dict, List, Type

import hydra
import omegaconf
import numpy as np
import cv2
import torch

import theseus as th
import theseus.utils.examples as theg

from modules.DataTypes.datatype import Scene, CameraData, BundleAdjustmentData, Points2D, Points3D, CameraPose, Calibration
from modules.baseclass import OptimizationClass


class BundleAdjustmentOptimizer(OptimizationClass):
    def __init__(self, 
                 scene: Scene | None = None,
                 cam_data: CameraData | None = None,
                 ):
        # super.__init__(cam_data = cam_data)

        self.module_name = "BundleAdjustmentOptimizer"
        self.description = f"""
Global Optimization tool using the non-linear least square optimization algorithm to optimize the reconstructed sparse 
scene using the 3D estimated points, 2D feature tracks for each 3D estimated point, and estimated camera poses of the 
monocular camera scene to optimize the reprojection error loss of each estimated 3D point. This tool applies the algorithm
Bundle Adjustment to globally optimize the scene. The output is a newly optimize sparse 3D reconstructed scene. The algorithm
optimizes the 3D points, camera poses, and distortion parameters of the calibrated camera. This algorithm keeps the focal length
of the original calibration

Initialization Parameters:
- calibration: Data type that stores the camera's calibration data initialized from the calibration 
reader module
    - default (Calibration): None - Include in initialization for usage
- scene: Data type that stores the camera pose and 3D point information of the reconstructed scene.
    - default (Scene): None - Include in initialization for usage

Function Calls:

- Function: prep_optimizer
    - Parameters:
        - max_iterations: Maximum number of iterations to run the bundle adjustment algorithm (larger number 
        guarantees convergence -- between 40 and 50 iterations)
            - Default (int) = 10,
        - ratio_known_cameras: Used if ground truth cameras are known prior, which can be used to optimize the reconstructed scen
        in training cases. Input should be percentage of known cameras (input < 1.0)
            - Default (float) = 0.1
        - optimizer_cls: Optimzer class utilized in the bundle adjustment algorithm. Gauss Newton is default due to 
        utilizing the non-linear least square optimization algorithm.
            - Default (str) = "GaussNewton"

- Function: Module call (Python __call__ function)
    - Parameters:
        -ba_file: Path to BAL file generated from SceneReconstruction module that stores the data from the reconstructed
        scene for bundle adjustment application.
            - Default (str): Not Applicable. Needs input to run.
"""

        self.example = f"""
Initialization: 
optimizer = BundleAdjustmentOptimizer(scene=sparse_scene, calibration=calibration_data)
optimizer.prep_optimizer(ratio_known_cameras=0.0)

Function Call: 
optimal_scene = optimizer(bal_path)
"""
        # Logger
        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger(__name__)

        # Collect Home Path
        file_path = os.path.realpath(__file__).split('\\')
        file_path[0] = file_path[0] + "\\"
        # Fixed Paths for Data
        self.result_path = os.path.join(*file_path[:7], "results", "scene_data")
        self.config_path = os.path.join(*file_path[:7], "results", "opt_configs")
        self.config_name = "config.yaml"

        # Active Scene Data for manipulation
        self.scene = scene
        # self.cal = calibration
        # self.bal = scene.bal_data # Bundle Adjustment Data for later use

        # Get Device
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Optimizer Setup Function. This function is for hyper parameter tuning of the optimizer
    def prep_optimizer(self, 
                        max_iterations: int = 10, 
                        step_size: float = 0.1, 
                        ratio_known_cameras: float = 0.1,
                        learning_rate: float = 0.1,
                        num_epochs: int = 20,
                        optimizer_cls: str = "GaussNewton") -> None:
        
        config_settings = {"inner_optim": {
                                "optimizer_cls": optimizer_cls,
                                "max_iters": max_iterations,
                                "step_size": step_size,
                                "backward_mode": "implicit",
                                "verbose": True,
                                "track_err_history": True,
                                "keep_step_size": True,
                                "regularize": True,
                                "ratio_known_cameras": ratio_known_cameras,
                                "reg_w": 1e-4},
                            "outer_optim": {
                                "lr": learning_rate,
                                "num_epochs": num_epochs},
                            "hydra": {
                                    "run":{
                                        "dir": "examples/outputs"}}}
        
        self.cfg = omegaconf.OmegaConf.create(config_settings)

    # Copyright (c) Meta Platforms, Inc. and affiliates.
    #
    # This source code is licensed under the MIT license found in the
    # LICENSE file in the root directory facebookresearch/theseus
    # Credit belongs to Facebook Research Team with Theseus library Examples:Bundle Adjustment

    ### HELPER FUNCTIONS ###
    
    def print_histogram(self,
        ba: theg.BundleAdjustmentDataset, var_dict: Dict[str, torch.Tensor], msg: str
    ):
        # print(ba.observations)
        self.log.info(msg)
        histogram = theg.ba_histogram(
            cameras=[
                theg.Camera(
                    th.SE3(tensor=var_dict[c.pose.name]),
                    c.focal_length,
                    c.calib_k1,
                    c.calib_k2,
                )
                for c in ba.cameras
            ],
            points=[th.Point3(tensor=var_dict[pt.name]) for pt in ba.points],
            observations=ba.observations,
        )
        for line in histogram.split("\n"):
            self.log.info(line)


    # loads (the only) batch
    def get_batch(self,
                ba: theg.BundleAdjustmentDataset,
                orig_poses: Dict[str, torch.Tensor],
                orig_points: Dict[str, torch.Tensor],
                ) -> Dict[str, torch.Tensor]:
        
        retv = {}
        for cam in ba.cameras:
            retv[cam.pose.name] = orig_poses[cam.pose.name].clone()
        for pt in ba.points:
            retv[pt.name] = orig_points[pt.name].clone()
        return retv

    def save_epoch(self,
                results_path: pathlib.Path,
                epoch: int,
                log_loss_radius: th.Vector,
                theseus_outputs: Dict[str, torch.Tensor],
                info: th.optimizer.OptimizerInfo,
                loss_value: float,
                total_time: float):
        
        def _clone(t_):
            return t_.detach().cpu().clone()

        results = {
            "log_loss_radius": _clone(log_loss_radius.tensor),
            "theseus_outputs": dict((s, _clone(t)) for s, t in theseus_outputs.items()),
            "err_history": info.err_history,  # type: ignore
            "loss": loss_value,
            "total_time": total_time,
        }
        torch.save(results, results_path / f"results_epoch{epoch}.pt")


    # def camera_loss(self,
    #     ba: theg.BundleAdjustmentDataset, camera_pose_vars: List[th.LieGroup]
    # ) -> torch.Tensor:
        
    #     loss: torch.Tensor = 0  # type:ignore
    #     for i in range(len(ba.cameras)):
    #         camera_loss = th.local(camera_pose_vars[i], ba.gt_cameras[i].pose).norm(dim=1)
    #         loss += camera_loss
    #     return loss


    def reprojection_loss(self,
                          ba: theg.BundleAdjustmentDataset,
                          theseus_outputs: Dict[str, torch.Tensor]
                          ) -> torch.Tensor:
        
        reproj_errors = []

        for obs in ba.observations:
            cam = ba.cameras[obs.camera_index]

            # Retrieve optimized pose and point variables by name
            # cam_pose_tensor = theseus_outputs[cam.pose.name]  # shape (6,) or (batch,6)
            # print(cam_pose_tensor)
            # print(cam.pose)
            # world_point_tensor = theseus_outputs[ba.points[obs.point_index].name]  # shape (3,) or (batch,3)
            # print(world_point_tensor)
            # print(ba.points[obs.point_index])

            # Project the 3D point using current optimized pose and intrinsics
            # error = th.eb.Reprojection(camera_pose = cam_pose_tensor, 
            #                              world_point = world_point_tensor,
            #                              image_feature_point = obs.image_feature_point,
            #                              focal_length = cam.focal_length,
            #                              calib_k1=cam.calib_k1,
            #                              calib_k2=cam.calib_k2).error()
            error = th.eb.Reprojection(camera_pose=cam.pose,
                                        world_point=ba.points[obs.point_index],
                                        focal_length=cam.focal_length,
                                        calib_k1=cam.calib_k1,
                                        calib_k2=cam.calib_k2,
                                        image_feature_point = obs.image_feature_point).error()

            # Compute Euclidean pixel error
            # error = (uv_pred - obs.image_feature_point).norm(dim=-1)  # if batch, else scalar
            # print(error)
            # print(error.norm(dim=-1))
            reproj_errors.append(error)

        errors_tensor = torch.stack(reproj_errors)  # shape (num_observations,) or (batch, num_obs)

        # print(errors_tensor.shape)
        return (errors_tensor ** 2).mean()
   

    ########################

    def __call__(self,
                 ba_file: str) -> Scene:
        
        # Get Result Path
        results_path = self._create_result_path("results/scene_data")

        # Read BAL dataset
        ba = theg.BundleAdjustmentDataset.load_from_file(ba_file)

        # hyper parameters (ie outer loop's parameters)
        log_loss_radius = th.Vector(1, name="log_loss_radius", dtype=torch.float64)

        # Set up objective
        objective = th.Objective(dtype=torch.float64)

        weight = th.ScaleCostWeight(torch.tensor(1.0).to(dtype=ba.cameras[0].pose.dtype, device=self.device))
        for obs in ba.observations:
            cam = ba.cameras[obs.camera_index]
            cost_function = th.eb.Reprojection(
                camera_pose=cam.pose,
                world_point=ba.points[obs.point_index],
                focal_length=cam.focal_length,
                calib_k1=cam.calib_k1,
                calib_k2=cam.calib_k2,
                image_feature_point=obs.image_feature_point,
                weight=weight,
            )
            robust_cost_function = th.RobustCostFunction(
                cost_function,
                th.HuberLoss,
                log_loss_radius,
                name=f"robust_{cost_function.name}",
            )
            objective.add(robust_cost_function)
        dtype = objective.dtype

        # Add regularization
        if self.cfg.inner_optim.regularize:
            zero_point3 = th.Point3(dtype=dtype, name="zero_point")
            identity_se3 = th.SE3(dtype=dtype, name="zero_se3")
            w = np.sqrt(self.cfg.inner_optim.reg_w)
            damping_weight = th.ScaleCostWeight(w * torch.ones(1, dtype=dtype, device=self.device))
            for name, var in objective.optim_vars.items():
                target: th.Manifold
                if isinstance(var, th.SE3):
                    target = identity_se3
                elif isinstance(var, th.Point3):
                    target = zero_point3
                else:
                    assert False
                objective.add(
                    th.Difference(var, target, damping_weight, name=f"reg_{name}")
                )

        # camera_pose_vars: List[th.LieGroup] = [
        #     objective.optim_vars[c.pose.name] for c in ba.cameras  # type: ignore
        # ]
        # if self.cfg.inner_optim.ratio_known_cameras > 0.0:
        #     w = 100.0
        #     camera_weight = th.ScaleCostWeight(100 * torch.ones(1, dtype=dtype))
        #     for i in range(len(ba.cameras)):
        #         if np.random.rand() > self.cfg.inner_optim.ratio_known_cameras:
        #             continue
        #         objective.add(
        #             th.Difference(
        #                 camera_pose_vars[i],
        #                 ba.gt_cameras[i].pose,
        #                 camera_weight,
        #                 name=f"camera_diff_{i}",
        #             )
        #         )

        # Create optimizer
        optimizer_cls: Type[th.NonlinearLeastSquares] = getattr(
            th, self.cfg.inner_optim.optimizer_cls
        )
        optimizer = optimizer_cls(
            objective,
            max_iterations=self.cfg.inner_optim.max_iters,
            step_size=self.cfg.inner_optim.step_size,
        )

        # Set up Theseus layer
        theseus_optim = th.TheseusLayer(optimizer)
        theseus_optim = theseus_optim.to(device=self.device)

        # copy the poses/pts to feed them to each outer iteration
        orig_poses = {cam.pose.name: cam.pose.tensor.clone() for cam in ba.cameras}
        orig_points = {pt.name: pt.tensor.clone() for pt in ba.points}

        # Outer optimization loop
        loss_radius_tensor = torch.nn.Parameter(torch.tensor([3.0], dtype=torch.float64, device=self.device))
        model_optimizer = torch.optim.Adam([loss_radius_tensor], lr=self.cfg.outer_optim.lr)

        num_epochs = self.cfg.outer_optim.num_epochs

        theseus_inputs = self.get_batch(ba, orig_poses, orig_points)
        theseus_inputs["log_loss_radius"] = loss_radius_tensor.unsqueeze(1).clone()

        # with torch.no_grad():
        #     #camera_loss_ref = self.camera_loss(ba, camera_pose_vars).item()
        #     reproj_loss = self.reprojection_loss(ba, theseus_inputs)
        # # self.log.info(f"CAMERA LOSS (no learning):  {camera_loss_ref: .3f}")
        # reproj_loss = objective.error_metric() 
        # reproj_loss = loss.sum()
        # self.log.info(f"Reprojection Loss (no learning):  {reproj_loss: .3f}")
        self.print_histogram(ba, theseus_inputs, "Input histogram:")

        for epoch in range(num_epochs):
            self.log.info(f" ******************* EPOCH {epoch} ******************* ")
            start_time = time.time_ns()
            model_optimizer.zero_grad()
            theseus_inputs["log_loss_radius"] = loss_radius_tensor.unsqueeze(1).clone()

            theseus_outputs, info = theseus_optim.forward(
                input_tensors=theseus_inputs,
                optimizer_kwargs={
                    "verbose": self.cfg.inner_optim.verbose,
                    "track_err_history": self.cfg.inner_optim.track_err_history,
                    "backward_mode": self.cfg.inner_optim.backward_mode,
                    "__keep_final_step_size__": self.cfg.inner_optim.keep_step_size,
                },
            )

            # print(theseus_outputs)
            # objective.update(theseus_outputs)
            loss = objective.error_metric(input_tensors=theseus_outputs) #(self.camera_loss(ba, camera_pose_vars) - camera_loss_ref) / camera_loss_ref
            loss = loss.sum()
            # loss = self.reprojection_loss(ba, theseus_outputs=theseus_outputs)
            # print(loss)
            # print(loss.device)
            # loss.backward()
            # print("AFTER BACK PROP")
            # model_optimizer.step()
            loss_value = torch.sum(loss.cpu().detach()).item()
            end_time = time.time_ns()
            
            self.print_histogram(ba, theseus_outputs, "Output histogram:")
            self.log.info(
                f"Epoch: {epoch} Loss: {loss_value} "
                f"Kernel Radius: exp({loss_radius_tensor.data.item()})="
                f"{torch.exp(loss_radius_tensor.data).item()}"
            )
            self.log.info(f"Epoch took {(end_time - start_time) / 1e9: .3f} seconds")

            self.save_epoch(
                results_path,
                epoch,
                log_loss_radius,
                theseus_outputs,
                info,
                loss_value,
                end_time - start_time,
            )

        poses = []
        # poses_track = set()
        points_3d = Points3D()

        # Rebuild Scene Datatype
        # for obs in ba.observations:
        #     cam = ba.cameras[obs.camera_index]

        #     cam_pose_tensor = theseus_outputs[cam.pose.name]  # shape (6,) or (batch,6)
        #     if cam.pose.name not in poses_track:
        #         print("Here")
        #         pose = cam_pose_tensor.cpu().detach().numpy()
        #         poses.append(pose)
        #         poses_track.add(cam.pose.name)
              
        #     world_point = theseus_outputs[ba.points[obs.point_index].name].cpu().detach().numpy()  # shape (3,) or (batch,3)
        #     points_3d.update_points(world_point)
        for key in theseus_outputs.keys():
            if 'Cam' in key:
                pose = theseus_outputs[key].cpu().detach().numpy()
                poses.append(pose)
            if 'Pt' in key:
                world_point = theseus_outputs[ba.points[obs.point_index].name].cpu().detach().numpy()  # shape (3,) or (batch,3)
                points_3d.update_points(world_point)
        
        # print(theseus_outputs)
        print(theseus_outputs.keys())
        print(len(poses))
        print(points_3d.points3D.shape)
        new_scene = Scene(points3D = points_3d,
                          cam_poses = poses,
                          representation = "point cloud")
        return new_scene
    
    def _create_result_path(self, result_dir: str) -> pathlib.Path:
        file_path = os.path.realpath(__file__).split('\\')
        file_path[0] = file_path[0] + "\\"
        home_dir = pathlib.Path(os.path.join(*file_path[:7])) # Sets Path to Breadth_agent
        result_path = home_dir / result_dir

        return result_path