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

import copy
import torch
import re

import theseus as th
import theseus.utils.examples as theg

from modules.DataTypes.datatype import Scene, CameraData, BundleAdjustmentData, Points2D, Points3D, CameraPose, Calibration
from modules.baseclass import OptimizationClass


class BundleAdjustmentOptimizerLeastSquares(OptimizationClass):
    def __init__(self, 
                 cam_data: CameraData,
                 max_iterations: int = 10, 
                 step_size: float = 0.1, 
                 learning_rate: float = 0.1,
                 num_epochs: int = 20,
                 optimizer_cls: str = "LevenbergMarquardt"
                 ):
        super().__init__(cam_data = cam_data)
        
        self.optimizer_choices = ["LevenbergMarquardt", "GaussNewton"]

        assert (optimizer_cls in self.optimizer_choices), "Must Choose Optimizer from supported classes: LevenbergMarquardt or GaussNewton"

        self.module_name = "BundleAdjustmentOptimizerLeastSquares"
        self.description = f"""
Global Optimization tool using the non-linear least square optimization algorithm to optimize the reconstructed sparse 
scene using the 3D estimated points, 2D feature tracks for each 3D estimated point, and estimated camera poses of the 
monocular camera scene to optimize the reprojection error loss of each estimated 3D point. This tool applies the algorithm
Bundle Adjustment to globally optimize the scene. The output is a newly optimize sparse 3D reconstructed scene. The algorithm
optimizes the 3D points, camera poses, and distortion parameters of the calibrated camera. This algorithm keeps the focal length
of the original calibration by normalizing the detected 2D feature points prior to optimization.

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
            - Default (float) = 0.0
        - optimizer_cls: Optimzer class utilized in the bundle adjustment algorithm. Levenberg-Marquardt is default due to 
        utilizing the non-linear least square optimization algorithm. Use Gauss Newton if initial data is less noisy and 
        can use a more aggressive optimizer, but it is less robust to outliers. 
            - Default (str) = "LevenbergMarquardt"
            - Options (str) ] ["LevenbergMarquardt", "GaussNewton"]

- Function: Module call (Python __call__ function)
    - Parameters:
        -ba_file: Path to BAL file generated from SceneReconstruction module that stores the data from the reconstructed
        scene for bundle adjustment application.
            - Default (str): Not Applicable. Needs input to run.
"""

        self.example = f"""
Initialization: 
optimizer = BundleAdjustmentOptimizerLeastSquares(scene=sparse_scene, 
                                                  cam_data=camera_data)
optimizer.prep_optimizer(ratio_known_cameras=0.0, 
                         max_iterations=30, 
                         num_epochs=1, 
                         step_size=0.1,
                         optimizer_cls="GaussNewton")

Function Call: 
optimal_scene = optimizer(bal_path)
"""
        # Logger
        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger(__name__)

        # Setup Path for Saving Results
        result_dir = "results/scene_data"
        file_path = os.path.realpath(__file__).split('\\')
        file_path[0] = file_path[0] + "\\"
        home_dir = pathlib.Path(os.path.join(*file_path[:7])) # Sets Path to Breadth_agent
        self.results_path = home_dir / result_dir

        # Get Device
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        config_settings = {"inner_optim": {
                                "optimizer_cls": optimizer_cls,
                                "max_iters": max_iterations,
                                "step_size": step_size,
                                "backward_mode": "implicit",
                                "verbose": True,
                                "track_err_history": True,
                                "keep_step_size": True,
                                "regularize": True,
                                "ratio_known_cameras": 0.0,
                                "reg_w": 1e-4},
                            "outer_optim": {
                                "lr": learning_rate,
                                "num_epochs": num_epochs},
                            "hydra": {
                                    "run":{
                                        "dir": "examples/outputs"}}}
        
        self.cfg = omegaconf.OmegaConf.create(config_settings)

        # End of Initialization

    ################################## HELPER FUNCTIONS ##################################
    # Copyright (c) Meta Platforms, Inc. and affiliates.
    #
    # This source code is licensed under the MIT license found in the
    # LICENSE file in the root directory facebookresearch/theseus
    # Credit belongs to Facebook Research Team with Theseus library Examples:Bundle Adjustment
    
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
    # End
    ################################## HELPER FUNCTIONS ##################################

    def __call__(self,
                 scene: Scene) -> Scene:

        # Setup Theseus BA dataset -- Read BAL dataset
        ba = copy.deepcopy(scene.bal_data.dataset) #theg.BundleAdjustmentDataset.load_from_file(ba_file) # Look into switching this

        # hyper parameters (ie outer loop's parameters) -> Not needed
        log_loss_radius = th.Vector(1, name="log_loss_radius", dtype=torch.float64)

        # Set up objective
        objective = th.Objective(dtype=torch.float64)

        pose_check = set()
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

        # Create optimizer
        optimizer_cls: Type[th.NonlinearLeastSquares] = getattr(
            th, self.cfg.inner_optim.optimizer_cls
        )
        # optimizer_cls: Type[th.LevenbergMarquardt] = getattr(
        #     th, self.cfg.inner_optim.optimizer_cls
        # )
        
        optimizer = optimizer_cls(
            objective,
            max_iterations=self.cfg.inner_optim.max_iters,
            step_size=self.cfg.inner_optim.step_size,
        )

        # Set up Theseus layer
        theseus_optim = th.TheseusLayer(optimizer)
        theseus_optim = theseus_optim.to(device=self.device)

        # copy the poses/pts to feed them to each outer iteration
        orig_poses = {cam.pose.name: cam.pose.tensor.clone().to(self.device) for cam in ba.cameras}
        orig_points = {pt.name: pt.tensor.clone() for pt in ba.points}
        print("CAMERAS", pose_check)
        # print(orig_poses)

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
            
            # self.print_histogram(ba, theseus_outputs, "Output histogram:")
            self.log.info(
                f"Epoch: {epoch} Loss: {loss_value} "
                f"Kernel Radius: exp({loss_radius_tensor.data.item()})="
                f"{torch.exp(loss_radius_tensor.data).item()}"
            )
            self.log.info(f"Epoch took {(end_time - start_time) / 1e9: .3f} seconds")

            self.save_epoch(
                self.results_path,
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
        print(theseus_outputs.keys())
        for key in theseus_outputs.keys():
            if 'Cam' in key:
                pose = theseus_outputs[key].cpu().detach().numpy()
                poses.append(pose)
            if 'Pt' in key:
                # print(obs.point_index)
                # print("KEY", int(re.findall(r'\d+',key)[0]))
                world_point = theseus_outputs[ba.points[int(re.findall(r'\d+',key)[0])].name].cpu().detach().numpy()  # shape (3,) or (batch,3)
                points_3d.update_points(world_point)#.astype(float))
        
        # print(theseus_outputs)
        # print(theseus_outputs.keys())
        # print(len(poses))
        print("Number of 3D Points", points_3d.points3D.shape)
        new_scene = Scene(points3D = points_3d,
                          cam_poses = poses,
                          representation = "point cloud")
        
        # print(points_3d.points3D)
        return new_scene
    
    # def _create_result_path(self, result_dir: str) -> pathlib.Path:
    #     file_path = os.path.realpath(__file__).split('\\')
    #     file_path[0] = file_path[0] + "\\"
    #     home_dir = pathlib.Path(os.path.join(*file_path[:7])) # Sets Path to Breadth_agent
    #     result_path = home_dir / result_dir

    #     return result_path