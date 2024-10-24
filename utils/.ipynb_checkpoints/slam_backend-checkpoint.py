import random
import time
import os 

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
import matplotlib.pyplot as plt 

from gaussian_splatting_2d.gaussian_renderer import render
from gaussian_splatting_2d.utils.loss_utils import l1_loss, ssim
from gaussian_splatting_2d.utils.system_utils import mkdir_p
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping


class BackEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        # general settings
        self.config          = config
        self.gaussians       = None
        self.pause           = False
        self.device          = "cuda:0"
        self.dtype           = torch.float32
        self.monocular       = False 
        self.initialized     = False 
        self.viewpoints      = {}
        self.iteration_count = 0

        # hyper-parameters 
        self.pipeline_params = None
        self.opt_params      = None
        self.background      = None
        self.cameras_extent  = None
        self.live_mode       = False

        # queues 
        self.frontend_queue = None
        self.backend_queue  = None

        # key-frame handling 
        self.last_sent            = 0
        self.occ_aware_visibility = {}
        self.current_window       = []
        self.keyframe_optimizers  = None

    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]

        self.init_itr_num           = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update   = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset    = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th       = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent   = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num        = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every  = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th            = self.config["Training"]["gaussian_th"]
        self.gaussian_extent        = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset         = self.config["Training"]["gaussian_reset"]
        self.size_threshold         = self.config["Training"]["size_threshold"]
        self.window_size            = self.config["Training"]["window_size"]

    def reset(self):
        self.iteration_count      = 0
        self.occ_aware_visibility = {}
        self.viewpoints           = {}
        self.current_window       = []
        self.initialized          = True
        self.keyframe_optimizers  = None

        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

    def add_next_kf(self, frame_idx, viewpoint, depth_map=None, init=False):
        self.gaussians.extend_from_pcd_seq(
            kf_id=frame_idx, cam_info=viewpoint, depthmap=depth_map, init=init
        )

    def initialize_map(self, cur_frame_idx, viewpoint):
        for mapping_iteration in range(self.init_itr_num):

            self.gaussians.update_learning_rate(self.iteration_count)

            self.iteration_count += 1
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["surf_depth"],
                render_pkg["rend_alpha"]
            )
            loss_init = get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity, initialization=True
            )
            loss_init.backward()

            with torch.no_grad():
                # update max radii, eventually decides to prune big splats 
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                #accumulates position gradient 
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                # densify and prune accordingly 
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )
                # reset opacity 
                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                        ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

                #self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long() <---- occlusion information 
        
        image = torch.clamp(image, 0, 1).detach().permute(1, 2, 0).cpu().numpy()
        path_tmp_frame = os.path.join(self.config["Results"]["save_dir"], "key_frames")
        mkdir_p(path_tmp_frame)
        plt.imsave(os.path.join(path_tmp_frame, f"initial_frame.jpg"), image)
        Log("Initialized map")
        return render_pkg

    def push_to_frontend(self, tag=None):
        self.last_sent = 0
        keyframes = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))
        if tag is None:
            tag = "sync_backend"

        msg = [tag, clone_obj(self.gaussians), self.occ_aware_visibility, keyframes]
        self.frontend_queue.put(msg)

    def run(self):
        while True:
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.01)
                    continue
                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue

            else:
                data = self.backend_queue.get()

                if data[0] == "stop":
                    break

                elif data[0] == "pause":
                    self.pause = True

                elif data[0] == "unpause":
                    self.pause = False

                if data[0] == "init":
                    cur_frame_idx = data[1]
                    viewpoint     = data[2]
                    depth_map     = data[3]
                    Log("Resetting the system")
                    self.reset()

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    )
                    self.initialize_map(cur_frame_idx, viewpoint)
                    self.push_to_frontend("init")
        
                else:
                    raise Exception("Unprocessed data", data)
                
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        return
    
