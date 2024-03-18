# Copyright (c) 2021-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE..

from enum import Enum
import numpy as np
import torch
import os
import json

from gym import spaces

from isaacgym import gymapi
from isaacgym import gymtorch
from torch.distributions import Uniform

from isaacgymenvs.tasks.amp.humanoid_fatigue_base import HumanoidFatigueBase, dof_to_obs
from isaacgymenvs.tasks.amp.utils_amp import gym_util
from isaacgymenvs.tasks.amp.utils_amp.motion_lib import MotionLib

from isaacgym.torch_utils import *
from isaacgymenvs.utils.torch_jit_utils import *


NUM_AMP_OBS_PER_STEP = 13 + 52 + 28 + 12 # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]


def to_unity_skeleton_desc(skeleton_tree):
    node_children = [[] for n in skeleton_tree.node_names]
    node_names =skeleton_tree.node_names
    local_translation = [[*map(float,o)] for o in skeleton_tree.local_translation.cpu().numpy()]
    for i, p in enumerate(skeleton_tree.parent_indices):
        if p >= 0:
            node_children[p].append(i)
    print(skeleton_tree.parent_indices)
    print(node_names)
    print(node_children)
    desc_list = []
    def node_to_joint_desc(node_idx):
        joint_desc = dict()
        joint_desc["name"] = node_names[node_idx]
        joint_desc["offset"] = local_translation[node_idx]
        joint_desc["rotation"] = [1,0,0,0]
        joint_desc["targetName"] = joint_desc["name"]
        joint_desc["children"] = []
        desc_list.append(joint_desc)
        for c in node_children[node_idx]:
            joint_desc["children"].append(node_names[c])
            node_to_joint_desc(c)
    node_to_joint_desc(0)
    data = dict()
    data["root"] = node_names[0]
    data["jointDescs"] = desc_list
    data["jointSequence"] = node_names
    return data


class HumanoidFatigue(HumanoidFatigueBase):

    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        state_init = cfg["env"]["stateInit"]
        self._state_init = HumanoidFatigue.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert(self._num_amp_obs_steps >= 2)

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        motion_file = cfg['env'].get('motion_file', "amp_humanoid_backflip.npy")
        motion_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets/amp/motions/" + motion_file)
        self._load_motion(motion_file_path)

        self.num_amp_obs = self._num_amp_obs_steps * NUM_AMP_OBS_PER_STEP

        self._amp_obs_space = spaces.Box(np.ones(self.num_amp_obs) * -np.Inf, np.ones(self.num_amp_obs) * np.Inf)

        self._amp_obs_buf = torch.zeros((self.num_envs, self._num_amp_obs_steps, NUM_AMP_OBS_PER_STEP), device=self.device, dtype=torch.float)
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]
        
        self._amp_obs_demo_buf = None

        self.skeleton_dict = to_unity_skeleton_desc(self._motion_lib._motions[0].skeleton_tree)
        self.enable_streaming = self.cfg["env"]["unity_stream"]
        self.sending_reset = False
        self.server = None

        self.json_frame_buffers = []

        if self.enable_streaming:
            from isaacgymenvs.utils.animation_tcp_server import AnimationTCPServer
            self.frame_buffer = None
            self.vis_env_id = 0
            self.frame_time = 1.0/30.0
            self.server = AnimationTCPServer(8889, self)
            print("start server--------------")
            self.server.start()

        return

    def post_physics_step(self):
        super().post_physics_step()
        
        self._update_hist_amp_obs()
        self._compute_amp_observations()

        amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        self.extras["amp_obs"] = amp_obs_flat
        self.extras["RC"] = self.RC.view(-1, self.num_dof)

        if self.cfg["env"]["record_poses"]:
            f = self._state_to_vis_frame()
            self.json_frame_buffers.append(f)
            if self.progress_buf[0] >= self.max_episode_length - 1:
                json_filename = self.cfg["env"]["motion_file"][:-4] \
                    + "_F=" + repr(self.cfg["env"]["fatigueF"]) \
                    + "_R=" + repr(self.cfg["env"]["fatigueR"]) \
                    + "_r=" + repr(self.cfg["env"]["fatigue_r"]) + ".json"
                with open(json_filename, 'wt') as file:
                    json.dump({"poseBuffer": self.json_frame_buffers}, file)
                print(f"Saved JSON file as {json_filename}")

        if self.enable_streaming:
            self._state_to_vis_frame()
            self._receive_FRr()


        return

    def get_num_amp_obs(self):
        return self.num_amp_obs

    @property
    def amp_observation_space(self):
        return self._amp_obs_space

    def fetch_amp_obs_demo(self, num_samples):
        return self.task.fetch_amp_obs_demo(num_samples)

    def fetch_amp_obs_demo(self, num_samples):
        dt = self.dt
        motion_ids = self._motion_lib.sample_motions(num_samples)

        if (self._amp_obs_demo_buf is None):
            self._build_amp_obs_demo_buf(num_samples)
        else:
            assert(self._amp_obs_demo_buf.shape[0] == num_samples)
            
        motion_times0 = self._motion_lib.sample_time(motion_ids)
        motion_ids = np.tile(np.expand_dims(motion_ids, axis=-1), [1, self._num_amp_obs_steps])
        motion_times = np.expand_dims(motion_times0, axis=-1)
        time_steps = -dt * np.arange(0, self._num_amp_obs_steps)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.flatten()
        motion_times = motion_times.flatten()
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
        amp_obs_demo = build_amp_observations(root_states, dof_pos, dof_vel, key_pos,
                                      self._local_root_obs)
        self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)

        amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self.get_num_amp_obs())
        return amp_obs_demo_flat

    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros((num_samples, self._num_amp_obs_steps, NUM_AMP_OBS_PER_STEP), device=self.device, dtype=torch.float)
        return

    def _load_motion(self, motion_file):
        self._motion_lib = MotionLib(motion_file=motion_file, 
                                     num_dofs=self.num_dof,
                                     key_body_ids=self._key_body_ids.cpu().numpy(), 
                                     device=self.device)
        return
    
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self._init_amp_obs(env_ids)
        if self.enable_streaming:
            self.sending_reset = True
        return
    
    def _state_to_vis_frame(self):
        f = {"rotations": [], "positions": []}
        positions = [[*map(float, p)] for p in self._rigid_body_pos[0, :, :].cpu().numpy()]
        rotations = [[*map(float, r)] for r in self._rigid_body_rot[0, :, :].cpu().numpy()]
        for i in range(len(positions)):
            p = positions[i]
            f["positions"].append({"x": p[0], "y": p[1], "z": p[2]})
            q = rotations[i]
            f["rotations"].append({"x": q[0], "y": q[1], "z": q[2], "w": q[3]})

        f["ma"] = self.MA[self.vis_env_id, :].tolist()
        f["mf"] = self.MF[self.vis_env_id, :].tolist()
        f["mr"] = self.MR[self.vis_env_id, :].tolist()
        f["tl"] = self.TL[self.vis_env_id, :].tolist()
        # f["reset"] = False
        f["reset"] = self.sending_reset
        if self.sending_reset:
            self.sending_reset = False

        self.frame_buffer = f
        return f
    
    def _receive_FRr(self):
        if self.server is not None:
            if 'fatigueF' in self.server.rev_data.keys():
                self.F = torch.ones_like(self.F) * self.server.rev_data['fatigueF']
                print('fatigueF: ', self.server.rev_data['fatigueF'])
            if 'fatigueR' in self.server.rev_data.keys():
                self.R = torch.ones_like(self.R) * self.server.rev_data['fatigueR']
                print("R: ", self.server.rev_data['fatigueR'])
            if 'fatigue_r' in self.server.rev_data:
                self.r = torch.ones_like(self.r) * self.server.rev_data['fatigue_r']
                print("r: ", self.server.rev_data['fatigue_r'])

    def _reset_actors(self, env_ids):
        if (self._state_init == HumanoidFatigue.StateInit.Default):
            self._reset_default(env_ids)
        elif (self._state_init == HumanoidFatigue.StateInit.Start
              or self._state_init == HumanoidFatigue.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        elif (self._state_init == HumanoidFatigue.StateInit.Hybrid):
            self._reset_hybrid_state_init(env_ids)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0

        self.set_targets(env_ids)

        if torch.rand(1) < float(self.cfg["env"]["randomizeFatigueProb"]):
            #print("random")
            self.MR[env_ids, :] = Uniform(0, 100).sample(self.MR[env_ids, :].shape).to(self.device)
            self.MF[env_ids, :] = Uniform(0, 100 - self.MR[env_ids, :]).sample().to(self.device)
            self.MA[env_ids, :] = 100 - self.MR[env_ids, :] - self.MF[env_ids, :]
            self.TL[env_ids, :] = Uniform(0, 100).sample(self.MR[env_ids, :].shape).to(self.device)
            self.RC[env_ids, :] = torch.ones_like(self.RC[env_ids, :], device=self.device) - (self.MF[env_ids, :] / 100)
        else:
            #print("not random")
            self.MF[env_ids, :] = 0
            self.MA[env_ids, :] = 0
            self.MR[env_ids, :] = 100
            self.TL[env_ids, :] = 0
            self.RC[env_ids, :] = 1

        return

    def _reset_default(self, env_ids):
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self._reset_default_env_ids = env_ids
        return

    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs)
        
        if (self._state_init == HumanoidFatigue.StateInit.Random
            or self._state_init == HumanoidFatigue.StateInit.Hybrid):
            motion_times = self._motion_lib.sample_time(motion_ids)
        elif (self._state_init == HumanoidFatigue.StateInit.Start):
            motion_times = np.zeros(num_envs)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        
        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times
        return

    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]
        if (len(ref_reset_ids) > 0):
            self._reset_ref_state_init(ref_reset_ids)

        default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if (len(default_reset_ids) > 0):
            self._reset_default(default_reset_ids)

        return

    def _init_amp_obs(self, env_ids):
        self._compute_amp_observations(env_ids)

        if (len(self._reset_default_env_ids) > 0):
            self._init_amp_obs_default(self._reset_default_env_ids)

        if (len(self._reset_ref_env_ids) > 0):
            self._init_amp_obs_ref(self._reset_ref_env_ids, self._reset_ref_motion_ids,
                                   self._reset_ref_motion_times)
        return

    def _init_amp_obs_default(self, env_ids):
        curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        self._hist_amp_obs_buf[env_ids] = curr_amp_obs
        return

    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times):
        dt = self.dt
        motion_ids = np.tile(np.expand_dims(motion_ids, axis=-1), [1, self._num_amp_obs_steps - 1])
        motion_times = np.expand_dims(motion_times, axis=-1)
        time_steps = -dt * (np.arange(0, self._num_amp_obs_steps - 1) + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.flatten()
        motion_times = motion_times.flatten()
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
        amp_obs_demo = build_amp_observations(root_states, dof_pos, dof_vel, key_pos,
                                      self._local_root_obs)
        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(self._hist_amp_obs_buf[env_ids].shape)
        return
    
    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._root_states[env_ids, 0:3] = root_pos
        self._root_states[env_ids, 3:7] = root_rot
        self._root_states[env_ids, 7:10] = root_vel
        self._root_states[env_ids, 10:13] = root_ang_vel

        root_states = self._root_states

        # to suppress setActorRootState error
        if self.cfg["env"]["showSkyWall"]:
            root_states = torch.cat([self._root_states, torch.zeros_like(self._root_states[0].reshape(1, -1))], dim=0)
        
        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(root_states), 
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return

    def _update_hist_amp_obs(self, env_ids=None):
        if (env_ids is None):
            self._hist_amp_obs_buf[:] = self._amp_obs_buf[:, 0:(self._num_amp_obs_steps - 1)]
        else:
            self._hist_amp_obs_buf[env_ids] = self._amp_obs_buf[env_ids, 0:(self._num_amp_obs_steps - 1)]
        return
    
    def _compute_amp_observations(self, env_ids=None):
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        if (env_ids is None):
            self._curr_amp_obs_buf[:] = build_amp_observations(self._root_states, self._dof_pos, self._dof_vel, key_body_pos,
                                                                self._local_root_obs)
        else:
            self._curr_amp_obs_buf[env_ids] = build_amp_observations(self._root_states[env_ids], self._dof_pos[env_ids], 
                                                                    self._dof_vel[env_ids], key_body_pos[env_ids],
                                                                    self._local_root_obs)
        return


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def build_amp_observations(root_states, dof_pos, dof_vel, key_body_pos, local_root_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    root_vel = root_states[:, 7:10]
    root_ang_vel = root_states[:, 10:13]

    root_h = root_pos[:, 2:3]
    heading_rot = calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = quat_to_tan_norm(root_rot_obs)

    local_root_vel = my_quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = my_quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    local_end_pos = my_quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])
    
    dof_obs = dof_to_obs(dof_pos)

    obs = torch.cat((root_h, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos), dim=-1)
    return obs