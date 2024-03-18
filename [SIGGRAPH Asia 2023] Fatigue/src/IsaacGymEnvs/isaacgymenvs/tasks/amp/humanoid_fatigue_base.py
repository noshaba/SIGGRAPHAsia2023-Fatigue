# Copyright (c) 2018-2022, NVIDIA Corporation
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
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
from torch.distributions import Uniform

from isaacgymenvs.utils.torch_jit_utils import *
from ..base.vec_task import VecTask

DOF_BODY_IDS = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
DOF_OFFSETS = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
NUM_OBS = 13 + 52 + 28 + 12  # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
NUM_ACTIONS = 28
SYMMETRIC_INDICES = [(0, 0), (2, 2), (3, 3), (5, 5), (6, 10), (7, 11), (8, 12), (9, 13), (14, 21), (15, 22), (16, 23), (17, 24), (18, 25), (19, 26), (20, 27)]
# backflip expert without fatigue uses these peak forces (estimated by PD control formula using stiffness and damping XML parameters as PD gains)
# so we want these numbers to be the ceiling, i.e. matching the gear values
# for symmetric body parts, whichever number is higher should be the ceiling. determine this.
# FORCE_MAXES = [254.86, 980.37, 305.61, 27.36, 85.49, 26.26, 634.45, 350.89, 423.95, 149.3, 475.57, 316.64, 542.98, 214.52, 339.97, 756.62, 406.06, 832.29, 195.34, 711.38, 253.84, 271.39, 782.34, 266.85, 853.5, 254.71, 996.18, 216.81]
# FORCE_MAXES = [242.46, 461.4, 242.26, 12.89, 70.31, 15.17, 376.72, 456.37, 308.52, 82.91, 361.62, 454.67, 476.79, 105.25, 223.04, 575.49, 281.23, 717.7, 312.14, 633.56, 59.03, 299.76, 741.56, 375.64, 600.71, 477.6, 747.98, 499.53]
# FORCE_MAXES = [543.63, 461.4, 595.45, 47.67, 70.31, 22.52, 376.72, 456.37, 337.97, 239.95, 361.62, 454.67, 476.79, 250.67, 741.63, 575.49, 639.56, 809.41, 795.66, 633.56, 469.7, 407.25, 741.56, 733.43, 663.8, 477.6, 747.98, 505.11]
# FORCE_MAXES = [654.99, 775.37, 1052.27, 53.88, 122.27, 76.79, 541.76, 463.44, 623.22, 293.65, 657.62, 501.9, 563.95, 365.63, 834.96, 1039.74, 616.19, 976.92, 506.7, 597.65, 484.93, 423.21, 793.3, 635.22, 843.99, 529.01, 1029.46, 573.2]
# FORCE_MAXES = [505.36, 670.29, 415.67, 59.89, 77.09, 63.66, 320.25, 362.03, 340.08, 164.28, 259.13, 458.2, 415.07, 222.6, 573.59, 653.57, 448.25, 583.76, 397.38, 407.15, 274.18, 459.52, 601.27, 253.79, 533.54, 343.64, 422.09, 249.86]
# FORCE_MAXES = [505.36, 835.53, 415.67, 59.89, 109.15, 63.66, 320.25, 362.03, 340.08, 182.72, 259.13, 458.2, 415.07, 222.6, 573.59, 799.18, 448.25, 583.76, 397.38, 472.46, 274.18, 459.52, 743.29, 253.79, 610.22, 343.64, 437.58, 249.86]
# FORCE_MAXES = [505.36, 835.53, 415.67, 59.89, 109.15, 63.66, 259.13, 362.03, 340.08, 182.72, 259.13, 362.03, 340.08, 182.72, 459.52, 743.29, 253.79, 583.76, 343.64, 437.58, 249.86, 459.52, 743.29, 253.79, 583.76, 343.64, 437.58, 249.86]
# FORCE_MAXES = [232.2, 247.42, 191.92, 48.67, 97.77, 94.3, 172.31, 135.34, 173.72, 59.95, 270.01, 243.31, 208.64, 172.55, 885.69, 1004.58, 396.58, 1010.32, 140.97, 385.93, 187.82, 461.29, 867.01, 409.85, 917.52, 322.43, 472.59, 242.71]
# FORCE_MAXES = [185.09, 271.23, 162.21, 58.66, 97.77, 53.56, 158.36, 153.73, 148.47, 96.19, 322.82, 289.02, 168.1, 160.23, 746.64, 839.08, 474.9, 1086.24, 140.62, 999.55, 189.51, 525.76, 892.67, 450.56, 795.27, 317.55, 257.49, 242.08]
FORCE_MAXES = [289.66, 315.79, 234.43, 78.61, 167.3, 61.56, 281.26, 184.85, 201.46, 237.31, 186.61, 263.18, 242.94, 152.18, 697.8, 796.42, 541.85, 872.66, 238.44, 597.67, 213.37, 614.75, 1120.94, 373.14, 901.95, 437.76, 455.27, 245.11]

KEY_BODY_NAMES = ["right_hand", "left_hand", "right_foot", "left_foot"]

DOF_TO_BODY_MAP = {
    'abdomen_x': {'parent': 'pelvis', 'child': 'torso'},
    'abdomen_y': {'parent': 'pelvis', 'child': 'torso'},
    'abdomen_z': {'parent': 'pelvis', 'child': 'torso'},

    'neck_x': {'parent': 'torso', 'child': 'head'},
    'neck_y': {'parent': 'torso', 'child': 'head'},
    'neck_z': {'parent': 'torso', 'child': 'head'},

    'right_shoulder_x': {'parent': 'torso', 'child': 'right_upper_arm'},
    'right_shoulder_y': {'parent': 'torso', 'child': 'right_upper_arm'},
    'right_shoulder_z': {'parent': 'torso', 'child': 'right_upper_arm'},

    'right_elbow': {'parent': 'right_upper_arm', 'child': 'right_lower_arm'},

    'left_shoulder_x': {'parent': 'torso', 'child': 'left_upper_arm'},
    'left_shoulder_y': {'parent': 'torso', 'child': 'left_upper_arm'},
    'left_shoulder_z': {'parent': 'torso', 'child': 'left_upper_arm'},

    'left_elbow': {'parent': 'left_upper_arm', 'child': 'left_lower_arm'},

    'right_hip_x': {'parent': 'torso', 'child': 'right_thigh'},
    'right_hip_y': {'parent': 'torso', 'child': 'right_thigh'},
    'right_hip_z': {'parent': 'torso', 'child': 'right_thigh'},

    'right_knee': {'parent': 'right_thigh', 'child': 'right_shin'},

    'right_ankle_x': {'parent': 'right_shin', 'child': 'right_foot'},
    'right_ankle_y': {'parent': 'right_shin', 'child': 'right_foot'},
    'right_ankle_z': {'parent': 'right_shin', 'child': 'right_foot'},

    'left_hip_x': {'parent': 'torso', 'child': 'left_thigh'},
    'left_hip_y': {'parent': 'torso', 'child': 'left_thigh'},
    'left_hip_z': {'parent': 'torso', 'child': 'left_thigh'},

    'left_knee': {'parent': 'left_thigh', 'child': 'left_shin'},

    'left_ankle_x': {'parent': 'left_shin', 'child': 'left_foot'},
    'left_ankle_y': {'parent': 'left_shin', 'child': 'left_foot'},
    'left_ankle_z': {'parent': 'left_shin', 'child': 'left_foot'}
}

JOINT_NAMES = [
    'abdomen_x',
    'abdomen_y',
    'abdomen_z',
    'neck_x',
    'neck_y',
    'neck_z',
    'right_shoulder_x',
    'right_shoulder_y',
    'right_shoulder_z',
    'right_elbow',
    'left_shoulder_x',
    'left_shoulder_y',
    'left_shoulder_z',
    'left_elbow',
    'right_hip_x',
    'right_hip_y',
    'right_hip_z',
    'right_knee',
    'right_ankle_x',
    'right_ankle_y',
    'right_ankle_z',
    'left_hip_x',
    'left_hip_y',
    'left_hip_z',
    'left_knee',
    'left_ankle_x',
    'left_ankle_y',
    'left_ankle_z'
]

def convert_to_body_map(dof_map):
    body_map = {}
    # initialize body_map
    for key, val in dof_map.items():
        if val['parent'] not in body_map:
            body_map[val['parent']] = {'parents': [], 'children': []}
        if val['child'] not in body_map:
            body_map[val['child']] = {'parents': [], 'children': []}

    for key, val in dof_map.items():
        body_map[val['parent']]['children'].append(key)
        body_map[val['child']]['parents'].append(key)
    return body_map


class HumanoidFatigueBase(VecTask):

    def __init__(self, config, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = config

        self._pd_control = self.cfg["env"]["pdControl"]
        self.power_scale = self.cfg["env"]["powerScale"]
        self.randomize = self.cfg["task"]["randomize"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.camera_follow = self.cfg["env"].get("cameraFollow", False)
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self._local_root_obs = self.cfg["env"]["localRootObs"]
        self._contact_bodies = self.cfg["env"]["contactBodies"]
        self._termination_height = self.cfg["env"]["terminationHeight"]
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"]

        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        dt = self.cfg["sim"]["dt"]
        self.dt = self.control_freq_inv * dt

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        sensors_per_env = 2
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.num_actors = actor_root_state.shape[0]

        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        if self.cfg["env"]["showSkyWall"]:
            self._root_states = self._root_states[:-1].reshape(self.num_envs,-1)

        self._initial_root_states = self._root_states.clone()
        self._initial_root_states[:, 7:13] = 0

        # create some wrapper tensors for different slices
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self._dof_pos = self._dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self._dof_vel = self._dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        self._initial_dof_pos = torch.zeros_like(self._dof_pos, device=self.device, dtype=torch.float)
        right_shoulder_x_handle = self.gym.find_actor_dof_handle(self.envs[0], self.humanoid_handles[0], "right_shoulder_x")
        left_shoulder_x_handle = self.gym.find_actor_dof_handle(self.envs[0], self.humanoid_handles[0], "left_shoulder_x")
        self._initial_dof_pos[:, right_shoulder_x_handle] = 0.5 * np.pi
        self._initial_dof_pos[:, left_shoulder_x_handle] = -0.5 * np.pi

        self._initial_dof_vel = torch.zeros_like(self._dof_vel, device=self.device, dtype=torch.float)

        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)[0:(self.num_bodies*self.num_envs), :]
        self._rigid_body_pos = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 0:3]
        self._rigid_body_rot = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 3:7]
        self._rigid_body_vel = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 7:10]
        self._rigid_body_ang_vel = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 10:13]
        self._contact_forces = gymtorch.wrap_tensor(contact_force_tensor)[0:(self.num_bodies*self.num_envs),:].view(self.num_envs, self.num_bodies, 3)

        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)

        # dynamically changing fatigue variables
        self.num3CC = 1
        self.MF = torch.zeros((self.num_envs, self.num_dof * self.num3CC), device=self.device, requires_grad=False)
        self.MA = torch.zeros((self.num_envs, self.num_dof * self.num3CC), device=self.device, requires_grad=False)
        self.MR = torch.ones((self.num_envs, self.num_dof * self.num3CC), device=self.device, requires_grad=False) * 100
        self.TL = torch.zeros((self.num_envs, self.num_dof * self.num3CC), device=self.device, requires_grad=False)
        self.RC = torch.ones((self.num_envs, self.num_dof * self.num3CC), device=self.device, requires_grad=False)

        # fatigue constants
        self.LD = 10
        self.LR = 10

        # to play around with per-part fatigue constants
        self.F = torch.ones(self.num_dof, device=self.device, requires_grad=False) * self.cfg["env"]["fatigueF"]
        self.r = torch.ones(self.num_dof, device=self.device, requires_grad=False) * self.cfg["env"]["fatigue_r"]
        self.R = torch.ones(self.num_dof, device=self.device, requires_grad=False) * self.F * self.cfg["env"]["fatigueR"]

        self.target_dir = torch.zeros((self.num_envs, 2), device=self.device, requires_grad=False)

        self._body_to_dof_map = convert_to_body_map(DOF_TO_BODY_MAP)

        if self.viewer != None:
            self._init_camera()

        return

    def get_obs_size(self):
        obs_size = 13 + 52 + 28 + 12
        if self.cfg["env"]["TLObs"]:
            obs_size += 28
        if self.cfg["env"]["MAObs"]:
            obs_size += 28
        if self.cfg["env"]["MRObs"]:
            obs_size += 28
        if self.cfg["env"]["MFObs"]:
            obs_size += 28
        if self.cfg["env"]["TargetObs"]:
            obs_size +=2

        return obs_size

    def get_action_size(self):
        action_size = 28
        if self.cfg["env"]["useTorqueCoeff"]:
            action_size += 1
        if self.cfg["env"]["useStiffnessCoeff"]:
            action_size += 1
        if self.cfg["env"]["useDampingCoeff"]:
            action_size += 1
        return action_size

    def create_sim(self):
        self.up_axis_idx = 2  # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        l_color = gymapi.Vec3(1, 1, 1)
        l_ambient = gymapi.Vec3(1, 1, 1)
        l_direction = gymapi.Vec3(0, 0, 1)
        self.gym.set_light_parameters(self.sim, 0, l_color, l_ambient, l_direction)

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        if self.cfg["env"]["showSkyWall"]:
            self._create_wall_box(self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        return
    
    def _create_wall_box(self, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        wall_env = self.gym.create_env(self.sim, lower, upper, num_per_row)
        wall_box_options = gymapi.AssetOptions()
        wall_box_options.disable_gravity = True
        box_W = 1000
        box_D = 1
        box_H = 1000
        wall_asset = self.gym.create_box(self.sim, box_W, box_D, box_H, wall_box_options)
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 100, box_H * 0.5)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        ahandle = self.gym.create_actor(wall_env, wall_asset, pose, "wall_box", -3, 0)
        self.gym.set_rigid_body_color(wall_env, ahandle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.7, 0.8, 0.8))

    def reset_idx(self, env_ids):
        self._reset_actors(env_ids)
        self._refresh_sim_tensors()
        self._compute_observations(env_ids)
        return

    def set_char_color(self, col):
        for i in range(self.num_envs):
            env_ptr = self.envs[i]
            handle = self.humanoid_handles[i]

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
                                              gymapi.Vec3(col[0], col[1], col[2]))

        return

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)
        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../assets')
        if self.cfg["env"]["strength"] == "weaker":
            print("weaker")
            asset_file = "mjcf/amp_humanoid_weaker.xml"
        elif self.cfg["env"]["strength"] == "male":
            print("male")
            asset_file = "mjcf/amp_humanoid_male_gains.xml"
        else:
            print("normal")
            asset_file = "mjcf/amp_humanoid.xml"


        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]

        # create force sensors at the feet
        right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "right_foot")
        left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "left_foot")
        sensor_pose = gymapi.Transform()

        self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)
        self.max_forces = self.motor_efforts.unsqueeze(0) * self.power_scale

        # want: peak torques to be matching max_forces values
        self.known_peak_torques = torch.tensor(FORCE_MAXES, device=self.device, requires_grad=False) * self.power_scale
        if "useMotionTLDenom" not in self.cfg["env"]:
            print("WARNING: useMotionTLDenom not specified. Using Max Across")
        else:
            if self.cfg["env"]["useMotionTLDenom"]:
                print("useMotionTLDenom")
                if self.cfg["env"]["motion_file"] == "amp_humanoid_walk.npy":
                    force_maxes = [36.7, 50.82, 45.37, 13.09, 20.85, 6.77, 23.25, 48.2, 46.28, 33.77, 23.94, 38.19, 31.21, 43.94, 90.83, 188.32, 54.09, 125.27, 22.2, 118.64, 61.07, 88.38, 190.43, 63.42, 117.43, 30.19, 160.67, 36.93]
                    self.known_peak_torques = torch.tensor(force_maxes, device=self.device, requires_grad=False) * self.power_scale

                if self.cfg["env"]["motion_file"] == "amp_humanoid_run.npy":
                    force_maxes = [44.86, 37.48, 56.89, 13.55, 26.97, 14.96, 19.55, 37.69, 22.42, 21.82, 20.35, 35.13, 12.77, 51.34, 150.29, 202.58, 132.12, 490.01, 23.25, 293.65, 37.95, 204.96, 300.73, 78.68, 420.83, 72.57, 227.65, 65.63]
                    self.known_peak_torques = torch.tensor(force_maxes, device=self.device, requires_grad=False) * self.power_scale

                if self.cfg["env"]["motion_file"] == "amp_humanoid_hop.npy":
                    force_maxes = [67.24, 196.68, 97.52, 25.52, 160.34, 15.98, 64.14, 98.05, 90.77, 106.93, 89.86, 135.4, 55.59, 106.75, 393.96, 466.35, 125.78, 872.66, 91.8, 264.36, 83.94, 444.96, 420.74, 100.95, 741.27, 122.53, 206.63, 69.69]
                    self.known_peak_torques = torch.tensor(force_maxes, device=self.device, requires_grad=False) * self.power_scale

                if self.cfg["env"]["motion_file"] == "amp_humanoid_cartwheel.npy":
                    force_maxes = [163.27, 234.43, 154.58, 58.88, 133.49, 61.56, 148.68, 89.79, 116.39, 101.56, 178.3, 162.26, 157.2, 80.32, 580.35, 796.42, 418.77, 642.08, 191.2, 319.82, 213.37, 614.75, 1120.94, 247.19, 496, 98.97, 119.59, 114.9]
                    self.known_peak_torques = torch.tensor(force_maxes, device=self.device, requires_grad=False) * self.power_scale

                if self.cfg["env"]["motion_file"] == "amp_humanoid_backflip.npy":
                    force_maxes = [162.91, 315.79, 141.48, 33.3, 167.3, 39.07, 281.26, 184.85, 201.46, 237.31, 186.61, 222.21, 210.41, 58.5, 287.28, 747.53, 296.95, 860.92, 238.44, 597.67, 209.73, 315.73, 708.42, 218.27, 901.95, 108.76, 342.87, 245.11]
                    self.known_peak_torques = torch.tensor(force_maxes, device=self.device, requires_grad=False) * self.power_scale

                if self.cfg["env"]["motion_file"] == "amp_humanoid_wushu_kick.npy":
                    force_maxes = [289.66, 181.37, 234.43, 78.61, 69.46, 46.88, 149.75, 170.81, 173.37, 136.1, 143.16, 263.18, 242.94, 152.18, 697.8, 691.47, 541.85, 857.04, 138.58, 451.8, 201.1, 598.84, 772.54, 373.14, 809.59, 437.76, 455.27, 224.82]
                    self.known_peak_torques = torch.tensor(force_maxes, device=self.device, requires_grad=False) * self.power_scale

        self.symmetric_known_peak_torques = self.known_peak_torques * 1  # copy
        for left, right in SYMMETRIC_INDICES:
            self.symmetric_known_peak_torques[left] = self.symmetric_known_peak_torques[right] = torch.minimum(self.known_peak_torques[left], self.known_peak_torques[right])
        # self.estimated_fudges = self.max_forces / self.symmetric_known_peak_torques

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*get_axis_params(0.89, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w], device=self.device)

        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
        self.stiffness_coefficients = to_torch(dof_prop["stiffness"], device=self.device)
        self.damping_coefficients = to_torch(dof_prop["damping"], device=self.device)

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            contact_filter = 0

            handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", i, contact_filter, 0)

            self.gym.enable_actor_dof_force_sensors(env_ptr, handle)

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.4706, 0.549, 0.6863))

            self.envs.append(env_ptr)
            self.humanoid_handles.append(handle)

            dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
            # disable to estimate fudge for "intended force"
            if self.cfg["env"]["effortLimiting"]:
                dof_prop["effort"][:] = self.motor_efforts.cpu() * self.power_scale
            if (self._pd_control):
                dof_prop["driveMode"] = gymapi.DOF_MODE_POS
                dof_prop["stiffness"] *= self.power_scale
                dof_prop["damping"] *= self.power_scale
                self.stiffness_coefficients = to_torch(dof_prop["stiffness"], device=self.device)
                self.damping_coefficients = to_torch(dof_prop["damping"], device=self.device)

            else:
                dof_prop["driveMode"] = gymapi.DOF_MODE_EFFORT
            self.gym.set_actor_dof_properties(env_ptr, handle, dof_prop)

        dof_prop = self.gym.get_actor_dof_properties(env_ptr, handle)
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        self._key_body_ids = self._build_key_body_ids_tensor(env_ptr, handle)
        self._contact_body_ids = self._build_contact_body_ids_tensor(env_ptr, handle)

        self._build_pd_action_offset_scale()

        # theoretical upper limit of torque estimate is already known by joint limits and velocity limit (in isaac it's 64 RAD/s)

        pd_tar_limits_upper = self._action_to_pd_targets(torch.ones(1, self.num_dof, device=self.device))
        pd_tar_limits_lower = self._action_to_pd_targets(-torch.ones(1, self.num_dof, device=self.device))
        self.max_pd_tar_dof_pos_diff = torch.maximum(torch.abs(pd_tar_limits_upper - self.dof_limits_lower), torch.abs(self.dof_limits_upper - pd_tar_limits_lower))

        self.force_estimate_upper_limit = self.max_pd_tar_dof_pos_diff * self.stiffness_coefficients + 5 * self.damping_coefficients

        return

    def _build_pd_action_offset_scale(self):
        num_joints = len(DOF_OFFSETS) - 1

        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        for j in range(num_joints):
            dof_offset = DOF_OFFSETS[j]
            dof_size = DOF_OFFSETS[j + 1] - DOF_OFFSETS[j]

            if (dof_size == 3):
                lim_low[dof_offset:(dof_offset + dof_size)] = -np.pi
                lim_high[dof_offset:(dof_offset + dof_size)] = np.pi

            elif (dof_size == 1):
                curr_low = lim_low[dof_offset]
                curr_high = lim_high[dof_offset]
                curr_mid = 0.5 * (curr_high + curr_low)

                # extend the action range to be a bit beyond the joint limits so that the motors
                # don't lose their strength as they approach the joint limits
                curr_scale = 0.7 * (curr_high - curr_low)
                curr_low = curr_mid - curr_scale
                curr_high = curr_mid + curr_scale

                lim_low[dof_offset] = curr_low
                lim_high[dof_offset] = curr_high

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)

        return

    def _compute_reward(self, actions):
        if self.cfg["env"]["regularize"] == "RC":
            print("RC")
            self.rew_buf[:] = compute_humanoid_reward(self.obs_buf) * torch.mean(self.RC, dim=1)
        elif self.cfg["env"]["regularize"] == "MF":
            print("MF")
            self.rew_buf[:] = -compute_humanoid_reward(self.obs_buf) * torch.mean(self.MF / 100, dim=1)
        elif self.cfg["env"]["regularize"] == "VB":
            print("VB")  # velocity reward, backward movement => encourage minus
            self.rew_buf[:] = torch.clip(-compute_humanoid_reward(self.obs_buf) * self._root_states[:, 7], 0, 1)
        elif self.cfg["env"]["regularize"] == "VF":
            print("VF")  # velocity reward, forward movement
            self.rew_buf[:] = torch.clip(compute_humanoid_reward(self.obs_buf) * self._root_states[:, 7], 0, 1)
        elif self.cfg["env"]["regularize"] == "TLMA_r":
            print("TLMA_r")
            # for 1 - (TL - MA)
            self.rew_buf[:] = compute_humanoid_reward(self.obs_buf) * torch.mean(1 - (self.TL - self.MA) / 100.0, dim=1)
        elif self.cfg["env"]["regularize"] == "TLMA_p":
            print("TLMA_p")
            # for -(TL - MA) = MA - TL
            self.rew_buf[:] = compute_humanoid_reward(self.obs_buf) * torch.mean((self.MA - self.TL) / 100.0, dim=1)
        elif self.cfg["env"]["regularize"] == "TLMA_f":
            print("TLMA_f")
            # for -(TL - MA) = MA - TL
            self.rew_buf[:] = compute_humanoid_reward(self.obs_buf) * torch.mean((self.TL - self.MA) / 100.0, dim=1)
        elif self.cfg["env"]["regularize"] == "TLMA_abs":
            print("TLMA_f")
            # for (TL - MA)            
            self.rew_buf[:] = compute_humanoid_reward(self.obs_buf) * torch.mean(torch.abs(self.MA - self.TL) / 100.0, dim=1)
        elif self.cfg["env"]["regularize"] == "TLMA_chi":
            self.rew_buf[:] = compute_humanoid_reward(self.obs_buf) * torch.mean(torch.exp(-((self.MA - self.TL) / 100)**2 / 0.12**2), dim=1)
        else:
            self.rew_buf[:] = compute_humanoid_reward(self.obs_buf)
        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                                           self._contact_forces, self._contact_body_ids,
                                                                           self._rigid_body_pos, self.max_episode_length,
                                                                           self._enable_early_termination, self._termination_height)
        return

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        return

    def _compute_observations(self, env_ids=None):
        obs = self._compute_humanoid_obs(env_ids)

        if (env_ids is None):
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs

        return

    def _compute_humanoid_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self._root_states
            dof_pos = self._dof_pos
            dof_vel = self._dof_vel
            key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        else:
            root_states = self._root_states[env_ids]
            dof_pos = self._dof_pos[env_ids]
            dof_vel = self._dof_vel[env_ids]
            key_body_pos = self._rigid_body_pos[env_ids][:, self._key_body_ids, :]

        obs = compute_humanoid_observations(root_states, dof_pos, dof_vel,
                                            key_body_pos, self._local_root_obs)
        if self.cfg["env"]["TLObs"]:
            if env_ids is None:
                obs = torch.cat([obs, self.TL], dim=-1)
            else:
                obs = torch.cat([obs, self.TL[env_ids]], dim=-1)
        if self.cfg["env"]["MAObs"]:
            if env_ids is None:
                obs = torch.cat([obs, self.MA], dim=-1)
            else:
                obs = torch.cat([obs, self.MA[env_ids]], dim=-1)
        if self.cfg["env"]["MFObs"]:
            if env_ids is None:
                obs = torch.cat([obs, self.MF], dim=-1)
            else:
                obs = torch.cat([obs, self.MF[env_ids]], dim=-1)
        if self.cfg["env"]["MRObs"]:
            if env_ids is None:
                obs = torch.cat([obs, self.MR], dim=-1)
            else:
                obs = torch.cat([obs, self.MR[env_ids]], dim=-1)
        if self.cfg["env"]["TargetObs"]:
            if env_ids is None:
                obs = torch.cat([obs, self.target_dir], dim=-1)
            else:
                obs = torch.cat([obs, self.target_dir[env_ids]], dim=-1)

        assert obs.shape[-1] == self.get_obs_size()

        return obs

    def _reset_actors(self, env_ids):
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        return
    
    def set_targets(self, env_ids):
        num_sets = len(env_ids)
        # set normalized target directions randomly with x in (0, 1) and y in (-1, 1)
        self.target_dir[env_ids] = torch.rand(num_sets, 2, device=self.device)
        self.target_dir[env_ids, 1] = self.target_dir[env_ids, 1] * 2 - 1
        self.target_dir[env_ids] = torch.nn.functional.normalize(self.target_dir[env_ids])
        
    ### ORIGINAL PRE_PHYSICS_STEP FROM AMP (NOT USED)

    def _pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()

        if (self._pd_control):
            pd_tar = self._action_to_pd_targets(self.actions)
            pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
            self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
        else:
            forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
            force_tensor = gymtorch.unwrap_tensor(forces)
            self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

        return
    
    ### OUR CHANGES WITH OUR MAIN CONTRIBUTIONS FOR FATIGUE MODELING

    def pre_physics_step(self, actions):
        set_target_ids = (self.progress_buf % 100 == 0).nonzero(as_tuple=False).squeeze(-1)
        if len(set_target_ids) > 0:
            self.set_targets(set_target_ids)
        self.actions = actions.to(self.device).clone()
        self.torque_mod = torch.ones(actions.shape[0], 1, device=self.device, requires_grad=False)
        self.stiffness_mod = torch.ones(actions.shape[0], 1, device=self.device, requires_grad=False)
        self.damping_mod = torch.ones(actions.shape[0], 1, device=self.device, requires_grad=False)

        # to optionally parse modulation coefficients
        if self.get_action_size() > self.num_dof:
            mods = self.actions[:, self.num_dof:]
            if self.cfg["env"]["useTorqueCoeff"]:
                # print("useTorqueCoeff")
                self.torque_mod = (mods[:, 0] + 1) * 0.5
                mods = mods[:, 1:]
            if self.cfg["env"]["useStiffnessCoeff"]:
                # print("useStiffnessCoeff")
                self.stiffness_mod = (mods[:, 0] + 1) * 0.5
                mods = mods[:, 1:]
            if self.cfg["env"]["useDampingCoeff"]:
                # print("useDampingCoeff")
                self.damping_mod = (mods[:, 0] + 1) * 0.5
                mods = mods[:, 1:]

        # for modulation of PD parameters (MetaPD)
        stiffness = self.stiffness_mod.reshape(-1, 1) * self.stiffness_coefficients
        damping = self.stiffness_mod.reshape(-1, 1) * self.damping_coefficients

        if self._pd_control:
            pd_tar = self._action_to_pd_targets(self.actions[:, :self.num_dof])
            pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
            intended_torques = (stiffness * (pd_tar - self._dof_pos) - damping * self._dof_vel)
            self.torques = intended_torques
            self.effective_torques = intended_torques

            if self.cfg["env"]["useFatigue"]:
                clipped_intended_torques = torch.clip(intended_torques, -self.symmetric_known_peak_torques, self.symmetric_known_peak_torques)  # the force the agent was going to use, ignoring fatigue
                self.compute_3CC(clipped_intended_torques)

            if self.cfg["env"]["randomizeEveryStep"]:
                print("randomizeEveryStep")
                self.MR[:, :] = Uniform(0, 100).sample(self.MR[:, :].shape).to(self.device)
                self.MF[:, :] = Uniform(0, 100 - self.MR[:, :]).sample().to(self.device)
                self.MA[:, :] = 100 - self.MR[:, :] - self.MF[:, :]
                self.RC[:, :] = torch.ones_like(self.RC[:, :], device=self.device) - (self.MF[:, :] / 100)

            if self.cfg["env"]["useFatigue"]:
                effective_torque_limits = self.RC * self.symmetric_known_peak_torques
                clipped_effective_torques = torch.clip(intended_torques, -effective_torque_limits, effective_torque_limits)

                # to allow for 0 intended torques
                clip_ratios = torch.where(
                    intended_torques == 0,
                    intended_torques,
                    torch.abs(clipped_effective_torques / intended_torques),
                )

                stiffness *= clip_ratios
                damping *= clip_ratios

                self.effective_torques = clipped_effective_torques
            else:
                t_max = self.max_forces * self.torque_mod.reshape(-1, 1)  # in Nm
                t_max_np = t_max.detach().cpu().numpy()

            stiffness_np = stiffness.detach().cpu().numpy()
            damping_np = damping.detach().cpu().numpy()

            if self.cfg["env"]["useFatigue"] or self.get_action_size() > self.num_dof:  # since using any modulation implies action size is larger
                for i, (env_ptr, handle) in enumerate(zip(self.envs, self.humanoid_handles)):
                    dof_prop = self.gym.get_actor_dof_properties(env_ptr, handle)
                    dof_prop["stiffness"][:] = stiffness_np[i]
                    dof_prop["damping"][:] = damping_np[i]
                    self.gym.set_actor_dof_properties(env_ptr, handle, dof_prop)
                    # print("B")

            self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
        return

    def post_physics_step(self):
        self.progress_buf += 1

        self._refresh_sim_tensors()
        self._compute_observations()
        self._compute_reward(self.actions)
        self._compute_reset()

        self.extras["terminate"] = self._terminate_buf

        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()

        return

    def render(self, **kwargs):
        if self.viewer and self.cfg["env"]["visualizeFatigue"]:
            self._visualize_fatigue()
        if self.viewer and self.camera_follow:
            self._update_camera()

        return super().render(**kwargs)

    def _build_key_body_ids_tensor(self, env_ptr, actor_handle):
        body_ids = []
        for body_name in KEY_BODY_NAMES:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert (body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_contact_body_ids_tensor(self, env_ptr, actor_handle):
        body_ids = []
        for body_name in self._contact_bodies:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert (body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _action_to_pd_targets(self, action):
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        return pd_tar

    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self._root_states[0, 0:3].cpu().numpy()

        cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0],
                              self._cam_prev_char_pos[1] - 3.0,
                              1.0)
        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0],
                                 self._cam_prev_char_pos[1],
                                 1.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._root_states[0, 0:3].cpu().numpy()

        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
        new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0],
                                  char_root_pos[1] + cam_delta[1],
                                  cam_pos[2])

        self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos
        return

    def _update_debug_viz(self):
        self.gym.clear_lines(self.viewer)
        return
    
    def _visualize_fatigue(self):
        for i in range(self.num_envs):
            env = self.envs[i]
            humanoid_handle = self.humanoid_handles[i]
            for body_name, dofs in self._body_to_dof_map.items():
                body_id = self.gym.find_actor_rigid_body_handle(
                    env, humanoid_handle, body_name)
                if body_id != -1 and len(dofs['parents']) > 0:
                    BE = 0
                    # print(BE)
                    for dof_name in dofs["parents"]:
                        dof_id = self.gym.find_actor_dof_handle(
                            env, humanoid_handle, dof_name)
                        num_dofs = 3. if (dof_name[-1] == 'x' or dof_name[-1] == 'y' or dof_name[-1] == 'z') else 1.

                        MF = self.MF[i][dof_id] / 100
                        BE += MF / num_dofs

                    self.gym.set_rigid_body_color(env, humanoid_handle, body_id, gymapi.MESH_VISUAL, gymapi.Vec3(BE, 0.5, 0.5))
                    if body_name == "left_lower_arm":
                        body_id = self.gym.find_actor_rigid_body_handle(
                            env, humanoid_handle, "left_hand")
                        self.gym.set_rigid_body_color(env, humanoid_handle, body_id, gymapi.MESH_VISUAL, gymapi.Vec3(BE, 0.5, 0.5))
                    if body_name == "right_lower_arm":
                        body_id = self.gym.find_actor_rigid_body_handle(
                            env, humanoid_handle, "right_hand")
                        self.gym.set_rigid_body_color(env, humanoid_handle, body_id, gymapi.MESH_VISUAL, gymapi.Vec3(BE, 0.5, 0.5))
                    if body_name == "torso":
                        body_id = self.gym.find_actor_rigid_body_handle(
                            env, humanoid_handle, "pelvis")
                        self.gym.set_rigid_body_color(env, humanoid_handle, body_id, gymapi.MESH_VISUAL, gymapi.Vec3(BE, 0.5, 0.5))

    def compute_3CC(self, forces):
        # forces = torch.clip(forces, -self.max_forces, self.max_forces)
        TL = abs(forces) / self.symmetric_known_peak_torques * 100
        # TL = abs(forces) / self.force_estimate_upper_limit * 100
        LD = self.LD * 1
        LR = self.LR * 1
        dt = self.dt * 1
        r = self.r * 1
        R = self.R * 1
        F = self.F * 1
        MA = self.MA * 1
        MR = self.MR * 1
        MF = self.MF * 1
        Ct = torch.zeros_like(TL, requires_grad=False)
        Ct = torch.where((MA < TL) * (MR > (TL - MA)), LD * (TL - MA), Ct)
        Ct = torch.where((MA < TL) * (MR < (TL - MA)), LD * MR, Ct)
        Ct = torch.where(MA >= TL, LR * (TL - MA), Ct)
        rR = torch.where(MA >= TL, r * R, R)
        self.MR += dt * (-Ct + rR * MF)
        self.MA += dt * (Ct - F * MA)
        self.MF += dt * (F * MA - rR * MF)
        self.RC.data = 1 - self.MF / 100
        self.TL.data = TL * 1


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def dof_to_obs(pose):
    # type: (Tensor) -> Tensor
    # dof_obs_size = 64
    # dof_offsets = [0, 3, 6, 9, 12, 13, 16, 19, 20, 23, 24, 27, 30, 31, 34]
    dof_obs_size = 52
    dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
    num_joints = len(dof_offsets) - 1

    dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)
    dof_obs = torch.zeros(dof_obs_shape, device=pose.device)
    dof_obs_offset = 0

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset:(dof_offset + dof_size)]

        # assume this is a spherical joint
        if (dof_size == 3):
            joint_pose_q = exp_map_to_quat(joint_pose)
            joint_dof_obs = quat_to_tan_norm(joint_pose_q)
            dof_obs_size = 6
        else:
            joint_dof_obs = joint_pose
            dof_obs_size = 1

        dof_obs[:, dof_obs_offset:(dof_obs_offset + dof_obs_size)] = joint_dof_obs
        dof_obs_offset += dof_obs_size

    return dof_obs


@torch.jit.script
def compute_humanoid_observations(root_states, dof_pos, dof_vel, key_body_pos, local_root_obs):
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

    # 1, 6, 3, 3, 52, 28, 12
    obs = torch.cat((root_h, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos), dim=-1)
    return obs


@torch.jit.script
def compute_humanoid_reward(obs_buf):
    # type: (Tensor) -> Tensor
    # reward = torch.ones_like(obs_buf[:, 0])
    root_dir = torch.nn.functional.normalize(obs_buf[:, 1:3])
    # print(root_dir)
    target_dir = obs_buf[:, -2:]
    dir_reward = (root_dir[:, 0] * target_dir[:, 0] + root_dir[:, 1] * target_dir[:, 1] + 1) * 0.5
    reward = dir_reward

    return reward


@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_height):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, float) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(masked_contact_buf > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_height
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(fall_contact, fall_height)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated
