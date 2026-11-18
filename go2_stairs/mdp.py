"""Task-specific MDP terms. Everything else comes from unitree_rl_lab / Isaac Lab."""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from unitree_rl_lab.tasks.locomotion.mdp import *  # noqa: F401, F403


def imu_ang_vel(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
    """Body-frame angular rate from the simulated IMU (rad/s)."""
    return env.scene.sensors[sensor_cfg.name].data.ang_vel_b


def imu_projected_gravity(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
    """Gravity direction in the IMU frame; its x/y components encode pitch/roll."""
    return env.scene.sensors[sensor_cfg.name].data.projected_gravity_b
