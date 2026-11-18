"""Go2 stair-climbing environment.

Built on Unitree's official Go2 velocity task (unitree_rl_lab) and changed
only where stairs and slopes require it:

* Observations: angular rate and gravity direction come from a simulated IMU
  on the trunk (names kept as ``base_ang_vel`` / ``projected_gravity`` so the
  Unitree C++ deploy code recognises them). The non-blind policy also gets a
  1.6 m x 1.0 m height scan in front of/under the robot.
* Terrain: stairs up/down, slopes, boxes and rough ground with a curriculum.
* Commands: mostly forward walking with a heading target.
* Rewards: Unitree's terms, retuned so the robot can move vertically and
  pitch on stairs, plus penalties for stubbing feet and knocking legs on edges.
* Random pushes during training teach recovery from near-falls.

Control loop: the policy runs at 50 Hz and outputs 12 joint position offsets
(scale 0.25 rad) around the standing pose; Unitree's Go2 motor model
tracks them with PD at the 200 Hz physics rate.
"""

from __future__ import annotations

import math

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ImuCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import RobotEnvCfg, RobotSceneCfg

from . import mdp
from .robot import GO2_CFG
from .terrains import STAIRS_TERRAINS_CFG, course_terrain_cfg


@configclass
class Go2StairsSceneCfg(RobotSceneCfg):
    imu = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/base")


@configclass
class Go2StairsEnvCfg(RobotEnvCfg):
    scene: Go2StairsSceneCfg = Go2StairsSceneCfg(num_envs=2048, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()

        # --- robot and terrain ---
        self.scene.robot = GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG.replace(curriculum=True)
        self.scene.terrain.max_init_terrain_level = 3
        self.scene.imu.update_period = self.sim.dt

        # --- observations (IMU-sourced attitude and rates) ---
        imu = SceneEntityCfg("imu")
        for group in (self.observations.policy, self.observations.critic):
            group.base_ang_vel.func = mdp.imu_ang_vel
            group.base_ang_vel.params = {"sensor_cfg": imu}
            group.projected_gravity.func = mdp.imu_projected_gravity
            group.projected_gravity.params = {"sensor_cfg": imu}
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-1.0, 1.0),
        )
        self.observations.policy.height_scan = height_scan
        self.observations.critic.height_scan = height_scan.replace(noise=None)

        # --- commands: walk forward through the terrain, steer to a heading ---
        self.commands.base_velocity = mdp.UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.05,
            rel_heading_envs=1.0,
            heading_command=True,
            heading_control_stiffness=0.5,
            debug_vis=True,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
            ),
        )
        self.curriculum.lin_vel_cmd_levels = None  # Unitree's command curriculum expects its own command type

        # --- rewards ---
        r = self.rewards
        r.base_linear_velocity.weight = -0.5  # stairs need vertical body motion
        r.flat_orientation_l2.weight = -1.0  # keep level, but allow pitch on stairs
        r.joint_pos.weight = -0.2  # stairs need large knee/hip excursions
        r.action_rate.weight = -0.05
        r.feet_air_time.weight = 0.25  # longer, higher swings clear step edges
        r.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=[".*_thigh", ".*_calf"]
        )
        r.feet_stumble = RewTerm(
            func=mdp.feet_stumble,
            weight=-0.5,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
        )

        # --- disturbances: random shoves every 5-10 s (balance recovery) ---
        self.events.push_robot.params["velocity_range"] = {"x": (-0.6, 0.6), "y": (-0.6, 0.6)}

        self.terminations.bad_orientation.params["limit_angle"] = 1.0


@configclass
class Go2StairsBlindEnvCfg(Go2StairsEnvCfg):
    """Proprioceptive policy (IMU + joints only). The critic still sees the terrain.

    This policy's observation set matches Unitree's C++ deploy pipeline
    (unitree_mujoco sim2sim and the physical Go2).
    """

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.height_scan = None


def apply_course(env_cfg: Go2StairsEnvCfg, step_height: float, slope_deg: float, speed: float, push: bool):
    """Turn a training config into the fixed evaluation course."""
    env_cfg.scene.terrain.terrain_generator = course_terrain_cfg(step_height, slope_deg)
    env_cfg.scene.terrain.max_init_terrain_level = None
    env_cfg.scene.terrain.debug_vis = False
    env_cfg.curriculum.terrain_levels = None
    env_cfg.curriculum.lin_vel_cmd_levels = None
    env_cfg.episode_length_s = 60.0
    env_cfg.observations.policy.enable_corruption = False

    cmd = env_cfg.commands.base_velocity
    cmd.resampling_time_range = (1.0e6, 1.0e6)
    cmd.rel_standing_envs = 0.0
    cmd.rel_heading_envs = 1.0
    cmd.heading_control_stiffness = 1.0
    cmd.ranges.lin_vel_x = (speed, speed)
    cmd.ranges.lin_vel_y = (0.0, 0.0)
    cmd.ranges.heading = (0.0, 0.0)

    env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (-0.3, 0.3), "yaw": (0.0, 0.0)}
    env_cfg.events.add_base_mass = None
    if not push:
        env_cfg.events.push_robot = None
    return env_cfg
