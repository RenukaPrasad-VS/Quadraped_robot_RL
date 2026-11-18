"""Go2 environments for live training.

Built on Isaac Lab's own Unitree Go2 locomotion tasks (velocity tracking, PPO-proven
rewards), using the Go2 model and DC-motor actuators from ``isaaclab_assets``.
Nothing here depends on Gazebo, ROS, or the sibling ``go2_stairs`` package.

Changes for live training:
* fewer robots by default, so the viewport stays smooth while you watch;
* the camera starts by following robot 0 (switch robots from the Live Training panel);
* a stair-focused terrain curriculum with random pushes.
"""

from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.envs import ViewerCfg
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg

FOLLOW_VIEW = ViewerCfg(eye=(-2.5, -2.5, 1.6), lookat=(0.0, 0.0, 0.3), origin_type="asset_root", asset_name="robot")


STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,  # difficulty levels (curriculum promotes/demotes robots between rows)
    num_cols=16,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.05),
        "stairs_up": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.30, step_height_range=(0.04, 0.18), step_width=0.30, platform_width=3.0, border_width=1.0
        ),
        "stairs_down": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.30, step_height_range=(0.04, 0.18), step_width=0.30, platform_width=3.0, border_width=1.0
        ),
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.10, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.10, grid_width=0.45, grid_height_range=(0.025, 0.10), platform_width=2.0
        ),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        ),
    },
)


@configclass
class Go2LiveFlatEnvCfg(UnitreeGo2FlatEnvCfg):
    viewer: ViewerCfg = FOLLOW_VIEW

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 256
        self.scene.env_spacing = 2.0


@configclass
class Go2LiveRoughEnvCfg(UnitreeGo2RoughEnvCfg):
    viewer: ViewerCfg = FOLLOW_VIEW

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 512


@configclass
class Go2LiveStairsEnvCfg(Go2LiveRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 2
        # stairs need the knees/hips to move further and the body to pitch
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.feet_air_time.weight = 0.1
        # shove robots every 10-15 s so they learn to recover
        from isaaclab.envs import mdp
        from isaaclab.managers import EventTermCfg as EventTerm

        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        )


def make_play_cfg(env_cfg):
    """Turn a training config into a viewing config: no noise, no pushes, robots spread over all levels."""
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.push_robot = None
    terrain = env_cfg.scene.terrain
    if terrain.terrain_generator is not None:
        terrain.max_init_terrain_level = None
        terrain.terrain_generator.num_rows = 5
        terrain.terrain_generator.num_cols = 5
        terrain.terrain_generator.curriculum = False
    return env_cfg
