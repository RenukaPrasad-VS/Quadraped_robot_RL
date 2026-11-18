"""PPO settings (Isaac Lab's Go2 defaults, saved more often so you can grab checkpoints while watching)."""

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2FlatPPORunnerCfg,
    UnitreeGo2RoughPPORunnerCfg,
)


@configclass
class Go2LiveFlatPPOCfg(UnitreeGo2FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "go2_live_flat"
        self.max_iterations = 1000
        self.save_interval = 50


@configclass
class Go2LiveRoughPPOCfg(UnitreeGo2RoughPPORunnerCfg):
    experiment_name = "go2_live_rough"
    max_iterations = 3000
    save_interval = 100


@configclass
class Go2LiveStairsPPOCfg(UnitreeGo2RoughPPORunnerCfg):
    experiment_name = "go2_live_stairs"
    max_iterations = 5000
    save_interval = 100
