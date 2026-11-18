"""Go2 stair-climbing tasks for Isaac Lab."""

import gymnasium as gym

_AGENTS = "go2_stairs.agents.ppo_cfg"

gym.register(
    id="Go2-Stairs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "go2_stairs.env_cfg:Go2StairsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}:Go2StairsPPORunnerCfg",
    },
)
gym.register(
    id="Go2-Stairs-Blind-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "go2_stairs.env_cfg:Go2StairsBlindEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}:Go2StairsBlindPPORunnerCfg",
    },
)
