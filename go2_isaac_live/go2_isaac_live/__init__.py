"""Live, in-viewport PPO training of Unitree Go2 robots in Isaac Sim / Isaac Lab (no Gazebo, no ROS).

Tasks (short names accepted by scripts/train_live.py and scripts/play.py):

    flat    Go2-Live-Flat-v0     flat ground, velocity tracking (learns to walk in ~10-20 min)
    rough   Go2-Live-Rough-v0    Isaac Lab rough-terrain curriculum (slopes, stairs, boxes, noise)
    stairs  Go2-Live-Stairs-v0   stair-heavy curriculum (up/down 4-18 cm) + random pushes
"""

import gymnasium as gym

TASKS = {
    "flat": "Go2-Live-Flat-v0",
    "rough": "Go2-Live-Rough-v0",
    "stairs": "Go2-Live-Stairs-v0",
}

for _short, _id in TASKS.items():
    _cls = _short.capitalize()
    gym.register(
        id=_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"go2_isaac_live.env_cfg:Go2Live{_cls}EnvCfg",
            "rsl_rl_cfg_entry_point": f"go2_isaac_live.ppo_cfg:Go2Live{_cls}PPOCfg",
        },
    )
