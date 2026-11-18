"""Unitree Go2 articulation.

Uses Unitree's own configuration from ``unitree_rl_lab`` unchanged: the Go2
motor model (UnitreeActuatorCfg_Go2HV, a torque-speed-limited PD drive with
kp=25, kd=0.5), the standing pose, and the SDK joint order the C++ deploy code
expects. Only the mesh source changes: the Go2 USD is streamed from the
Isaac Lab asset server instead of a local ``unitree_model`` download.
"""

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG, UnitreeUsdFileCfg

GO2_CFG = UNITREE_GO2_CFG.replace(
    spawn=UnitreeUsdFileCfg(usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd"),
)
