"""Unitree's Go2 URDF (unitree_ros/go2_description) + the Gazebo pieces the policy needs.

Actuation matches Unitree's Go2 motor model used in training,
tau = kp (q* - q) - kd q_dot, clipped to the motor torque limit:

* kp: Gazebo JointPositionController with p_gain = kp. Its force is computed
  every physics step and limited to +/- effort_limit.
* kd: joint viscous damping (URDF <dynamics damping=kd>). That gives exactly
  -kd*q_dot. The controller's own d_gain differentiates the position *error*
  and would kick on every 50 Hz target step.
"""

from __future__ import annotations

import os
import re

from .policy import SDK_JOINTS


def go2_urdf(
    description_dir: str,
    kp: float = 25.0,
    kd: float = 0.5,
    effort_limit: float = 23.5,
    foot_mu: float = 1.0,
    model_name: str = "go2",
) -> str:
    with open(os.path.join(description_dir, "urdf", "go2_description.urdf")) as f:
        urdf = f.read()
    urdf = urdf.replace("package://go2_description/", f"file://{os.path.abspath(description_dir)}/")

    # Viscous damping = the motor model's kd term, on the 12 actuated joints.
    def add_damping(m: re.Match) -> str:
        block = re.sub(r"\s*<dynamics[^>]*/>", "", m.group(0))
        return block.replace("</joint>", f'  <dynamics damping="{kd}" friction="0.0"/>\n  </joint>')

    urdf = re.sub(r'<joint name="(?:FL|FR|RL|RR)_(?:hip|thigh|calf)_joint" type="revolute">.*?</joint>',
                  add_damping, urdf, flags=re.S)

    gz = []
    for j in SDK_JOINTS:
        gz.append(f"""
  <gazebo>
    <plugin filename="ignition-gazebo-joint-position-controller-system" name="ignition::gazebo::systems::JointPositionController">
      <joint_name>{j}</joint_name>
      <topic>/{model_name}/joint/{j}/cmd_pos</topic>
      <p_gain>{kp}</p_gain><i_gain>0</i_gain><d_gain>0</d_gain>
      <cmd_max>{effort_limit}</cmd_max><cmd_min>{-effort_limit}</cmd_min>
    </plugin>
  </gazebo>""")
    for leg in ("FL", "FR", "RL", "RR"):
        gz.append(f"""
  <gazebo reference="{leg}_foot"><mu1>{foot_mu}</mu1><mu2>{foot_mu}</mu2></gazebo>""")
    gz.append(f"""
  <gazebo reference="base">
    <sensor name="imu" type="imu">
      <always_on>true</always_on>
      <update_rate>200</update_rate>
      <topic>/{model_name}/imu</topic>
      <gz_frame_id>base</gz_frame_id>
    </sensor>
  </gazebo>
  <gazebo>
    <plugin filename="ignition-gazebo-joint-state-publisher-system" name="ignition::gazebo::systems::JointStatePublisher"/>
  </gazebo>
  <gazebo>
    <plugin filename="ignition-gazebo-odometry-publisher-system" name="ignition::gazebo::systems::OdometryPublisher">
      <odom_frame>odom</odom_frame>
      <robot_base_frame>base</robot_base_frame>
      <odom_publish_frequency>50</odom_publish_frequency>
      <dimensions>3</dimensions>
    </plugin>
  </gazebo>
""")
    return urdf.replace("</robot>", "".join(gz) + "\n</robot>")
