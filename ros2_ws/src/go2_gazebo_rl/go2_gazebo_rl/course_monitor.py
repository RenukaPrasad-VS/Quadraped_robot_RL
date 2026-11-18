"""Score a Gazebo run on the course, using the same metrics as scripts/course.py in Isaac Lab.

Starts when the policy reports active. Ends when the robot passes the end of
the second side slope (finished), tilts past 1 rad (fell, matching the
training termination), or hits the time limit. Prints a status line every
2 s and writes a JSON report to <project>/logs/gazebo_eval/.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool

from .course_world import Course
from .paths import PROJECT_ROOT
from .policy_node import quat_rotate_inverse

import numpy as np


class CourseMonitor(Node):
    def __init__(self):
        super().__init__("go2_course_monitor")
        self.declare_parameter("step_height", 0.12)
        self.declare_parameter("slope_deg", 8.0)
        self.declare_parameter("timeout", 60.0)
        self.declare_parameter("fall_angle", 1.0)
        self.declare_parameter("model_name", "go2")
        self.declare_parameter("policy_path", "")
        self.course = Course(step_height=self.get_parameter("step_height").value,
                             slope_deg=self.get_parameter("slope_deg").value)
        model = self.get_parameter("model_name").value
        self.create_subscription(Bool, f"/{model}/policy_active", self.on_active, 10)
        self.create_subscription(Imu, f"/{model}/imu", self.on_imu, qos_profile_sensor_data)
        self.create_subscription(Odometry, f"/model/{model}/odometry", self.on_odom, qos_profile_sensor_data)
        self.t0 = None
        self.x = self.y = self.z = 0.0
        self.max_x = 0.0
        self.max_roll = self.max_pitch = self.max_rate = 0.0
        self.sq_roll = self.sq_pitch = 0.0
        self.samples = 0
        self.done = False
        self.status = "waiting for policy"
        self.create_timer(2.0, self.report)
        self.get_logger().info(
            f"course: {self.course.num_steps} steps x {self.course.step_height * 100:.0f} cm, side slopes "
            f"±{self.course.slope_deg:.1f}°, goal x = {self.course.goal_x:.1f} m"
        )

    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_active(self, msg: Bool):
        if msg.data and self.t0 is None:
            self.t0 = self.now()
            self.status = "running"

    def on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        self.x, self.y, self.z = p.x, p.y, p.z
        if self.t0 is not None and not self.done:
            self.max_x = max(self.max_x, self.x)

    def on_imu(self, msg: Imu):
        if self.t0 is None or self.done:
            return
        o, w = msg.orientation, msg.angular_velocity
        g = quat_rotate_inverse(np.array([o.x, o.y, o.z, o.w]), np.array([0.0, 0.0, -1.0]))
        roll = abs(math.atan2(g[1], -g[2]))
        pitch = abs(math.asin(max(-1.0, min(1.0, -g[0]))))
        self.max_roll, self.max_pitch = max(self.max_roll, roll), max(self.max_pitch, pitch)
        self.max_rate = max(self.max_rate, math.hypot(w.x, w.y))
        self.sq_roll += roll * roll
        self.sq_pitch += pitch * pitch
        self.samples += 1
        if self.x >= self.course.goal_x:
            self.finish("finished")
        elif math.acos(max(-1.0, min(1.0, -g[2]))) > self.get_parameter("fall_angle").value:
            self.finish(f"fell on {self.course.segment_at(self.x)}")
        elif self.now() - self.t0 > self.get_parameter("timeout").value:
            self.finish(f"stopped on {self.course.segment_at(self.x)} (time limit)")

    def report(self):
        if self.t0 is None or self.done:
            return
        self.get_logger().info(
            f"t={self.now() - self.t0:5.1f}s x={self.x:5.2f} m ({self.course.segment_at(self.x)}) "
            f"max roll={math.degrees(self.max_roll):4.1f}° max pitch={math.degrees(self.max_pitch):4.1f}°"
        )

    def finish(self, status: str):
        self.done, self.status = True, status
        k = max(self.samples, 1)
        result = {
            "status": status,
            "progress_m": round(self.max_x - self.course.spawn_x, 2),
            "goal_m": round(self.course.goal_x - self.course.spawn_x, 2),
            "time_s": round(self.now() - self.t0, 1),
            "max_roll_deg": round(math.degrees(self.max_roll), 1),
            "max_pitch_deg": round(math.degrees(self.max_pitch), 1),
            "rms_roll_deg": round(math.degrees(math.sqrt(self.sq_roll / k)), 1),
            "rms_pitch_deg": round(math.degrees(math.sqrt(self.sq_pitch / k)), 1),
            "max_tilt_rate_dps": round(math.degrees(self.max_rate)),
            "step_height_m": self.course.step_height,
            "slope_deg": self.course.slope_deg,
            "policy": self.get_parameter("policy_path").value,
        }
        out_dir = os.path.join(PROJECT_ROOT, "logs", "gazebo_eval")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".json")
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        self.get_logger().info("RESULT " + json.dumps(result))
        self.get_logger().info(f"report -> {out}")


def main():
    rclpy.init()
    node = CourseMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
