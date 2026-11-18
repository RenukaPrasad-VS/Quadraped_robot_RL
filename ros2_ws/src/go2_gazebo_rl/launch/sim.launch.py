"""Gazebo Fortress: Go2 on the stairs/side-slope course, driven by the Isaac Lab-trained policy.

ros2 launch go2_gazebo_rl sim.launch.py [policy:=...] [step_height:=0.12] [slope_deg:=8]
                                        [speed:=0.5] [gui:=true] [policy_enabled:=true]
"""

import os
import subprocess
import tempfile

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, RegisterEventHandler,
                            SetEnvironmentVariable, TimerAction)
from launch.event_handlers import OnProcessExit, OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

from go2_gazebo_rl.course_world import Course, world_sdf
from go2_gazebo_rl.paths import GO2_DESCRIPTION, deploy_cfg_for, newest_checkpoint
from go2_gazebo_rl.policy import SDK_JOINTS, DeployCfg
from go2_gazebo_rl.robot_description import go2_urdf

MODEL = "go2"
WORLD = "course"


def setup(context):
    arg = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731
    step_height, slope_deg = float(arg("step_height")), float(arg("slope_deg"))
    gui = arg("gui").lower() == "true"
    policy_enabled = arg("policy_enabled").lower() == "true"
    path = lambda name: os.path.expanduser(arg(name))  # noqa: E731  (allow policy:=~/...)
    policy = path("policy") or newest_checkpoint()
    deploy_cfg = path("deploy_cfg") or (deploy_cfg_for(policy) if policy else "")

    # Joint gains/limit from the training deploy config, so Gazebo matches Isaac Lab.
    kp, kd = 25.0, 0.5
    if deploy_cfg:
        cfg = DeployCfg(deploy_cfg)
        kp, kd = float(cfg.kp[0]), float(cfg.kd[0])

    course = Course(step_height=step_height, slope_deg=slope_deg)
    world_file = os.path.join(tempfile.gettempdir(), f"go2_course_{step_height:.3f}_{slope_deg:.1f}_{PARTITION}.sdf")
    with open(world_file, "w") as f:
        f.write(world_sdf(course))
    urdf = go2_urdf(path("go2_description") or GO2_DESCRIPTION, kp=kp, kd=kd)

    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")),
        launch_arguments={"gz_args": f"-r {'' if gui else '-s '}{world_file}"}.items(),
    )
    rsp = Node(package="robot_state_publisher", executable="robot_state_publisher",
               parameters=[{"robot_description": urdf, "use_sim_time": True}], output="log")
    spawn = Node(package="ros_gz_sim", executable="create", output="screen",
                 arguments=["-world", WORLD, "-name", MODEL, "-topic", "robot_description",
                            "-x", str(course.spawn_x), "-y", "0.0", "-z", "0.48"])
    bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge", output="log",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            f"/{MODEL}/imu@sensor_msgs/msg/Imu[gz.msgs.IMU",
            f"/model/{MODEL}/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry",
            f"/world/{WORLD}/model/{MODEL}/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model",
            *[f"/{MODEL}/joint/{j}/cmd_pos@std_msgs/msg/Float64]gz.msgs.Double" for j in SDK_JOINTS],
        ],
        remappings=[(f"/world/{WORLD}/model/{MODEL}/joint_state", f"/{MODEL}/joint_states")],
    )
    monitor = Node(package="go2_gazebo_rl", executable="course_monitor", output="screen",
                   parameters=[{"use_sim_time": True, "step_height": step_height, "slope_deg": slope_deg,
                                "policy_path": policy, "timeout": float(arg("timeout"))}])
    policy_node = Node(
        package="go2_gazebo_rl", executable="policy_node", output="screen",
        parameters=[{"use_sim_time": True, "policy_path": policy, "deploy_cfg": deploy_cfg,
                     "cmd_vx": float(arg("speed")), "stand_only": not policy_enabled}])
    after_spawn = [monitor, policy_node]
    # The `ign gazebo` wrapper does not always forward shutdown to the server: stop it by its unique world file.
    stop_gz = RegisterEventHandler(OnShutdown(on_shutdown=[OpaqueFunction(
        function=lambda _ctx: subprocess.run(["pkill", "-INT", "-f", os.path.basename(world_file)]) and [])]))
    return [
        stop_gz, gz, rsp, bridge,
        TimerAction(period=3.0, actions=[spawn]),
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=after_spawn)),
    ]


# Unique Gazebo transport partition per launch: leftover/other Gazebo servers can never cross-talk.
PARTITION = f"go2_stairs_rl_{os.getpid()}"


def generate_launch_description():
    return LaunchDescription([
        # Private Gazebo transport partition: ignore any other Gazebo instance on this machine.
        SetEnvironmentVariable("IGN_PARTITION", PARTITION),
        SetEnvironmentVariable("GZ_PARTITION", PARTITION),
        DeclareLaunchArgument("policy", default_value="", description="model_*.pt or policy.onnx (default: newest blind checkpoint)"),
        DeclareLaunchArgument("deploy_cfg", default_value="", description="deploy.yaml (default: from the policy's run)"),
        DeclareLaunchArgument("step_height", default_value="0.12"),
        DeclareLaunchArgument("slope_deg", default_value="8.0"),
        DeclareLaunchArgument("speed", default_value="0.5", description="forward speed command (m/s)"),
        DeclareLaunchArgument("timeout", default_value="60.0"),
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("policy_enabled", default_value="true", description="false = stand in the training pose without running the policy"),
        DeclareLaunchArgument("go2_description", default_value="", description="unitree_ros go2_description dir"),
        OpaqueFunction(function=setup),
    ])
