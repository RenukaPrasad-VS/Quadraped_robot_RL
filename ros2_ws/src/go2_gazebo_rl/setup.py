from glob import glob

from setuptools import setup

package_name = "go2_gazebo_rl"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/config", glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="Run the Isaac Lab-trained Go2 stair policy in Gazebo Fortress.",
    license="BSD-3-Clause",
    entry_points={
        "console_scripts": [
            "policy_node = go2_gazebo_rl.policy_node:main",
            "course_monitor = go2_gazebo_rl.course_monitor:main",
            "make_world = go2_gazebo_rl.course_world:main",
            "gazebo_trainer = go2_gazebo_rl.gazebo_trainer:main",
        ],
    },
)
