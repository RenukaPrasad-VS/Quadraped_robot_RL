"""Training terrain mix and a fixed evaluation course.

Training: a curriculum grid (rows = difficulty levels) of up/down stairs,
5-15 degree slopes, random boxes and rough ground. Robots that walk far
enough are promoted to harder rows; robots that fall are demoted.

Evaluation course (one long tile, walked along +x):

    flat | stairs up | landing | stairs down | flat | side slope (left) | flat | side slope (right) | flat

The two side slopes tilt the ground across the walking direction so the robot
tends to roll toward the downhill side; descending stairs pitch it forward.
"""

from __future__ import annotations

import math

import numpy as np
import trimesh

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass


def _deg(angle: float) -> float:
    return math.tan(math.radians(angle))


STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.05),
        # Robot spawns on the top platform and walks down/out (descending stairs).
        "stairs_down": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.04, 0.18),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # Robot spawns in the pit and walks up/out (ascending stairs).
        "stairs_up": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.04, 0.18),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # Slopes in every direction relative to the robot heading, so both
        # side-slope (roll) and up/down-slope (pitch) disturbances are seen.
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.10, slope_range=(_deg(2.0), _deg(15.0)), platform_width=2.0, border_width=0.25
        ),
        "slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.10, slope_range=(_deg(2.0), _deg(15.0)), platform_width=2.0, border_width=0.25
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15, grid_width=0.45, grid_height_range=(0.025, 0.10), platform_width=2.0
        ),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.10, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        ),
    },
)


# ---------------------------------------------------------------------------
# Evaluation course
# ---------------------------------------------------------------------------

_BOTTOM = -1.0  # all blocks are solid down to this depth


def _block(x0: float, x1: float, y0: float, y1: float, top: float) -> trimesh.Trimesh:
    extents = (x1 - x0, y1 - y0, top - _BOTTOM)
    center = ((x0 + x1) / 2, (y0 + y1) / 2, (top + _BOTTOM) / 2)
    return trimesh.creation.box(extents, trimesh.transformations.translation_matrix(center))


def _side_slope(x0: float, x1: float, width: float, angle_deg: float) -> trimesh.Trimesh:
    """Prism whose top surface is tilted about the x axis, level (z=0) on the centreline."""
    grade = _deg(angle_deg)
    corners = []
    for x in (x0, x1):
        for y in (0.0, width):
            corners.append((x, y, grade * (y - width / 2)))
            corners.append((x, y, _BOTTOM))
    return trimesh.convex.convex_hull(np.array(corners))


def stair_slope_course(difficulty: float, cfg: StairSlopeCourseCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    length, width = cfg.size
    h, d, n = cfg.step_height, cfg.step_depth, cfg.num_steps
    meshes = []
    x = 0.0

    def flat(seg_len: float, top: float = 0.0):
        nonlocal x
        meshes.append(_block(x, x + seg_len, 0.0, width, top))
        x += seg_len

    flat(cfg.start_length)
    for i in range(n):  # up
        meshes.append(_block(x, x + d, 0.0, width, (i + 1) * h))
        x += d
    flat(cfg.landing_length, n * h)
    for i in range(n - 1):  # down
        meshes.append(_block(x, x + d, 0.0, width, (n - 1 - i) * h))
        x += d
    flat(2.0)
    meshes.append(_side_slope(x, x + cfg.slope_length, width, cfg.slope_deg))
    x += cfg.slope_length
    flat(1.0)
    meshes.append(_side_slope(x, x + cfg.slope_length, width, -cfg.slope_deg))
    x += cfg.slope_length
    flat(max(length - x, 1.0))

    origin = np.array([cfg.start_length / 2.0, width / 2.0, 0.0])
    return meshes, origin


def course_length(cfg: StairSlopeCourseCfg) -> float:
    """x distance from the spawn origin to the end of the second side slope."""
    n, d = cfg.num_steps, cfg.step_depth
    end = cfg.start_length + n * d + cfg.landing_length + (n - 1) * d + 2.0 + 2 * cfg.slope_length + 1.0
    return end - cfg.start_length / 2.0


@configclass
class StairSlopeCourseCfg(SubTerrainBaseCfg):
    function = stair_slope_course
    size: tuple[float, float] = (26.0, 3.0)
    step_height: float = 0.12
    step_depth: float = 0.30
    num_steps: int = 6
    landing_length: float = 2.0
    start_length: float = 3.0
    slope_deg: float = 8.0
    slope_length: float = 4.0


def course_terrain_cfg(step_height: float, slope_deg: float) -> TerrainGeneratorCfg:
    course = StairSlopeCourseCfg(step_height=step_height, slope_deg=slope_deg)
    return TerrainGeneratorCfg(
        size=course.size,
        border_width=5.0,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        curriculum=False,
        use_cache=False,
        sub_terrains={"course": course},
    )
