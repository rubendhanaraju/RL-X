"""JAX/MJX Booster T1 locomotion primitives.

This package is intentionally separate from the legacy NAO/RoboCup behavior
stack.  It starts a faithful port of the NAO Walk_RL3 control abstraction:

    step phase generator -> foot-space residual action -> leg IK -> controls
"""

from .constants import T1Actuator, T1Joint
from .ik import IKSolverConfig, solve_leg_ik, solve_two_leg_ik
from .model import T1MjxMetadata, T1WalkDefaults, load_t1_mjx
from .step_generator import StepGeneratorConfig, StepGeneratorState, init_step_state, step_generator
from .walk_core import (
    WalkCoreConfig,
    WalkCoreState,
    WalkTargets,
    init_walk_core_state,
    jitted_walk_core_step,
    walk_core_step,
)

__all__ = [
    "IKSolverConfig",
    "StepGeneratorConfig",
    "StepGeneratorState",
    "T1Actuator",
    "T1Joint",
    "T1MjxMetadata",
    "T1WalkDefaults",
    "WalkCoreConfig",
    "WalkCoreState",
    "WalkTargets",
    "init_step_state",
    "init_walk_core_state",
    "jitted_walk_core_step",
    "load_t1_mjx",
    "solve_leg_ik",
    "solve_two_leg_ik",
    "step_generator",
    "walk_core_step",
]
