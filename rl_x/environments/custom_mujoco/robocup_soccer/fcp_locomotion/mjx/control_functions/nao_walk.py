import jax.numpy as jnp

from ..t1_walk.ik import IKSolverConfig
from ..t1_walk.walk_core import (
    WalkCoreConfig,
    init_walk_core_state,
    walk_core_step,
)


class NaoWalkControl:
    def __init__(self, env):
        walk_config = env.env_config["walk"]
        self.env = env
        self.ts_per_step = jnp.array(walk_config["ts_per_step"], dtype=jnp.int32)
        self.z_span = jnp.array(walk_config["swing_height"], dtype=jnp.float32)
        self.z_extension = (
            jnp.array(walk_config["z_extension"], dtype=jnp.float32)
            if "z_extension" in walk_config
            else jnp.array(env.t1.defaults.z_extension, dtype=jnp.float32)
        )
        feet_y_dev_scale = jnp.array(
            walk_config.get("feet_y_dev_scale", 1.0), dtype=jnp.float32
        )
        z0_override = float(walk_config.get("z0", -1.0))
        gravity_override = float(walk_config.get("gravity", -1.0))
        self.step_config = env.t1.defaults.step_config._replace(
            feet_y_dev=env.t1.defaults.step_config.feet_y_dev * feet_y_dev_scale,
            z0=(
                jnp.array(z0_override, dtype=jnp.float32)
                if z0_override > 0.0
                else env.t1.defaults.step_config.z0
            ),
            gravity=(
                jnp.array(gravity_override, dtype=jnp.float32)
                if gravity_override > 0.0
                else env.t1.defaults.step_config.gravity
            ),
        )
        self.walk_config = WalkCoreConfig(
            left_foot_home_waist=env.t1.defaults.left_foot_home_waist,
            right_foot_home_waist=env.t1.defaults.right_foot_home_waist,
            action_scale=jnp.array(walk_config["action_scale"], dtype=jnp.float32),
            action_smoothing_new_weight=jnp.array(
                walk_config.get("action_smoothing_new_weight", 0.2),
                dtype=jnp.float32,
            ),
            foot_x_bias=jnp.array(
                walk_config.get("foot_x_bias", 0.0), dtype=jnp.float32
            ),
            foot_position_scales=jnp.array(
                walk_config.get("foot_position_scales", [0.02, 0.02, 0.01]),
                dtype=jnp.float32,
            ),
            foot_rotation_scales_deg=jnp.array(
                walk_config.get("foot_rotation_scales_deg", [3.0, 3.0, 5.0]),
                dtype=jnp.float32,
            ),
            foot_yaw_bias_deg=jnp.array(
                walk_config.get("foot_yaw_bias_deg", 7.0), dtype=jnp.float32
            ),
            min_abs_foot_y=jnp.array(
                walk_config.get("min_abs_foot_y", 0.01), dtype=jnp.float32
            ),
            arm_base=jnp.array(
                walk_config.get(
                    "arm_base",
                    [0.0, -1.4, 0.0, -0.4, 0.0, 1.4, 0.0, 0.4],
                ),
                dtype=jnp.float32,
            ),
            arm_swing_scale=jnp.array(
                walk_config.get("arm_swing_scale", 0.10), dtype=jnp.float32
            ),
            arm_action_scale=jnp.array(
                walk_config.get("arm_action_scale", 0.08), dtype=jnp.float32
            ),
        )
        self.ik_config = IKSolverConfig(
            iterations=int(walk_config["ik_iterations"]),
            damping=float(walk_config["ik_damping"]),
            max_delta=float(walk_config["ik_max_delta"]),
            rotation_weight=float(walk_config["ik_rotation_weight"]),
        )

    def init_state(self):
        return init_walk_core_state(
            int(self.ts_per_step),
            float(self.z_span),
            float(self.z_extension),
        )

    def process_action(
        self,
        mjx_model,
        data,
        walk_core_state,
        action,
        internal_target,
        reset=jnp.bool_(False),
    ):
        internal_dist = jnp.linalg.norm(internal_target)
        action_mult = jnp.where(
            internal_dist > 0.2,
            1.0,
            (0.7 / 0.2) * internal_dist + 0.3,
        )
        data, walk_core_state, targets = walk_core_step(
            mjx_model,
            data,
            self.env.t1.ids,
            walk_core_state,
            action * action_mult,
            reset,
            self.ts_per_step,
            self.z_span,
            self.z_extension,
            self.step_config,
            self.walk_config,
            self.ik_config,
        )
        return data, walk_core_state, targets
