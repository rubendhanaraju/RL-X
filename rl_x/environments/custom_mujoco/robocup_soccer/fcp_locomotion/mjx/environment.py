from functools import partial

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from .box_space import BoxSpace
from .control_functions.handler import get_control_function
from .math_functions.rotation import yaw_from_mat_deg, yaw_to_quat_wxyz
from .observation_functions.handler import get_observation_function
from .reward_functions.handler import get_reward_function
from .state import State
from .target_functions.handler import get_target_function
from .termination_functions.handler import get_termination_function
from .t1_walk.model import load_t1_mjx
from .viewer import MujocoViewer


class FcpLocomotionEnv:
    """Target-walk task using the NAO-style T1 step generator and IK core."""

    def __init__(self, robot_config, runner_mode, render, env_config, nr_envs):
        self.robot_config = robot_config
        self.runner_mode = runner_mode
        self.should_render = render
        self.env_config = env_config
        self.nr_envs = nr_envs

        xml_path = (self.robot_config["directory_path"] / "data" / "plane.xml").as_posix()
        self.control_frequency_hz = int(env_config["control_frequency_hz"])
        self.dt = 1.0 / self.control_frequency_hz
        self.nr_substeps = int(round(self.dt / env_config["timestep"]))
        self.dt = env_config["timestep"] * self.nr_substeps
        self.horizon = int(
            round(env_config["episode_length_in_seconds"] * self.control_frequency_hz)
        )

        self.t1 = load_t1_mjx(
            xml_path,
            control_dt=self.dt,
            timestep=float(env_config["timestep"]),
            p_gain=float(env_config["control"]["p_gain"]),
            d_gain=float(env_config["control"]["d_gain"]),
            solver_iterations=int(env_config["control"]["solver_iterations"]),
            solver_ls_iterations=int(env_config["control"]["solver_ls_iterations"]),
        )
        self.initial_mj_model = self.t1.mj_model
        self.initial_mjx_model = self.t1.mjx_model
        self.mjx_data = mjx.forward(self.initial_mjx_model, self.t1.mjx_data)

        self.gravity_world = jnp.array([0.0, 0.0, -1.0], dtype=jnp.float32)
        self.initial_qpos = jnp.array(self.initial_mj_model.keyframe("home").qpos)

        self.trunk_body_id = mujoco.mj_name2id(
            self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, "trunk"
        )
        self.head_body_id = mujoco.mj_name2id(
            self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, "H2"
        )
        self.imu_site_id = mujoco.mj_name2id(
            self.initial_mj_model, mujoco.mjtObj.mjOBJ_SITE, "imu"
        )
        self.nominal_imu_height = self.mjx_data.site_xpos[self.imu_site_id, 2]
        self.floor_geom_id = mujoco.mj_name2id(
            self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        self.foot_geom_indices = jnp.array(
            [
                mujoco.mj_name2id(
                    self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot"
                ),
                mujoco.mj_name2id(
                    self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot"
                ),
            ],
            dtype=jnp.int32,
        )
        self.foot_site_indices = jnp.array(
            [self.t1.ids.left.site_id, self.t1.ids.right.site_id],
            dtype=jnp.int32,
        )

        self.actuator_joint_names = [
            mujoco.mj_id2name(
                self.initial_mj_model, mujoco.mjtObj.mjOBJ_JOINT, actuator_trnid[0]
            )
            for actuator_trnid in self.initial_mj_model.actuator_trnid
        ]
        self.actuator_joint_mask_joints = jnp.array(
            [
                self.initial_mj_model.joint(joint_name).id
                for joint_name in self.actuator_joint_names
            ],
            dtype=jnp.int32,
        )
        self.actuator_joint_mask_qpos = jnp.array(
            [
                self.initial_mj_model.joint(joint_name).qposadr[0]
                for joint_name in self.actuator_joint_names
            ],
            dtype=jnp.int32,
        )
        self.actuator_joint_mask_qvel = jnp.array(
            [
                self.initial_mj_model.joint(joint_name).dofadr[0]
                for joint_name in self.actuator_joint_names
            ],
            dtype=jnp.int32,
        )
        self.nr_actuator_joints = len(self.actuator_joint_names)
        self.nr_walk_actions = 16
        self.actuator_joint_limits = jnp.array(
            self.initial_mj_model.jnt_range[self.actuator_joint_mask_joints],
            dtype=jnp.float32,
        )
        self.actuator_joint_midpoints = jnp.mean(self.actuator_joint_limits, axis=1)
        self.actuator_joint_half_ranges = jnp.maximum(
            (self.actuator_joint_limits[:, 1] - self.actuator_joint_limits[:, 0]) / 2.0,
            1e-6,
        )
        self.actuator_joint_max_velocities = jnp.array(
            robot_config["actuator_joint_max_velocities"], dtype=jnp.float32
        )
        self.walk_rl3_arm_ctrl_ids = jnp.array([2, 6, 3, 7], dtype=jnp.int32)
        self.walk_rl3_arm_qpos_ids = self.actuator_joint_mask_qpos[
            self.walk_rl3_arm_ctrl_ids
        ]
        self.walk_rl3_speed_ctrl_ids = jnp.concatenate(
            [
                self.t1.ids.left_leg_ctrl_ids,
                self.t1.ids.right_leg_ctrl_ids,
                self.walk_rl3_arm_ctrl_ids,
            ]
        )

        imu_angular_velocity_sensor_id = self.initial_mj_model.sensor(
            "imu_angular_velocity"
        ).id
        self.imu_angular_velocity_sensor_adr = self.initial_mj_model.sensor_adr[
            imu_angular_velocity_sensor_id
        ]
        self.imu_angular_velocity_sensor_dim = self.initial_mj_model.sensor_dim[
            imu_angular_velocity_sensor_id
        ]
        imu_linear_velocity_sensor_id = self.initial_mj_model.sensor(
            "imu_linear_velocity"
        ).id
        self.imu_linear_velocity_sensor_adr = self.initial_mj_model.sensor_adr[
            imu_linear_velocity_sensor_id
        ]
        self.imu_linear_velocity_sensor_dim = self.initial_mj_model.sensor_dim[
            imu_linear_velocity_sensor_id
        ]

        self.initial_ctrl = self.initial_qpos[self.actuator_joint_mask_qpos]
        self.nominal_root_height = jnp.float32(env_config["reset"]["root_position_xyz"][2])
        self.reset_root_position = jnp.array(
            env_config["reset"]["root_position_xyz"], dtype=jnp.float32
        )
        self.reset_random_yaw = bool(env_config["reset"]["random_yaw"])
        self.reset_settle_steps = int(env_config["reset"]["settle_steps"])

        self.control_function = get_control_function(env_config["walk"]["type"], self)
        self.target_function = get_target_function(env_config["target"]["type"], self)
        self.reward_function = get_reward_function(env_config["reward"]["type"], self)
        self.termination_function = get_termination_function(
            env_config["termination"]["type"], self
        )
        self.observation_function = get_observation_function(
            env_config["observation"]["type"], self
        )
        self.history_length = self.observation_function.history_length
        self.base_observation_dim = self.observation_function.base_observation_dim

        self.single_action_space = BoxSpace(
            low=-jnp.ones(self.nr_walk_actions, dtype=jnp.float32),
            high=jnp.ones(self.nr_walk_actions, dtype=jnp.float32),
            shape=(self.nr_walk_actions,),
            dtype=jnp.float32,
            center=jnp.zeros(self.nr_walk_actions, dtype=jnp.float32),
            scale=jnp.ones(self.nr_walk_actions, dtype=jnp.float32),
        )
        self.single_observation_space = self.get_observation_space()

        if self.should_render:
            import pygame

            self.pygame = pygame
            self.viewer = MujocoViewer(self.initial_mj_model, self.dt)
            self.pygame.init()

    def render(self, state):
        data = mjx.get_data(self.viewer.model, state.data)[0]
        self.viewer.render(data)
        return state

    def _empty_info(self):
        return self.reward_function.empty_info()

    @partial(jax.vmap, in_axes=(None, 0, None))
    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, eval_mode):
        next_observation = jnp.zeros(self.single_observation_space.shape, dtype=jnp.float32)
        reward = jnp.float32(0.0)
        terminated = jnp.bool_(False)
        truncated = jnp.bool_(False)

        state = State(
            self.initial_mjx_model,
            self.mjx_data,
            next_observation,
            next_observation,
            reward,
            terminated,
            truncated,
            self._empty_info(),
            {
                "episode_return": jnp.float32(0.0),
                "episode_step": jnp.int32(0),
            },
            {
                "in_eval_mode": jnp.bool_(eval_mode),
                "last_action": jnp.zeros(self.nr_walk_actions, dtype=jnp.float32),
                "second_last_action": jnp.zeros(self.nr_walk_actions, dtype=jnp.float32),
                "obs_history": jnp.zeros(
                    (self.history_length, self.base_observation_dim), dtype=jnp.float32
                ),
                "previous_head_z": jnp.float32(0.0),
                "previous_imu_linear_velocity": jnp.zeros(3, dtype=jnp.float32),
                "previous_forward_x": jnp.float32(0.0),
                "walk_core_state": self.control_function.init_state(),
                "last_joint_target_speed": jnp.zeros(16, dtype=jnp.float32),
                "virtual_target": jnp.zeros(2, dtype=jnp.float32),
                "virtual_target_velocity": jnp.zeros(2, dtype=jnp.float32),
                "virtual_orientation": jnp.float32(0.0),
                "virtual_orientation_speed": jnp.float32(0.0),
                "virtual_orientation_ignore": jnp.bool_(True),
                "internal_target": jnp.zeros(2, dtype=jnp.float32),
                "internal_rel_orientation": jnp.float32(0.0),
                "internal_target_velocity": jnp.zeros(2, dtype=jnp.float32),
                "internal_abs_target": jnp.zeros(2, dtype=jnp.float32),
                "internal_linear_distance": jnp.float32(0.0),
                "internal_abs_orientation": jnp.float32(0.0),
            },
            key,
        )

        return self._reset(state)

    @partial(jax.vmap, in_axes=(None, 0, 0))
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        return self._step(state, action)

    @partial(jax.jit, static_argnums=(0,))
    def _reset(self, state):
        key, reset_key, target_key = jax.random.split(state.key, 3)
        state = state.replace(key=key)

        qpos, qvel = self._sample_reset_state(reset_key)
        data = self.mjx_data.replace(qpos=qpos, qvel=qvel, ctrl=self.initial_ctrl)
        data = mjx.forward(state.mjx_model, data)
        data = self._settle_data(state.mjx_model, data)

        last_action = jnp.zeros(self.nr_walk_actions, dtype=jnp.float32)
        obs_history = jnp.zeros(
            (self.history_length, self.base_observation_dim), dtype=jnp.float32
        )
        walk_core_state = self.control_function.init_state()
        target_state = self.target_function.reset_state(
            data, target_key, state.internal_state["in_eval_mode"]
        )
        internal_state = {
            "in_eval_mode": state.internal_state["in_eval_mode"],
            "last_action": last_action,
            "second_last_action": last_action,
            "obs_history": obs_history,
            "previous_head_z": data.xpos[self.head_body_id, 2],
            "previous_imu_linear_velocity": jnp.zeros(3, dtype=jnp.float32),
            "previous_forward_x": data.qpos[0].astype(jnp.float32),
            "walk_core_state": walk_core_state,
            "last_joint_target_speed": jnp.zeros(16, dtype=jnp.float32),
            **target_state,
        }
        (
            internal_target,
            internal_rel_orientation,
            internal_target_velocity,
            _walk_rel_target,
            _walk_distance,
            _walk_rel_orientation,
        ) = self.target_function.observe_update(data, internal_state, jnp.bool_(True))
        internal_state["internal_target"] = internal_target
        internal_state["internal_rel_orientation"] = internal_rel_orientation
        internal_state["internal_target_velocity"] = internal_target_velocity
        internal_state["internal_abs_target"] = self.target_function.internal_abs_target(
            data, internal_target
        )
        internal_state["internal_linear_distance"] = jnp.linalg.norm(internal_target)
        internal_state["internal_abs_orientation"] = (
            self.target_function.internal_abs_orientation(data, internal_rel_orientation)
        )

        current_base_observation, current_head_z, current_imu_linear_velocity = (
            self.observation_function.build_current_base_observation(
                data=data,
                init=jnp.bool_(True),
                step_counter=jnp.int32(0),
                internal_target=internal_target,
                internal_rel_orientation=internal_rel_orientation,
                internal_target_velocity=internal_target_velocity,
                walk_core_state=walk_core_state,
                last_joint_target_speed=internal_state["last_joint_target_speed"],
                previous_head_z=internal_state["previous_head_z"],
                previous_imu_linear_velocity=internal_state[
                    "previous_imu_linear_velocity"
                ],
            )
        )
        next_observation = self.observation_function.compose(
            current_base_observation, obs_history
        )
        internal_state["previous_head_z"] = current_head_z
        internal_state["previous_imu_linear_velocity"] = current_imu_linear_velocity

        return state.replace(
            data=data,
            next_observation=next_observation,
            actual_next_observation=next_observation,
            reward=jnp.float32(0.0),
            terminated=jnp.bool_(False),
            truncated=jnp.bool_(False),
            info=self._empty_info(),
            info_episode_store={
                "episode_return": jnp.float32(0.0),
                "episode_step": jnp.int32(0),
            },
            internal_state=internal_state,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, state, action):
        key, target_key = jax.random.split(state.key, 2)
        state = state.replace(key=key)

        chosen_action = jnp.clip(action[: self.nr_walk_actions], -1.0, 1.0)
        previous_action = state.internal_state["last_action"]

        data, walk_core_state, _ = self.control_function.process_action(
            state.mjx_model,
            state.data,
            state.internal_state["walk_core_state"],
            chosen_action,
            state.internal_state["internal_target"],
        )
        last_joint_target_speed = jnp.clip(
            (data.ctrl[self.walk_rl3_speed_ctrl_ids] - state.data.ctrl[self.walk_rl3_speed_ctrl_ids])
            / self.dt,
            -6.1395,
            6.1395,
        )
        data = self._apply_control_targets(state.mjx_model, data)

        episode_step = state.info_episode_store["episode_step"] + 1
        terminated = self.termination_function.should_terminate(
            data,
            walk_core_state.step_counter,
            state.internal_state["in_eval_mode"],
        )
        reward, reward_info = self.reward_function.reward_and_info(
            data=data,
            action=chosen_action,
            internal_abs_target=state.internal_state["internal_abs_target"],
            internal_linear_distance=state.internal_state["internal_linear_distance"],
            internal_abs_orientation=state.internal_state["internal_abs_orientation"],
            previous_forward_x=state.internal_state["previous_forward_x"],
        )

        random_target_state = self.target_function.update_virtual_target(
            state.internal_state, target_key
        )
        eval_target_state = self.target_function.evaluation_update(
            state.internal_state, walk_core_state.step_counter
        )
        virtual_target_state = jax.lax.cond(
            state.internal_state["in_eval_mode"],
            lambda _: eval_target_state,
            lambda _: random_target_state,
            operand=None,
        )
        target_internal_state = {
            **state.internal_state,
            **virtual_target_state,
        }
        (
            internal_target,
            internal_rel_orientation,
            internal_target_velocity,
            _walk_rel_target,
            _walk_distance,
            _walk_rel_orientation,
        ) = self.target_function.observe_update(
            data, target_internal_state, jnp.bool_(False)
        )
        internal_abs_target = self.target_function.internal_abs_target(
            data, internal_target
        )
        internal_abs_orientation = self.target_function.internal_abs_orientation(
            data, internal_rel_orientation
        )

        current_base_observation, current_head_z, current_imu_linear_velocity = (
            self.observation_function.build_current_base_observation(
                data=data,
                init=jnp.bool_(False),
                step_counter=walk_core_state.step_counter,
                internal_target=internal_target,
                internal_rel_orientation=internal_rel_orientation,
                internal_target_velocity=internal_target_velocity,
                walk_core_state=walk_core_state,
                last_joint_target_speed=last_joint_target_speed,
                previous_head_z=state.internal_state["previous_head_z"],
                previous_imu_linear_velocity=state.internal_state[
                    "previous_imu_linear_velocity"
                ],
            )
        )
        next_observation = self.observation_function.compose(
            current_base_observation, state.internal_state["obs_history"]
        )
        next_obs_history = self.observation_function.push_history(
            state.internal_state["obs_history"], current_base_observation
        )

        episode_return = state.info_episode_store["episode_return"] + reward
        truncated = episode_step >= self.horizon
        done = terminated | truncated

        transition_info = {
            "rollout/episode_return": jnp.where(
                done, episode_return, state.info["rollout/episode_return"]
            ),
            "rollout/episode_length": jnp.where(
                done, episode_step, state.info["rollout/episode_length"]
            ),
            "env_info/internal_linear_distance": reward_info["internal_linear_distance"],
            "env_info/linear_distance": reward_info["linear_distance"],
            "env_info/angular_distance": reward_info["angular_distance"],
            "env_info/root_height": data.site_xpos[self.imu_site_id, 2],
            "env_info/forward_x": data.qpos[0].astype(jnp.float32),
            "reward/total": reward,
            "reward/progress": reward_info["progress"],
            "reward/orientation_multiplier": reward_info["orientation_multiplier"],
            "reward/idle": reward_info["idle"],
            "reward/forward_displacement": reward_info["forward_displacement"],
        }
        next_internal_state = {
            "in_eval_mode": state.internal_state["in_eval_mode"],
            "last_action": chosen_action,
            "second_last_action": previous_action,
            "obs_history": next_obs_history,
            "previous_head_z": current_head_z,
            "previous_imu_linear_velocity": current_imu_linear_velocity,
            "previous_forward_x": data.qpos[0].astype(jnp.float32),
            "walk_core_state": walk_core_state,
            "last_joint_target_speed": last_joint_target_speed,
            "virtual_target": virtual_target_state["virtual_target"],
            "virtual_target_velocity": virtual_target_state["virtual_target_velocity"],
            "virtual_orientation": virtual_target_state["virtual_orientation"],
            "virtual_orientation_speed": virtual_target_state[
                "virtual_orientation_speed"
            ],
            "virtual_orientation_ignore": virtual_target_state[
                "virtual_orientation_ignore"
            ],
            "internal_target": internal_target,
            "internal_rel_orientation": internal_rel_orientation,
            "internal_target_velocity": internal_target_velocity,
            "internal_abs_target": internal_abs_target,
            "internal_linear_distance": jnp.linalg.norm(internal_target),
            "internal_abs_orientation": internal_abs_orientation,
        }
        next_info_episode_store = {
            "episode_return": episode_return,
            "episode_step": episode_step,
        }

        def when_done(_):
            start_state = self._reset(state)
            return start_state.replace(
                actual_next_observation=next_observation,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=transition_info,
            )

        def when_not_done(_):
            return state.replace(
                data=data,
                next_observation=next_observation,
                actual_next_observation=next_observation,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=transition_info,
                info_episode_store=next_info_episode_store,
                internal_state=next_internal_state,
            )

        return jax.lax.cond(done, when_done, when_not_done, operand=None)

    def _sample_reset_state(self, key):
        qpos = self.initial_qpos
        qpos = qpos.at[:3].set(self.reset_root_position)
        if self.reset_random_yaw:
            yaw = jax.random.uniform(key, minval=-jnp.pi, maxval=jnp.pi)
        else:
            yaw = jnp.float32(0.0)
        qpos = qpos.at[3:7].set(yaw_to_quat_wxyz(yaw))
        qpos = qpos.at[self.actuator_joint_mask_qpos].set(self.initial_ctrl)
        qvel = jnp.zeros(self.initial_mj_model.nv, dtype=jnp.float32)
        return qpos, qvel

    def _settle_data(self, mjx_model, data):
        def settle_fn(settle_data, _):
            next_data = self._apply_fixed_ctrl(mjx_model, settle_data, self.initial_ctrl)
            return next_data, None

        data, _ = jax.lax.scan(settle_fn, data, xs=None, length=self.reset_settle_steps)
        return data

    def _apply_control_targets(self, mjx_model, data):
        data = self._apply_fixed_ctrl(mjx_model, data, data.ctrl)
        max_qvel = 100.0 * jnp.ones(self.initial_mj_model.nv, dtype=jnp.float32)
        max_qvel = max_qvel.at[self.actuator_joint_mask_qvel].set(
            self.actuator_joint_max_velocities
        )
        return data.replace(qvel=jnp.clip(data.qvel, -max_qvel, max_qvel))

    def _apply_fixed_ctrl(self, mjx_model, data, ctrl):
        def substep_fn(step_data, _):
            step_data = step_data.replace(ctrl=ctrl)
            return mjx.step(mjx_model, step_data), None

        data, _ = jax.lax.scan(substep_fn, data, xs=None, length=self.nr_substeps)
        return data

    def get_torso_yaw_deg(self, data):
        return yaw_from_mat_deg(data.xmat[self.trunk_body_id])

    def get_observation_space(self):
        observation_space = self.observation_function.get_observation_space()
        observation_dim = observation_space.shape[0]
        self.policy_observation_indices = jnp.arange(observation_dim, dtype=int)
        self.critic_observation_indices = jnp.arange(observation_dim, dtype=int)
        return observation_space

    def close(self):
        if self.should_render:
            self.viewer.close()
            self.pygame.quit()
