import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint
from flax.training import orbax_utils
from orbax.checkpoint import args as orbax_args

from rl_x.algorithms.ppo_gru.flax_full_jit.default_config import (
    get_config as get_algorithm_config,
)
from rl_x.algorithms.ppo_gru.flax_full_jit.policy import get_policy
from rl_x.environments.custom_mujoco.robocup_soccer.fcp_locomotion.mjx.t1_walk.constants import (
    LEFT_LEG_ACTUATORS,
    RIGHT_LEG_ACTUATORS,
)
from rl_x.environments.custom_mujoco.robocup_soccer.locomotion.mjx.create_env import (
    create_train_and_eval_env,
)
from rl_x.environments.custom_mujoco.robocup_soccer.locomotion.mjx.default_config import (
    get_config as get_environment_config,
)


DEFAULT_CHECKPOINT = (
    "rl_x/environments/custom_mujoco/robocup_soccer/latest.model"
)
DEFAULT_OUTPUT_DIR = (
    "rl_x/environments/custom_mujoco/robocup_soccer/"
    "fcp_locomotion/mjx/t1_walk/assets"
)
DEFAULT_PREFIX = "robocup_soccer_locomotion_forward_cycle"
FCP_RESIDUAL_CTRL_IDS = np.array(
    list(LEFT_LEG_ACTUATORS) + list(RIGHT_LEG_ACTUATORS) + [2, 6, 3, 7],
    dtype=np.int32,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Distill a phase-indexed forward gait cycle from the root "
            "robocup_soccer PPO-GRU locomotion checkpoint."
        )
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--cmd-vx", type=float, default=0.5)
    parser.add_argument("--cmd-vy", type=float, default=0.0)
    parser.add_argument("--cmd-wz", type=float, default=0.0)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--cycles", type=float, default=4.0)
    parser.add_argument("--warmup-steps", type=int, default=250)
    parser.add_argument("--collect-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    parser.add_argument(
        "--clean-rollout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable command resampling, noise, action delay, and random pushes.",
    )
    parser.add_argument(
        "--eval-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic eval gait phase/frequency.",
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write a small diagnostic PNG.",
    )
    return parser.parse_args()


def _zero_numeric_leaves(section):
    for key in section:
        value = section[key]
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            section[key] = 0
        elif isinstance(value, float):
            section[key] = 0.0


def make_clean_rollout_config(env_config):
    env_config.command.sampling_type = "none"

    dr = env_config.domain_randomization
    dr.sampling_type = "none"
    dr.initial_state.type = "default"
    dr.action_delay.type = "none"
    dr.observation_noise.type = "none"
    dr.perturbation.type = "none"
    dr.perturbation.sampling_type = "none"
    dr.joint_dropout.type = "none"

    # Keep the default seen/unseen functions because they initialize required
    # internal fields such as scaling_factor and nominal robot height.
    _zero_numeric_leaves(dr.mujoco_model)
    _zero_numeric_leaves(dr.seen_robot)
    _zero_numeric_leaves(dr.unseen_robot)

    return env_config


def make_env(args):
    env_config = get_environment_config(
        "custom_mujoco.robocup_soccer.locomotion.mjx"
    )
    env_config.nr_envs = 1
    env_config.seed = args.seed
    env_config.render = False
    env_config.device = args.device
    if args.clean_rollout:
        env_config = make_clean_rollout_config(env_config)

    config = SimpleNamespace(
        environment=env_config,
        runner=SimpleNamespace(mode="test"),
    )
    env, _ = create_train_and_eval_env(config)
    return env_config, env


def load_policy_params(checkpoint_path, config, env, initial_observation):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    tmp_dir = tempfile.mkdtemp(prefix="rlx_locomotion_cycle_")
    try:
        shutil.unpack_archive(checkpoint_path.as_posix(), tmp_dir, "zip")
        with open(Path(tmp_dir) / "config_algorithm.json", "r") as config_file:
            loaded_algorithm_config = json.load(config_file)
        for key, value in loaded_algorithm_config.items():
            if key in config.algorithm:
                config.algorithm[key] = value

        policy, process_action = get_policy(config, env)
        dummy_obs = jnp.asarray(initial_observation, dtype=jnp.float32)
        dummy_carry = policy.initialize_carry(1)
        init_params = policy.init(
            jax.random.PRNGKey(config.environment.seed),
            dummy_obs,
            dummy_carry,
            method=policy.apply_one_step,
        )
        target = {"policy": {"params": init_params}}
        restore_args = orbax_utils.restore_args_from_target(target)
        restored = orbax.checkpoint.PyTreeCheckpointer().restore(
            tmp_dir,
            args=orbax_args.PyTreeRestore(
                item=target,
                restore_args=restore_args,
                partial_restore=True,
            ),
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return policy, process_action, restored["policy"]["params"]


def make_fixed_command_fn(env, command):
    @jax.jit
    def apply_fixed_command(state):
        def per_env(data, mjx_model, internal_state, key):
            goal_velocities = jnp.clip(
                command,
                -internal_state["max_command_velocity"],
                internal_state["max_command_velocity"],
            )
            goal_velocities = jnp.where(
                jnp.abs(goal_velocities)
                < (
                    env.command_function.zero_clip_threshold_percentage
                    * internal_state["max_command_velocity"]
                ),
                0.0,
                goal_velocities,
            )
            internal_state["goal_velocities"] = goal_velocities
            internal_state["actuator_joint_keep_nominal"] = jnp.where(
                jnp.all(goal_velocities == 0.0),
                jnp.ones(env.nr_actuator_joints, dtype=bool),
                env.command_function.default_actuator_joint_keep_nominal,
            )
            observation = env.get_observation(
                data,
                mjx_model,
                internal_state,
                key,
                internal_state["last_action"],
            )
            return internal_state, observation

        internal_state, observation = jax.vmap(per_env)(
            state.data,
            state.mjx_model,
            state.internal_state,
            state.key,
        )
        return state.replace(
            internal_state=internal_state,
            next_observation=observation,
            actual_next_observation=observation,
        )

    return apply_fixed_command


def first_env(value):
    return np.asarray(value)[0]


def body_frame_positions(world_positions, body_position, body_xmat):
    body_rotation = body_xmat.reshape(3, 3)
    return (world_positions - body_position) @ body_rotation


def extract_sample(env, state, action_mean, action, step):
    data = state.data
    qpos = first_env(data.qpos)
    qvel = first_env(data.qvel)
    ctrl = first_env(data.ctrl)
    sensordata = first_env(data.sensordata)

    phase = first_env(state.internal_state["gait_phase"])
    phase_fraction = ((phase[0] + np.pi) % (2.0 * np.pi)) / (2.0 * np.pi)
    foot_world = np.asarray(first_env(data.geom_xpos) [np.asarray(env.foot_geom_indices)])
    trunk_pos = np.asarray(first_env(data.xpos)[int(env.trunk_body_id)])
    trunk_xmat = np.asarray(first_env(data.xmat)[int(env.trunk_body_id)])
    foot_trunk = body_frame_positions(foot_world, trunk_pos, trunk_xmat)

    observation = first_env(state.next_observation)
    feet_contact = (observation[np.asarray(env.feet_ground_contact_obs_idx)] + 1.0) * 0.5

    return {
        "step": np.asarray(step, dtype=np.int32),
        "phase_fraction": np.asarray(phase_fraction, dtype=np.float32),
        "gait_phase": phase.astype(np.float32),
        "goal_velocity": first_env(state.internal_state["goal_velocities"]).astype(
            np.float32
        ),
        "action_mean": first_env(action_mean).astype(np.float32),
        "action": first_env(action).astype(np.float32),
        "ctrl": ctrl.astype(np.float32),
        "joint_position": qpos[np.asarray(env.actuator_joint_mask_qpos)].astype(
            np.float32
        ),
        "joint_velocity": qvel[np.asarray(env.actuator_joint_mask_qvel)].astype(
            np.float32
        ),
        "fcp_residual_ctrl": ctrl[FCP_RESIDUAL_CTRL_IDS].astype(np.float32),
        "root_position": qpos[:3].astype(np.float32),
        "root_quat": qpos[3:7].astype(np.float32),
        "root_linear_velocity": qvel[:3].astype(np.float32),
        "root_angular_velocity": qvel[3:6].astype(np.float32),
        "imu_linear_velocity": sensordata[
            int(env.imu_linear_velocity_sensor_adr) : int(
                env.imu_linear_velocity_sensor_adr
                + env.imu_linear_velocity_sensor_dim
            )
        ].astype(np.float32),
        "imu_angular_velocity": sensordata[
            int(env.imu_angular_velocity_sensor_adr) : int(
                env.imu_angular_velocity_sensor_adr
                + env.imu_angular_velocity_sensor_dim
            )
        ].astype(np.float32),
        "foot_world": foot_world.astype(np.float32),
        "foot_trunk": foot_trunk.astype(np.float32),
        "feet_contact": feet_contact.astype(np.float32),
        "reward": first_env(state.reward).astype(np.float32),
    }


def stack_samples(samples):
    keys = samples[0].keys()
    return {key: np.stack([sample[key] for sample in samples], axis=0) for key in keys}


def aggregate_by_phase(rollout, frame_count):
    phase = rollout["phase_fraction"] % 1.0
    bin_ids = np.floor(phase * frame_count).astype(np.int32) % frame_count
    centers = (np.arange(frame_count, dtype=np.float32) + 0.5) / frame_count
    counts = np.zeros(frame_count, dtype=np.int32)

    frame_data = {}
    for key, values in rollout.items():
        if key in {"step", "phase_fraction", "gait_phase"}:
            continue
        frames = []
        for frame_id, center in enumerate(centers):
            matching = np.flatnonzero(bin_ids == frame_id)
            if matching.size == 0:
                circular_dist = np.abs(((phase - center + 0.5) % 1.0) - 0.5)
                matching = np.array([int(np.argmin(circular_dist))], dtype=np.int32)
            else:
                counts[frame_id] = matching.size
            frames.append(values[matching].mean(axis=0))
        frame_data[key] = np.stack(frames, axis=0).astype(np.float32)

    phase_left = (centers * 2.0 * np.pi - np.pi).astype(np.float32)
    phase_right = ((phase_left - np.pi + np.pi) % (2.0 * np.pi) - np.pi).astype(
        np.float32
    )
    return {
        "phase_fraction": centers.astype(np.float32),
        "phase_left": phase_left,
        "phase_right": phase_right,
        "source_count": counts,
        **frame_data,
    }


def sanitize(name):
    return str(name).replace("/", "_").replace(" ", "_").replace("-", "_")


def add_vector(row, prefix, values, names=None):
    values = np.asarray(values)
    if names is None:
        names = [str(i) for i in range(values.shape[0])]
    for name, value in zip(names, values):
        row[f"{prefix}_{sanitize(name)}"] = float(value)


def add_xyz(row, prefix, value):
    for axis, scalar in zip(("x", "y", "z"), value):
        row[f"{prefix}_{axis}"] = float(scalar)


def write_csv(path, frames, actuator_names, fcp_names, foot_names):
    rows = []
    for frame_id in range(frames["phase_fraction"].shape[0]):
        row = {
            "frame": frame_id,
            "phase_fraction": float(frames["phase_fraction"][frame_id]),
            "phase_left_rad": float(frames["phase_left"][frame_id]),
            "phase_right_rad": float(frames["phase_right"][frame_id]),
            "source_count": int(frames["source_count"][frame_id]),
            "reward": float(frames["reward"][frame_id]),
        }
        add_vector(row, "action", frames["action"][frame_id], actuator_names)
        add_vector(row, "ctrl", frames["ctrl"][frame_id], actuator_names)
        add_vector(
            row,
            "joint_position",
            frames["joint_position"][frame_id],
            actuator_names,
        )
        add_vector(
            row,
            "joint_velocity",
            frames["joint_velocity"][frame_id],
            actuator_names,
        )
        add_vector(
            row,
            "fcp_residual_ctrl",
            frames["fcp_residual_ctrl"][frame_id],
            fcp_names,
        )
        add_xyz(row, "root_position", frames["root_position"][frame_id])
        add_xyz(row, "root_linear_velocity", frames["root_linear_velocity"][frame_id])
        add_xyz(row, "root_angular_velocity", frames["root_angular_velocity"][frame_id])
        add_xyz(row, "imu_linear_velocity", frames["imu_linear_velocity"][frame_id])
        add_xyz(row, "imu_angular_velocity", frames["imu_angular_velocity"][frame_id])
        add_vector(row, "goal_velocity", frames["goal_velocity"][frame_id], ["x", "y", "yaw"])
        add_vector(row, "feet_contact", frames["feet_contact"][frame_id], foot_names)
        for foot_id, foot_name in enumerate(foot_names):
            add_xyz(
                row,
                f"foot_world_{sanitize(foot_name)}",
                frames["foot_world"][frame_id, foot_id],
            )
            add_xyz(
                row,
                f"foot_trunk_{sanitize(foot_name)}",
                frames["foot_trunk"][frame_id, foot_id],
            )
        rows.append(row)

    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_npz(path, frames, rollout, metadata):
    np.savez_compressed(
        path,
        metadata_json=np.asarray(json.dumps(metadata, indent=2)),
        frame_phase_fraction=frames["phase_fraction"],
        frame_phase_left=frames["phase_left"],
        frame_phase_right=frames["phase_right"],
        frame_source_count=frames["source_count"],
        frame_action=frames["action"],
        frame_action_mean=frames["action_mean"],
        frame_ctrl=frames["ctrl"],
        frame_joint_position=frames["joint_position"],
        frame_joint_velocity=frames["joint_velocity"],
        frame_fcp_residual_ctrl=frames["fcp_residual_ctrl"],
        frame_root_position=frames["root_position"],
        frame_root_quat=frames["root_quat"],
        frame_root_linear_velocity=frames["root_linear_velocity"],
        frame_root_angular_velocity=frames["root_angular_velocity"],
        frame_imu_linear_velocity=frames["imu_linear_velocity"],
        frame_imu_angular_velocity=frames["imu_angular_velocity"],
        frame_goal_velocity=frames["goal_velocity"],
        frame_foot_world=frames["foot_world"],
        frame_foot_trunk=frames["foot_trunk"],
        frame_feet_contact=frames["feet_contact"],
        frame_reward=frames["reward"],
        rollout_phase_fraction=rollout["phase_fraction"],
        rollout_gait_phase=rollout["gait_phase"],
        rollout_ctrl=rollout["ctrl"],
        rollout_action=rollout["action"],
        actuator_joint_names=np.asarray(metadata["actuator_joint_names"]),
        foot_names=np.asarray(metadata["foot_names"]),
        fcp_residual_ctrl_ids=np.asarray(metadata["fcp_residual_ctrl_ids"], dtype=np.int32),
        fcp_residual_ctrl_names=np.asarray(metadata["fcp_residual_ctrl_names"]),
    )


def maybe_write_plot(path, frames, foot_names):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plot, matplotlib unavailable: {exc}")
        return

    phase = frames["phase_fraction"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for foot_id, foot_name in enumerate(foot_names):
        axes[0].plot(
            phase,
            frames["foot_trunk"][:, foot_id, 2],
            label=f"{foot_name} trunk z",
        )
    axes[0].set_ylabel("foot z in trunk frame (m)")
    axes[0].legend(loc="best")

    for joint_id in range(frames["fcp_residual_ctrl"].shape[1]):
        axes[1].plot(
            phase,
            frames["fcp_residual_ctrl"][:, joint_id],
            linewidth=0.8,
            alpha=0.75,
        )
    axes[1].set_ylabel("FCP 16 ctrl (rad)")

    axes[2].plot(phase, frames["imu_linear_velocity"][:, 0], label="imu vx")
    axes[2].plot(phase, frames["imu_linear_velocity"][:, 1], label="imu vy")
    axes[2].plot(phase, frames["imu_angular_velocity"][:, 2], label="imu wz")
    axes[2].set_ylabel("velocity")
    axes[2].set_xlabel("phase fraction")
    axes[2].legend(loc="best")

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    args = parse_args()
    if args.device == "cpu":
        jax.config.update("jax_platform_name", "cpu")

    env_config, env = make_env(args)
    fixed_command = jnp.array([args.cmd_vx, args.cmd_vy, args.cmd_wz], dtype=jnp.float32)
    apply_fixed_command = make_fixed_command_fn(env, fixed_command)

    key = jax.random.PRNGKey(args.seed)
    key, reset_key = jax.random.split(key)
    state = env.reset(jax.random.split(reset_key, 1), args.eval_mode)
    state = apply_fixed_command(state)

    algorithm_config = get_algorithm_config("ppo_gru.flax_full_jit")
    config = SimpleNamespace(algorithm=algorithm_config, environment=env_config)
    policy, process_action, policy_params = load_policy_params(
        args.checkpoint,
        config,
        env,
        state.next_observation,
    )
    policy_carry = policy.initialize_carry(1)

    @jax.jit
    def rollout_step(state, carry):
        action_mean, _, next_carry = policy.apply(
            policy_params,
            state.next_observation,
            carry,
            method=policy.apply_one_step,
        )
        action = process_action(action_mean)
        next_state = env.step(state, action)
        next_state = apply_fixed_command(next_state)
        return next_state, next_carry, action_mean, action

    steps_per_cycle = int(round(float(env_config.gait_manager.gait_period) / env.dt))
    collect_steps = (
        args.collect_steps
        if args.collect_steps > 0
        else int(round(args.cycles * steps_per_cycle))
    )
    total_steps = args.warmup_steps + collect_steps

    samples = []
    for step in range(total_steps):
        state, policy_carry, action_mean, action = rollout_step(state, policy_carry)
        done = bool(np.asarray(state.terminated[0] | state.truncated[0]))
        if done:
            raise RuntimeError(
                f"Teacher terminated at step {step}. Try lower --cmd-vx or "
                "disable --clean-rollout to match the training setup more closely."
            )
        if step >= args.warmup_steps:
            samples.append(extract_sample(env, state, action_mean, action, step))

    rollout = stack_samples(samples)
    frames = aggregate_by_phase(rollout, args.frames)

    actuator_names = list(env.actuator_joint_names)
    foot_names = list(env.feet_names)
    fcp_names = [actuator_names[int(ctrl_id)] for ctrl_id in FCP_RESIDUAL_CTRL_IDS]
    metadata = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "environment": env_config.name,
        "robot": env_config.train_robot,
        "command_velocity": [args.cmd_vx, args.cmd_vy, args.cmd_wz],
        "control_frequency_hz": float(env.control_frequency_hz),
        "dt": float(env.dt),
        "gait_period": float(env_config.gait_manager.gait_period),
        "steps_per_cycle": steps_per_cycle,
        "warmup_steps": args.warmup_steps,
        "collect_steps": collect_steps,
        "frames": args.frames,
        "clean_rollout": bool(args.clean_rollout),
        "eval_mode": bool(args.eval_mode),
        "actuator_joint_names": actuator_names,
        "foot_names": foot_names,
        "fcp_residual_ctrl_ids": FCP_RESIDUAL_CTRL_IDS.tolist(),
        "fcp_residual_ctrl_names": fcp_names,
    }

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{args.prefix}.npz"
    csv_path = output_dir / f"{args.prefix}.csv"
    png_path = output_dir / f"{args.prefix}.png"

    write_npz(npz_path, frames, rollout, metadata)
    write_csv(csv_path, frames, actuator_names, fcp_names, foot_names)
    if args.plot:
        maybe_write_plot(png_path, frames, foot_names)

    left_z = frames["foot_trunk"][:, 0, 2]
    right_z = frames["foot_trunk"][:, 1, 2]
    print(f"Wrote target table: {npz_path}")
    print(f"Wrote target CSV:   {csv_path}")
    if args.plot:
        print(f"Wrote diagnostic:   {png_path}")
    print(
        "Summary: "
        f"frames={args.frames}, collect_steps={collect_steps}, "
        f"full_ctrl_dim={frames['ctrl'].shape[1]}, "
        f"fcp_ctrl_dim={frames['fcp_residual_ctrl'].shape[1]}, "
        f"left_foot_trunk_z_range={left_z.min():.4f}..{left_z.max():.4f}, "
        f"right_foot_trunk_z_range={right_z.min():.4f}..{right_z.max():.4f}"
    )


if __name__ == "__main__":
    main()
