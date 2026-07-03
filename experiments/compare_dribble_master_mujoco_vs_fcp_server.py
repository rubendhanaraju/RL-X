import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from experiments.visualize_dribble_master_drag_ball import (
    make_env,
    place_ball_on_ring,
    refresh_root_state,
)
from rl_x.environments.custom_mujoco.robocup_soccer.rcssservermj_model import (
    set_server_pd_gains,
)


FCP_ROOT = Path("/home/ruben/Documents/GitHub/fcp")
DEFAULT_SERVER_BIN = FCP_ROOT / "venv" / "bin" / "rcssservermj"


def add_fcp_to_path():
    fcp_root = FCP_ROOT.as_posix()
    if fcp_root not in sys.path:
        sys.path.insert(0, fcp_root)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare a scripted low-level DribbleMaster rollout in the local "
            "MuJoCo rcssservermj-model adapter and in the live FCP rcssservermj server."
        )
    )
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--startup-steps", type=int, default=120)
    parser.add_argument("--server-port", type=int, default=6140)
    parser.add_argument("--monitor-port", type=int, default=6141)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--team", default="FCPortugal")
    parser.add_argument("--number", type=int, default=2)
    parser.add_argument("--field", default="fifa7vs7")
    parser.add_argument("--config", default=(FCP_ROOT / "config.toml").as_posix())
    parser.add_argument("--server-bin", default=DEFAULT_SERVER_BIN.as_posix())
    parser.add_argument("--rcssservermj-root", default="/home/ruben/Documents/GitHub/RoboCup/rcssservermj")
    parser.add_argument(
        "--sequential",
        dest="sequential",
        action="store_true",
        help="Start rcssservermj in its sequential agent loop for RL-style timing.",
    )
    parser.add_argument(
        "--no-sequential",
        dest="sequential",
        action="store_false",
        help="Use rcssservermj's parallel competition-style loop.",
    )
    parser.set_defaults(sequential=True)
    parser.add_argument("--start", default="-8.0,0.0,0.0")
    parser.add_argument("--start-z", type=float, default=0.65)
    parser.add_argument("--ball-distance", type=float, default=10.0)
    parser.add_argument("--ball-z", type=float, default=0.11)
    parser.add_argument("--scripted-action-amplitude", type=float, default=0.25)
    parser.add_argument("--mode", choices=("nominal", "scripted", "policy"), default="scripted")
    parser.add_argument(
        "--sync-local-from-server-after-startup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After the startup ramp, overwrite the local MuJoCo root/joint state "
            "with the server perceptors before the compared rollout."
        ),
    )
    parser.add_argument(
        "--state-sync-mode",
        choices=("full", "perception"),
        default="full",
        help=(
            "How to sync local MuJoCo after startup. 'full' uses the debug "
            "dumpState monitor command; 'perception' uses the lossy FCP sensors."
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("videos/dribble_master_6pqn4igo/dynamics_compare.json"))
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument(
        "--monitor-command-settle",
        type=float,
        default=0.005,
        help="Small delay after debug monitor commands so the server queues them before the sync step.",
    )
    parser.add_argument(
        "--local-observation-noise",
        action="store_true",
        help="Keep RL-X local observation noise enabled. Disabled by default for deployment parity.",
    )
    parser.add_argument(
        "--local-world-source",
        choices=("server", "rlx"),
        default="rlx",
        help="Local rcssservermj model world source: full server soccer world or lightweight RL-X world.",
    )
    parser.add_argument(
        "--server-policy-entry",
        choices=("execute", "manual"),
        default="execute",
        help=(
            "For policy mode, either call the real FCP SkillsManager/DribbleMaster "
            "execute path or use the older hand-copied policy loop."
        ),
    )
    parser.add_argument(
        "--server-use-local-policy-targets",
        action="store_true",
        help=(
            "In policy mode, compute the policy only on the local side and send "
            "the resulting joint targets to the FCP server. This isolates physics "
            "drift from closed-loop policy/action drift."
        ),
    )
    return parser.parse_args()


def start_server(args):
    cmd = [
        args.server_bin,
        "--host",
        args.host,
        "--aport",
        str(args.server_port),
        "--mport",
        str(args.monitor_port),
        "--field",
        args.field,
        "--rules",
        "ssim",
        "--no-referee",
        "--cheats",
        "--no-render",
        "--no-realtime",
    ]
    if args.sequential:
        cmd.append("--sequential")
    cmd.append("--sync")
    args.server_command = cmd
    print("$ " + " ".join(cmd), flush=True)
    process = subprocess.Popen(cmd, cwd=FCP_ROOT)
    time.sleep(2.0)
    return process


def make_local_env(args):
    env_args = SimpleNamespace(
        env_package="dribble_master",
        model_source="rcssservermj",
        rcssservermj_root=args.rcssservermj_root,
        rcssservermj_world_source=args.local_world_source,
        stage="stage_1",
        seed=0,
        device="cpu",
        episode_length_seconds=20,
    )
    _, env = make_env(env_args)
    if not args.local_observation_noise:
        disable_local_observation_noise(env)
    return env


def disable_local_observation_noise(env):
    env.observation_noise_function.modify_observation = lambda observation: observation


def set_local_root(env, args):
    x, y, yaw = (float(v) for v in args.start.split(","))
    data = env.internal_state["data"]
    data.qpos[0:3] = np.array([x, y, args.start_z], dtype=np.float64)
    data.qpos[3:7] = np.array(
        [np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)],
        dtype=np.float64,
    )
    data.qvel[0:6] = 0.0


def reset_local_like_fcp_beam(env, args):
    data = env.internal_state["data"]
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    set_local_root(env, args)
    data.qpos[env.ball_qposadr : env.ball_qposadr + 7] = np.array(
        [-8.0 + args.ball_distance, 0.0, args.ball_z, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    data.qvel[env.ball_qveladr : env.ball_qveladr + 6] = 0.0
    data.ctrl = env.zero_ctrl()
    refresh_root_state(env)
    place_ball_on_ring(env, args.ball_distance, 0.0)


def local_set_targets_and_step(env, target_joint_positions, kp, kd):
    set_server_pd_gains(env.internal_state["mj_model"], env.server_position_actuator_ids, kp=kp, kd=kd)
    ctrl = env.zero_ctrl()
    ctrl[env.server_position_actuator_ids] = np.asarray(target_joint_positions, dtype=np.float64)
    env.internal_state["data"].ctrl = ctrl
    mujoco.mj_step(env.internal_state["mj_model"], env.internal_state["data"], env.nr_substeps)
    refresh_root_state(env)


def contact_pairs_from_model_data(model, data):
    pairs = []
    for idx in range(data.ncon):
        contact = data.contact[idx]
        geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
        geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
        pairs.append(tuple(sorted((strip_server_prefix(geom1), strip_server_prefix(geom2)))))
    return sorted(set(pairs))


def contact_pairs_from_state_dump(state_path):
    if state_path is None:
        return []
    state_path = Path(state_path)
    if not state_path.is_file():
        return []
    with np.load(state_path, allow_pickle=False) as state:
        if "contact_geom1_names" not in state or "contact_geom2_names" not in state:
            return []
        pairs = []
        for geom1, geom2 in zip(state["contact_geom1_names"], state["contact_geom2_names"]):
            pairs.append(tuple(sorted((strip_server_prefix(str(geom1)), strip_server_prefix(str(geom2))))))
    return sorted(set(pairs))


def server_set_targets_and_step(agent, target_joint_positions, kp, kd, monitor=None, state_path=None, command_settle=0.0):
    if monitor is not None and state_path is not None:
        state_path = Path(state_path)
        if state_path.exists():
            state_path.unlink()
        monitor.dump_state(state_path.as_posix())
        if command_settle > 0.0:
            time.sleep(command_settle)

    robot = agent.world.robot
    for idx, target in enumerate(target_joint_positions):
        robot.set_motor_target_position(
            robot.ROBOT_MOTORS[idx],
            float(target),
            kp=kp,
            kd=kd,
        )
    agent.act()
    agent.observe()
    if monitor is not None and state_path is not None:
        wait_for_file(state_path)
        update_agent_world_from_state_dump(agent, state_path)


def server_step_current_targets(agent, monitor=None, state_path=None):
    agent.act()
    agent.observe()
    if monitor is not None and state_path is not None:
        wait_for_file(state_path)
        update_agent_world_from_state_dump(agent, state_path)


def place_local_ball(env, target_world):
    data = env.internal_state["data"]
    data.qpos[env.ball_qposadr : env.ball_qposadr + 7] = np.array(
        [
            float(target_world[0]),
            float(target_world[1]),
            float(target_world[2]),
            1.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    data.qvel[env.ball_qveladr : env.ball_qveladr + 6] = 0.0
    mujoco.mj_forward(env.internal_state["mj_model"], data)
    refresh_root_state(env)


def set_local_policy_context(env, target_world, command):
    place_local_ball(env, target_world)
    env.internal_state["ball_velocity_command"] = np.asarray(command, dtype=np.float32)
    env.internal_state["ball_visible"] = True
    env.internal_state["time_since_ball_seen"] = 0.0
    env.internal_state["ball_unseen_too_long"] = False


def init_local_policy_state(controller):
    from utils.neural_network import initialize_recurrent_state

    return {
        "carry": initialize_recurrent_state(controller.model, batch_size=1),
        "previous_action": np.zeros_like(controller.previous_action, dtype=np.float32),
        "action_history": np.zeros_like(controller.action_history, dtype=np.float32),
    }


def local_policy_step(env, controller, policy_state, target_world, command):
    from utils.neural_network import run_recurrent_network

    set_local_policy_context(env, target_world, command)
    observation_full = env.get_observation(policy_state["previous_action"])
    observation = observation_full[np.asarray(env.policy_observation_indices, dtype=int)].astype(np.float32)
    action, policy_state["carry"] = run_recurrent_network(
        obs=observation,
        carry=policy_state["carry"],
        model=controller.model,
    )
    action = np.asarray(action, dtype=np.float32)
    policy_state["action_history"] = np.roll(policy_state["action_history"], -1, axis=0)
    policy_state["action_history"][-1] = action.copy()
    delayed_action = policy_state["action_history"][-1 - controller.action_delay_steps]
    target_joint_positions = (
        controller.joint_nominal_position
        + (controller.scaling_factor * delayed_action * controller.action_control_mask)
    ) * controller.train_sim_flip

    policy_state["previous_action"] = action.copy()
    local_set_targets_and_step(env, target_joint_positions, kp=25.0, kd=0.6)
    env.gait_manager_function.step()
    env.internal_state["second_last_action"] = env.internal_state["last_action"].copy()
    env.internal_state["last_action"] = action.copy()
    env.internal_state["info_episode_store"]["episode_step"] += 1
    return {
        "observation": observation,
        "action": action,
        "target_joint_positions": target_joint_positions,
        "contacts": contact_pairs_from_model_data(
            env.internal_state["mj_model"],
            env.internal_state["data"],
        ),
    }


def server_policy_step(agent, monitor, controller, target_world, command, args):
    from utils.neural_network import run_recurrent_network

    post_step_state_path = None
    if args.state_sync_mode == "full":
        post_step_state_path = Path(args.server_step_state_path)
        if post_step_state_path.exists():
            post_step_state_path.unlink()
        monitor.place_ball_and_dump_state(
            pos_xyz=tuple(map(float, target_world)),
            vel_xyz=(0.0, 0.0, 0.0),
            state_path=post_step_state_path.as_posix(),
        )
        if args.monitor_command_settle > 0.0:
            time.sleep(args.monitor_command_settle)
    else:
        monitor.place_ball(
            pos_xyz=tuple(map(float, target_world)),
            vel_xyz=(0.0, 0.0, 0.0),
        )

    if args.server_policy_entry == "execute":
        agent.skills_manager.execute(
            "DribbleMaster",
            target_2d=np.asarray(target_world[:2], dtype=np.float32),
            is_target_absolute=True,
            command_2d=np.asarray(command, dtype=np.float32),
            is_command_absolute=True,
            target_z=np.float32(target_world[2]),
            force_visible=True,
        )
        server_step_current_targets(
            agent,
            monitor=monitor if post_step_state_path is not None else None,
            state_path=post_step_state_path,
        )
        if controller.last_observation is None:
            raise RuntimeError(
                "DribbleMaster.execute did not expose last_observation. "
                "Update the FCP skill diagnostics before using --server-policy-entry execute."
            )
        return {
            "observation": controller.last_observation.copy(),
            "action": controller.last_action.copy(),
            "target_joint_positions": controller.last_target_joint_positions.copy(),
            "contacts": contact_pairs_from_state_dump(post_step_state_path),
        }

    observation = controller._build_observation(
        target_world=np.asarray(target_world, dtype=np.float32),
        ball_velocity_command=np.asarray(command, dtype=np.float32),
        ball_visible=True,
    )
    action, controller.carry = run_recurrent_network(
        obs=observation,
        carry=controller.carry,
        model=controller.model,
    )
    action = np.asarray(action, dtype=np.float32)
    controller.action_history = np.roll(controller.action_history, -1, axis=0)
    controller.action_history[-1] = action.copy()
    delayed_action = controller.action_history[-1 - controller.action_delay_steps]
    target_joint_positions = (
        controller.joint_nominal_position
        + (controller.scaling_factor * delayed_action * controller.action_control_mask)
    ) * controller.train_sim_flip

    controller.previous_action = action.copy()
    controller._advance_policy_time()
    server_set_targets_and_step(
        agent,
        target_joint_positions,
        kp=25.0,
        kd=0.6,
    )
    if post_step_state_path is not None:
        wait_for_file(post_step_state_path)
        update_agent_world_from_state_dump(agent, post_step_state_path)
    return {
        "observation": observation,
        "action": action,
        "target_joint_positions": target_joint_positions,
        "contacts": contact_pairs_from_state_dump(post_step_state_path),
    }


def server_policy_same_targets_step(agent, monitor, target_world, target_joint_positions, args):
    post_step_state_path = None
    if args.state_sync_mode == "full":
        post_step_state_path = Path(args.server_step_state_path)
        if post_step_state_path.exists():
            post_step_state_path.unlink()
        monitor.place_ball_and_dump_state(
            pos_xyz=tuple(map(float, target_world)),
            vel_xyz=(0.0, 0.0, 0.0),
            state_path=post_step_state_path.as_posix(),
        )
        if args.monitor_command_settle > 0.0:
            time.sleep(args.monitor_command_settle)
    else:
        monitor.place_ball(
            pos_xyz=tuple(map(float, target_world)),
            vel_xyz=(0.0, 0.0, 0.0),
        )
    server_set_targets_and_step(
        agent,
        target_joint_positions,
        kp=25.0,
        kd=0.6,
    )
    if post_step_state_path is not None:
        wait_for_file(post_step_state_path)
        update_agent_world_from_state_dump(agent, post_step_state_path)
    return {
        "observation": np.zeros(86, dtype=np.float32),
        "action": np.zeros_like(target_joint_positions, dtype=np.float32),
        "target_joint_positions": np.asarray(target_joint_positions, dtype=np.float32),
        "contacts": contact_pairs_from_state_dump(post_step_state_path),
    }


def diff_policy_step(local, server):
    obs = np.asarray(local["observation"]) - np.asarray(server["observation"])
    action = np.asarray(local["action"]) - np.asarray(server["action"])
    target = angle_wrap_diff(local["target_joint_positions"], server["target_joint_positions"])
    obs_groups = {
        "joint_pos": slice(0, 23),
        "joint_vel": slice(23, 46),
        "prev_action": slice(46, 69),
        "imu_ang_vel": slice(69, 72),
        "body_orientation": slice(72, 75),
        "command": slice(75, 77),
        "relative_target": slice(77, 80),
        "visible": slice(80, 81),
        "gait": slice(81, 83),
        "gravity": slice(83, 86),
    }
    obs_group_diff = {}
    for name, group_slice in obs_groups.items():
        group = obs[group_slice]
        obs_group_diff[name] = {
            "max_abs": float(np.max(np.abs(group))),
            "l2": float(np.linalg.norm(group)),
        }
    local_contacts = {tuple(pair) for pair in local.get("contacts", [])}
    server_contacts = {tuple(pair) for pair in server.get("contacts", [])}
    return {
        "obs_max_abs": float(np.max(np.abs(obs))),
        "obs_l2": float(np.linalg.norm(obs)),
        "obs_groups": obs_group_diff,
        "action_max_abs": float(np.max(np.abs(action))),
        "action_l2": float(np.linalg.norm(action)),
        "target_max_abs": float(np.max(np.abs(target))),
        "local_contacts": [list(pair) for pair in sorted(local_contacts)],
        "server_contacts": [list(pair) for pair in sorted(server_contacts)],
        "contacts_only_local": [list(pair) for pair in sorted(local_contacts - server_contacts)],
        "contacts_only_server": [list(pair) for pair in sorted(server_contacts - local_contacts)],
    }


def smoothstep(alpha):
    alpha = np.clip(float(alpha), 0.0, 1.0)
    return alpha * alpha * (3.0 - 2.0 * alpha)


def scripted_action(step, action_dim, action_control_mask, amplitude):
    action = np.zeros(action_dim, dtype=np.float32)
    phase = 2.0 * np.pi * step / 32.0
    # Small alternating leg/head commands. This stays far from joint limits but
    # is enough to expose actuator/order/physics divergence.
    action[0] = 0.4 * np.sin(phase)
    action[1] = 0.25 * np.cos(phase)
    left = np.array([11, 12, 13, 14, 15, 16])
    right = np.array([17, 18, 19, 20, 21, 22])
    leg_pattern = np.array([0.6, 0.25, -0.15, -0.5, 0.35, -0.2], dtype=np.float32)
    action[left] = leg_pattern * np.sin(phase)
    action[right] = -leg_pattern * np.sin(phase)
    return amplitude * action * action_control_mask


def local_snapshot(env):
    data = env.internal_state["data"]
    quat_wxyz = data.qpos[3:7].copy()
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    return {
        "pos": data.qpos[0:3].astype(float).tolist(),
        "rpy": Rotation.from_quat(quat_xyzw).as_euler("xyz").astype(float).tolist(),
        "joints": data.qpos[env.actuator_joint_mask_qpos].astype(float).tolist(),
        "joint_vel": data.qvel[env.actuator_joint_mask_qvel].astype(float).tolist(),
    }


def server_snapshot(agent):
    robot = agent.world.robot
    return {
        "pos": np.asarray(agent.world.global_position, dtype=float).tolist(),
        "rpy": np.asarray(robot.global_orientation_euler, dtype=float).tolist(),
        "joints": np.asarray(
            [robot.motor_positions[motor] for motor in robot.ROBOT_MOTORS],
            dtype=float,
        ).tolist(),
        "joint_vel": np.asarray(
            [robot.motor_speeds[motor] for motor in robot.ROBOT_MOTORS],
            dtype=float,
        ).tolist(),
    }


def sync_local_from_server(env, agent, args):
    robot = agent.world.robot
    data = env.internal_state["data"]
    data.qpos[0:3] = np.asarray(agent.world.global_position, dtype=np.float64)

    quat_xyzw = np.asarray(robot.torso_cheat_quat_ori, dtype=np.float64)
    data.qpos[3:7] = np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float64,
    )

    data.qpos[env.actuator_joint_mask_qpos] = np.asarray(
        [robot.motor_positions[motor] for motor in robot.ROBOT_MOTORS],
        dtype=np.float64,
    )
    data.qvel[:] = 0.0
    data.qvel[env.actuator_joint_mask_qvel] = np.asarray(
        [robot.motor_speeds[motor] for motor in robot.ROBOT_MOTORS],
        dtype=np.float64,
    )
    data.ctrl = env.zero_ctrl()
    data.qpos[env.ball_qposadr : env.ball_qposadr + 7] = np.array(
        [
            float(agent.world.global_position[0]) + args.ball_distance,
            float(agent.world.global_position[1]),
            args.ball_z,
            1.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    data.qvel[env.ball_qveladr : env.ball_qveladr + 6] = 0.0
    refresh_root_state(env)


def strip_server_prefix(name):
    if not name:
        return name
    parts = str(name).split("-", 3)
    if (
        len(parts) == 4
        and parts[0] in {"l", "r"}
        and parts[1].isdigit()
        and parts[2].isdigit()
    ):
        return parts[3]
    return str(name)


def name_lookup(names):
    lookup = {}
    for idx, name in enumerate(names):
        name = str(name)
        for candidate in (name, strip_server_prefix(name)):
            if candidate and candidate not in lookup:
                lookup[candidate] = idx
    return lookup


def joint_widths(model):
    qpos_widths = []
    qvel_widths = []
    for joint_type in model.jnt_type:
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            qpos_widths.append(7)
            qvel_widths.append(6)
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            qpos_widths.append(4)
            qvel_widths.append(3)
        else:
            qpos_widths.append(1)
            qvel_widths.append(1)
    return np.asarray(qpos_widths, dtype=np.int32), np.asarray(qvel_widths, dtype=np.int32)


def model_names(model, obj_type, count):
    return [
        mujoco.mj_id2name(model, obj_type, idx) or ""
        for idx in range(count)
    ]


def wait_for_file(path, timeout=5.0):
    start = time.time()
    while time.time() - start < timeout:
        if path.is_file() and path.stat().st_size > 0:
            return
        time.sleep(0.01)
    raise TimeoutError(f"Timed out waiting for {path}")


def request_server_state_dump(monitor, agent, path):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    monitor.dump_state(path.as_posix())
    agent.act()
    agent.observe()
    wait_for_file(path)
    return path


def load_local_from_named_state(env, state_path):
    model = env.internal_state["mj_model"]
    data = env.internal_state["data"]
    local_joint_names = model_names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    local_actuator_names = model_names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
    local_body_names = model_names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    local_joint_lookup = name_lookup(local_joint_names)
    local_actuator_lookup = name_lookup(local_actuator_names)
    local_body_lookup = name_lookup(local_body_names)
    local_qpos_width, local_qvel_width = joint_widths(model)

    with np.load(state_path, allow_pickle=False) as state:
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        if "ctrl" in state:
            data.ctrl[:] = 0.0
        if "qacc_warmstart" in state:
            data.qacc_warmstart[:] = 0.0
        if "qfrc_applied" in state:
            data.qfrc_applied[:] = 0.0
        if "xfrc_applied" in state:
            data.xfrc_applied[:] = 0.0

        source_joint_names = [str(name) for name in state["joint_names"]]
        for source_idx, source_name in enumerate(source_joint_names):
            local_idx = local_joint_lookup.get(strip_server_prefix(source_name))
            if local_idx is None:
                continue

            source_qpos_adr = int(state["joint_qposadr"][source_idx])
            source_qvel_adr = int(state["joint_dofadr"][source_idx])
            source_qpos_width = int(state["joint_qpos_width"][source_idx])
            source_qvel_width = int(state["joint_qvel_width"][source_idx])
            local_qpos_adr = int(model.jnt_qposadr[local_idx])
            local_qvel_adr = int(model.jnt_dofadr[local_idx])

            if source_qpos_width != int(local_qpos_width[local_idx]):
                raise ValueError(
                    f"qpos width mismatch for joint {source_name}: "
                    f"{source_qpos_width} vs {int(local_qpos_width[local_idx])}"
                )
            if source_qvel_width != int(local_qvel_width[local_idx]):
                raise ValueError(
                    f"qvel width mismatch for joint {source_name}: "
                    f"{source_qvel_width} vs {int(local_qvel_width[local_idx])}"
                )

            data.qpos[local_qpos_adr:local_qpos_adr + source_qpos_width] = state["qpos"][
                source_qpos_adr:source_qpos_adr + source_qpos_width
            ]
            data.qvel[local_qvel_adr:local_qvel_adr + source_qvel_width] = state["qvel"][
                source_qvel_adr:source_qvel_adr + source_qvel_width
            ]
            if "qacc_warmstart" in state:
                data.qacc_warmstart[local_qvel_adr:local_qvel_adr + source_qvel_width] = state["qacc_warmstart"][
                    source_qvel_adr:source_qvel_adr + source_qvel_width
                ]
            if "qfrc_applied" in state:
                data.qfrc_applied[local_qvel_adr:local_qvel_adr + source_qvel_width] = state["qfrc_applied"][
                    source_qvel_adr:source_qvel_adr + source_qvel_width
                ]

        if "ctrl" in state and "actuator_names" in state:
            for source_idx, source_name in enumerate(str(name) for name in state["actuator_names"]):
                local_idx = local_actuator_lookup.get(strip_server_prefix(source_name))
                if local_idx is not None:
                    data.ctrl[local_idx] = state["ctrl"][source_idx]

        if "xfrc_applied" in state and "body_names" in state:
            for source_idx, source_name in enumerate(str(name) for name in state["body_names"]):
                local_idx = local_body_lookup.get(strip_server_prefix(source_name))
                if local_idx is not None:
                    data.xfrc_applied[local_idx] = state["xfrc_applied"][source_idx]

        warmstart = data.qacc_warmstart.copy()
        if "time" in state:
            data.time = float(state["time"][0])

    mujoco.mj_forward(model, data)
    data.qacc_warmstart[:] = warmstart
    refresh_root_state(env)


def update_agent_world_from_state_dump(agent, state_path):
    robot = agent.world.robot
    readable_to_motor = dict(robot.MOTOR_FROM_READABLE_TO_SERVER)
    readable_to_motor["AAHead_yaw"] = "he1"

    with np.load(state_path, allow_pickle=False) as state:
        if "sensordata" in state and "sensor_names" in state:
            for source_idx, source_name in enumerate(str(name) for name in state["sensor_names"]):
                if strip_server_prefix(source_name) != "torso_gyro":
                    continue
                source_adr = int(state["sensor_adr"][source_idx])
                source_dim = int(state["sensor_dim"][source_idx])
                if source_dim >= 3:
                    robot.gyroscope = state["sensordata"][source_adr:source_adr + 3].astype(float)
                break

        source_joint_names = [str(name) for name in state["joint_names"]]
        for source_idx, source_name in enumerate(source_joint_names):
            joint_name = strip_server_prefix(source_name)
            source_qpos_adr = int(state["joint_qposadr"][source_idx])
            source_qvel_adr = int(state["joint_dofadr"][source_idx])
            source_qpos_width = int(state["joint_qpos_width"][source_idx])
            source_qvel_width = int(state["joint_qvel_width"][source_idx])

            if joint_name == "root" and source_qpos_width == 7 and source_qvel_width == 6:
                root_qpos = state["qpos"][source_qpos_adr:source_qpos_adr + 7]
                quat_xyzw = np.asarray(root_qpos[3:7])[[1, 2, 3, 0]]
                agent.world.global_position = root_qpos[:3].astype(float)
                agent.world.torso_abs_cheat_pos = root_qpos[:3].astype(float)
                robot.torso_cheat_quat_ori = quat_xyzw.astype(float)
                robot.global_orientation_quat = quat_xyzw.astype(float)
                robot.global_orientation_euler = Rotation.from_quat(quat_xyzw).as_euler("xyz")
                continue

            motor = readable_to_motor.get(joint_name)
            if motor is None:
                continue
            robot.motor_positions[motor] = float(state["qpos"][source_qpos_adr])
            robot.motor_speeds[motor] = float(state["qvel"][source_qvel_adr])

    robot.update_pose()


def sync_local_from_server_full_state(env, monitor, agent, args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    state_path = (args.output.parent / f".server_state_{os.getpid()}_{args.server_port}.npz").resolve()
    request_server_state_dump(monitor, agent, state_path)
    load_local_from_named_state(env, state_path)
    update_agent_world_from_state_dump(agent, state_path)
    return state_path


def angle_wrap_diff(a, b):
    return (np.asarray(a) - np.asarray(b) + np.pi) % (2.0 * np.pi) - np.pi


def diff_snapshot(local, server):
    pos = np.asarray(local["pos"]) - np.asarray(server["pos"])
    rpy = angle_wrap_diff(local["rpy"], server["rpy"])
    joints = angle_wrap_diff(local["joints"], server["joints"])
    joint_vel = np.asarray(local["joint_vel"]) - np.asarray(server["joint_vel"])
    return {
        "pos_norm": float(np.linalg.norm(pos)),
        "z": float(pos[2]),
        "rpy_norm": float(np.linalg.norm(rpy)),
        "joint_max_abs": float(np.max(np.abs(joints))),
        "joint_vel_max_abs": float(np.max(np.abs(joint_vel))),
    }


def main():
    args = parse_args()
    add_fcp_to_path()

    from agent.common.agent import Agent
    from comms.monitor import Monitor

    server_process = start_server(args)
    agent = None
    monitor = None
    report = {
        "mode": args.mode,
        "server": {
            "sequential": bool(args.sequential),
            "sync": True,
            "realtime": False,
            "command": list(args.server_command),
        },
        "state_sync_mode": args.state_sync_mode,
        "local_observation_noise": bool(args.local_observation_noise),
        "local_world_source": args.local_world_source,
        "server_policy_entry": args.server_policy_entry,
        "server_use_local_policy_targets": bool(args.server_use_local_policy_targets),
        "samples": [],
    }
    try:
        agent = Agent(
            team_name=args.team,
            number=args.number,
            host=args.host,
            server_port=args.server_port,
            monitor_port=args.monitor_port,
            field=args.field,
            config_file=args.config,
        )
        agent.config.debug.draw = False
        monitor = Monitor(host=args.host, port=args.monitor_port)
        controller = agent.skills_manager.get_skill_object("DribbleMaster")

        env = make_local_env(args)
        reset_local_like_fcp_beam(env, args)

        robot = agent.world.robot
        report["joint_order"] = {
            "local": list(env.actuator_joint_names),
            "server": list(robot.ROBOT_MOTORS),
        }
        report["model_options"] = {
            "local_timestep": float(env.internal_state["mj_model"].opt.timestep),
            "local_iterations": int(env.internal_state["mj_model"].opt.iterations),
            "local_ls_iterations": int(env.internal_state["mj_model"].opt.ls_iterations),
        }

        start_pose = tuple(float(v) for v in args.start.split(","))
        monitor.beam_2d(args.number, args.team, start_pose)
        monitor.place_ball(
            pos_xyz=(start_pose[0] + args.ball_distance, start_pose[1], args.ball_z),
            vel_xyz=(0.0, 0.0, 0.0),
        )
        agent.observe()
        server_start_joints = np.asarray(
            [robot.motor_positions[motor] for motor in robot.ROBOT_MOTORS],
            dtype=np.float32,
        )
        local_start_joints = env.internal_state["data"].qpos[env.actuator_joint_mask_qpos].astype(np.float32)
        nominal = controller.joint_nominal_position.astype(np.float32)
        mask = controller.action_control_mask.astype(np.float32)

        report["initial_joint_max_abs_diff"] = float(
            np.max(np.abs(local_start_joints - server_start_joints))
        )

        for step in range(args.startup_steps):
            monitor.beam_2d(args.number, args.team, start_pose)
            monitor.place_ball(
                pos_xyz=(start_pose[0] + args.ball_distance, start_pose[1], args.ball_z),
                vel_xyz=(0.0, 0.0, 0.0),
            )
            set_local_root(env, args)
            alpha = smoothstep((step + 1) / max(1, args.startup_steps))
            local_target = local_start_joints + alpha * (nominal - local_start_joints)
            server_target = server_start_joints + alpha * (nominal - server_start_joints)
            local_set_targets_and_step(env, local_target, kp=60.0, kd=1.0)
            server_set_targets_and_step(agent, server_target, kp=60.0, kd=1.0)

        report["post_startup_pre_sync_diff"] = diff_snapshot(
            local_snapshot(env),
            server_snapshot(agent),
        )
        if args.sync_local_from_server_after_startup:
            if args.state_sync_mode == "full":
                state_path = sync_local_from_server_full_state(env, monitor, agent, args)
                report["server_state_dump"] = str(state_path.resolve())
                args.server_step_state_path = (
                    args.output.parent / f".server_step_state_{os.getpid()}_{args.server_port}.npz"
                ).resolve()
            else:
                sync_local_from_server(env, agent, args)
                args.server_step_state_path = None
            report["post_startup_post_sync_diff"] = diff_snapshot(
                local_snapshot(env),
                server_snapshot(agent),
            )
        else:
            args.server_step_state_path = None

        local_policy_state = None
        fixed_target_world = None
        fixed_command = None
        last_policy_diff = None
        if args.mode == "policy":
            controller._reset_policy_state()
            agent.skills_manager.current_skill_name = None
            local_policy_state = init_local_policy_state(controller)
            fixed_target_world = np.array(
                [
                    float(agent.world.global_position[0]) + args.ball_distance,
                    float(agent.world.global_position[1]),
                    args.ball_z,
                ],
                dtype=np.float32,
            )
            fixed_command = np.array([1.0, 0.0], dtype=np.float32)
            set_local_policy_context(env, fixed_target_world, fixed_command)
            monitor.place_ball(
                pos_xyz=tuple(map(float, fixed_target_world)),
                vel_xyz=(0.0, 0.0, 0.0),
            )

        for step in range(args.steps + 1):
            local = local_snapshot(env)
            server = server_snapshot(agent)
            diff = diff_snapshot(local, server)
            if step % max(1, args.print_every) == 0 or step == args.steps:
                policy_text = ""
                if last_policy_diff is not None:
                    policy_text = (
                        f" obs_diff={last_policy_diff['obs_max_abs']:.3f}"
                        f" act_diff={last_policy_diff['action_max_abs']:.3f}"
                    )
                print(
                    f"step={step:04d} "
                    f"local_z={local['pos'][2]:+.3f} server_z={server['pos'][2]:+.3f} "
                    f"pos_diff={diff['pos_norm']:.4f} rpy_diff={diff['rpy_norm']:.4f} "
                    f"joint_max={diff['joint_max_abs']:.4f}"
                    f"{policy_text}",
                    flush=True,
                )
                sample = {
                    "step": step,
                    "local": local,
                    "server": server,
                    "diff": diff,
                }
                if last_policy_diff is not None:
                    sample["policy_diff_from_previous_step"] = last_policy_diff
                report["samples"].append(sample)
            if step == args.steps:
                break

            if args.mode == "policy":
                local_policy_info = local_policy_step(
                    env,
                    controller,
                    local_policy_state,
                    fixed_target_world,
                    fixed_command,
                )
                if args.server_use_local_policy_targets:
                    server_policy_info = server_policy_same_targets_step(
                        agent,
                        monitor,
                        fixed_target_world,
                        local_policy_info["target_joint_positions"],
                        args,
                    )
                    last_policy_diff = None
                else:
                    server_policy_info = server_policy_step(
                        agent,
                        monitor,
                        controller,
                        fixed_target_world,
                        fixed_command,
                        args,
                    )
                    last_policy_diff = diff_policy_step(local_policy_info, server_policy_info)
            elif args.mode == "nominal":
                action = np.zeros_like(nominal)
            else:
                action = scripted_action(
                    step,
                    action_dim=nominal.shape[0],
                    action_control_mask=mask,
                    amplitude=args.scripted_action_amplitude,
                )
            if args.mode != "policy":
                target = nominal + controller.scaling_factor * action * mask
                local_set_targets_and_step(env, target, kp=25.0, kd=0.6)
                server_set_targets_and_step(
                    agent,
                    target,
                    kp=25.0,
                    kd=0.6,
                    monitor=monitor if args.state_sync_mode == "full" else None,
                    state_path=args.server_step_state_path if args.state_sync_mode == "full" else None,
                    command_settle=args.monitor_command_settle if args.state_sync_mode == "full" else 0.0,
                )

        all_diffs = [sample["diff"] for sample in report["samples"]]
        report["sampled_max"] = {
            key: float(max(diff[key] for diff in all_diffs))
            for key in all_diffs[0]
        }
        report["sampled_mean"] = {
            key: float(np.mean([diff[key] for diff in all_diffs]))
            for key in all_diffs[0]
        }
        policy_diffs = [
            sample["policy_diff_from_previous_step"]
            for sample in report["samples"]
            if "policy_diff_from_previous_step" in sample
        ]
        if policy_diffs:
            numeric_policy_keys = [
                key for key, value in policy_diffs[0].items()
                if isinstance(value, (int, float, np.floating))
            ]
            report["sampled_policy_max"] = {
                key: float(max(diff[key] for diff in policy_diffs))
                for key in numeric_policy_keys
            }
            report["sampled_policy_mean"] = {
                key: float(np.mean([diff[key] for diff in policy_diffs]))
                for key in numeric_policy_keys
            }
            group_names = policy_diffs[0].get("obs_groups", {}).keys()
            report["sampled_policy_obs_group_max"] = {
                name: {
                    metric: float(max(
                        diff["obs_groups"][name][metric]
                        for diff in policy_diffs
                    ))
                    for metric in ("max_abs", "l2")
                }
                for name in group_names
            }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.output.resolve()}", flush=True)
    finally:
        if monitor is not None:
            monitor.close()
        if agent is not None:
            try:
                agent.shutdown()
            except Exception:
                pass
        server_process.terminate()
        try:
            server_process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            server_process.kill()


if __name__ == "__main__":
    main()
