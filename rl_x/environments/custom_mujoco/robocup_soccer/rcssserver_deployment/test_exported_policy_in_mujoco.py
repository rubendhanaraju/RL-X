import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch

from rl_x.environments.custom_mujoco.robocup_soccer.locomotion.mujoco.default_config import (
    get_config,
)
from rl_x.environments.custom_mujoco.robocup_soccer.locomotion.mujoco.environment import (
    LocomotionEnv,
)
from rl_x.environments.custom_mujoco.robocup_soccer.rcssserver_deployment.torch_policy import (
    load_policy_from_files,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("torch", "onnx"), default="torch")
    parser.add_argument("--weights", type=Path, default=Path("locomotion_nn.pth"))
    parser.add_argument("--meta", type=Path, default=Path("locomotion_nn_meta.json"))
    parser.add_argument("--onnx", type=Path, default=Path("locomotion_nn.onnx"))
    parser.add_argument("--robot", type=str, default="booster_t1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--startup-settle-steps", type=int, default=50)
    parser.add_argument("--command-vx", type=float, default=0.5)
    parser.add_argument("--command-vy", type=float, default=0.0)
    parser.add_argument("--command-wz", type=float, default=0.0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--show-goal-arrow", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--eval-mode",
        action="store_true",
        help=(
            "Turn on env eval mode. Note: in the current RoboCup MuJoCo env this still enables "
            "some unseen-robot randomization at reset, so it is off by default for export smoke tests."
        ),
    )
    return parser.parse_args()


def build_env(args):
    env_config = get_config("custom_mujoco.robocup_soccer.locomotion.mujoco")
    env_config.seed = args.seed
    env_config.nr_envs = 1
    env_config.render = args.render
    env_config.train_robot = args.robot
    env_config.add_goal_arrow = args.render and args.show_goal_arrow

    robot_config_module = importlib.import_module(
        f"rl_x.environments.custom_mujoco.robocup_soccer.robots.{args.robot}.robot_config"
    )
    robot_config = dict(robot_config_module.robot_config)
    robot_config["directory_path"] = (
        Path(__file__).resolve().parent.parent / "robots" / args.robot
    )

    env = LocomotionEnv(
        robot_config=robot_config,
        runner_mode="test",
        seed=args.seed,
        render=args.render,
        env_config=env_config,
        nr_envs=1,
    )
    env.internal_state["in_eval_mode"] = args.eval_mode

    # Keep the commanded velocity under our control instead of letting the env
    # resample random commands in the middle of the test rollout.
    env.command_sampling_function.step = lambda: False

    return env


def set_velocity_command(env, command):
    command = np.asarray(command, dtype=np.float32)
    zero_threshold = (
        env.command_function.zero_clip_threshold_percentage
        * env.internal_state["max_command_velocity"]
    )
    command = np.where(np.abs(command) < zero_threshold, 0.0, command)
    command = np.clip(
        command,
        -env.internal_state["max_command_velocity"],
        env.internal_state["max_command_velocity"],
    )
    env.internal_state["goal_velocities"] = command
    env.internal_state["actuator_joint_keep_nominal"] = np.where(
        np.all(command == 0.0),
        np.ones(env.nr_actuator_joints, dtype=bool),
        env.command_function.default_actuator_joint_keep_nominal,
    )


def format_vec(vec):
    return "[" + ", ".join(f"{value:.3f}" for value in vec) + "]"


class TorchPolicyRunner:
    def __init__(self, weights_path, meta_path, device_name):
        self.device = torch.device(device_name)
        self.model, self.meta = load_policy_from_files(weights_path, meta_path, self.device)
        self.runtime_name = f"torch:{self.device.type}"

    def initialize_carry(self):
        return self.model.initialize_carry(batch_size=1, device=self.device)

    def act(self, policy_obs, carry):
        policy_obs_tensor = torch.from_numpy(policy_obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_mean, _, next_carry = self.model.forward_step(policy_obs_tensor, carry)
        return action_mean.squeeze(0).cpu().numpy(), next_carry


class OnnxPolicyRunner:
    def __init__(self, onnx_path, meta_path):
        with meta_path.open("r", encoding="utf-8") as f:
            self.meta = json.load(f)

        try:
            import onnxruntime as ort

            self.session = ort.InferenceSession(
                onnx_path.as_posix(),
                providers=["CPUExecutionProvider"],
            )
            self.input_names = [tensor.name for tensor in self.session.get_inputs()]
            self.output_names = [tensor.name for tensor in self.session.get_outputs()]
            self.uses_onnxruntime = True
            self.runtime_name = "onnxruntime"
        except ImportError:
            import onnx
            from onnx.reference import ReferenceEvaluator

            self.session = ReferenceEvaluator(onnx.load(onnx_path.as_posix()))
            self.input_names = ["obs", "carry"]
            self.output_names = ["action", "next_carry"]
            self.uses_onnxruntime = False
            self.runtime_name = "onnx-reference"

    def initialize_carry(self):
        return np.zeros((1, self.meta["gru_hidden_dim"]), dtype=np.float32)

    def act(self, policy_obs, carry):
        output_values = self.session.run(
            self.output_names,
            {
                self.input_names[0]: policy_obs[None, :].astype(np.float32),
                self.input_names[1]: carry.astype(np.float32),
            },
        )
        action = np.asarray(output_values[0][0], dtype=np.float32)
        next_carry = np.asarray(output_values[1], dtype=np.float32)
        return action, next_carry


def build_policy_runner(args):
    if args.backend == "onnx":
        return OnnxPolicyRunner(args.onnx, args.meta)
    return TorchPolicyRunner(args.weights, args.meta, args.device)


def main():
    args = parse_args()

    policy_runner = build_policy_runner(args)
    meta = policy_runner.meta
    env = build_env(args)

    command = np.array([args.command_vx, args.command_vy, args.command_wz], dtype=np.float32)

    obs, _ = env.reset()
    carry = policy_runner.initialize_carry()
    start_xy = env.internal_state["data"].qpos[:2].copy()
    episode_return = 0.0
    terminated = False
    truncated = False
    executed_steps = 0

    if args.backend == "onnx":
        print(f"Loaded ONNX:    {args.onnx.resolve()}")
    else:
        print(f"Loaded weights: {args.weights.resolve()}")
    print(f"Loaded meta:    {args.meta.resolve()}")
    print(f"Backend:        {args.backend}")
    print(f"Runtime:        {policy_runner.runtime_name}")
    print(f"Policy obs dim: {meta['expected_policy_obs_dim']}")
    print(f"Action dim:     {meta['action_dim']}")
    print(f"Full obs dim:   {obs.shape[0]}")
    print(f"Policy obs idx: {env.policy_observation_indices.shape[0]}")
    print(
        f"Command plan:   {args.startup_settle_steps} settle steps, then {format_vec(command)}"
    )

    try:
        for step in range(args.steps):
            current_command = (
                np.zeros(3, dtype=np.float32)
                if step < args.startup_settle_steps
                else command
            )
            set_velocity_command(env, current_command)

            policy_obs = obs[env.policy_observation_indices].astype(np.float32, copy=False)
            action, carry = policy_runner.act(policy_obs, carry)
            obs, reward, terminated, truncated, _ = env.step(action)

            episode_return += float(reward)
            executed_steps = step + 1

            if terminated or truncated:
                break
    finally:
        final_xy = env.internal_state["data"].qpos[:2].copy()
        displacement_xy = final_xy - start_xy
        final_height = float(env.internal_state["data"].qpos[2])
        final_euler = env.internal_state["imu_orientation_euler"]
        env.close()

    print(f"Executed steps: {executed_steps}")
    print(f"Terminated:     {bool(terminated)}")
    print(f"Truncated:      {bool(truncated)}")
    print(f"Episode return: {episode_return:.3f}")
    print(f"XY displacement:{format_vec(displacement_xy)}")
    print(f"Final height:   {final_height:.3f}")
    print(f"Final rpy(rad): {format_vec(final_euler)}")


if __name__ == "__main__":
    main()
