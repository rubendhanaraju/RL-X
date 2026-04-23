import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch

from rl_x.environments.custom_mujoco.robocup_soccer.get_up.mujoco.default_config import (
    get_config,
)
from rl_x.environments.custom_mujoco.robocup_soccer.get_up.mujoco.environment import (
    GetUpEnv,
)
from rl_x.environments.custom_mujoco.robocup_soccer.rcssserver_deployment.torch_policy import (
    load_policy_from_files,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("torch", "onnx"), default="torch")
    parser.add_argument("--weights", type=Path, default=Path("getup_nn.pth"))
    parser.add_argument("--meta", type=Path, default=Path("getup_nn_meta.json"))
    parser.add_argument("--onnx", type=Path, default=Path("getup_nn.onnx"))
    parser.add_argument("--robot", type=str, default="booster_t1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--history-length", type=int, default=None)
    parser.add_argument("--settle-steps", type=int, default=None)
    parser.add_argument("--root-x", type=float, default=None)
    parser.add_argument("--root-y", type=float, default=None)
    parser.add_argument("--root-z", type=float, default=None)
    return parser.parse_args()


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
            self.runtime_name = "onnxruntime"
        except ImportError:
            import onnx
            from onnx.reference import ReferenceEvaluator

            self.session = ReferenceEvaluator(onnx.load(onnx_path.as_posix()))
            self.input_names = ["obs", "carry"]
            self.output_names = ["action", "next_carry"]
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


def build_env(args):
    env_config = get_config("custom_mujoco.robocup_soccer.get_up.mujoco")
    env_config.seed = args.seed
    env_config.nr_envs = 1
    env_config.render = args.render
    env_config.train_robot = args.robot

    if args.history_length is not None:
        env_config.observation.history_length = args.history_length
    if args.settle_steps is not None:
        env_config.reset.settle_steps = args.settle_steps
    if args.root_x is not None:
        env_config.reset.root_position_xyz[0] = args.root_x
    if args.root_y is not None:
        env_config.reset.root_position_xyz[1] = args.root_y
    if args.root_z is not None:
        env_config.reset.root_position_xyz[2] = args.root_z

    robot_config_module = importlib.import_module(
        f"rl_x.environments.custom_mujoco.robocup_soccer.robots.{args.robot}.robot_config"
    )
    robot_config = dict(robot_config_module.robot_config)
    robot_config["directory_path"] = (
        Path(__file__).resolve().parent.parent / "robots" / args.robot
    )

    return GetUpEnv(
        robot_config=robot_config,
        runner_mode="test",
        seed=args.seed,
        render=args.render,
        env_config=env_config,
        nr_envs=1,
    )


def get_policy_observation_indices(meta, full_obs_dim):
    if "policy_observation_indices" in meta:
        policy_obs_indices = np.asarray(meta["policy_observation_indices"], dtype=np.int64)
    else:
        policy_obs_indices = np.arange(full_obs_dim, dtype=np.int64)

    expected_policy_obs_dim = int(meta["expected_policy_obs_dim"])
    if policy_obs_indices.shape[0] != expected_policy_obs_dim:
        raise ValueError(
            f"Meta mismatch: expected_policy_obs_dim={expected_policy_obs_dim} but "
            f"policy_observation_indices has length {policy_obs_indices.shape[0]}."
        )
    if policy_obs_indices.ndim != 1:
        raise ValueError(
            f"policy_observation_indices must be 1D, got shape {policy_obs_indices.shape}."
        )
    if np.any(policy_obs_indices < 0) or np.any(policy_obs_indices >= full_obs_dim):
        raise ValueError(
            f"policy_observation_indices must lie in [0, {full_obs_dim}), got "
            f"min={policy_obs_indices.min()}, max={policy_obs_indices.max()}."
        )

    return policy_obs_indices


def main():
    args = parse_args()

    policy_runner = build_policy_runner(args)
    meta = policy_runner.meta
    env = build_env(args)

    obs, _ = env.reset()
    policy_obs_indices = get_policy_observation_indices(meta, obs.shape[0])
    carry = policy_runner.initialize_carry()
    episode_return = 0.0
    terminated = False
    truncated = False
    executed_steps = 0
    last_info = {}

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
    print(f"Policy obs idx: {policy_obs_indices.shape[0]}")
    print(f"History length: {env.history_length}")
    print(f"Settle steps:   {env.reset_settle_steps}")
    print(f"Reset root xyz: {format_vec(env.reset_root_position)}")

    try:
        for step in range(args.steps):
            policy_obs = obs[policy_obs_indices].astype(np.float32, copy=False)
            action, carry = policy_runner.act(policy_obs, carry)
            obs, reward, terminated, truncated, last_info = env.step(action)

            episode_return += float(reward)
            executed_steps = step + 1

            if terminated or truncated:
                break
    finally:
        data = env.internal_state["data"]
        final_root_xyz = data.qpos[:3].copy()
        final_head_xyz = data.xpos[env.head_body_id].copy()
        env.close()

    print(f"Executed steps: {executed_steps}")
    print(f"Terminated:     {bool(terminated)}")
    print(f"Truncated:      {bool(truncated)}")
    print(f"Episode return: {episode_return:.3f}")
    print(f"Final root xyz: {format_vec(final_root_xyz)}")
    print(f"Final head xyz: {format_vec(final_head_xyz)}")
    print(f"Final height:   {float(last_info.get('env_info/height', final_head_xyz[2])):.3f}")
    print(f"Success:        {bool(last_info.get('env_info/is_success', False))}")


if __name__ == "__main__":
    main()
