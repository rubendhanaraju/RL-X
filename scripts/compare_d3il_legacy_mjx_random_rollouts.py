#!/usr/bin/env python3
"""Compare legacy D3IL MuJoCo rollouts with RL-X D3IL MJX rollouts.

The legacy fork imports a few optional packages that are not needed for this
headless comparison.  This script shims those imports in-process, patches the
old XML option that modern MuJoCo rejects, and then runs both simulators from
the same robot/object state under the same randomly generated target sequence.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
from pathlib import Path
import sys
import types

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx


ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = Path("/tmp/d3il_legacy")
sys.path.insert(0, str(ROOT))

TASK_SPECS = {
    "avoiding": {
        "module": "environments.d3il.envs.gym_avoiding_env.gym_avoiding.envs.avoiding",
        "class": "ObstacleAvoidanceEnv",
        "kwargs": {},
        "mode": "cartesian",
    },
    "pushing": {
        "module": "environments.d3il.envs.gym_pushing_env.gym_pushing.envs.pushing",
        "class": "Block_Push_Env",
        "kwargs": {},
        "mode": "cartesian",
    },
    "aligning": {
        "module": "environments.d3il.envs.gym_aligning_env.gym_aligning.envs.aligning",
        "class": "Robot_Push_Env",
        "kwargs": {},
        "mode": "cartesian",
    },
    "sorting": {
        "module": "environments.d3il.envs.gym_sorting_env.gym_sorting.envs.sorting",
        "class": "Sorting_Env",
        "kwargs": {"num_boxes": 2},
        "mode": "cartesian",
    },
    "stacking": {
        "module": "environments.d3il.envs.gym_stacking_env.gym_stacking.envs.stacking",
        "class": "CubeStacking_Env",
        "kwargs": {},
        "mode": "joint",
    },
    "inserting": {
        "module": "environments.d3il.envs.gym_inserting_env.gym_inserting.envs.gate_insertion",
        "class": "Gate_Insertion_Env",
        "kwargs": {},
        "mode": "cartesian",
    },
}

OBJECT_NAME_MAP = {}


def install_legacy_shims() -> None:
    np.NAN = np.nan

    config = {}

    def parse_config_file(path):
        for raw in open(path):
            line = raw.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            try:
                config[key] = ast.literal_eval(value)
            except Exception:
                try:
                    config[key] = float(value)
                except Exception:
                    config[key] = value

    def configurable(cls):
        original_init = cls.__init__
        signature = inspect.signature(original_init)
        parameters = [p for p in signature.parameters.values() if p.name != "self"]

        def wrapped_init(self, *args, **kwargs):
            if not args:
                fq_name = f"{cls.__module__}.{cls.__name__}"
                for parameter in parameters:
                    key = f"{fq_name}.{parameter.name}"
                    if parameter.name not in kwargs and key in config:
                        kwargs[parameter.name] = config[key]
            return original_init(self, *args, **kwargs)

        cls.__init__ = wrapped_init
        return cls

    gin = types.ModuleType("gin")
    gin.configurable = configurable
    gin.parse_config_file = parse_config_file
    sys.modules["gin"] = gin

    class GymBox:
        def __init__(self, low, high, shape=None, dtype=np.float32):
            self.low = np.array(low if shape is None else np.full(shape, low), dtype=dtype)
            self.high = np.array(high if shape is None else np.full(shape, high), dtype=dtype)
            self.shape = tuple(shape) if shape is not None else self.low.shape
            self.dtype = dtype

        def sample(self):
            return np.random.uniform(self.low, self.high).astype(self.dtype)

    gym = types.ModuleType("gym")
    gym.Env = type("Env", (), {})
    spaces = types.ModuleType("gym.spaces")
    spaces.Box = GymBox
    gym.spaces = spaces
    utils = types.ModuleType("gym.utils")
    seeding = types.ModuleType("gym.utils.seeding")
    seeding.np_random = lambda seed=None: (np.random.default_rng(seed), seed)
    utils.seeding = seeding
    envs = types.ModuleType("gym.envs")
    registration = types.ModuleType("gym.envs.registration")
    registration.register = lambda *args, **kwargs: None
    envs.registration = registration
    for name, module in [
        ("gym", gym),
        ("gym.spaces", spaces),
        ("gym.utils", utils),
        ("gym.utils.seeding", seeding),
        ("gym.envs", envs),
        ("gym.envs.registration", registration),
    ]:
        sys.modules[name] = module

    cv2 = types.ModuleType("cv2")
    cv2.COLOR_RGB2BGR = 0
    cv2.cvtColor = lambda image, code: image
    cv2.bilateralFilter = lambda z, *args, **kwargs: z
    cv2.imwrite = lambda *args, **kwargs: True
    sys.modules["cv2"] = cv2


def bootstrap_legacy() -> None:
    install_legacy_shims()
    for path in (LEGACY_ROOT, LEGACY_ROOT / "environments"):
        sys.path.insert(0, str(path))

    for name in [
        "environments.d3il.d3il_sim.sims.pybullet",
        "environments.d3il.d3il_sim.sims.mujoco",
        "environments.d3il.d3il_sim.sims.sl",
    ]:
        module = types.ModuleType(name)
        module.__all__ = []
        sys.modules[name] = module

    from environments.d3il.d3il_sim.sims.mj_beta.mj_utils import mj_scene_parser

    original_init = mj_scene_parser.MjSceneParser.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        option = self._root.find("option")
        if option is not None:
            option.attrib.pop("collision", None)
            option.attrib.pop("apirate", None)

    mj_scene_parser.MjSceneParser.__init__ = patched_init

    modules_to_alias = [
        "environments.d3il.d3il_sim",
        "environments.d3il.d3il_sim.utils.sim_path",
        "environments.d3il.d3il_sim.controllers.Controller",
        "environments.d3il.d3il_sim.core",
        "environments.d3il.d3il_sim.core.logger",
        "environments.d3il.d3il_sim.gyms.gym_env_wrapper",
        "environments.d3il.d3il_sim.gyms.gym_utils.helpers",
        "environments.d3il.d3il_sim.utils.geometric_transformation",
        "environments.d3il.d3il_sim.sims",
        "environments.d3il.d3il_sim.sims.mj_beta.MjRobot",
        "environments.d3il.d3il_sim.sims.mj_beta.MjFactory",
        "environments.d3il.d3il_sim.sims.universal_sim.PrimitiveObjects",
        "environments.d3il.d3il_sim.gyms.gym_controllers",
    ]
    for module_name in modules_to_alias:
        importlib.import_module(module_name)

    prefix = "environments.d3il.d3il_sim"
    for name, module in list(sys.modules.items()):
        if name == prefix or name.startswith(prefix + "."):
            sys.modules["d3il_sim" + name[len(prefix) :]] = module


def make_legacy_env(task: str, seed: int):
    np.random.seed(seed)
    spec = TASK_SPECS[task]
    cls = getattr(importlib.import_module(spec["module"]), spec["class"])
    env = cls(render=False, **spec["kwargs"])
    env.start()
    try:
        env.reset(random=True)
    except TypeError:
        env.reset()
    return env


def make_mjx_env(task: str):
    cfg_mod = importlib.import_module(f"rl_x.environments.custom_mujoco.d3il.{task}.mjx.default_config")
    env_mod = importlib.import_module(f"rl_x.environments.custom_mujoco.d3il.{task}.mjx.environment")
    cls = getattr(env_mod, f"{task.capitalize()}Mjx")
    cfg = cfg_mod.get_config(f"custom_mujoco.d3il.{task}.mjx")
    cfg.nr_envs = 1
    cfg.render = False
    return cls(cfg)


def make_mjx_runner(task: str):
    mjx_env = make_mjx_env(task)
    step_physics = jax.jit(lambda d, target, oq, ov: mjx_env._step_physics(d, target, oq, ov))
    return mjx_env, step_physics


def legacy_name(task: str, local_name: str) -> str:
    return OBJECT_NAME_MAP.get(task, {}).get(local_name, local_name)


def read_legacy_pose(legacy_env, name: str):
    pos = np.asarray(legacy_env.scene.get_obj_pos(obj_name=name), dtype=np.float64)
    quat = np.asarray(legacy_env.scene.get_obj_quat(obj_name=name), dtype=np.float64)
    return pos, quat


def set_local_state_from_legacy(task: str, mjx_env, legacy_env):
    qpos = mjx_env.initial_qpos
    qvel = mjx_env.initial_qvel
    qpos = qpos.at[mjx_env.robot_qpos_adrs].set(jnp.asarray(legacy_env.robot.current_j_pos, dtype=qpos.dtype))
    qvel = qvel.at[mjx_env.robot_dof_adrs].set(jnp.asarray(legacy_env.robot.current_j_vel, dtype=qvel.dtype))
    if mjx_env.finger_qpos_adrs.shape[0] == 2:
        qpos = qpos.at[mjx_env.finger_qpos_adrs].set(jnp.asarray(legacy_env.robot.current_fing_pos, dtype=qpos.dtype))
        qvel = qvel.at[mjx_env.finger_dof_adrs].set(jnp.asarray(legacy_env.robot.current_fing_vel, dtype=qvel.dtype))

    for index, local_name in enumerate(mjx_env.object_body_names):
        try:
            pos, quat = read_legacy_pose(legacy_env, legacy_name(task, local_name))
        except Exception:
            continue
        adr = mjx_env.object_qpos_adrs[index]
        qpos = qpos.at[adr : adr + 3].set(jnp.asarray(pos, dtype=qpos.dtype))
        qpos = qpos.at[adr + 3 : adr + 7].set(jnp.asarray(quat, dtype=qpos.dtype))

    for index, local_name in enumerate(mjx_env.target_body_names):
        try:
            pos, quat = read_legacy_pose(legacy_env, legacy_name(task, local_name))
        except Exception:
            continue
        adr = mjx_env.target_qpos_adrs[index]
        qpos = qpos.at[adr : adr + 3].set(jnp.asarray(pos, dtype=qpos.dtype))
        qpos = qpos.at[adr + 3 : adr + 7].set(jnp.asarray(quat, dtype=qpos.dtype))

    return mjx.forward(mjx_env.mjx_model, mjx_env.mjx_data.replace(qpos=qpos, qvel=qvel, ctrl=mjx_env.initial_ctrl))


def legacy_cartesian_step(env, target_xy, z):
    action = np.array([target_xy[0], target_xy[1], z, 0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    env.robot.open_fingers()
    env.controller.setSetPoint(action)
    env.controller.executeControllerTimeSteps(env.robot, env.n_substeps, block=False)
    for _ in range(env.n_substeps):
        env.scene.next_step(log=False)
    env.env_step_counter += 1


def legacy_joint_step(env, action):
    if action[-1] > 0.075:
        env.robot.open_fingers()
    else:
        env.robot.close_fingers(duration=0.0)
    env.controller.setSetPoint(action[:-1])
    env.controller.executeControllerTimeSteps(env.robot, env.n_substeps, block=False)
    for _ in range(env.n_substeps):
        env.scene.next_step(log=False)
    env.env_step_counter += 1


def object_error(task: str, mjx_env, data, legacy_env):
    errors = []
    for index, local_name in enumerate(mjx_env.object_body_names):
        try:
            legacy_pos, _ = read_legacy_pose(legacy_env, legacy_name(task, local_name))
        except Exception:
            continue
        adr = mjx_env.object_qpos_adrs[index]
        local_pos = np.asarray(data.qpos[adr : adr + 3])
        errors.append(float(np.linalg.norm(local_pos - legacy_pos)))
    return float(max(errors)) if errors else None


def summarise(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": None, "p95": None, "max": None, "final": None}
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "final": float(values[-1]),
    }


def clip_delta(origin, value, radius):
    delta = value - origin
    norm = np.linalg.norm(delta)
    if norm > radius:
        return origin + delta * (radius / norm)
    return value


def run_rollout(task: str, seed: int, steps: int, max_delta: float, wander_radius: float, joint_delta: float, mjx_env=None, step_physics=None, return_runner=False):
    rng = np.random.default_rng(seed)
    legacy_env = make_legacy_env(task, seed)
    if mjx_env is None or step_physics is None:
        mjx_env, step_physics = make_mjx_runner(task)
    data = set_local_state_from_legacy(task, mjx_env, legacy_env)
    old_q = jnp.asarray(legacy_env.robot.current_j_pos, dtype=data.qpos.dtype)
    old_des_vel = jnp.zeros((7,), dtype=data.qvel.dtype)

    spec = TASK_SPECS[task]
    tcp_errors = []
    tcp_xy_errors = []
    joint_errors = []
    joint_vel_errors = []
    object_errors = []

    if spec["mode"] == "cartesian":
        target_xy = np.asarray(legacy_env.robot.current_c_pos[:2], dtype=np.float64)
        origin_xy = target_xy.copy()
        workspace_low = np.asarray([0.2, -0.45], dtype=np.float64)
        workspace_high = np.asarray([0.8, 0.5], dtype=np.float64)
        target_arg = None
    else:
        joint_low = np.asarray(legacy_env.robot.joint_pos_min, dtype=np.float64)
        joint_high = np.asarray(legacy_env.robot.joint_pos_max, dtype=np.float64)
        target_joint = np.asarray(legacy_env.robot.current_j_pos, dtype=np.float64)
        target_arg = None

    for _ in range(steps):
        if spec["mode"] == "cartesian":
            target_xy = target_xy + rng.uniform(-max_delta, max_delta, size=2)
            target_xy = clip_delta(origin_xy, target_xy, wander_radius)
            target_xy = np.clip(target_xy, workspace_low, workspace_high)
            legacy_cartesian_step(legacy_env, target_xy, float(mjx_env.task.control_agent_z))
            target_arg = jnp.asarray(target_xy, dtype=data.qpos.dtype)
        else:
            target_joint = target_joint + rng.uniform(-joint_delta, joint_delta, size=7)
            target_joint = np.clip(target_joint, joint_low, joint_high)
            action = np.concatenate([target_joint, [0.08]]).astype(np.float64)
            legacy_joint_step(legacy_env, action)
            target_arg = jnp.asarray(action, dtype=data.qpos.dtype)

        data, old_q, old_des_vel = step_physics(data, target_arg, old_q, old_des_vel)
        data.qpos.block_until_ready()

        local_tcp = np.asarray(data.xpos[mjx_env.tcp_body_id])
        legacy_tcp = np.asarray(legacy_env.robot.current_c_pos)
        tcp_errors.append(float(np.linalg.norm(local_tcp - legacy_tcp)))
        tcp_xy_errors.append(float(np.linalg.norm(local_tcp[:2] - legacy_tcp[:2])))
        joint_errors.append(float(np.max(np.abs(np.asarray(data.qpos[mjx_env.robot_qpos_adrs]) - legacy_env.robot.current_j_pos))))
        joint_vel_errors.append(float(np.max(np.abs(np.asarray(data.qvel[mjx_env.robot_dof_adrs]) - legacy_env.robot.current_j_vel))))
        obj_error = object_error(task, mjx_env, data, legacy_env)
        if obj_error is not None:
            object_errors.append(obj_error)

    result = {
        "task": task,
        "seed": seed,
        "steps": steps,
        "tcp_l2": summarise(tcp_errors),
        "tcp_xy_l2": summarise(tcp_xy_errors),
        "joint_max_abs": summarise(joint_errors),
        "joint_vel_max_abs": summarise(joint_vel_errors),
        "object_l2": summarise(object_errors),
    }
    if return_runner:
        return result, mjx_env, step_physics
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="avoiding,pushing,aligning,sorting,stacking,inserting")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--max-delta", type=float, default=0.008)
    parser.add_argument("--wander-radius", type=float, default=0.12)
    parser.add_argument("--joint-delta", type=float, default=0.01)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if not LEGACY_ROOT.exists():
        raise SystemExit(f"legacy checkout not found at {LEGACY_ROOT}")

    bootstrap_legacy()

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    results = []
    print("task seed steps tcp_mean tcp_p95 tcp_max tcp_final joint_max object_max", flush=True)
    for task in tasks:
        mjx_env = None
        step_physics = None
        for seed in range(args.seeds):
            result, mjx_env, step_physics = run_rollout(
                task,
                seed,
                args.steps,
                args.max_delta,
                args.wander_radius,
                args.joint_delta,
                mjx_env,
                step_physics,
                return_runner=True,
            )
            results.append(result)
            object_max = result["object_l2"]["max"]
            print(
                f"{task} {seed} {args.steps} "
                f"{result['tcp_l2']['mean']:.6f} {result['tcp_l2']['p95']:.6f} "
                f"{result['tcp_l2']['max']:.6f} {result['tcp_l2']['final']:.6f} "
                f"{result['joint_max_abs']['max']:.6f} "
                f"{object_max if object_max is not None else float('nan'):.6f}",
                flush=True,
            )

    if args.json is not None:
        args.json.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
