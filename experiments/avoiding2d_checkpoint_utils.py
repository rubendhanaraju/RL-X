from __future__ import annotations

import importlib
import json
import zipfile
from pathlib import Path

from ml_collections import config_dict

from rl_x.algorithms.algorithm_manager import get_algorithm_config, get_algorithm_model_class


def read_checkpoint_json(checkpoint_path: str | Path, filename: str):
    with zipfile.ZipFile(checkpoint_path) as archive:
        with archive.open(filename) as config_file:
            return json.load(config_file)


def read_json_config(path: str | Path):
    with open(path, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


def to_config_dict(payload):
    return config_dict.ConfigDict(payload)


def infer_algorithm_name(checkpoint_path: str | Path) -> str:
    algorithm_config = read_checkpoint_json(checkpoint_path, "config_algorithm.json")

    algorithm_name = algorithm_config.get("name")
    if not algorithm_name:
        raise ValueError(f"Could not infer algorithm name from {checkpoint_path}")
    return algorithm_name


def resolve_algorithm_name(checkpoint_path: str | Path, algorithm_name: str) -> str:
    if algorithm_name and algorithm_name != "auto":
        return algorithm_name
    return infer_algorithm_name(checkpoint_path)


def import_algorithm(algorithm_name: str) -> None:
    importlib.import_module(f"rl_x.algorithms.{algorithm_name}")


def load_algorithm_config(algorithm_name: str):
    import_algorithm(algorithm_name)
    return get_algorithm_config(algorithm_name)


def load_checkpoint_algorithm_config(checkpoint_path: str | Path, algorithm_name: str):
    config = load_algorithm_config(algorithm_name)
    config.update(read_checkpoint_json(checkpoint_path, "config_algorithm.json"))
    return config


def load_saved_config(path: str | Path):
    return to_config_dict(read_json_config(path))


def load_algorithm_model_class(algorithm_name: str):
    import_algorithm(algorithm_name)
    return get_algorithm_model_class(algorithm_name)


def select_eval_action(model, actor_params, observation, key):
    if hasattr(model, "select_eval_action"):
        return model.select_eval_action(actor_params, observation, key)
    return model.policy.deterministic_action(actor_params, observation, key)


def clip_action(model, action):
    if hasattr(model, "clip_action"):
        return model.clip_action(action)
    return action
