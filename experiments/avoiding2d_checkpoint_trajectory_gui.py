#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from avoiding2d_trajectory_gui import (
    ENV_META,
    choose_port,
    config_get,
    first_done_length,
    json_safe,
    load_json,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "wandb_map_outputs" / "avoiding2d"
STEP_DIR_RE = re.compile(r"^step_(\d+)$")


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def run_metadata(output_dir: Path) -> dict[str, str]:
    result = {"id": "", "name": "", "url": ""}
    for filename in ("checkpoint_renders_result.json", "render_result.json", "run_manifest.json"):
        path = output_dir / filename
        if not path.exists():
            continue
        try:
            payload = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        run_info = payload.get("run") if isinstance(payload, dict) else None
        if isinstance(run_info, dict):
            result["id"] = result["id"] or str(run_info.get("id") or "")
            result["name"] = result["name"] or str(run_info.get("name") or "")
            result["url"] = result["url"] or str(run_info.get("url") or "")
        elif isinstance(payload, dict):
            result["id"] = result["id"] or str(payload.get("id") or "")
            result["name"] = result["name"] or str(payload.get("name") or "")
            result["url"] = result["url"] or str(payload.get("url") or "")
    return result


def checkpoint_step_from_dir(step_dir: Path) -> int | None:
    match = STEP_DIR_RE.match(step_dir.name)
    if match is None:
        return None
    return int(match.group(1))


def checkpoint_result_stats(step_dir: Path) -> dict[str, Any]:
    path = step_dir / "render_result.json"
    if not path.exists():
        return {}
    try:
        payload = load_json(path)
    except (OSError, json.JSONDecodeError):
        return {}
    render = payload.get("render", {}) if isinstance(payload, dict) else {}
    return render if isinstance(render, dict) else {}


def checkpoint_result_step(step_dir: Path) -> int | None:
    path = step_dir / "render_result.json"
    if not path.exists():
        return None
    try:
        payload = load_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    checkpoint = payload.get("checkpoint") if isinstance(payload, dict) else None
    if not isinstance(checkpoint, dict):
        return None
    step = checkpoint.get("step")
    try:
        return int(step)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class CheckpointRecord:
    id: str
    step: int
    step_label: str
    output_dir: Path
    npz_path: Path
    png_path: Path
    return_mean: Any
    return_std: Any
    return_min: Any
    return_max: Any
    reached_mode_counts: Any

    def to_json(self, output_root: Path) -> dict[str, Any]:
        return {
            "id": self.id,
            "step": self.step,
            "step_label": self.step_label,
            "output_dir": relative_path(self.output_dir, output_root),
            "npz": relative_path(self.npz_path, output_root),
            "png": relative_path(self.png_path, output_root) if self.png_path.exists() else "",
            "return_mean": self.return_mean,
            "return_std": self.return_std,
            "return_min": self.return_min,
            "return_max": self.return_max,
            "reached_mode_counts": self.reached_mode_counts,
        }


@dataclass(frozen=True)
class TrainingRunRecord:
    id: str
    run_folder: str
    run_id: str
    run_name: str
    run_url: str
    algorithm_name: str
    ent_start: Any
    ent_target_mult: Any
    lmbda: Any
    nr_steps: Any
    output_dir: Path
    checkpoints: tuple[CheckpointRecord, ...]

    def to_json(self, output_root: Path) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_folder": self.run_folder,
            "run_id": self.run_id,
            "run_name": self.run_name,
            "run_url": self.run_url,
            "algorithm_name": self.algorithm_name,
            "ent_start": self.ent_start,
            "ent_target_mult": self.ent_target_mult,
            "lmbda": self.lmbda,
            "nr_steps": self.nr_steps,
            "output_dir": relative_path(self.output_dir, output_root),
            "checkpoint_count": len(self.checkpoints),
            "first_step": self.checkpoints[0].step if self.checkpoints else None,
            "last_step": self.checkpoints[-1].step if self.checkpoints else None,
            "checkpoints": [checkpoint.to_json(output_root) for checkpoint in self.checkpoints],
        }


def sort_value(value: Any) -> tuple[int, float, str]:
    if value in ("", None):
        return (2, 0.0, "")
    try:
        return (0, float(value), "")
    except (TypeError, ValueError):
        return (1, 0.0, str(value))


def scan_checkpoint_runs(output_root: Path) -> list[TrainingRunRecord]:
    grouped: dict[Path, list[Path]] = {}
    for npz_path in sorted(output_root.glob("**/checkpoints/*/trajectories.npz")):
        output_dir = npz_path.parent.parent.parent
        if not (output_dir / "config.json").exists():
            continue
        grouped.setdefault(output_dir, []).append(npz_path)

    records: list[TrainingRunRecord] = []
    for output_dir, npz_paths in sorted(grouped.items(), key=lambda item: item[0].as_posix()):
        try:
            config = load_json(output_dir / "config.json")
        except (OSError, json.JSONDecodeError):
            continue
        meta = run_metadata(output_dir)
        checkpoints: list[CheckpointRecord] = []
        for npz_path in npz_paths:
            step_dir = npz_path.parent
            step = checkpoint_result_step(step_dir)
            if step is None:
                step = checkpoint_step_from_dir(step_dir)
            if step is None:
                continue
            stats = checkpoint_result_stats(step_dir)
            checkpoints.append(
                CheckpointRecord(
                    id=npz_path.relative_to(output_root).as_posix(),
                    step=step,
                    step_label=step_dir.name,
                    output_dir=step_dir,
                    npz_path=npz_path,
                    png_path=step_dir / "trajectories.png",
                    return_mean=stats.get("return_mean", ""),
                    return_std=stats.get("return_std", ""),
                    return_min=stats.get("return_min", ""),
                    return_max=stats.get("return_max", ""),
                    reached_mode_counts=stats.get("reached_mode_counts", ""),
                )
            )
        checkpoints.sort(key=lambda checkpoint: checkpoint.step)
        if not checkpoints:
            continue
        records.append(
            TrainingRunRecord(
                id=output_dir.relative_to(output_root).as_posix(),
                run_folder=output_dir.parent.name,
                run_id=meta["id"],
                run_name=meta["name"],
                run_url=meta["url"],
                algorithm_name=str(config_get(config, "algorithm.name", "")),
                ent_start=config_get(config, "algorithm.ent_start", ""),
                ent_target_mult=config_get(config, "algorithm.ent_target_mult", ""),
                lmbda=config_get(config, "algorithm.lmbda", ""),
                nr_steps=config_get(config, "algorithm.nr_steps", ""),
                output_dir=output_dir,
                checkpoints=tuple(checkpoints),
            )
        )

    records.sort(
        key=lambda record: (
            record.algorithm_name,
            sort_value(record.ent_start),
            sort_value(record.ent_target_mult),
            sort_value(record.lmbda),
            sort_value(record.nr_steps),
            record.run_folder,
        )
    )
    return records


def first_episode_returns(rewards: np.ndarray, done: np.ndarray) -> np.ndarray:
    done_int = done.astype(np.int32)
    done_count_before_step = np.cumsum(done_int, axis=0) - done_int
    first_episode_mask = done_count_before_step == 0
    return np.sum(np.where(first_episode_mask, rewards, 0.0), axis=0)


def load_trajectories(record: CheckpointRecord, max_trajectories: int, stride: int) -> dict[str, Any]:
    max_trajectories = max(1, min(int(max_trajectories), 4096))
    stride = max(1, int(stride))
    with np.load(record.npz_path) as data:
        points = np.asarray(data["points"], dtype=np.float32)
        done = np.asarray(data["done"], dtype=np.bool_)
        rewards = np.asarray(data["reward"], dtype=np.float32)
        final_modes = np.asarray(data.get("final_mode_encoding", np.zeros((points.shape[1], 0))), dtype=np.float32)

    nr_available = points.shape[1]
    count = min(max_trajectories, nr_available)
    if count >= nr_available:
        indices = np.arange(nr_available, dtype=np.int32)
    else:
        indices = np.linspace(0, nr_available - 1, count, dtype=np.int32)

    trajectories = []
    for index in indices:
        length = first_done_length(done[:, index], points.shape[0])
        trajectory = points[:length:stride, index, :]
        if trajectory.shape[0] < 2 and length >= 2:
            trajectory = points[:length, index, :]
        trajectories.append(np.round(trajectory, 5).tolist())

    selected_rewards = rewards[:, indices]
    selected_done = done[:, indices]
    returns = first_episode_returns(selected_rewards, selected_done)
    reached_modes = final_modes[indices] > 0.5 if final_modes.size else np.zeros((indices.shape[0], 0), dtype=bool)
    return {
        "id": record.id,
        "step": record.step,
        "step_label": record.step_label,
        "trajectories": trajectories,
        "stats": {
            "loaded_trajectories": int(indices.shape[0]),
            "available_trajectories": int(nr_available),
            "points_per_full_trajectory": int(points.shape[0]),
            "return_mean": float(np.mean(returns)),
            "return_std": float(np.std(returns)),
            "return_min": float(np.min(returns)),
            "return_max": float(np.max(returns)),
            "reached_mode_counts": reached_modes.sum(axis=0).astype(int).tolist(),
        },
    }


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Avoiding2D Checkpoint Trajectories</title>
  <style>
    :root {
      --bg: #f5f7fa;
      --panel: #ffffff;
      --line: #d8dee8;
      --text: #1f2630;
      --muted: #697386;
      --accent: #1769aa;
      --accent-soft: #e8f2fb;
      --danger: #b42318;
      --sidebar-width: 460px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: var(--bg);
    }
    .app {
      display: grid;
      grid-template-columns: var(--sidebar-width) 6px minmax(0, 1fr);
      min-height: 100vh;
    }
    aside {
      border-right: 1px solid var(--line);
      background: var(--panel);
      padding: 14px;
      overflow-y: auto;
      max-height: 100vh;
      min-width: 0;
    }
    .sidebar-resizer {
      background: linear-gradient(to right, transparent, var(--line), transparent);
      cursor: col-resize;
      min-height: 100vh;
    }
    .sidebar-resizer:hover,
    body.resizing-sidebar .sidebar-resizer {
      background: var(--accent);
    }
    main {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      min-width: 0;
    }
    .toolbar {
      display: flex;
      gap: 12px;
      align-items: center;
      justify-content: space-between;
      padding: 10px 14px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    .toolbar .left,
    .toolbar .right {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }
    .canvas-wrap {
      position: relative;
      padding: 14px;
      min-height: 0;
    }
    canvas {
      width: 100%;
      height: calc(100vh - 78px);
      display: block;
      background: #fbfcfd;
      border: 1px solid var(--line);
    }
    h1 {
      font-size: 18px;
      line-height: 1.2;
      margin: 0 0 10px;
    }
    h2 {
      font-size: 13px;
      margin: 16px 0 8px;
      color: #303741;
    }
    label {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 4px;
    }
    select,
    input,
    button {
      font: inherit;
    }
    select,
    input[type="text"],
    input[type="number"] {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px 8px;
      background: #fff;
      color: var(--text);
    }
    input[type="range"] {
      width: 100%;
    }
    input[type="color"] {
      width: 40px;
      height: 30px;
      padding: 1px;
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 6px;
    }
    input[type="checkbox"] {
      width: 16px;
      height: 16px;
      margin: 0;
    }
    button {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px 10px;
      background: #fff;
      color: var(--text);
      cursor: pointer;
    }
    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }
    button.danger {
      color: var(--danger);
    }
    button:disabled {
      cursor: wait;
      opacity: 0.65;
    }
    .tabs {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
      margin: 10px 0 12px;
    }
    .tab-button {
      background: #fff;
      color: var(--text);
    }
    .tab-button.active {
      background: var(--accent-soft);
      border-color: var(--accent);
      color: #124d7d;
    }
    .tab-panel {
      display: none;
    }
    .tab-panel.active {
      display: block;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .field {
      margin-bottom: 8px;
    }
    .inline-field {
      display: grid;
      grid-template-columns: 18px minmax(0, 1fr);
      gap: 8px;
      align-items: center;
      margin-bottom: 8px;
    }
    .select-run {
      height: 220px;
      font-size: 12px;
    }
    .muted {
      color: var(--muted);
      font-size: 12px;
    }
    .layer {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px;
      margin-bottom: 8px;
      background: #fff;
    }
    .layer-top {
      display: grid;
      grid-template-columns: 18px minmax(0, 1fr) 40px auto;
      gap: 8px;
      align-items: center;
    }
    .layer-title {
      min-width: 0;
      font-size: 12px;
      line-height: 1.25;
    }
    .layer-title strong,
    .layer-title span {
      display: block;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .layer-controls {
      display: grid;
      grid-template-columns: 1fr 48px;
      gap: 8px;
      align-items: center;
      margin-top: 8px;
    }
    .stats {
      margin-top: 7px;
      font-size: 11px;
      color: var(--muted);
      line-height: 1.35;
    }
    .status {
      font-size: 12px;
      color: var(--muted);
    }
    .legend {
      position: absolute;
      top: 24px;
      right: 24px;
      background: rgba(255,255,255,0.92);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      max-width: 380px;
      font-size: 12px;
      pointer-events: none;
    }
    .legend-row {
      display: grid;
      grid-template-columns: 14px minmax(0, 1fr);
      gap: 6px;
      align-items: center;
      margin: 2px 0;
    }
    .swatch {
      width: 12px;
      height: 12px;
      border-radius: 3px;
      border: 1px solid rgba(0,0,0,0.15);
    }
    .step-label {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
      margin-top: -2px;
      margin-bottom: 8px;
    }
    @media (max-width: 960px) {
      .app {
        grid-template-columns: 1fr;
      }
      .sidebar-resizer {
        display: none;
      }
      aside {
        max-height: none;
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }
      canvas {
        height: 70vh;
      }
    }
  </style>
</head>
<body>
<div class="app">
  <aside>
    <h1>Avoiding2D Checkpoints</h1>
    <div class="muted" id="runCount">Loading runs...</div>

    <div class="tabs">
      <button id="compareTab" class="tab-button active" type="button">Compare</button>
      <button id="evolutionTab" class="tab-button" type="button">Evolution</button>
    </div>

    <h2>Render Settings</h2>
    <div class="grid">
      <div class="field">
        <label for="maxTraj">Trajectories</label>
        <input id="maxTraj" type="number" min="1" max="4096" value="1024">
      </div>
      <div class="field">
        <label for="stride">Point stride</label>
        <input id="stride" type="number" min="1" max="20" value="1">
      </div>
    </div>
    <div class="field">
      <label for="lineWidth">Line width</label>
      <input id="lineWidth" type="range" min="0.2" max="3" step="0.1" value="0.8">
    </div>

    <section id="comparePanel" class="tab-panel active">
      <h2>Checkpoint</h2>
      <div class="field">
        <label for="compareStep">Step</label>
        <select id="compareStep"></select>
      </div>

      <h2>Filter Runs</h2>
      <div class="field">
        <label for="algorithmFilter">Algorithm</label>
        <select id="algorithmFilter"></select>
      </div>
      <div class="grid">
        <div class="field">
          <label for="entFilter">ent_start</label>
          <select id="entFilter"></select>
        </div>
        <div class="field">
          <label for="targetFilter">ent_target_mult</label>
          <select id="targetFilter"></select>
        </div>
        <div class="field">
          <label for="lambdaFilter">lmbda</label>
          <select id="lambdaFilter"></select>
        </div>
        <div class="field">
          <label for="stepsFilter">nr_steps</label>
          <select id="stepsFilter"></select>
        </div>
      </div>
      <div class="field">
        <label for="searchFilter">Search</label>
        <input id="searchFilter" type="text" placeholder="run id, run folder, algorithm">
      </div>
      <div class="field">
        <label for="runSelect">Runs</label>
        <select id="runSelect" class="select-run" size="12" multiple></select>
      </div>
      <div class="grid">
        <button id="addCompareRun" type="button">Add selected</button>
        <button id="clearCompareLayers" class="danger" type="button">Clear</button>
      </div>
      <button id="renderCompareButton" class="primary" type="button">Render selected</button>
      <div class="status" id="compareStatus"></div>

      <h2>Layers</h2>
      <div id="compareLayers"></div>
    </section>

    <section id="evolutionPanel" class="tab-panel">
      <h2>Run</h2>
      <div class="field">
        <label for="evolutionSearch">Search</label>
        <input id="evolutionSearch" type="text" placeholder="run id, run folder, algorithm">
      </div>
      <div class="field">
        <label for="evolutionRun">Run</label>
        <select id="evolutionRun"></select>
      </div>

      <h2>Training Step</h2>
      <div class="field">
        <label for="evolutionStep">Checkpoint</label>
        <input id="evolutionStep" type="range" min="0" max="0" step="1" value="0">
      </div>
      <div class="step-label">
        <span id="evolutionStepText"></span>
        <span id="evolutionStepCount"></span>
      </div>

      <div class="inline-field">
        <input id="historyEnabled" type="checkbox">
        <label for="historyEnabled">Plot previous checkpoints</label>
      </div>
      <div class="grid">
        <div class="field">
          <label for="historyCount">Last checkpoints</label>
          <input id="historyCount" type="number" min="1" max="50" value="5">
        </div>
        <div class="field">
          <label for="historyMinOpacity">Oldest opacity</label>
          <input id="historyMinOpacity" type="number" min="0.01" max="1" step="0.01" value="0.08">
        </div>
      </div>
      <div class="grid">
        <div class="field">
          <label for="evolutionOpacity">Current opacity</label>
          <input id="evolutionOpacity" type="number" min="0.01" max="1" step="0.01" value="0.35">
        </div>
        <div class="field">
          <label for="evolutionColor">Color</label>
          <input id="evolutionColor" type="color" value="#1769aa">
        </div>
      </div>
      <button id="renderEvolutionButton" class="primary" type="button">Render step</button>
      <div class="status" id="evolutionStatus"></div>
    </section>
  </aside>
  <div id="sidebarResizer" class="sidebar-resizer" title="Drag to resize"></div>

  <main>
    <div class="toolbar">
      <div class="left">
        <button id="redrawButton" type="button">Redraw</button>
        <button id="savePng" type="button">Save PNG</button>
      </div>
      <div class="right">
        <span class="status" id="canvasStatus"></span>
      </div>
    </div>
    <div class="canvas-wrap">
      <canvas id="plot"></canvas>
      <div id="legend" class="legend"></div>
    </div>
  </main>
</div>

<script>
const COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2"];
const state = {
  runs: [],
  runsById: new Map(),
  checkpointsById: new Map(),
  filteredCompareRuns: [],
  compareLayers: [],
  evolutionLayers: [],
  activeTab: "compare",
  env: null,
  nextColor: 0,
  trajectoryCache: new Map(),
  evolutionTimer: null
};

const el = {
  runCount: document.getElementById("runCount"),
  compareTab: document.getElementById("compareTab"),
  evolutionTab: document.getElementById("evolutionTab"),
  comparePanel: document.getElementById("comparePanel"),
  evolutionPanel: document.getElementById("evolutionPanel"),
  maxTraj: document.getElementById("maxTraj"),
  stride: document.getElementById("stride"),
  lineWidth: document.getElementById("lineWidth"),
  compareStep: document.getElementById("compareStep"),
  algorithmFilter: document.getElementById("algorithmFilter"),
  entFilter: document.getElementById("entFilter"),
  targetFilter: document.getElementById("targetFilter"),
  lambdaFilter: document.getElementById("lambdaFilter"),
  stepsFilter: document.getElementById("stepsFilter"),
  searchFilter: document.getElementById("searchFilter"),
  runSelect: document.getElementById("runSelect"),
  addCompareRun: document.getElementById("addCompareRun"),
  clearCompareLayers: document.getElementById("clearCompareLayers"),
  renderCompareButton: document.getElementById("renderCompareButton"),
  compareStatus: document.getElementById("compareStatus"),
  compareLayers: document.getElementById("compareLayers"),
  evolutionSearch: document.getElementById("evolutionSearch"),
  evolutionRun: document.getElementById("evolutionRun"),
  evolutionStep: document.getElementById("evolutionStep"),
  evolutionStepText: document.getElementById("evolutionStepText"),
  evolutionStepCount: document.getElementById("evolutionStepCount"),
  historyEnabled: document.getElementById("historyEnabled"),
  historyCount: document.getElementById("historyCount"),
  historyMinOpacity: document.getElementById("historyMinOpacity"),
  evolutionOpacity: document.getElementById("evolutionOpacity"),
  evolutionColor: document.getElementById("evolutionColor"),
  renderEvolutionButton: document.getElementById("renderEvolutionButton"),
  evolutionStatus: document.getElementById("evolutionStatus"),
  sidebarResizer: document.getElementById("sidebarResizer"),
  redrawButton: document.getElementById("redrawButton"),
  savePng: document.getElementById("savePng"),
  canvasStatus: document.getElementById("canvasStatus"),
  canvas: document.getElementById("plot"),
  legend: document.getElementById("legend")
};

function fmt(value) {
  if (value === null || value === undefined) return "";
  return String(value);
}

function sortValue(value) {
  const text = fmt(value);
  if (text === "") return Number.POSITIVE_INFINITY;
  const number = Number(text);
  return Number.isFinite(number) ? number : text;
}

function stepText(step) {
  const number = Number(step);
  if (!Number.isFinite(number)) return fmt(step);
  return number.toLocaleString("en-US");
}

function runLabel(run) {
  const pieces = [
    run.algorithm_name,
    `ent=${run.ent_start}`,
    `target=${run.ent_target_mult}`,
    `lambda=${run.lmbda}`,
    `steps=${run.nr_steps}`,
    `${run.checkpoint_count} ckpts`,
    run.run_folder
  ];
  return pieces.join(" | ");
}

function checkpointForStep(run, step) {
  const numericStep = Number(step);
  return run.checkpoints.find(checkpoint => Number(checkpoint.step) === numericStep);
}

async function api(path) {
  const response = await fetch(path);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

function setActiveTab(tab) {
  state.activeTab = tab;
  el.compareTab.classList.toggle("active", tab === "compare");
  el.evolutionTab.classList.toggle("active", tab === "evolution");
  el.comparePanel.classList.toggle("active", tab === "compare");
  el.evolutionPanel.classList.toggle("active", tab === "evolution");
  draw();
}

function uniqueValues(key) {
  return [...new Set(state.runs.map(run => fmt(run[key])).filter(Boolean))]
    .sort((a, b) => {
      const av = sortValue(a);
      const bv = sortValue(b);
      if (typeof av === "number" && typeof bv === "number") return av - bv;
      return String(av).localeCompare(String(bv));
    });
}

function uniqueSteps() {
  return [...new Set(state.runs.flatMap(run => run.checkpoints.map(checkpoint => Number(checkpoint.step))))]
    .filter(step => Number.isFinite(step))
    .sort((a, b) => a - b);
}

function fillFilter(select, values, label) {
  const current = select.value;
  select.innerHTML = "";
  const all = document.createElement("option");
  all.value = "";
  all.textContent = label;
  select.appendChild(all);
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.appendChild(option);
  }
  if ([...select.options].some(option => option.value === current)) {
    select.value = current;
  }
}

function initFilters() {
  fillFilter(el.algorithmFilter, uniqueValues("algorithm_name"), "All algorithms");
  fillFilter(el.entFilter, uniqueValues("ent_start"), "All");
  fillFilter(el.targetFilter, uniqueValues("ent_target_mult"), "All");
  fillFilter(el.lambdaFilter, uniqueValues("lmbda"), "All");
  fillFilter(el.stepsFilter, uniqueValues("nr_steps"), "All");

  const steps = uniqueSteps();
  el.compareStep.innerHTML = "";
  for (const step of steps) {
    const option = document.createElement("option");
    option.value = String(step);
    option.textContent = stepText(step);
    el.compareStep.appendChild(option);
  }
  if (steps.length) {
    el.compareStep.value = String(steps[steps.length - 1]);
  }
}

function compareRunPassesFilters(run) {
  const step = el.compareStep.value;
  if (step && !checkpointForStep(run, step)) return false;
  if (el.algorithmFilter.value && fmt(run.algorithm_name) !== el.algorithmFilter.value) return false;
  if (el.entFilter.value && fmt(run.ent_start) !== el.entFilter.value) return false;
  if (el.targetFilter.value && fmt(run.ent_target_mult) !== el.targetFilter.value) return false;
  if (el.lambdaFilter.value && fmt(run.lmbda) !== el.lambdaFilter.value) return false;
  if (el.stepsFilter.value && fmt(run.nr_steps) !== el.stepsFilter.value) return false;
  const search = el.searchFilter.value.trim().toLowerCase();
  if (search) {
    const haystack = [
      run.algorithm_name,
      run.run_folder,
      run.run_id,
      run.run_name,
      run.ent_start,
      run.ent_target_mult,
      run.lmbda,
      run.nr_steps
    ].map(fmt).join(" ").toLowerCase();
    if (!haystack.includes(search)) return false;
  }
  return true;
}

function filterCompareRuns() {
  const selectedLayerIds = new Set(state.compareLayers.map(layer => layer.key));
  state.filteredCompareRuns = state.runs.filter(run => {
    const checkpoint = checkpointForStep(run, el.compareStep.value);
    if (!checkpoint) return false;
    if (selectedLayerIds.has(`${run.id}::${checkpoint.step}`)) return false;
    return compareRunPassesFilters(run);
  });

  state.filteredCompareRuns.sort((a, b) => {
    const keys = [
      ["algorithm_name", false],
      ["ent_start", true],
      ["ent_target_mult", true],
      ["lmbda", true],
      ["nr_steps", true],
      ["run_folder", false]
    ];
    for (const [key, numeric] of keys) {
      const av = numeric ? sortValue(a[key]) : fmt(a[key]);
      const bv = numeric ? sortValue(b[key]) : fmt(b[key]);
      if (typeof av === "number" && typeof bv === "number" && av !== bv) return av - bv;
      const cmp = String(av).localeCompare(String(bv));
      if (cmp !== 0) return cmp;
    }
    return 0;
  });
  renderCompareRunSelect();
}

function renderCompareRunSelect() {
  el.runSelect.innerHTML = "";
  for (const run of state.filteredCompareRuns) {
    const option = document.createElement("option");
    option.value = run.id;
    option.textContent = runLabel(run);
    option.title = runLabel(run);
    el.runSelect.appendChild(option);
  }
  el.runCount.textContent = `${state.runs.length} runs, ${state.filteredCompareRuns.length} shown`;
}

function selectedCompareRuns() {
  return [...el.runSelect.selectedOptions]
    .map(option => state.runsById.get(option.value))
    .filter(Boolean);
}

function addSelectedCompareRuns() {
  const step = el.compareStep.value;
  const runs = selectedCompareRuns();
  if (!runs.length || !step) return;
  const selectedKeys = new Set(state.compareLayers.map(layer => layer.key));
  for (const run of runs) {
    const checkpoint = checkpointForStep(run, step);
    if (!checkpoint) continue;
    const key = `${run.id}::${checkpoint.step}`;
    if (selectedKeys.has(key)) continue;
    selectedKeys.add(key);
    const color = COLORS[state.nextColor % COLORS.length];
    state.nextColor += 1;
    state.compareLayers.push({
      key,
      run,
      checkpoint,
      label: `${run.algorithm_name} ent=${run.ent_start} target=${run.ent_target_mult} lambda=${run.lmbda} step=${stepText(checkpoint.step)}`,
      color,
      opacity: 0.35,
      visible: true,
      loaded: false,
      loading: false,
      data: null
    });
  }
  filterCompareRuns();
  renderCompareLayers();
  draw();
}

function renderCompareLayers() {
  el.compareLayers.innerHTML = "";
  if (!state.compareLayers.length) {
    const empty = document.createElement("div");
    empty.className = "muted";
    empty.textContent = "No layers selected.";
    el.compareLayers.appendChild(empty);
    return;
  }
  state.compareLayers.forEach((layer, index) => {
    const item = document.createElement("div");
    item.className = "layer";

    const top = document.createElement("div");
    top.className = "layer-top";

    const visible = document.createElement("input");
    visible.type = "checkbox";
    visible.checked = layer.visible;
    visible.addEventListener("input", () => {
      layer.visible = visible.checked;
      draw();
    });

    const title = document.createElement("div");
    title.className = "layer-title";
    title.title = `${layer.run.run_folder} | ${layer.checkpoint.step_label}`;
    title.innerHTML = `<strong>${layer.run.algorithm_name}</strong><span>${layer.run.run_folder} | ${stepText(layer.checkpoint.step)}</span>`;

    const color = document.createElement("input");
    color.type = "color";
    color.value = layer.color;
    color.addEventListener("input", () => {
      layer.color = color.value;
      draw();
    });

    const remove = document.createElement("button");
    remove.className = "danger";
    remove.textContent = "Remove";
    remove.addEventListener("click", () => {
      state.compareLayers.splice(index, 1);
      filterCompareRuns();
      renderCompareLayers();
      draw();
    });

    top.appendChild(visible);
    top.appendChild(title);
    top.appendChild(color);
    top.appendChild(remove);

    const controls = document.createElement("div");
    controls.className = "layer-controls";
    const opacity = document.createElement("input");
    opacity.type = "range";
    opacity.min = "0";
    opacity.max = "1";
    opacity.step = "0.01";
    opacity.value = String(layer.opacity);
    const opacityText = document.createElement("span");
    opacityText.className = "muted";
    opacityText.textContent = layer.opacity.toFixed(2);
    opacity.addEventListener("input", () => {
      layer.opacity = Number(opacity.value);
      opacityText.textContent = layer.opacity.toFixed(2);
      draw();
    });
    controls.appendChild(opacity);
    controls.appendChild(opacityText);

    const stats = document.createElement("div");
    stats.className = "stats";
    const load = layer.loaded ? `${layer.data.stats.loaded_trajectories}/${layer.data.stats.available_trajectories} loaded` : "not loaded";
    const returns = layer.loaded ? `return ${layer.data.stats.return_mean.toFixed(2)} +- ${layer.data.stats.return_std.toFixed(2)}` : "";
    stats.textContent = `step ${stepText(layer.checkpoint.step)} | ${load}${returns ? " | " + returns : ""}`;

    item.appendChild(top);
    item.appendChild(controls);
    item.appendChild(stats);
    el.compareLayers.appendChild(item);
  });
}

function trajectoryCacheKey(checkpointId) {
  return `${checkpointId}|${el.maxTraj.value}|${el.stride.value}`;
}

async function loadCheckpointData(checkpoint) {
  const key = trajectoryCacheKey(checkpoint.id);
  if (state.trajectoryCache.has(key)) {
    return state.trajectoryCache.get(key);
  }
  const params = new URLSearchParams({
    id: checkpoint.id,
    max_trajectories: el.maxTraj.value,
    stride: el.stride.value
  });
  const data = await api(`/api/trajectories?${params.toString()}`);
  state.trajectoryCache.set(key, data);
  return data;
}

async function loadCompareLayer(layer) {
  if (layer.loaded || layer.loading) return;
  layer.loading = true;
  renderCompareLayers();
  layer.data = await loadCheckpointData(layer.checkpoint);
  layer.loaded = true;
  layer.loading = false;
}

async function renderCompareSelected() {
  if (!state.compareLayers.length) {
    el.compareStatus.textContent = "Add at least one run.";
    return;
  }
  el.renderCompareButton.disabled = true;
  try {
    for (let i = 0; i < state.compareLayers.length; i += 1) {
      const layer = state.compareLayers[i];
      el.compareStatus.textContent = `Loading ${i + 1}/${state.compareLayers.length}: ${layer.run.run_folder}`;
      layer.loaded = false;
      layer.data = null;
      await loadCompareLayer(layer);
    }
    el.compareStatus.textContent = "Ready.";
    renderCompareLayers();
    draw();
  } catch (error) {
    el.compareStatus.textContent = `Error: ${error.message}`;
  } finally {
    el.renderCompareButton.disabled = false;
  }
}

function evolutionRunPassesSearch(run) {
  const search = el.evolutionSearch.value.trim().toLowerCase();
  if (!search) return true;
  const haystack = [
    run.algorithm_name,
    run.run_folder,
    run.run_id,
    run.run_name,
    run.ent_start,
    run.ent_target_mult,
    run.lmbda,
    run.nr_steps
  ].map(fmt).join(" ").toLowerCase();
  return haystack.includes(search);
}

function populateEvolutionRuns() {
  const current = el.evolutionRun.value;
  el.evolutionRun.innerHTML = "";
  const runs = state.runs.filter(evolutionRunPassesSearch);
  for (const run of runs) {
    const option = document.createElement("option");
    option.value = run.id;
    option.textContent = runLabel(run);
    option.title = runLabel(run);
    el.evolutionRun.appendChild(option);
  }
  if ([...el.evolutionRun.options].some(option => option.value === current)) {
    el.evolutionRun.value = current;
  } else if (el.evolutionRun.options.length) {
    el.evolutionRun.value = el.evolutionRun.options[0].value;
  }
  updateEvolutionRun();
}

function selectedEvolutionRun() {
  return state.runsById.get(el.evolutionRun.value) || null;
}

function updateEvolutionRun() {
  const run = selectedEvolutionRun();
  if (!run) {
    el.evolutionStep.max = "0";
    el.evolutionStep.value = "0";
    el.evolutionStepText.textContent = "";
    el.evolutionStepCount.textContent = "";
    return;
  }
  const maxIndex = Math.max(0, run.checkpoints.length - 1);
  el.evolutionStep.max = String(maxIndex);
  if (Number(el.evolutionStep.value) > maxIndex) {
    el.evolutionStep.value = String(maxIndex);
  }
  if (el.evolutionStep.value === "0" && run.checkpoints.length > 1) {
    el.evolutionStep.value = String(maxIndex);
  }
  updateEvolutionStepText();
}

function currentEvolutionCheckpointIndex() {
  const run = selectedEvolutionRun();
  if (!run) return 0;
  return Math.min(Math.max(0, Number(el.evolutionStep.value)), run.checkpoints.length - 1);
}

function updateEvolutionStepText() {
  const run = selectedEvolutionRun();
  if (!run) {
    el.evolutionStepText.textContent = "";
    el.evolutionStepCount.textContent = "";
    return;
  }
  const index = currentEvolutionCheckpointIndex();
  const checkpoint = run.checkpoints[index];
  el.evolutionStepText.textContent = `step ${stepText(checkpoint.step)}`;
  el.evolutionStepCount.textContent = `${index + 1}/${run.checkpoints.length}`;
}

function evolutionLayerWindow() {
  const run = selectedEvolutionRun();
  if (!run) return [];
  const currentIndex = currentEvolutionCheckpointIndex();
  if (!el.historyEnabled.checked) {
    return [{run, checkpoint: run.checkpoints[currentIndex], rank: 0, count: 1}];
  }
  const count = Math.max(1, Math.min(Number(el.historyCount.value) || 1, run.checkpoints.length));
  const start = Math.max(0, currentIndex - count + 1);
  const checkpoints = run.checkpoints.slice(start, currentIndex + 1);
  return checkpoints.map((checkpoint, index) => ({run, checkpoint, rank: index, count: checkpoints.length}));
}

function historyOpacity(rank, count) {
  const currentOpacity = Math.max(0.01, Math.min(1, Number(el.evolutionOpacity.value) || 0.35));
  if (count <= 1) return currentOpacity;
  const minOpacity = Math.max(0.01, Math.min(currentOpacity, Number(el.historyMinOpacity.value) || 0.08));
  const t = rank / (count - 1);
  return minOpacity + (currentOpacity - minOpacity) * t;
}

function hslToHex(h, s, l) {
  const a = s * Math.min(l, 1 - l);
  const f = n => {
    const k = (n + h / 30) % 12;
    const color = l - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
    return Math.round(255 * color).toString(16).padStart(2, "0");
  };
  return `#${f(0)}${f(8)}${f(4)}`;
}

function evolutionCheckpointColor(rank, count) {
  if (count <= 1) return el.evolutionColor.value;
  const hue = (205 + rank * 137.508) % 360;
  return hslToHex(hue, 0.68, 0.45);
}

async function renderEvolution() {
  const run = selectedEvolutionRun();
  if (!run) {
    el.evolutionStatus.textContent = "Select a run.";
    return;
  }
  el.renderEvolutionButton.disabled = true;
  try {
    const layerSpecs = evolutionLayerWindow();
    const nextLayers = [];
    for (let i = 0; i < layerSpecs.length; i += 1) {
      const spec = layerSpecs[i];
      el.evolutionStatus.textContent = `Loading ${i + 1}/${layerSpecs.length}: step ${stepText(spec.checkpoint.step)}`;
      const data = await loadCheckpointData(spec.checkpoint);
      const isCurrent = i === layerSpecs.length - 1;
      nextLayers.push({
        key: `${spec.run.id}::${spec.checkpoint.step}`,
        run: spec.run,
        checkpoint: spec.checkpoint,
        label: `${isCurrent ? "current" : "previous"} step=${stepText(spec.checkpoint.step)} | ${spec.run.run_folder}`,
        color: evolutionCheckpointColor(spec.rank, spec.count),
        opacity: historyOpacity(spec.rank, spec.count),
        visible: true,
        loaded: true,
        data
      });
    }
    state.evolutionLayers = nextLayers;
    el.evolutionStatus.textContent = "Ready.";
    draw();
  } catch (error) {
    el.evolutionStatus.textContent = `Error: ${error.message}`;
  } finally {
    el.renderEvolutionButton.disabled = false;
  }
}

function scheduleEvolutionRender() {
  updateEvolutionStepText();
  window.clearTimeout(state.evolutionTimer);
  state.evolutionTimer = window.setTimeout(() => {
    if (state.activeTab === "evolution") renderEvolution();
  }, 160);
}

function activeLayers() {
  return state.activeTab === "compare" ? state.compareLayers : state.evolutionLayers;
}

function setupCanvas() {
  const rect = el.canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(320, Math.floor(rect.width * dpr));
  const height = Math.max(320, Math.floor(rect.height * dpr));
  if (el.canvas.width !== width || el.canvas.height !== height) {
    el.canvas.width = width;
    el.canvas.height = height;
  }
  const ctx = el.canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return {ctx, width: rect.width, height: rect.height};
}

function transform(width, height) {
  const view = state.env.view;
  const pad = 34;
  const sx = (width - pad * 2) / (view.x_max - view.x_min);
  const sy = (height - pad * 2) / (view.y_max - view.y_min);
  const scale = Math.min(sx, sy);
  const plotW = (view.x_max - view.x_min) * scale;
  const plotH = (view.y_max - view.y_min) * scale;
  const ox = (width - plotW) / 2;
  const oy = (height - plotH) / 2;
  return {
    toX: x => ox + (x - view.x_min) * scale,
    toY: y => oy + plotH - (y - view.y_min) * scale,
    scale,
    ox,
    oy,
    plotW,
    plotH
  };
}

function drawEnvironment(ctx, tx) {
  const view = state.env.view;
  ctx.save();
  ctx.lineWidth = 1;
  ctx.strokeStyle = "#9aa4b2";
  ctx.strokeRect(tx.toX(view.x_min), tx.toY(view.y_max), tx.plotW, tx.plotH);

  ctx.strokeStyle = "#2f855a";
  ctx.setLineDash([6, 5]);
  ctx.beginPath();
  ctx.moveTo(tx.toX(view.x_min), tx.toY(state.env.goal_y));
  ctx.lineTo(tx.toX(view.x_max), tx.toY(state.env.goal_y));
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = "#111827";
  for (const obstacle of state.env.obstacles) {
    ctx.beginPath();
    ctx.arc(tx.toX(obstacle.x), tx.toY(obstacle.y), obstacle.r * tx.scale, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.fillStyle = "#0f766e";
  ctx.beginPath();
  ctx.arc(tx.toX(state.env.start[0]), tx.toY(state.env.start[1]), 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function hexToRgb(hex) {
  const clean = hex.replace("#", "");
  return {
    r: parseInt(clean.slice(0, 2), 16),
    g: parseInt(clean.slice(2, 4), 16),
    b: parseInt(clean.slice(4, 6), 16)
  };
}

function drawLayer(ctx, tx, layer) {
  if (!layer.visible || !layer.loaded || !layer.data) return;
  const rgb = hexToRgb(layer.color);
  ctx.save();
  ctx.strokeStyle = `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${layer.opacity})`;
  ctx.lineWidth = Number(el.lineWidth.value);
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  for (const trajectory of layer.data.trajectories) {
    if (trajectory.length < 2) continue;
    ctx.beginPath();
    ctx.moveTo(tx.toX(trajectory[0][0]), tx.toY(trajectory[0][1]));
    for (let i = 1; i < trajectory.length; i += 1) {
      ctx.lineTo(tx.toX(trajectory[i][0]), tx.toY(trajectory[i][1]));
    }
    ctx.stroke();
  }
  ctx.restore();
}

function renderLegend() {
  const visible = activeLayers().filter(layer => layer.visible && layer.loaded);
  if (!visible.length) {
    el.legend.style.display = "none";
    return;
  }
  el.legend.style.display = "block";
  el.legend.innerHTML = visible.map(layer => {
    return `<div class="legend-row"><span class="swatch" style="background:${layer.color}; opacity:${Math.max(layer.opacity, 0.2)}"></span><span>${layer.label}</span></div>`;
  }).join("");
}

function draw() {
  if (!state.env) return;
  const {ctx, width, height} = setupCanvas();
  ctx.clearRect(0, 0, width, height);
  const tx = transform(width, height);
  drawEnvironment(ctx, tx);
  for (const layer of activeLayers()) {
    drawLayer(ctx, tx, layer);
  }
  renderLegend();
  const layers = activeLayers();
  const loaded = layers.filter(layer => layer.loaded).length;
  el.canvasStatus.textContent = `${loaded}/${layers.length} layers loaded`;
}

function initSidebarResizer() {
  const savedWidth = Number(localStorage.getItem("avoiding2dCheckpointSidebarWidth"));
  if (Number.isFinite(savedWidth) && savedWidth > 280) {
    document.documentElement.style.setProperty("--sidebar-width", `${savedWidth}px`);
  }
  let startX = 0;
  let startWidth = 0;

  function onMove(event) {
    const nextWidth = Math.min(Math.max(startWidth + event.clientX - startX, 320), Math.max(560, window.innerWidth - 360));
    document.documentElement.style.setProperty("--sidebar-width", `${nextWidth}px`);
    localStorage.setItem("avoiding2dCheckpointSidebarWidth", String(Math.round(nextWidth)));
    draw();
  }

  function onUp() {
    document.body.classList.remove("resizing-sidebar");
    window.removeEventListener("pointermove", onMove);
    window.removeEventListener("pointerup", onUp);
  }

  el.sidebarResizer.addEventListener("pointerdown", event => {
    if (window.innerWidth <= 960) return;
    startX = event.clientX;
    startWidth = document.querySelector("aside").getBoundingClientRect().width;
    document.body.classList.add("resizing-sidebar");
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  });
}

function wireEvents() {
  el.compareTab.addEventListener("click", () => setActiveTab("compare"));
  el.evolutionTab.addEventListener("click", () => setActiveTab("evolution"));
  for (const node of [el.compareStep, el.algorithmFilter, el.entFilter, el.targetFilter, el.lambdaFilter, el.stepsFilter]) {
    node.addEventListener("change", filterCompareRuns);
  }
  el.searchFilter.addEventListener("input", filterCompareRuns);
  el.addCompareRun.addEventListener("click", addSelectedCompareRuns);
  el.clearCompareLayers.addEventListener("click", () => {
    state.compareLayers = [];
    filterCompareRuns();
    renderCompareLayers();
    draw();
  });
  el.renderCompareButton.addEventListener("click", renderCompareSelected);
  el.evolutionSearch.addEventListener("input", populateEvolutionRuns);
  el.evolutionRun.addEventListener("change", () => {
    updateEvolutionRun();
    renderEvolution();
  });
  el.evolutionStep.addEventListener("input", scheduleEvolutionRender);
  for (const node of [el.historyEnabled, el.historyCount, el.historyMinOpacity, el.evolutionOpacity, el.evolutionColor]) {
    node.addEventListener("input", () => {
      if (state.activeTab === "evolution") renderEvolution();
    });
  }
  el.renderEvolutionButton.addEventListener("click", renderEvolution);
  el.redrawButton.addEventListener("click", draw);
  el.lineWidth.addEventListener("input", draw);
  el.maxTraj.addEventListener("change", () => {
    for (const layer of state.compareLayers) layer.loaded = false;
    state.evolutionLayers = [];
    renderCompareLayers();
    draw();
  });
  el.stride.addEventListener("change", () => {
    for (const layer of state.compareLayers) layer.loaded = false;
    state.evolutionLayers = [];
    renderCompareLayers();
    draw();
  });
  el.savePng.addEventListener("click", () => {
    const link = document.createElement("a");
    link.download = state.activeTab === "compare" ? "avoiding2d_checkpoint_compare.png" : "avoiding2d_training_evolution.png";
    link.href = el.canvas.toDataURL("image/png");
    link.click();
  });
  window.addEventListener("resize", draw);
}

async function init() {
  wireEvents();
  state.env = await api("/api/env");
  const payload = await api("/api/runs");
  state.runs = payload.runs;
  state.runsById = new Map(state.runs.map(run => [run.id, run]));
  state.checkpointsById = new Map(state.runs.flatMap(run => run.checkpoints.map(checkpoint => [checkpoint.id, checkpoint])));
  initFilters();
  filterCompareRuns();
  renderCompareLayers();
  populateEvolutionRuns();
  draw();
}

initSidebarResizer();
init().catch(error => {
  el.compareStatus.textContent = `Error: ${error.message}`;
  el.evolutionStatus.textContent = `Error: ${error.message}`;
});
</script>
</body>
</html>
"""


class CheckpointTrajectoryGuiHandler(BaseHTTPRequestHandler):
    records: list[TrainingRunRecord] = []
    checkpoint_records_by_id: dict[str, CheckpointRecord] = {}
    output_root: Path = DEFAULT_OUTPUT_ROOT

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.log_date_time_string(), fmt % args))

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(json_safe(payload), separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, text: str, content_type: str = "text/html; charset=utf-8") -> None:
        body = text.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status: HTTPStatus, message: str) -> None:
        self.send_json({"error": message}, status=status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path in ("", "/"):
                self.send_text(HTML)
                return
            if parsed.path == "/api/env":
                self.send_json(ENV_META)
                return
            if parsed.path == "/api/runs":
                self.send_json({"runs": [record.to_json(self.output_root) for record in self.records]})
                return
            if parsed.path == "/api/trajectories":
                query = parse_qs(parsed.query)
                checkpoint_id = query.get("id", [""])[0]
                record = self.checkpoint_records_by_id.get(checkpoint_id)
                if record is None:
                    self.send_error_json(HTTPStatus.NOT_FOUND, f"unknown checkpoint id: {checkpoint_id}")
                    return
                max_trajectories = int(query.get("max_trajectories", ["1024"])[0])
                stride = int(query.get("stride", ["1"])[0])
                self.send_json(load_trajectories(record, max_trajectories, stride))
                return
            self.send_error_json(HTTPStatus.NOT_FOUND, "not found")
        except Exception as exc:
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local GUI for checkpointed Avoiding2D trajectory NPZs.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Mapped Avoiding2D output root.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    if not output_root.exists():
        raise FileNotFoundError(f"output root does not exist: {output_root}")

    records = scan_checkpoint_runs(output_root)
    if not records:
        raise RuntimeError(f"no checkpoint trajectories.npz files found under {output_root}")

    checkpoint_records_by_id = {
        checkpoint.id: checkpoint
        for record in records
        for checkpoint in record.checkpoints
    }

    handler_class = CheckpointTrajectoryGuiHandler
    handler_class.records = records
    handler_class.checkpoint_records_by_id = checkpoint_records_by_id
    handler_class.output_root = output_root

    port = choose_port(args.host, args.port)
    server = ThreadingHTTPServer((args.host, port), handler_class)
    actual_host, actual_port = server.server_address
    checkpoint_count = sum(len(record.checkpoints) for record in records)
    print(f"Loaded {len(records)} runs and {checkpoint_count} checkpoint trajectory files from {output_root}", flush=True)
    print(f"Open http://{actual_host}:{actual_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
