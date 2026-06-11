#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "wandb_map_outputs" / "avoiding2d"


ENV_META = {
    "view": {"x_min": 0.2, "x_max": 0.8, "y_min": -0.35, "y_max": 0.42},
    "start": [0.5, -0.28],
    "goal_y": 0.35,
    "obstacles": [
        {"x": 0.32, "y": 0.08, "r": 0.025, "layer": 2},
        {"x": 0.44, "y": 0.08, "r": 0.025, "layer": 2},
        {"x": 0.56, "y": 0.08, "r": 0.025, "layer": 2},
        {"x": 0.68, "y": 0.08, "r": 0.025, "layer": 2},
        {"x": 0.26, "y": 0.26, "r": 0.025, "layer": 3},
        {"x": 0.38, "y": 0.26, "r": 0.06, "layer": 3},
        {"x": 0.50, "y": 0.26, "r": 0.025, "layer": 3},
        {"x": 0.62, "y": 0.26, "r": 0.06, "layer": 3},
        {"x": 0.74, "y": 0.26, "r": 0.025, "layer": 3},
    ],
}


def unwrap(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value and len(value) <= 3:
        return value["value"]
    return value


def config_get(config: dict[str, Any], dotted_key: str, default: Any = "") -> Any:
    if dotted_key in config:
        return unwrap(config[dotted_key])
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = unwrap(current[part])
    return current


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return str(value)


def run_metadata(output_dir: Path) -> dict[str, str]:
    result = {"id": "", "name": "", "url": ""}
    for filename in ("render_result.json", "value_result.json", "run_manifest.json"):
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


@dataclass(frozen=True)
class RunRecord:
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
    return_mean: Any
    return_std: Any
    return_min: Any
    return_max: Any
    reached_mode_counts: Any
    output_dir: Path
    npz_path: Path
    png_path: Path

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
            "return_mean": self.return_mean,
            "return_std": self.return_std,
            "return_min": self.return_min,
            "return_max": self.return_max,
            "reached_mode_counts": self.reached_mode_counts,
            "output_dir": self.output_dir.relative_to(output_root).as_posix(),
            "npz": self.npz_path.relative_to(output_root).as_posix(),
            "png": self.png_path.relative_to(output_root).as_posix() if self.png_path.exists() else "",
        }


def render_result_stats(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "render_result.json"
    if not path.exists():
        return {}
    try:
        payload = load_json(path)
    except (OSError, json.JSONDecodeError):
        return {}
    render = payload.get("render", {}) if isinstance(payload, dict) else {}
    return render if isinstance(render, dict) else {}


def scan_runs(output_root: Path) -> list[RunRecord]:
    records: list[RunRecord] = []
    for npz_path in sorted(output_root.glob("**/trajectories.npz")):
        output_dir = npz_path.parent
        config_path = output_dir / "config.json"
        if not config_path.exists():
            continue
        try:
            config = load_json(config_path)
        except (OSError, json.JSONDecodeError):
            continue
        meta = run_metadata(output_dir)
        stats = render_result_stats(output_dir)
        relative_id = npz_path.relative_to(output_root).as_posix()
        records.append(
            RunRecord(
                id=relative_id,
                run_folder=output_dir.parent.name,
                run_id=meta["id"],
                run_name=meta["name"],
                run_url=meta["url"],
                algorithm_name=str(config_get(config, "algorithm.name", "")),
                ent_start=config_get(config, "algorithm.ent_start", ""),
                ent_target_mult=config_get(config, "algorithm.ent_target_mult", ""),
                lmbda=config_get(config, "algorithm.lmbda", ""),
                nr_steps=config_get(config, "algorithm.nr_steps", ""),
                return_mean=stats.get("return_mean", ""),
                return_std=stats.get("return_std", ""),
                return_min=stats.get("return_min", ""),
                return_max=stats.get("return_max", ""),
                reached_mode_counts=stats.get("reached_mode_counts", ""),
                output_dir=output_dir,
                npz_path=npz_path,
                png_path=output_dir / "trajectories.png",
            )
        )
    return records


def first_done_length(done_column: np.ndarray, max_points: int) -> int:
    done_indices = np.flatnonzero(done_column)
    if done_indices.size == 0:
        return max_points
    return min(int(done_indices[0]) + 2, max_points)


def load_trajectories(record: RunRecord, max_trajectories: int, stride: int) -> dict[str, Any]:
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

    returns = np.sum(rewards[:, indices], axis=0)
    reached_modes = final_modes[indices] > 0.5 if final_modes.size else np.zeros((indices.shape[0], 0), dtype=bool)
    return {
        "id": record.id,
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
  <title>Avoiding2D Trajectory Overlay</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --panel: #ffffff;
      --line: #d7dce3;
      --text: #20242b;
      --muted: #6a7280;
      --accent: #1769aa;
      --danger: #b42318;
      --sidebar-width: 430px;
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
      margin: 0 0 12px;
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
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .field {
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
      max-width: 360px;
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
    <h1>Avoiding2D Trajectories</h1>
    <div class="muted" id="runCount">Loading runs...</div>

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
      <button id="addRun">Add selected</button>
      <button id="clearLayers" class="danger">Clear</button>
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
    <button id="renderButton" class="primary">Render selected</button>
    <div class="status" id="status"></div>

    <h2>Layers</h2>
    <div id="layers"></div>
  </aside>
  <div id="sidebarResizer" class="sidebar-resizer" title="Drag to resize"></div>

  <main>
    <div class="toolbar">
      <div class="left">
        <button id="redrawButton">Redraw</button>
        <button id="savePng">Save PNG</button>
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
  filtered: [],
  layers: [],
  env: null,
  nextColor: 0
};

const el = {
  runCount: document.getElementById("runCount"),
  algorithmFilter: document.getElementById("algorithmFilter"),
  entFilter: document.getElementById("entFilter"),
  targetFilter: document.getElementById("targetFilter"),
  lambdaFilter: document.getElementById("lambdaFilter"),
  stepsFilter: document.getElementById("stepsFilter"),
  searchFilter: document.getElementById("searchFilter"),
  runSelect: document.getElementById("runSelect"),
  sidebarResizer: document.getElementById("sidebarResizer"),
  addRun: document.getElementById("addRun"),
  clearLayers: document.getElementById("clearLayers"),
  maxTraj: document.getElementById("maxTraj"),
  stride: document.getElementById("stride"),
  lineWidth: document.getElementById("lineWidth"),
  renderButton: document.getElementById("renderButton"),
  redrawButton: document.getElementById("redrawButton"),
  savePng: document.getElementById("savePng"),
  status: document.getElementById("status"),
  canvasStatus: document.getElementById("canvasStatus"),
  layers: document.getElementById("layers"),
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

function runLabel(run) {
  const pieces = [
    run.algorithm_name,
    `ent=${run.ent_start}`,
    `target=${run.ent_target_mult}`,
    `lambda=${run.lmbda}`,
    `steps=${run.nr_steps}`,
    run.run_folder
  ];
  return pieces.join(" | ");
}

function setStatus(text) {
  el.status.textContent = text;
}

async function api(path) {
  const response = await fetch(path);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `${response.status} ${response.statusText}`);
  }
  return response.json();
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
}

function filterRuns() {
  const algorithm = el.algorithmFilter.value;
  const ent = el.entFilter.value;
  const target = el.targetFilter.value;
  const lambda = el.lambdaFilter.value;
  const steps = el.stepsFilter.value;
  const search = el.searchFilter.value.trim().toLowerCase();
  const selectedIds = new Set(state.layers.map(layer => layer.run.id));

  state.filtered = state.runs.filter(run => {
    if (selectedIds.has(run.id)) return false;
    if (algorithm && fmt(run.algorithm_name) !== algorithm) return false;
    if (ent && fmt(run.ent_start) !== ent) return false;
    if (target && fmt(run.ent_target_mult) !== target) return false;
    if (lambda && fmt(run.lmbda) !== lambda) return false;
    if (steps && fmt(run.nr_steps) !== steps) return false;
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
  });

  state.filtered.sort((a, b) => {
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
  renderRunSelect();
}

function renderRunSelect() {
  el.runSelect.innerHTML = "";
  for (const run of state.filtered) {
    const option = document.createElement("option");
    option.value = run.id;
    option.textContent = runLabel(run);
    option.title = runLabel(run);
    el.runSelect.appendChild(option);
  }
  el.runCount.textContent = `${state.runs.length} trajectory files, ${state.filtered.length} shown`;
}

function selectedRuns() {
  return [...el.runSelect.selectedOptions]
    .map(option => state.runs.find(run => run.id === option.value))
    .filter(Boolean);
}

function addSelectedRun() {
  const runs = selectedRuns();
  if (runs.length === 0) return;
  const selectedIds = new Set(state.layers.map(layer => layer.run.id));
  for (const run of runs) {
    if (selectedIds.has(run.id)) continue;
    selectedIds.add(run.id);
    const color = COLORS[state.nextColor % COLORS.length];
    state.nextColor += 1;
    state.layers.push({
      run,
      color,
      opacity: 0.35,
      visible: true,
      loaded: false,
      loading: false,
      data: null
    });
  }
  filterRuns();
  renderLayers();
  draw();
}

function renderLayers() {
  el.layers.innerHTML = "";
  if (state.layers.length === 0) {
    const empty = document.createElement("div");
    empty.className = "muted";
    empty.textContent = "No layers selected.";
    el.layers.appendChild(empty);
    return;
  }
  state.layers.forEach((layer, index) => {
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
    title.title = `${layer.run.algorithm_name} | ${layer.run.run_folder}`;
    title.innerHTML = `<strong>${layer.run.algorithm_name}</strong><span>${layer.run.run_folder}</span>`;

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
      state.layers.splice(index, 1);
      filterRuns();
      renderLayers();
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
    const params = `ent=${layer.run.ent_start}, target=${layer.run.ent_target_mult}, lambda=${layer.run.lmbda}, steps=${layer.run.nr_steps}`;
    const load = layer.loaded ? `${layer.data.stats.loaded_trajectories}/${layer.data.stats.available_trajectories} loaded` : "not loaded";
    const returns = layer.loaded ? `return ${layer.data.stats.return_mean.toFixed(2)} +- ${layer.data.stats.return_std.toFixed(2)}` : "";
    stats.textContent = `${params} | ${load}${returns ? " | " + returns : ""}`;

    item.appendChild(top);
    item.appendChild(controls);
    item.appendChild(stats);
    el.layers.appendChild(item);
  });
}

async function loadLayer(layer) {
  if (layer.loaded || layer.loading) return;
  layer.loading = true;
  renderLayers();
  const params = new URLSearchParams({
    id: layer.run.id,
    max_trajectories: el.maxTraj.value,
    stride: el.stride.value
  });
  layer.data = await api(`/api/trajectories?${params.toString()}`);
  layer.loaded = true;
  layer.loading = false;
}

async function renderSelected() {
  if (state.layers.length === 0) {
    setStatus("Add at least one run.");
    return;
  }
  el.renderButton.disabled = true;
  try {
    for (let i = 0; i < state.layers.length; i += 1) {
      const layer = state.layers[i];
      setStatus(`Loading ${i + 1}/${state.layers.length}: ${layer.run.run_folder}`);
      if (layer.loaded) {
        layer.loaded = false;
        layer.data = null;
      }
      await loadLayer(layer);
    }
    setStatus("Ready.");
    renderLayers();
    draw();
  } catch (error) {
    setStatus(`Error: ${error.message}`);
  } finally {
    el.renderButton.disabled = false;
  }
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
  const visible = state.layers.filter(layer => layer.visible && layer.loaded);
  if (!visible.length) {
    el.legend.style.display = "none";
    return;
  }
  el.legend.style.display = "block";
  el.legend.innerHTML = visible.map(layer => {
    const label = `${layer.run.algorithm_name} ent=${layer.run.ent_start} target=${layer.run.ent_target_mult} lambda=${layer.run.lmbda} steps=${layer.run.nr_steps}`;
    return `<div class="legend-row"><span class="swatch" style="background:${layer.color}; opacity:${Math.max(layer.opacity, 0.2)}"></span><span>${label}</span></div>`;
  }).join("");
}

function draw() {
  if (!state.env) return;
  const {ctx, width, height} = setupCanvas();
  ctx.clearRect(0, 0, width, height);
  const tx = transform(width, height);
  drawEnvironment(ctx, tx);
  for (const layer of state.layers) {
    drawLayer(ctx, tx, layer);
  }
  renderLegend();
  const loaded = state.layers.filter(layer => layer.loaded).length;
  const selected = state.layers.length;
  el.canvasStatus.textContent = `${loaded}/${selected} layers loaded`;
}

function initSidebarResizer() {
  const savedWidth = Number(localStorage.getItem("avoiding2dSidebarWidth"));
  if (Number.isFinite(savedWidth) && savedWidth > 260) {
    document.documentElement.style.setProperty("--sidebar-width", `${savedWidth}px`);
  }
  let startX = 0;
  let startWidth = 0;

  function onMove(event) {
    const nextWidth = Math.min(Math.max(startWidth + event.clientX - startX, 300), Math.max(520, window.innerWidth - 360));
    document.documentElement.style.setProperty("--sidebar-width", `${nextWidth}px`);
    localStorage.setItem("avoiding2dSidebarWidth", String(Math.round(nextWidth)));
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

async function init() {
  state.env = await api("/api/env");
  const payload = await api("/api/runs");
  state.runs = payload.runs;
  initFilters();
  filterRuns();
  renderLayers();
  draw();
}

for (const node of [el.algorithmFilter, el.entFilter, el.targetFilter, el.lambdaFilter, el.stepsFilter]) {
  node.addEventListener("change", filterRuns);
}
el.searchFilter.addEventListener("input", filterRuns);
el.addRun.addEventListener("click", addSelectedRun);
el.clearLayers.addEventListener("click", () => {
  state.layers = [];
  filterRuns();
  renderLayers();
  draw();
});
el.renderButton.addEventListener("click", renderSelected);
el.redrawButton.addEventListener("click", draw);
el.lineWidth.addEventListener("input", draw);
el.savePng.addEventListener("click", () => {
  const link = document.createElement("a");
  link.download = "avoiding2d_overlay.png";
  link.href = el.canvas.toDataURL("image/png");
  link.click();
});
window.addEventListener("resize", draw);

initSidebarResizer();
init().catch(error => setStatus(`Error: ${error.message}`));
</script>
</body>
</html>
"""


class TrajectoryGuiHandler(BaseHTTPRequestHandler):
    records: list[RunRecord] = []
    records_by_id: dict[str, RunRecord] = {}
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
                record_id = query.get("id", [""])[0]
                record = self.records_by_id.get(record_id)
                if record is None:
                    self.send_error_json(HTTPStatus.NOT_FOUND, f"unknown run id: {record_id}")
                    return
                max_trajectories = int(query.get("max_trajectories", ["1024"])[0])
                stride = int(query.get("stride", ["1"])[0])
                self.send_json(load_trajectories(record, max_trajectories, stride))
                return
            self.send_error_json(HTTPStatus.NOT_FOUND, "not found")
        except Exception as exc:
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))


def port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def choose_port(host: str, requested_port: int) -> int:
    if requested_port == 0:
        return 0
    for port in range(requested_port, requested_port + 50):
        if port_is_free(host, port):
            return port
    raise RuntimeError(f"could not find a free port starting at {requested_port}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local GUI for overlaying saved Avoiding2D trajectory NPZs.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Mapped Avoiding2D output root.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    if not output_root.exists():
        raise FileNotFoundError(f"output root does not exist: {output_root}")

    records = scan_runs(output_root)
    if not records:
        raise RuntimeError(f"no trajectories.npz files with config.json found under {output_root}")

    handler_class = TrajectoryGuiHandler
    handler_class.records = records
    handler_class.records_by_id = {record.id: record for record in records}
    handler_class.output_root = output_root

    port = choose_port(args.host, args.port)
    server = ThreadingHTTPServer((args.host, port), handler_class)
    actual_host, actual_port = server.server_address
    print(f"Loaded {len(records)} trajectory files from {output_root}", flush=True)
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
