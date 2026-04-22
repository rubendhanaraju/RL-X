import argparse
import json
from datetime import datetime
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from rl_x.environments.custom_mujoco.robocup_soccer.locomotion.mjx.viewer import (
    MujocoViewer,
)


ROOT_POS_NAMES = ("root_x", "root_y", "root_z")
ROOT_EULER_NAMES = ("root_roll", "root_pitch", "root_yaw")


def wxyz_to_xyzw(quat_wxyz):
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    return np.array(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64
    )


def xyzw_to_wxyz(quat_xyzw):
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
    return np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64
    )


def euler_xyz_to_wxyz(euler_xyz):
    quat_xyzw = Rotation.from_euler("xyz", euler_xyz).as_quat()
    return xyzw_to_wxyz(quat_xyzw)


def wxyz_to_euler_xyz(quat_wxyz):
    quat_xyzw = wxyz_to_xyzw(quat_wxyz)
    return Rotation.from_quat(quat_xyzw).as_euler("xyz")


def format_qpos(qpos):
    return " ".join(f"{value:.6f}" for value in qpos)


class PoseVisualizer:
    def __init__(self, xml_path, load_path=None):
        self.xml_path = Path(xml_path).resolve()
        self.model = mujoco.MjModel.from_xml_path(self.xml_path.as_posix())
        self.data = mujoco.MjData(self.model)
        self.viewer = MujocoViewer(self.model, dt=1.0 / 60.0)

        self.home_qpos = np.array(self.model.keyframe("home").qpos, dtype=np.float64)
        self.actuator_joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, trnid[0])
            for trnid in self.model.actuator_trnid
        ]
        self.actuator_joint_qpos_idx = np.array(
            [self.model.joint(name).qposadr[0] for name in self.actuator_joint_names],
            dtype=np.int32,
        )
        self.actuator_joint_limits = np.array(
            [self.model.jnt_range[self.model.joint(name).id] for name in self.actuator_joint_names],
            dtype=np.float64,
        )

        self.root = tk.Tk()
        self.root.title("RoboCup Pose Visualizer")
        self.root.geometry("620x980")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.running = True

        self.root_vars = {}
        self.joint_vars = {}
        self.status_var = tk.StringVar(value=f"Loaded {self.xml_path.name}")
        self.keyframe_text = tk.StringVar()
        self.keyframes = []
        self.keyframe_duration_var = tk.DoubleVar(value=0.35)
        self.playback_loop_var = tk.BooleanVar(value=True)
        self.is_playing = False
        self.playback_segment_index = 0
        self.playback_segment_started_at = 0.0
        self.playback_qpos = None
        self.history_undo = []
        self.history_redo = []
        self.current_snapshot = None
        self.history_debounce_ms = 250
        self.history_suspended = 1
        self.pending_history_commit_id = None

        self._build_ui()
        self._set_pose_from_qpos(self.home_qpos)
        self.history_suspended = 0
        self.current_snapshot = self._snapshot_state()

        if load_path is not None:
            self._load_pose_file(Path(load_path), record_history=False)

    def _build_ui(self):
        top_actions = tk.Frame(self.root)
        top_actions.pack(fill="x", padx=8, pady=8)

        tk.Button(top_actions, text="Home Pose", command=self._reset_home_pose).pack(
            side="left", padx=4
        )
        tk.Button(top_actions, text="Zero Joints", command=self._zero_joints).pack(
            side="left", padx=4
        )
        tk.Button(top_actions, text="Load File", command=self._load_pose_dialog).pack(
            side="left", padx=4
        )
        tk.Button(
            top_actions, text="Export Pose", command=self._export_pose_dialog
        ).pack(side="left", padx=4)
        tk.Button(
            top_actions, text="Copy Keyframe", command=self._copy_keyframe
        ).pack(side="left", padx=4)
        self.root.bind_all("<Control-z>", self._undo_event)
        self.root.bind_all("<Control-y>", self._redo_event)
        self.root.bind_all("<Control-Shift-Z>", self._redo_event)

        pose_frame = tk.LabelFrame(self.root, text="Root Pose")
        pose_frame.pack(fill="x", padx=8, pady=(0, 8))

        self._add_slider_group(
            parent=pose_frame,
            names=ROOT_POS_NAMES,
            ranges=[(-3.0, 3.0), (-3.0, 3.0), (0.0, 1.5)],
            target_dict=self.root_vars,
        )
        self._add_slider_group(
            parent=pose_frame,
            names=ROOT_EULER_NAMES,
            ranges=[(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)],
            target_dict=self.root_vars,
        )

        timeline_frame = tk.LabelFrame(self.root, text="Animation / Keyframes")
        timeline_frame.pack(fill="x", padx=8, pady=(0, 8))

        timeline_actions_top = tk.Frame(timeline_frame)
        timeline_actions_top.pack(fill="x", padx=6, pady=(6, 2))
        tk.Button(
            timeline_actions_top, text="Add Current", command=self._add_keyframe
        ).pack(side="left", padx=4)
        tk.Button(
            timeline_actions_top,
            text="Update Selected",
            command=self._update_selected_keyframe,
        ).pack(side="left", padx=4)
        tk.Button(
            timeline_actions_top,
            text="Load Selected",
            command=self._load_selected_keyframe,
        ).pack(side="left", padx=4)
        tk.Button(
            timeline_actions_top, text="Remove", command=self._remove_selected_keyframe
        ).pack(side="left", padx=4)

        timeline_actions_bottom = tk.Frame(timeline_frame)
        timeline_actions_bottom.pack(fill="x", padx=6, pady=(2, 2))
        tk.Button(
            timeline_actions_bottom, text="Move Up", command=lambda: self._move_keyframe(-1)
        ).pack(side="left", padx=4)
        tk.Button(
            timeline_actions_bottom, text="Move Down", command=lambda: self._move_keyframe(1)
        ).pack(side="left", padx=4)
        tk.Button(
            timeline_actions_bottom, text="New Anim", command=self._new_animation
        ).pack(side="left", padx=4)
        tk.Button(
            timeline_actions_bottom,
            text="Export Anim",
            command=self._export_animation_dialog,
        ).pack(side="left", padx=4)
        tk.Button(
            timeline_actions_bottom,
            text="Copy MJCF Keys",
            command=self._copy_animation_keyframes,
        ).pack(side="left", padx=4)

        timeline_settings = tk.Frame(timeline_frame)
        timeline_settings.pack(fill="x", padx=6, pady=(2, 4))
        tk.Label(
            timeline_settings, text="Duration to next (s)", anchor="w"
        ).pack(side="left")
        tk.Entry(
            timeline_settings, width=8, textvariable=self.keyframe_duration_var
        ).pack(side="left", padx=(6, 12))
        self.keyframe_duration_var.trace_add("write", self._on_editor_var_changed)
        tk.Checkbutton(
            timeline_settings, text="Loop", variable=self.playback_loop_var
        ).pack(side="left")
        self.playback_loop_var.trace_add("write", self._on_editor_var_changed)
        tk.Button(
            timeline_settings, text="Play / Stop", command=self._toggle_playback
        ).pack(side="left", padx=10)

        self.keyframe_listbox = tk.Listbox(timeline_frame, height=7, exportselection=False)
        self.keyframe_listbox.pack(fill="x", padx=6, pady=(0, 6))
        self.keyframe_listbox.bind("<<ListboxSelect>>", self._on_keyframe_select)

        joint_container = tk.LabelFrame(self.root, text="Actuator Joints")
        joint_container.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        canvas = tk.Canvas(joint_container, highlightthickness=0)
        scrollbar = tk.Scrollbar(
            joint_container, orient="vertical", command=canvas.yview
        )
        scrollable = tk.Frame(canvas)
        scrollable.bind(
            "<Configure>",
            lambda event: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for index, joint_name in enumerate(self.actuator_joint_names):
            lower, upper = self.actuator_joint_limits[index]
            self._add_single_slider(
                parent=scrollable,
                name=joint_name,
                lower=lower,
                upper=upper,
                target_dict=self.joint_vars,
            )

        keyframe_frame = tk.LabelFrame(self.root, text="Current Keyframe qpos")
        keyframe_frame.pack(fill="x", padx=8, pady=(0, 8))
        entry = tk.Entry(
            keyframe_frame, textvariable=self.keyframe_text, state="readonly"
        )
        entry.pack(fill="x", padx=6, pady=6)

        status_label = tk.Label(
            self.root, textvariable=self.status_var, anchor="w", justify="left"
        )
        status_label.pack(fill="x", padx=8, pady=(0, 8))

    def _add_slider_group(self, parent, names, ranges, target_dict):
        for name, (lower, upper) in zip(names, ranges):
            self._add_single_slider(parent, name, lower, upper, target_dict)

    def _add_single_slider(self, parent, name, lower, upper, target_dict):
        row = tk.Frame(parent)
        row.pack(fill="x", padx=6, pady=2)

        label = tk.Label(row, text=name, width=22, anchor="w")
        label.pack(side="left")

        variable = tk.DoubleVar(value=0.0)
        target_dict[name] = variable
        variable.trace_add("write", self._on_editor_var_changed)

        slider = tk.Scale(
            row,
            from_=lower,
            to=upper,
            resolution=0.001,
            orient=tk.HORIZONTAL,
            variable=variable,
            length=330,
        )
        slider.pack(side="left", fill="x", expand=True)

        value_entry = tk.Entry(row, width=10, textvariable=variable)
        value_entry.pack(side="left", padx=(8, 0))

    def _current_qpos(self):
        qpos = self.home_qpos.copy()
        qpos[0] = self.root_vars["root_x"].get()
        qpos[1] = self.root_vars["root_y"].get()
        qpos[2] = self.root_vars["root_z"].get()

        root_euler = np.array(
            [
                self.root_vars["root_roll"].get(),
                self.root_vars["root_pitch"].get(),
                self.root_vars["root_yaw"].get(),
            ],
            dtype=np.float64,
        )
        qpos[3:7] = euler_xyz_to_wxyz(root_euler)

        for joint_name, qpos_idx in zip(
            self.actuator_joint_names, self.actuator_joint_qpos_idx
        ):
            qpos[qpos_idx] = self.joint_vars[joint_name].get()

        return qpos

    def _set_pose_from_qpos(self, qpos):
        qpos = np.asarray(qpos, dtype=np.float64)
        self.root_vars["root_x"].set(qpos[0])
        self.root_vars["root_y"].set(qpos[1])
        self.root_vars["root_z"].set(qpos[2])

        root_euler = wxyz_to_euler_xyz(qpos[3:7])
        self.root_vars["root_roll"].set(root_euler[0])
        self.root_vars["root_pitch"].set(root_euler[1])
        self.root_vars["root_yaw"].set(root_euler[2])

        for joint_name, qpos_idx in zip(
            self.actuator_joint_names, self.actuator_joint_qpos_idx
        ):
            self.joint_vars[joint_name].set(qpos[qpos_idx])

        self.keyframe_text.set(format_qpos(qpos))

    def _pose_payload(self):
        qpos = self._current_qpos()
        root_euler = np.array(
            [
                self.root_vars["root_roll"].get(),
                self.root_vars["root_pitch"].get(),
                self.root_vars["root_yaw"].get(),
            ],
            dtype=np.float64,
        )

        return {
            "xml_path": self.xml_path.as_posix(),
            "pos_xyz": qpos[:3].tolist(),
            "quat_wxyz": qpos[3:7].tolist(),
            "euler_xyz": root_euler.tolist(),
            "joint_positions": {
                joint_name: float(self.joint_vars[joint_name].get())
                for joint_name in self.actuator_joint_names
            },
            "full_qpos": qpos.tolist(),
            "mujoco_keyframe_qpos": format_qpos(qpos),
        }

    def _animation_payload(self):
        return {
            "xml_path": self.xml_path.as_posix(),
            "keyframes": [
                {
                    "name": keyframe["name"],
                    "duration_to_next_s": keyframe["duration_to_next_s"],
                    "full_qpos": keyframe["qpos"].tolist(),
                    "mujoco_keyframe_qpos": format_qpos(keyframe["qpos"]),
                }
                for keyframe in self.keyframes
            ],
        }

    def _snapshot_state(self):
        selected_index = self._selected_keyframe_index()
        return {
            "editor_qpos": self._current_qpos().copy(),
            "keyframes": [
                {
                    "name": keyframe["name"],
                    "duration_to_next_s": float(keyframe["duration_to_next_s"]),
                    "qpos": keyframe["qpos"].copy(),
                }
                for keyframe in self.keyframes
            ],
            "selected_index": selected_index,
            "duration_var": float(self.keyframe_duration_var.get()),
            "playback_loop": bool(self.playback_loop_var.get()),
        }

    def _snapshots_equal(self, left, right):
        if left is None or right is None:
            return left is right
        if not np.allclose(left["editor_qpos"], right["editor_qpos"]):
            return False
        if left["selected_index"] != right["selected_index"]:
            return False
        if left["playback_loop"] != right["playback_loop"]:
            return False
        if not np.isclose(left["duration_var"], right["duration_var"]):
            return False
        if len(left["keyframes"]) != len(right["keyframes"]):
            return False
        for keyframe_left, keyframe_right in zip(left["keyframes"], right["keyframes"]):
            if keyframe_left["name"] != keyframe_right["name"]:
                return False
            if not np.isclose(
                keyframe_left["duration_to_next_s"],
                keyframe_right["duration_to_next_s"],
            ):
                return False
            if not np.allclose(keyframe_left["qpos"], keyframe_right["qpos"]):
                return False
        return True

    def _cancel_pending_history_commit(self):
        if self.pending_history_commit_id is None:
            return
        self.root.after_cancel(self.pending_history_commit_id)
        self.pending_history_commit_id = None

    def _on_editor_var_changed(self, *args):
        if self.history_suspended:
            return
        self._cancel_pending_history_commit()
        self.pending_history_commit_id = self.root.after(
            self.history_debounce_ms, self._commit_pending_history_change
        )

    def _commit_pending_history_change(self):
        self.pending_history_commit_id = None
        snapshot = self._snapshot_state()
        if self.current_snapshot is None:
            self.current_snapshot = snapshot
            return
        if self._snapshots_equal(snapshot, self.current_snapshot):
            return
        self.history_undo.append(self.current_snapshot)
        self.current_snapshot = snapshot
        self.history_redo.clear()

    def _restore_snapshot(self, snapshot):
        self._stop_playback()
        self._cancel_pending_history_commit()
        self.history_suspended += 1
        try:
            self._set_pose_from_qpos(snapshot["editor_qpos"])
            self.keyframes = [
                {
                    "name": keyframe["name"],
                    "duration_to_next_s": float(keyframe["duration_to_next_s"]),
                    "qpos": keyframe["qpos"].copy(),
                }
                for keyframe in snapshot["keyframes"]
            ]
            self._refresh_keyframe_list()
            self.keyframe_duration_var.set(snapshot["duration_var"])
            self.playback_loop_var.set(snapshot["playback_loop"])

            self.keyframe_listbox.selection_clear(0, tk.END)
            if snapshot["selected_index"] is not None and self.keyframes:
                selected_index = min(snapshot["selected_index"], len(self.keyframes) - 1)
                self.keyframe_listbox.selection_set(selected_index)
                self.keyframe_listbox.activate(selected_index)
        finally:
            self.history_suspended -= 1

    def _apply_explicit_state_change(self, mutator, record_history=True):
        self._cancel_pending_history_commit()
        before = self.current_snapshot
        self.history_suspended += 1
        try:
            mutator()
        finally:
            self.history_suspended -= 1

        after = self._snapshot_state()
        if not record_history or before is None:
            self.current_snapshot = after
            return
        if self._snapshots_equal(after, before):
            self.current_snapshot = before
            return
        self.history_undo.append(before)
        self.current_snapshot = after
        self.history_redo.clear()

    def _undo_event(self, event=None):
        self._undo()
        return "break"

    def _redo_event(self, event=None):
        self._redo()
        return "break"

    def _undo(self):
        self._cancel_pending_history_commit()
        if not self.history_undo:
            self.status_var.set("Nothing to undo.")
            return
        if self.current_snapshot is not None:
            self.history_redo.append(self.current_snapshot)
        snapshot = self.history_undo.pop()
        self._restore_snapshot(snapshot)
        self.current_snapshot = snapshot
        self.status_var.set("Undo.")

    def _redo(self):
        self._cancel_pending_history_commit()
        if not self.history_redo:
            self.status_var.set("Nothing to redo.")
            return
        if self.current_snapshot is not None:
            self.history_undo.append(self.current_snapshot)
        snapshot = self.history_redo.pop()
        self._restore_snapshot(snapshot)
        self.current_snapshot = snapshot
        self.status_var.set("Redo.")

    def _reset_home_pose(self):
        def mutator():
            self._stop_playback()
            self._set_pose_from_qpos(self.home_qpos)
            self.status_var.set("Reset to MuJoCo home pose.")

        self._apply_explicit_state_change(mutator)

    def _zero_joints(self):
        def mutator():
            self._stop_playback()
            qpos = self._current_qpos()
            qpos[self.actuator_joint_qpos_idx] = 0.0
            self._set_pose_from_qpos(qpos)
            self.status_var.set("Set actuator joints to zero.")

        self._apply_explicit_state_change(mutator)

    def _copy_keyframe(self):
        qpos_string = format_qpos(self._current_qpos())
        self.root.clipboard_clear()
        self.root.clipboard_append(qpos_string)
        self.status_var.set("Copied keyframe qpos string to clipboard.")

    def _export_pose_dialog(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"robocup_pose_{timestamp}.json"
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export pose",
            defaultextension=".json",
            initialfile=default_name,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        self._write_pose_file(Path(path))

    def _write_pose_file(self, path):
        payload = self._pose_payload()
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self.status_var.set(f"Exported pose to {path}.")

    def _export_animation_dialog(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"robocup_animation_{timestamp}.json"
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export animation",
            defaultextension=".json",
            initialfile=default_name,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return

        payload = self._animation_payload()
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self.status_var.set(f"Exported animation to {path}.")

    def _load_pose_dialog(self):
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Load pose",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        self._load_pose_file(Path(path))

    def _load_pose_file(self, path, record_history=True):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if "keyframes" in payload:
            def mutator():
                self._stop_playback()
                self.keyframes = []
                for index, keyframe_payload in enumerate(payload["keyframes"]):
                    qpos = np.asarray(keyframe_payload["full_qpos"], dtype=np.float64)
                    self.keyframes.append(
                        {
                            "name": keyframe_payload.get("name", f"kf_{index:02d}"),
                            "duration_to_next_s": float(
                                keyframe_payload.get("duration_to_next_s", 0.35)
                            ),
                            "qpos": qpos,
                        }
                    )
                self._refresh_keyframe_list()
                if self.keyframes:
                    self.keyframe_listbox.selection_set(0)
                    self.keyframe_listbox.activate(0)
                    self._on_keyframe_select()
                    self._set_pose_from_qpos(self.keyframes[0]["qpos"])
                self.status_var.set(f"Loaded animation from {path}.")

            self._apply_explicit_state_change(mutator, record_history=record_history)
            return

        if "full_qpos" in payload:
            qpos = np.asarray(payload["full_qpos"], dtype=np.float64)
        else:
            qpos = self.home_qpos.copy()
            if "pos_xyz" in payload:
                qpos[:3] = np.asarray(payload["pos_xyz"], dtype=np.float64)
            if "quat_wxyz" in payload:
                qpos[3:7] = np.asarray(payload["quat_wxyz"], dtype=np.float64)
            elif "euler_xyz" in payload:
                qpos[3:7] = euler_xyz_to_wxyz(payload["euler_xyz"])
            for joint_name, value in payload.get("joint_positions", {}).items():
                if joint_name in self.joint_vars:
                    qpos_idx = self.actuator_joint_qpos_idx[
                        self.actuator_joint_names.index(joint_name)
                    ]
                    qpos[qpos_idx] = value

        def mutator():
            self._stop_playback()
            self._set_pose_from_qpos(qpos)
            self.status_var.set(f"Loaded pose from {path}.")

        self._apply_explicit_state_change(mutator, record_history=record_history)

    def _selected_keyframe_index(self):
        selection = self.keyframe_listbox.curselection()
        if not selection:
            return None
        return int(selection[0])

    def _refresh_keyframe_list(self):
        self.keyframe_listbox.delete(0, tk.END)
        for index, keyframe in enumerate(self.keyframes):
            label = (
                f"{index:02d} | {keyframe['name']} | "
                f"{keyframe['duration_to_next_s']:.2f}s to next"
            )
            self.keyframe_listbox.insert(tk.END, label)

    def _on_keyframe_select(self, event=None):
        index = self._selected_keyframe_index()
        if index is None:
            return
        self.history_suspended += 1
        self.keyframe_duration_var.set(self.keyframes[index]["duration_to_next_s"])
        self.history_suspended -= 1

    def _add_keyframe(self):
        def mutator():
            qpos = self._current_qpos()
            index = len(self.keyframes)
            self.keyframes.append(
                {
                    "name": f"kf_{index:02d}",
                    "duration_to_next_s": float(self.keyframe_duration_var.get()),
                    "qpos": qpos,
                }
            )
            self._refresh_keyframe_list()
            self.keyframe_listbox.selection_clear(0, tk.END)
            self.keyframe_listbox.selection_set(index)
            self.keyframe_listbox.activate(index)
            self.status_var.set(f"Added keyframe {index:02d}.")

        self._apply_explicit_state_change(mutator)

    def _update_selected_keyframe(self):
        index = self._selected_keyframe_index()
        if index is None:
            messagebox.showwarning("No keyframe selected", "Select a keyframe first.")
            return
        def mutator():
            self.keyframes[index]["qpos"] = self._current_qpos()
            self.keyframes[index]["duration_to_next_s"] = float(
                self.keyframe_duration_var.get()
            )
            self._refresh_keyframe_list()
            self.keyframe_listbox.selection_set(index)
            self.keyframe_listbox.activate(index)
            self.status_var.set(f"Updated keyframe {index:02d}.")

        self._apply_explicit_state_change(mutator)

    def _load_selected_keyframe(self):
        index = self._selected_keyframe_index()
        if index is None:
            messagebox.showwarning("No keyframe selected", "Select a keyframe first.")
            return
        def mutator():
            self._stop_playback()
            self._set_pose_from_qpos(self.keyframes[index]["qpos"])
            self.status_var.set(f"Loaded keyframe {index:02d} into editor.")

        self._apply_explicit_state_change(mutator)

    def _remove_selected_keyframe(self):
        index = self._selected_keyframe_index()
        if index is None:
            messagebox.showwarning("No keyframe selected", "Select a keyframe first.")
            return
        def mutator():
            self._stop_playback()
            del self.keyframes[index]
            self._refresh_keyframe_list()
            if self.keyframes:
                new_index = min(index, len(self.keyframes) - 1)
                self.keyframe_listbox.selection_set(new_index)
                self.keyframe_listbox.activate(new_index)
                self._on_keyframe_select()
            self.status_var.set(f"Removed keyframe {index:02d}.")

        self._apply_explicit_state_change(mutator)

    def _move_keyframe(self, offset):
        index = self._selected_keyframe_index()
        if index is None:
            messagebox.showwarning("No keyframe selected", "Select a keyframe first.")
            return
        new_index = index + offset
        if new_index < 0 or new_index >= len(self.keyframes):
            return
        def mutator():
            self._stop_playback()
            self.keyframes[index], self.keyframes[new_index] = (
                self.keyframes[new_index],
                self.keyframes[index],
            )
            self._refresh_keyframe_list()
            self.keyframe_listbox.selection_set(new_index)
            self.keyframe_listbox.activate(new_index)
            self.status_var.set(f"Moved keyframe to position {new_index:02d}.")

        self._apply_explicit_state_change(mutator)

    def _new_animation(self):
        def mutator():
            self._stop_playback()
            self.keyframes = []
            self._refresh_keyframe_list()
            self.status_var.set("Cleared animation timeline.")

        self._apply_explicit_state_change(mutator)

    def _copy_animation_keyframes(self):
        if not self.keyframes:
            messagebox.showwarning("No keyframes", "Add at least one keyframe first.")
            return
        lines = ["<keyframe>"]
        for keyframe in self.keyframes:
            lines.append(
                f'  <key name="{keyframe["name"]}" qpos="{format_qpos(keyframe["qpos"])}"/>'
            )
        lines.append("</keyframe>")
        keyframe_block = "\n".join(lines)
        self.root.clipboard_clear()
        self.root.clipboard_append(keyframe_block)
        self.status_var.set("Copied MJCF keyframe block to clipboard.")

    def _toggle_playback(self):
        if self.is_playing:
            self._stop_playback()
            self.status_var.set("Playback stopped.")
            return
        if not self.keyframes:
            messagebox.showwarning("No keyframes", "Add at least one keyframe first.")
            return
        self.is_playing = True
        self.playback_segment_index = 0
        self.playback_segment_started_at = time.time()
        self.playback_qpos = self.keyframes[0]["qpos"].copy()
        self.status_var.set("Playback started.")

    def _stop_playback(self):
        self.is_playing = False
        self.playback_qpos = None

    def _interpolate_qpos(self, qpos_a, qpos_b, alpha):
        alpha = float(np.clip(alpha, 0.0, 1.0))
        interpolated = (1.0 - alpha) * qpos_a + alpha * qpos_b

        rotations = Rotation.from_quat(
            np.stack([wxyz_to_xyzw(qpos_a[3:7]), wxyz_to_xyzw(qpos_b[3:7])], axis=0)
        )
        slerp = Slerp([0.0, 1.0], rotations)
        interpolated[3:7] = xyzw_to_wxyz(slerp([alpha]).as_quat()[0])
        return interpolated

    def _update_playback(self):
        if not self.is_playing:
            return
        if len(self.keyframes) == 1:
            self.playback_qpos = self.keyframes[0]["qpos"].copy()
            return

        now = time.time()
        while True:
            current = self.keyframes[self.playback_segment_index]
            duration = max(float(current["duration_to_next_s"]), 1e-3)
            elapsed = now - self.playback_segment_started_at
            if elapsed <= duration:
                break
            self.playback_segment_started_at += duration
            if self.playback_segment_index == len(self.keyframes) - 1:
                if self.playback_loop_var.get():
                    self.playback_segment_index = 0
                else:
                    self.playback_qpos = self.keyframes[-1]["qpos"].copy()
                    self._stop_playback()
                    self.status_var.set("Playback finished.")
                    return
            else:
                self.playback_segment_index += 1

        next_index = (self.playback_segment_index + 1) % len(self.keyframes)
        current_qpos = self.keyframes[self.playback_segment_index]["qpos"]
        next_qpos = self.keyframes[next_index]["qpos"]
        duration = max(
            float(self.keyframes[self.playback_segment_index]["duration_to_next_s"]), 1e-3
        )
        alpha = (now - self.playback_segment_started_at) / duration
        self.playback_qpos = self._interpolate_qpos(current_qpos, next_qpos, alpha)

    def _apply_pose_to_sim(self):
        qpos = self.playback_qpos if self.playback_qpos is not None else self._current_qpos()
        self.data.qpos = qpos
        self.data.qvel = np.zeros_like(self.data.qvel)
        self.data.ctrl = qpos[self.actuator_joint_qpos_idx]
        mujoco.mj_forward(self.model, self.data)
        self.keyframe_text.set(format_qpos(qpos))

    def _on_close(self):
        self.running = False

    def run(self):
        while self.running:
            try:
                self.root.update_idletasks()
                self.root.update()
            except tk.TclError:
                break

            self._update_playback()
            self._apply_pose_to_sim()
            self.viewer.render(self.data)

        try:
            self.viewer.close()
        except Exception:
            pass

        try:
            self.root.destroy()
        except tk.TclError:
            pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive pose editor for the RoboCup MJX/MuJoCo robot."
    )
    parser.add_argument(
        "--xml",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "robots"
            / "booster_t1"
            / "data"
            / "plane.xml"
        ),
        help="Path to the MuJoCo XML file to visualize.",
    )
    parser.add_argument(
        "--load",
        type=Path,
        default=None,
        help="Optional JSON pose file to load on startup.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    visualizer = PoseVisualizer(xml_path=args.xml, load_path=args.load)
    visualizer.run()


if __name__ == "__main__":
    main()
