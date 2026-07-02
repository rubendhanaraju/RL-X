import argparse
import subprocess
import sys
from pathlib import Path


POINT_MASTER_OBS_DIM = 85
POINT_MASTER_ACTION_DIM = 23
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[7] / "fcp" / "skills" / "run"


def run_command(cmd):
    cmd = [str(part) for part in cmd]
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a point_master PPO-GRU checkpoint for the FCP Run skill."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = output_dir / "point_master_nn.pth"
    meta_path = output_dir / "point_master_nn_meta.json"
    onnx_path = output_dir / "point_master_nn.onnx"

    run_command(
        [
            args.python,
            script_dir / "convert.py",
            "--model",
            args.model.expanduser().resolve(),
            "--output",
            weights_path,
            "--meta-output",
            meta_path,
            "--obs-dim",
            POINT_MASTER_OBS_DIM,
            "--action-dim",
            POINT_MASTER_ACTION_DIM,
        ]
    )
    run_command(
        [
            args.python,
            "-m",
            "rl_x.environments.custom_mujoco.robocup_soccer.rcssserver_deployment.export_onnx",
            "--weights",
            weights_path,
            "--meta",
            meta_path,
            "--output",
            onnx_path,
            "--device",
            args.device,
            "--opset",
            args.opset,
        ]
    )

    print(f"FCP Run ONNX: {onnx_path}", flush=True)
    print(f"FCP Run meta: {meta_path}", flush=True)


if __name__ == "__main__":
    main()
