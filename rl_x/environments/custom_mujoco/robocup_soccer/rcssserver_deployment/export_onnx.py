import argparse
from pathlib import Path

import torch

from torch_policy import load_policy_from_files


def main():
    parser = argparse.ArgumentParser(description="Export converted RoboCup GRU policy to ONNX.")
    parser.add_argument("--weights", type=str, required=True, help="Path to .pth produced by convert.py")
    parser.add_argument("--meta", type=str, required=True, help="Path to meta JSON produced by convert.py")
    parser.add_argument("--output", type=str, default="locomotion_nn.onnx", help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    weights_path = Path(args.weights).resolve()
    meta_path = Path(args.meta).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    model, meta = load_policy_from_files(weights_path, meta_path, device)

    obs_dim = len(meta["policy_observation_indices"])
    hidden_dim = int(meta["gru_hidden_dim"])

    dummy_obs = torch.zeros(1, obs_dim, dtype=torch.float32, device=device)
    dummy_carry = torch.zeros(1, hidden_dim, dtype=torch.float32, device=device)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_obs, dummy_carry),
            output_path.as_posix(),
            input_names=["obs", "carry_in"],
            output_names=["action_mean", "carry_out"],
            dynamic_axes={
                "obs": {0: "batch"},
                "carry_in": {0: "batch"},
                "action_mean": {0: "batch"},
                "carry_out": {0: "batch"},
            },
            opset_version=args.opset,
        )

    print(f"Exported ONNX model to: {output_path}")


if __name__ == "__main__":
    main()
