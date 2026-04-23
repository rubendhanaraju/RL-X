import argparse
import importlib.util
from pathlib import Path

import numpy as np
import onnx
import torch

from rl_x.environments.custom_mujoco.robocup_soccer.rcssserver_deployment.torch_policy_feedforward import (
    load_policy_from_files,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=Path("policy_ff.pth"))
    parser.add_argument("--meta", type=Path, default=Path("policy_ff_meta.json"))
    parser.add_argument("--output", type=Path, default=Path("policy_ff.onnx"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def maybe_compare_with_onnxruntime(output_path, obs, torch_action):
    if importlib.util.find_spec("onnxruntime") is None:
        print("onnxruntime not installed in this interpreter; skipped runtime parity check.")
        return

    import onnxruntime as ort

    session = ort.InferenceSession(
        output_path.as_posix(),
        providers=["CPUExecutionProvider"],
    )
    (ort_action,) = session.run(
        None,
        {
            session.get_inputs()[0].name: obs.cpu().numpy(),
        },
    )

    action_diff = float(np.max(np.abs(torch_action.cpu().numpy() - ort_action)))
    print(f"ONNXRuntime parity check max |action diff|: {action_diff:.8f}")


def main():
    args = parse_args()

    device = torch.device(args.device)
    model, meta = load_policy_from_files(args.weights, args.meta, device)
    model.eval()

    batch_size = 1
    obs = torch.zeros(
        (batch_size, meta["expected_policy_obs_dim"]),
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        torch_action = model(obs)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (obs,),
        args.output.as_posix(),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={
            "obs": {0: "batch"},
            "action": {0: "batch"},
        },
    )

    onnx_model = onnx.load(args.output.as_posix())
    onnx.checker.check_model(onnx_model)

    print(f"Saved ONNX model to: {args.output.resolve()}")
    print(f"Policy obs dim:      {meta['expected_policy_obs_dim']}")
    print(f"Action dim:          {meta['action_dim']}")

    maybe_compare_with_onnxruntime(
        args.output,
        obs,
        torch_action,
    )


if __name__ == "__main__":
    main()
