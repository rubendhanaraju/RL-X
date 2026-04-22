import argparse
import importlib.util
from pathlib import Path

import numpy as np
import onnx
import torch

from rl_x.environments.custom_mujoco.robocup_soccer.rcssserver_deployment.torch_policy import (
    load_policy_from_files,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=Path("locomotion_nn.pth"))
    parser.add_argument("--meta", type=Path, default=Path("locomotion_nn_meta.json"))
    parser.add_argument("--output", type=Path, default=Path("locomotion_nn.onnx"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def maybe_compare_with_onnxruntime(output_path, obs, carry, torch_action, torch_next_carry):
    if importlib.util.find_spec("onnxruntime") is None:
        print("onnxruntime not installed in this interpreter; skipped runtime parity check.")
        return

    import onnxruntime as ort

    session = ort.InferenceSession(
        output_path.as_posix(),
        providers=["CPUExecutionProvider"],
    )
    ort_action, ort_next_carry = session.run(
        None,
        {
            session.get_inputs()[0].name: obs.cpu().numpy(),
            session.get_inputs()[1].name: carry.cpu().numpy(),
        },
    )

    action_diff = float(np.max(np.abs(torch_action.cpu().numpy() - ort_action)))
    carry_diff = float(np.max(np.abs(torch_next_carry.cpu().numpy() - ort_next_carry)))
    print(f"ONNXRuntime parity check max |action diff|: {action_diff:.8f}")
    print(f"ONNXRuntime parity check max |carry diff|:  {carry_diff:.8f}")


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
    carry = model.initialize_carry(batch_size=batch_size, device=device)

    with torch.no_grad():
        torch_action, torch_next_carry = model(obs, carry)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (obs, carry),
        args.output.as_posix(),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["obs", "carry"],
        output_names=["action", "next_carry"],
        dynamic_axes={
            "obs": {0: "batch"},
            "carry": {0: "batch"},
            "action": {0: "batch"},
            "next_carry": {0: "batch"},
        },
    )

    onnx_model = onnx.load(args.output.as_posix())
    onnx.checker.check_model(onnx_model)

    print(f"Saved ONNX model to: {args.output.resolve()}")
    print(f"Policy obs dim:      {meta['expected_policy_obs_dim']}")
    print(f"GRU hidden dim:      {meta['gru_hidden_dim']}")
    print(f"Action dim:          {meta['action_dim']}")

    maybe_compare_with_onnxruntime(
        args.output,
        obs,
        carry,
        torch_action,
        torch_next_carry,
    )


if __name__ == "__main__":
    main()
