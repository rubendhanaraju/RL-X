import json
import math
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class TorchPolicyFeedforward(nn.Module):
    def __init__(
        self,
        action_dim: int,
        std_dev: float,
        policy_observation_indices: Sequence[int],
    ):
        super().__init__()

        self.action_dim = int(action_dim)

        indices = torch.as_tensor(policy_observation_indices, dtype=torch.long)
        self.register_buffer("policy_observation_indices", indices, persistent=False)
        policy_obs_dim = int(indices.numel())

        self.dense1 = nn.Linear(policy_obs_dim, 512)
        self.ln1 = nn.LayerNorm(512, eps=1e-6)
        self.dense2 = nn.Linear(512, 256)
        self.dense3 = nn.Linear(256, 128)
        self.mean_head = nn.Linear(128, self.action_dim)
        self.policy_logstd = nn.Parameter(
            torch.full((1, self.action_dim), float(math.log(std_dev)), dtype=torch.float32)
        )

    @classmethod
    def from_meta(cls, meta: dict) -> "TorchPolicyFeedforward":
        return cls(
            action_dim=meta["action_dim"],
            std_dev=meta.get("std_dev", 1.0),
            policy_observation_indices=meta["policy_observation_indices"],
        )

    def _select_policy_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.index_select(dim=-1, index=self.policy_observation_indices)

    def forward_with_logstd(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._select_policy_obs(obs)
        x = self.dense1(x)
        x = self.ln1(x)
        x = F.elu(x)
        x = self.dense2(x)
        x = F.elu(x)
        x = self.dense3(x)
        x = F.elu(x)
        mean = self.mean_head(x)
        logstd = self.policy_logstd.expand(mean.shape[0], -1)
        return mean, logstd

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward_with_logstd(obs)
        return mean


def load_policy_from_files(
    weights_path: str | Path,
    meta_path: str | Path,
    device: torch.device,
) -> tuple[TorchPolicyFeedforward, dict]:
    weights_path = Path(weights_path)
    meta_path = Path(meta_path)

    with meta_path.open("r") as f:
        meta = json.load(f)

    model = TorchPolicyFeedforward.from_meta(meta).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, meta
