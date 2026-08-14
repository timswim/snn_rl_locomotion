"""Гибрид: SNN-актор + ANN-критик."""

import torch
import torch.nn as nn
from torch.distributions import Normal

from norse.torch.module.encode import ConstantCurrentLIFEncoder

from .ann import build_mlp
from .snn import build_snn_sequential, make_lif_params


class ActorCritic(nn.Module):
    """SNN-актор и ANN-критик. ``forward`` возвращает состояние только актора."""

    def __init__(
        self,
        num_inputs,
        num_outputs,
        hidden_sizes,
        T=16,
        alpha=1.0,
        std=0.0,
        lif_v_th=0.4,
        dt=0.01,
    ):
        super().__init__()
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        self.lif_params = make_lif_params(alpha, lif_v_th)
        self.dt = dt
        self.constant_current_encoder = ConstantCurrentLIFEncoder(
            T, p=self.lif_params, dt=self.dt
        )
        self.T = T
        self.alpha = alpha
        self.critic = build_mlp(num_inputs, hidden_sizes, 1)
        self.actor = build_snn_sequential(
            num_inputs, hidden_sizes, num_outputs, self.lif_params, self.dt
        )

    def forward(self, x, actor_state=None, critic_state=None, **kwargs):
        """
        Прямой проход: критик по сырому наблюдению, актор по T микрошагам SNN.

        Returns:
            dist, value, actor_state, critic_state — critic_state всегда None.
        """
        value = self.critic(x)
        x_enc = self.constant_current_encoder(x)
        for t in range(self.T):
            mu, actor_state = self.actor(x_enc[t, :, :], actor_state)
            mu = torch.tanh(mu)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist, value, actor_state, None
