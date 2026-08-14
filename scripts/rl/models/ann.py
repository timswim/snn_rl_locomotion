"""Полносвязный актор-критик (ANN)."""

import torch
import torch.nn as nn
from torch.distributions import Normal


def build_mlp(input_size, hidden_sizes, output_size):
    """Строит MLP: Linear-ReLU блоки и линейный выход."""
    layers_list = []
    in_size = input_size
    for h in hidden_sizes:
        layers_list.append(nn.Linear(in_size, h))
        layers_list.append(nn.ReLU())
        in_size = h
    layers_list.append(nn.Linear(in_size, output_size))
    return nn.Sequential(*layers_list)


class ActorCritic(nn.Module):
    """ANN актор-критик. Скрытого состояния нет: ``forward`` возвращает ``None, None``."""

    def __init__(self, num_inputs, num_outputs, hidden_sizes, std=0.0):
        super().__init__()
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        self.critic = build_mlp(num_inputs, hidden_sizes, 1)
        self.actor = build_mlp(num_inputs, hidden_sizes, num_outputs)

    def forward(self, x, actor_state=None, critic_state=None, **kwargs):
        """
        Прямой проход.

        Returns:
            dist, value, actor_state, critic_state — состояния всегда None.
        """
        value = self.critic(x)
        mu = self.actor(x)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist, value, None, None
