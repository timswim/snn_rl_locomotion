"""SNN актор-критик на Norse (LIF + LILinear readout)."""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from norse.torch.functional.lif import LIFParameters
from norse.torch.module.encode import ConstantCurrentLIFEncoder
from norse.torch.module.leaky_integrator import LILinearCell
from norse.torch.module.lif import LIFCell
from norse.torch.module.sequential import SequentialState


def make_lif_params(alpha, lif_v_th):
    """
    Параметры LIF.

    v_th=1.0 (дефолт Norse) слишком высок для кодов в [0, 1]:
    скрытые LIF не спайкают, LILinearCell на выходе не получает входа → mu = 0.
    """
    return LIFParameters(
        method="triangle",
        alpha=alpha,
        v_th=torch.as_tensor(lif_v_th),
    )


def build_snn_sequential(input_size, hidden_sizes, output_size, lif_params, dt):
    """SequentialState: Linear-LIF блоки и LILinearCell на выходе."""
    layers_list = []
    in_size = input_size
    for h in hidden_sizes:
        layers_list.append(nn.Linear(in_size, h))
        layers_list.append(LIFCell(p=lif_params, dt=dt))
        in_size = h
    layers_list.append(LILinearCell(in_size, output_size))
    return SequentialState(*layers_list)


def forward_sequential_state_with_lif_spikes(module, input_tensor, state=None):
    """
    Forward через SequentialState с возвратом спайков после каждого LIFCell.

    Returns:
        output, state, lif_spikes — список тензоров (batch, n_hidden) по порядку LIF-слоёв.
    """
    state = [None] * len(module) if state is None else state
    lif_spikes = []
    for index, layer in enumerate(module):
        if module.stateful_layers[index]:
            input_tensor, s = layer(input_tensor, state[index])
            state[index] = s
            if isinstance(layer, LIFCell):
                lif_spikes.append(input_tensor)
        else:
            input_tensor = layer(input_tensor)
    return input_tensor, state, lif_spikes


def lif_layer_spike_fraction_pct(spikes):
    """Средняя доля спайкующих нейронов за один микрошаг, в процентах (batch × neurons)."""
    return (spikes > 0).float().mean().item() * 100.0


def aggregate_spike_activity_over_T(lif_spikes_per_t):
    """
    Усредняет долю спайков по T микрошагам для каждого LIF-слоя.

    lif_spikes_per_t: список длины T; каждый элемент — список LIF-спайков по слоям.
    Returns:
        список из L float — средняя доля спайков (%) по T для каждого слоя.
    """
    if not lif_spikes_per_t:
        return []
    num_layers = len(lif_spikes_per_t[0])
    out = []
    for layer_idx in range(num_layers):
        fracs = [
            lif_layer_spike_fraction_pct(spikes[layer_idx])
            for spikes in lif_spikes_per_t
        ]
        out.append(float(np.mean(fracs)))
    return out


class ActorCritic(nn.Module):
    """SNN актор-критик. ``forward`` возвращает состояния актора и критика."""

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
        self.critic = build_snn_sequential(
            num_inputs, hidden_sizes, 1, self.lif_params, self.dt
        )
        self.actor = build_snn_sequential(
            num_inputs, hidden_sizes, num_outputs, self.lif_params, self.dt
        )

    def forward(
        self,
        x,
        actor_state=None,
        critic_state=None,
        return_mu_trace=False,
        return_spike_activity=False,
        **kwargs,
    ):
        """
        Прямой проход по T микрошагам.

        Returns:
            dist, value, actor_state, critic_state
            и опционально mu_trace / spike_activity, если запрошены адаптером.
        """
        x = self.constant_current_encoder(x)
        mu_trace = [] if return_mu_trace else None
        actor_spikes_per_t = [] if return_spike_activity else None
        for t in range(self.T):
            x_t = x[t, :, :]
            if return_spike_activity:
                value, critic_state, _ = forward_sequential_state_with_lif_spikes(
                    self.critic, x_t, critic_state
                )
                mu, actor_state, actor_spikes = forward_sequential_state_with_lif_spikes(
                    self.actor, x_t, actor_state
                )
                actor_spikes_per_t.append(actor_spikes)
            else:
                value, critic_state = self.critic(x_t, critic_state)
                mu, actor_state = self.actor(x_t, actor_state)
            mu = torch.tanh(mu)
            if return_mu_trace:
                mu_trace.append(mu)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        extras = []
        if return_mu_trace:
            extras.append(torch.stack(mu_trace, dim=0))
        if return_spike_activity:
            extras.append(aggregate_spike_activity_over_T(actor_spikes_per_t))
        if extras:
            return (dist, value, actor_state, critic_state, *extras)
        return dist, value, actor_state, critic_state
