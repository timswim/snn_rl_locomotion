"""
Гибрид SNN-актор + ANN-критик, GAE и PPO с учётом скрытых состояний Norse только у актора.
"""
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal

from norse.torch.functional.lif import LIFParameters
from norse.torch.module.lif import LIFCell
from norse.torch.module.sequential import SequentialState
from norse.torch.module.leaky_integrator import LILinearCell
from norse.torch.module.encode import ConstantCurrentLIFEncoder

from models.SNN_PPO import (
    compute_gae,
    detach_state,
    index_state_batch,
    reset_state_batch_indices,
    stack_rollout_states,
    structural_zeros_like,
)

__all__ = [
    "ActorCritic",
    "compute_gae",
    "detach_state",
    "initial_zero_hidden",
    "ppo_update",
    "reset_state_batch_indices",
    "stack_rollout_states",
]


def initial_zero_hidden(model, num_envs, num_inputs, device):
    """Формы скрытого состояния SNN-актера после forward; нули как «холодный старт»."""
    with torch.no_grad():
        z = torch.zeros(num_envs, num_inputs, device=device)
        _, _, actor_state, _ = model(z, None)
    return detach_state(structural_zeros_like(actor_state))


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_sizes, T=16, alpha=1.0, std=0.0, lif_v_th=0.4, dt=0.01):
        super(ActorCritic, self).__init__()

        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

        # v_th=1.0 (дефолт Norse) слишком высок для population-кодов в [0, 1]:
        # скрытые LIF не спайкают, LILinearCell на выходе не получает входа → mu = 0.
        self.lif_params = LIFParameters(
            method="triangle",
            alpha=alpha,
            v_th=torch.as_tensor(lif_v_th),
        )
        self.dt = dt
        self.constant_current_encoder = ConstantCurrentLIFEncoder(T, p=self.lif_params, dt=self.dt)
        self.T = T
        self.alpha = alpha

        self.critic = self.build_ann_network(num_inputs, hidden_sizes, 1)
        self.actor = self.build_snn_network(num_inputs, hidden_sizes, num_outputs)

    def build_ann_network(self, input_size, hidden_sizes, output_size):
        layers_list = []
        in_size = input_size

        for h in hidden_sizes:
            layers_list.append(nn.Linear(in_size, h))
            layers_list.append(nn.ReLU())
            in_size = h

        layers_list.append(nn.Linear(in_size, output_size))

        return nn.Sequential(*layers_list)

    def build_snn_network(self, input_size, hidden_sizes, output_size):
        layers_list = []
        in_size = input_size

        for h in hidden_sizes:
            layers_list.append(nn.Linear(in_size, h))
            layers_list.append(LIFCell(p=self.lif_params, dt=self.dt))
            in_size = h

        layers_list.append(LILinearCell(in_size, output_size))

        return SequentialState(*layers_list)

    def forward(self, x, actor_state=None, critic_state=None):
        value = self.critic(x)

        x_enc = self.constant_current_encoder(x)
        for t in range(self.T):
            mu, actor_state = self.actor(x_enc[t, :, :], actor_state)
            mu = torch.tanh(mu)

        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist, value, actor_state, None


def ppo_iter(
    mini_batch_size,
    states,
    actions,
    log_probs,
    returns,
    advantage,
    actor_states_flat,
):
    """Итератор по мини-батчам rollout; к каждому батчу подмешивает срезы скрытых состояний актора."""
    batch_size = states.size(0)
    dev = states.device
    ids = np.random.permutation(batch_size)
    ids = np.split(ids[: batch_size // mini_batch_size * mini_batch_size], batch_size // mini_batch_size)
    for i in range(len(ids)):
        idx = ids[i]
        yield (
            states[idx, :],
            actions[idx, :],
            log_probs[idx, :],
            returns[idx, :],
            advantage[idx, :],
            index_state_batch(actor_states_flat, idx, device=dev),
        )


def ppo_update(
    model,
    optimizer,
    ppo_epochs,
    mini_batch_size,
    states,
    actions,
    log_probs,
    returns,
    advantages,
    actor_states_flat,
    clip_param=0.2,
):
    """Один цикл обновления PPO; передаёт в модель сохранённые скрытые состояния только актора."""
    actor_loss_arr = []
    critic_loss_arr = []
    loss_arr = []
    entropy_arr = []
    for _ in range(ppo_epochs):
        for (
            state,
            action,
            old_log_probs,
            return_,
            advantage,
            actor_s,
        ) in ppo_iter(
            mini_batch_size,
            states,
            actions,
            log_probs,
            returns,
            advantages,
            actor_states_flat,
        ):
            dist, value, _, _ = model(state, actor_s)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            actor_loss_arr.append(actor_loss)
            critic_loss_arr.append(critic_loss)
            loss_arr.append(loss)
            entropy_arr.append(entropy)

    mean_actor_loss = torch.mean(torch.stack(actor_loss_arr), dim=0).item()
    mean_critic_loss = torch.mean(torch.stack(critic_loss_arr), dim=0).item()
    mean_loss = torch.mean(torch.stack(loss_arr), dim=0).item()
    mean_entropy = torch.mean(torch.stack(entropy_arr), dim=0).item()

    return mean_actor_loss, mean_critic_loss, mean_loss, mean_entropy
