"""
Модель SNN (актор-критик), GAE и обновление PPO с учётом скрытых состояний Norse.
"""
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal

from norse.torch.functional.lif import LIFParameters
from norse.torch.module.lif import LIFCell
from norse.torch.module.sequential import SequentialState
from norse.torch.module.leaky_integrator import LILinearCell
from norse.torch.module.encode import ConstantCurrentLIFEncoder, PopulationEncoder


class DeviceAwarePopulationEncoder(PopulationEncoder):
    """Norse PopulationEncoder создаёт centres на CPU — переносим на device входа."""

    def forward(self, input_tensor):
        size = input_tensor.shape + (self.out_features,)
        scale = self.scale if self.scale is not None else input_tensor.max()
        centres = torch.linspace(
            0,
            scale,
            self.out_features,
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        ).expand(size)
        x = input_tensor.unsqueeze(-1).expand(size)
        distances = self.distance_function(x, centres) * scale
        return self.kernel(distances)

# BPTT / detach:
# - Внутри одного forward() градиенты идут по микрошагам t=0..T-1 (динамика мембраны).
# - Между шагами среды скрытое состояние отсоединяем (detach), чтобы граф не тянулся
#   на всю длину rollout (экономия памяти). Градиенты по параметрам считаются по текущему
#   наблюдению и отсоединённым потенциалам на входе (нет dL/dv между шагами среды).
# - PPO пересчитывает каждый переход с сохранённым скрытым состоянием, чтобы шаг
#   оптимизации видел то же представление, что и при сборе данных.

def structural_zeros_like(state):
    """Рекурсивно строит дерево нулевых тензоров той же структуры, что и состояние Norse (список / named tuple)."""
    if state is None:
        return None
    if isinstance(state, list):
        return [structural_zeros_like(s) for s in state]
    if hasattr(state, "_fields"):
        return type(state)(
            *(torch.zeros_like(t) if isinstance(t, torch.Tensor) else t for t in state)
        )
    if isinstance(state, torch.Tensor): # Тут и в следующих функциях это запасной вариант, сюда даже не заходит программа.
        return torch.zeros_like(state)
    return state


def reset_state_batch_indices(state, indices):
    """
    Обнуляет скрытое состояние Norse по размерности батча (dim=0) для указанных индексов сред.
    Вызывать при завершении эпизода (terminated / truncated / time limit), когда среда автоматически
    перезапускается, чтобы память SNN не переносилась между эпизодами.
    """
    if state is None:
        return
    if not isinstance(indices, torch.Tensor) or indices.numel() == 0:
        return
    t0 = next(_state_tensors(state), None)
    if t0 is not None:
        indices = indices.to(device=t0.device, dtype=torch.long)
    if isinstance(state, list):
        for s in state:
            reset_state_batch_indices(s, indices)
        return
    if hasattr(state, "_fields"):
        for t in state:
            if isinstance(t, torch.Tensor):
                t[indices] = 0
        return
    if isinstance(state, torch.Tensor):
        state[indices] = 0


def detach_state(state):
    """Отсоединяет тензоры состояния (градиент не идёт через предыдущий шаг)."""
    if state is None:
        return None
    if isinstance(state, list):
        return [detach_state(s) for s in state]
    if hasattr(state, "_fields"):
        return type(state)(*(t.detach() if isinstance(t, torch.Tensor) else t for t in state))
    if isinstance(state, torch.Tensor):
        return state.detach()
    return state


def stack_rollout_states(states_per_step):
    """
    Склеивает список состояний по шагам (батч = num_envs) в одно состояние
    с батчем num_steps * num_envs (тот же порядок, что у torch.cat по списку наблюдений).
    """
    if not states_per_step:
        return None
    if states_per_step[0] is None:
        return None
    if isinstance(states_per_step[0], list):
        return [
            stack_rollout_states([s[i] for s in states_per_step])
            for i in range(len(states_per_step[0]))
        ]
    if hasattr(states_per_step[0], "_fields"):
        fields = states_per_step[0]._fields
        out = []
        for f in fields:
            parts = [getattr(s, f) for s in states_per_step]
            if parts[0] is None:
                out.append(None)
            else:
                out.append(torch.cat(parts, dim=0))
        return type(states_per_step[0])(*out)
    if isinstance(states_per_step[0], torch.Tensor):
        return torch.cat(states_per_step, dim=0)
    return states_per_step[0]


def index_state_batch(state, indices, device=None):
    """Индексация по размерности 0 у всех тензоров в state (indices: long tensor или ndarray)."""
    if state is None:
        return None
    if isinstance(indices, np.ndarray):
        dev = device
        if dev is None:
            t0 = next(_state_tensors(state), None)
            dev = t0.device if t0 is not None else torch.device("cpu")
        indices = torch.as_tensor(indices, dtype=torch.long, device=dev)
    if isinstance(state, list):
        return [index_state_batch(s, indices) for s in state]
    if hasattr(state, "_fields"):
        return type(state)(
            *(
                t[indices] if isinstance(t, torch.Tensor) else t
                for t in state
            )
        )
    if isinstance(state, torch.Tensor):
        return state[indices]
    return state


def _state_tensors(state):
    """Обёртка по листьям-тензорам для определения устройства."""
    if state is None:
        return
    if isinstance(state, list):
        for s in state:
            yield from _state_tensors(s)
    elif hasattr(state, "_fields"):
        for t in state:
            if isinstance(t, torch.Tensor):
                yield t
    elif isinstance(state, torch.Tensor):
        yield state


def forward_sequential_state_with_lif_spikes(module, input_tensor, state=None):
    """
    Forward через SequentialState с возвратом спайков после каждого LIFCell.
    Returns: output, state, lif_spikes — список тензоров (batch, n_hidden) по порядку LIF-слоёв.
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
    lif_spikes_per_t: список длины T; каждый элемент — список LIF-спайков по слоям.
    Returns: список из L float — средняя доля спайков (%) по T микрошагам для каждого слоя.
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


def initial_zero_hidden(model, num_envs, num_inputs, device):
    """Один раз: формы как у скрытого состояния после forward; нули как «холодный старт» (аналог None)."""
    with torch.no_grad():
        z = torch.zeros(num_envs, num_inputs, device=device)
        _, _, a, c = model(z, None, None)
    return detach_state(structural_zeros_like(a)), detach_state(structural_zeros_like(c))


# Нейросеть (актор-критик)
class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_sizes, T=16, alpha=1.0, std=0.0, lif_v_th=0.2, dt=0.01):
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
        #self.population_encoder = DeviceAwarePopulationEncoder(5)
        self.T = T
        self.alpha = alpha

        self.critic = self.build_network(num_inputs, hidden_sizes, 1)
        self.actor = self.build_network(num_inputs, hidden_sizes, num_outputs)

    def build_network(self, input_size, hidden_sizes, output_size):
        layers_list = []
        in_size = input_size

        for h in hidden_sizes:
            layers_list.append(nn.Linear(in_size, h))
            layers_list.append(LIFCell(p=self.lif_params, dt=self.dt))
            in_size = h

        layers_list.append(LILinearCell(in_size, output_size)) 

        return SequentialState(*layers_list)

    def forward(
        self,
        x,
        actor_state=None,
        critic_state=None,
        return_mu_trace=False,
        return_spike_activity=False,
    ):
        #x = self.population_encoder(x)
        #x = x.reshape(x.shape[0], -1)
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

# GAE — обобщённая оценка преимущества (Generalized Advantage Estimation)
def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

# PPO — Proximal Policy Optimization (https://arxiv.org/abs/1707.06347)
def ppo_iter(
    mini_batch_size,
    states,
    actions,
    log_probs,
    returns,
    advantage,
    actor_states_flat,
    critic_states_flat,
):
    """Итератор по мини-батчам rollout; к каждому батчу подмешивает срезы скрытых состояний."""
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
            index_state_batch(critic_states_flat, idx, device=dev),
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
        critic_states_flat,
        clip_param=0.2,
    ):
    """Один цикл обновления PPO по накопленному rollout; передаёт в модель сохранённые скрытые состояния."""
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
            critic_s,
        ) in ppo_iter(
            mini_batch_size,
            states,
            actions,
            log_probs,
            returns,
            advantages,
            actor_states_flat,
            critic_states_flat,
        ):
            dist, value, _, _ = model(state, actor_s, critic_s)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
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
