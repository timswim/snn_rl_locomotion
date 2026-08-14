"""GAE и обновление PPO. Единое ядро для ANN / SNN / hybrid."""

import numpy as np
import torch

from .snn_state import index_state_batch


def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    """Обобщённая оценка преимущества (Generalized Advantage Estimation)."""
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


def ppo_iter(
    mini_batch_size,
    states,
    actions,
    log_probs,
    returns,
    advantage,
    actor_states_flat=None,
    critic_states_flat=None,
):
    """
    Итератор по мини-батчам rollout.

    К каждому батчу подмешивает срезы скрытых состояний (None, если сеть без состояния).
    """
    batch_size = states.size(0)
    dev = states.device
    ids = np.random.permutation(batch_size)
    ids = np.split(
        ids[: batch_size // mini_batch_size * mini_batch_size],
        batch_size // mini_batch_size,
    )
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
    actor_states_flat=None,
    critic_states_flat=None,
    clip_param=0.2,
    entropy_coef=0.001,
):
    """
    Один цикл обновления PPO по накопленному rollout.

    Модель вызывается с единым контрактом
    ``forward(x, actor_state, critic_state) -> dist, value, actor_state, critic_state``.
    Скрытые состояния могут быть None (ANN).
    """
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

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - entropy_coef * entropy

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
