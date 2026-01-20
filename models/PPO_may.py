import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


# Neural Network

class ActorCritic(nn.Module): # Описание сетей системы, в данном случае они практически идентичные. имеют один скрытый слой
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs)
        )

        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

    # Функция прямого прохода данных по сети
    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)  # Дисперсия
        dist  = Normal(mu, std) # Наше распределение
        return dist, value
    
# GAE (generalised advantage estimation)
def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


# Proximal Policy Optimization Algorithm
# Arxiv: "https://arxiv.org/abs/1707.06347"
# Итерируемо возвращает набор величин, нужных для одного прохода по сетке
def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    ids = np.random.permutation(batch_size)
    ids = np.split(ids[:batch_size // mini_batch_size * mini_batch_size], batch_size // mini_batch_size)
    for i in range(len(ids)):
        yield states[ids[i], :], actions[ids[i], :], log_probs[ids[i], :], returns[ids[i], :], advantage[ids[i], :]

# ppo_update разу работает с целым батчем значений, возможно раз в эпизод
def ppo_update(model, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
    total_entropy = []
    total_actor_loss = []
    total_critic_loss = []
    total_loss = []
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            dist, value = model(state) # Модель не является воходным параметром а просто как глобальная вызывается? Возвращает распределение и выход функции ценности
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()
            # Для данного степа считается ошибка (почему не для батча?) и выполняется оптимизация
            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy
            total_loss.append(loss)
            total_entropy.append(entropy)
            total_actor_loss.append(actor_loss)
            total_critic_loss.append(critic_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()