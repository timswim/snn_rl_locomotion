import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal

from norse.torch.functional.lif import LIFParameters
from norse.torch.module.lif import LIFCell, SequentialState
from norse.torch.module.leaky_integrator import LILinearCell
from norse.torch.module.encode import ConstantCurrentLIFEncoder


# Neural Network
class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_sizes, T=16, std=0.0):
        super(ActorCritic, self).__init__()

        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

        self.constant_current_encoder = ConstantCurrentLIFEncoder(T)
        self.T = T

        self.critic = self.build_network(num_inputs, hidden_sizes, 1)
        self.actor = self.build_network(num_inputs, hidden_sizes, num_outputs) 
        
    def build_network(self, input_size, hidden_sizes, output_size): 
        layers_list = []
        in_size = input_size 
        
        for h in hidden_sizes: 
            layers_list.append(nn.Linear(in_size, h))
            layers_list.append(LIFCell(p=LIFParameters(method="super", alpha=100.0))) 
            in_size = h 
        
        layers_list.append(LILinearCell(in_size, output_size)) 
            
        return SequentialState(*layers_list)

    def forward(self, x):
        x = self.constant_current_encoder(x)
        for t in range(self.T):
            self.critic(x[t, :, :])
            self.actor(x[t, :, :])
        value = self.critic[-1].v
        mu = self.actor[-1].v
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist, value

# GAE
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
def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    ids = np.random.permutation(batch_size)
    ids = np.split(ids[:batch_size // mini_batch_size * mini_batch_size], batch_size // mini_batch_size)
    for i in range(len(ids)):
        yield states[ids[i], :], actions[ids[i], :], log_probs[ids[i], :], returns[ids[i], :], advantage[ids[i], :]

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
        clip_param=0.2,
    ):
    actor_loss_arr = []
    critic_loss_arr = []
    loss_arr = []
    entropy_arr = []
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            dist, value = model(state)
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