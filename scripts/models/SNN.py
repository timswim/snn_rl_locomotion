import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal

from norse.torch.functional.lif import LIFParameters
from norse.torch.module.lif import LIFCell
from norse.torch.module.sequential import SequentialState
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
