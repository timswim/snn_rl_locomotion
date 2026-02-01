"""Launch Isaac Sim Simulator first."""


import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Cartpole-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, _ = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
from datetime import datetime

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

#---------------------------
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from tensorboardX import SummaryWriter

from models.PPO import ActorCritic, compute_gae, ppo_update

# Use CUDA
use_cuda = torch.cuda.is_available()
device   = torch.device('cuda' if use_cuda else 'cpu')

# Set Seed
seed = 1

random.seed(seed)
np.random.seed(seed)

torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

#-----------------------------

def main():

    # --------Classic workflow--------

    # read the seed from command line
    args_cli_seed = args_cli.seed

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric
    )

    # ---------- LOGGING ----------
    log_dir = os.path.join(
        "logs",
        args_cli.task,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    # create checkpoint dir
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)

    writer = SummaryWriter(logdir=log_dir)

    # ---------- ENV ----------
    # create isaac environment
    envs = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        envs = gym.wrappers.RecordVideo(envs, **video_kwargs)
   
    num_inputs  = envs.observation_space.spaces['policy'].shape[1]
    num_outputs = 2 # сейчас я знаю что это 2, но на будющее надо будет как-то эту размерность вытянуть
    #num_outputs = env.action_space.n
    
    print('State Num: %d, Action Num: %d' % (num_inputs, num_outputs))

    # Hyper params:
    hidden_sizes      = [75, 75]   # 32
    lr               = 2e-3 # 1e-3
    num_steps        = 500  # 128
    mini_batch_size  = 440  # 256
    ppo_epochs       = 45   # 30
    #T = 10                  # 16

    # Исходя из даданных параметров формируется модель
    model = ActorCritic(num_inputs, num_outputs, hidden_sizes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    max_steps = 40000
    step_idx  = 0
    # Обнуляем значение среды
    state = envs.reset()
    state = state[0]['policy']

    #free_reward = [1 for i in range(args_cli.num_envs)]
    #free_reward = torch.FloatTensor(free_reward).to(device)

    while step_idx < max_steps: # Сам цикл обучения
        # обнуляем все массивы в начале эпизода
        log_probs = []
        values    = []
        states    = []
        actions   = []
        rewards   = []
        masks     = []
        entropy = 0

        total_reward = 0

        for _ in range(num_steps): # Почему-то считается эпизод должен уложиться в 128 steps
            dist, value = model(state) # Получаем распределение и оценку функции ценности относительно текущего состояния

            action = dist.sample() # Выбираем случайное? действие  torch.max(action, 1)[1].cpu().numpy()
            wrap_action_0 = torch.max(action, 1, keepdim =True)[0] # вероятноси или что это, непонятно
            wrap_action_1 = torch.max(action, 1, keepdim =True)[1] # само действие
            next_state, reward, terminated, truncated, _  = envs.step(wrap_action_0) # Получаем ответ среды, интересно как тут "_" работает
            total_reward = total_reward + torch.sum(reward).item()
            #reward = free_reward
            done = torch.logical_or(terminated, truncated)
            # Что-то для расчета алгоритма
            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()
            # Запоминаем все
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.unsqueeze(reward.detach().clone(), 1)) # Эту строку надо будет проверить
            #rewards.append(torch.unsqueeze(reward, 1))
            masks.append(torch.unsqueeze(torch.logical_not(done), 1))

            states.append(state)
            actions.append(action)

            state = next_state['policy'] # Обновляем значение состояния
            step_idx += 1

            if step_idx % 250 == 0: # Каждые 100 степов тестируем сетку
                print('Step: %d, Mean total reward: %.2f' % (step_idx, total_reward/args_cli.num_envs))
                #test_reward = test_env()
                #print('Step: %d, Reward: %.2f' % (step_idx, test_reward))
                #writer.add_scalar('Spiking-PPO-' + args_cli.task + '/Reward', test_reward, step_idx)

        # Вычисляем какие-то дополнительные величины для обучения
        next_state = next_state['policy']
        _, next_value = model(next_state)
        returns = compute_gae(next_value, rewards, masks, values)
        '''
        # Visualise data for one example
        example_rewards = torch.cat([i[0] for i in rewards])
        example_values = torch.cat([i[0] for i in values]).detach()
        example_returns = torch.cat([i[0] for i in returns]).detach()
        example_masks = torch.cat([i[0] for i in masks]).detach()
        #example_advantage = example_returns - example_values

        
        x = [i for i in range(num_steps)]
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(x, example_rewards.cpu().numpy())
        axs[0, 0].set_title('example_rewards')
        axs[0, 1].plot(x, example_values.cpu().numpy(), 'tab:orange')
        axs[0, 1].set_title('example_values')
        axs[1, 0].plot(x, example_returns.cpu().numpy(), 'tab:green')
        axs[1, 0].set_title('example_returns')
        axs[1, 1].plot(x, example_masks.cpu().numpy(), 'tab:red')
        axs[1, 1].set_title('example_masks')
        plt.show()
        '''
        # Какая-то постобработка данных
        returns   = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs).detach()
        values    = torch.cat(values).detach()
        states    = torch.cat(states)
        actions   = torch.cat(actions)
        advantage = returns - values

        actor_loss, critic_loss, loss, entropy = ppo_update(
            model,
            optimizer,
            ppo_epochs,
            mini_batch_size,
            states,
            actions,
            log_probs,
            returns,
            advantage
        )

    print('----------------------------')
    print('Complete')

    #writer.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
