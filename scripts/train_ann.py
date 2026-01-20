"""
Script to train ANN RL agent with torch
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an ANN RL agent with torch.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Cartpole-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.") # Не знаю пока что это
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the agent.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, _ = parser.parse_known_args()
# always enable cameras to record video
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

# PLACEHOLDER: Extension template (do not remove this comment)

#-----------------------------

import random
import numpy as np

import torch
import torch.optim as optim

from tensorboardX import SummaryWriter

#from spikePPO import ActorCritic, compute_gae, ppo_update
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
    # ---------- ENV CONFIG ----------
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs, 
        use_fabric=not args_cli.disable_fabric,
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
    
    # ---------- SPACES ----------
    num_inputs  = envs.observation_space.spaces['policy'].shape[1]
    num_outputs = 12 # сейчас я знаю что это 12, но на будющее надо будет как-то эту размерность вытянуть
    #num_outputs = env.action_space.n
    act_dim = envs.action_space.shape[1]  # ← ВАЖНО: больше не хардкодим
    
    print('State Num: %d, Action Num: %d' % (num_inputs, num_outputs))

    # Hyper params:
    hidden_sizes     = [512, 256, 128]  #[512, 256, 128]  
    lr               = 1e-3 # 1e-3
    num_steps        = 24  # 24
    mini_batch_size  = 192  # 192
    ppo_epochs       = 5   # 30

    clip_param = 0.2

    # test params:
    model_test_frec  = 500
    max_reward = -100

    # Исходя из даданных параметров формируется модель
    model = ActorCritic(
        num_inputs,
        num_outputs,
        hidden_sizes
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #actor = Actor(num_inputs=num_inputs, num_outputs=num_outputs, hidden_sizes=[512, 256, 128]).to(device)
    #critic = Critic(num_inputs=num_inputs, hidden_sizes=[512, 256, 128]).to(device)
    #optimizer_actor = torch.optim.Adam(actor.parameters(), lr=1e-4)
    #optimizer_critic = torch.optim.Adam(critic.parameters(), lr=1e-4)

    max_steps = 100000
    step_idx  = 0
    # Обнуляем значение среды
    state = envs.reset()
    state = state[0]['policy']

    from running_ms import RunningMeanStd
    obs_rms = RunningMeanStd(shape=num_inputs, device=device)
    rts_rms = RunningMeanStd(shape=1, device=device)

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
            
            # === 1. Сохраняем текущие эпизодические суммы до step ===
            cached_sums = {}
            for term_name, buf in envs.env.reward_manager._episode_sums.items():
                cached_sums[term_name] = buf.clone()
            
            # 🔹 Обновляем статистику по текущему батчу состояний
            obs_rms.update(state)  # [num_envs, obs_dim]
            # 🔹 Нормализуем состояния перед подачей в модель
            norm_state = obs_rms.normalize(state)

            # Получаем распределение и оценку функции ценности относительно текущего состояния
            dist, value = model(state)
            
            action = dist.sample() # Выбираем случайное? действие  torch.max(action, 1)[1].cpu().numpy()
            wrap_action_0 = torch.max(action, 1, keepdim =True)[0] # вероятноси или что это, непонятно
            wrap_action_1 = torch.max(action, 1, keepdim =True)[1] # само действие
            # === 2. Делаем шаг ===
            next_state, reward, terminated, truncated, _  = envs.step(action) # Получаем ответ среды, интересно как тут "_" работает
            total_reward = total_reward + torch.sum(reward).item()
            # debug
            if (total_reward) < -3000:
                print('alert')
            # debug
            done = torch.logical_or(terminated, truncated)
            # Что-то для расчета алгоритма
            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean() # Зачем тут вообще эта энтропия
            # Запоминаем все
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.unsqueeze(reward.detach().clone(), 1))
            #rewards.append(torch.unsqueeze(reward, 1))
            masks.append(torch.unsqueeze(torch.logical_not(done), 1))

            states.append(state)
            actions.append(action)

            state = next_state['policy'] # Обновляем значение состояния
            step_idx += 1

            if step_idx % model_test_frec == 0: # Каждые model_test_frec степов тестируем сетку
                print('Step: %d, Mean total reward: %.2f' % (step_idx, total_reward/args_cli.num_envs))
                mean_total_reward = total_reward/(args_cli.num_envs * step_idx)
                #print('Step: %d, Reward: %.2f' % (step_idx, test_reward))
                writer.add_scalar('Reward/' + 'mean_total_reward', mean_total_reward, step_idx)
            
            if terminated.any():
                env_ids = terminated.nonzero(as_tuple=False).squeeze(-1)

                for i in env_ids:
                    env_i = i.item()
                    for term_name in cached_sums:
                        episodic_value = cached_sums[term_name][env_i] / envs.env.max_episode_length_s
                        writer.add_scalar(f"Episode_Reward/{term_name}", episodic_value.item(), step_idx)

        # Вычисляем какие-то дополнительные величины для обучения
        next_state = next_state['policy']
        _, next_value = model(next_state)

        '''
        # Нормализация rewards
        # Преобразуем список тензоров rewards -> [T, 1] тензор
        rewards_tensor = torch.cat(rewards, dim=0)
        # Z-score нормализация по rollout
        mean_reward = rewards_tensor.mean()
        std_reward = rewards_tensor.std() + 1e-8
        normalized_rewards_tensor = (rewards_tensor - mean_reward) / std_reward
        # Разбиваем обратно на список тензоров поэлементно (чтобы не менять compute_gae)
        normalized_rewards = torch.split(normalized_rewards_tensor, args_cli.num_envs, dim=0)
        rewards = [r.clone()/20 for r in normalized_rewards]  # пересохраняем обратно в список
        '''
        '''
        rewards_tensor = torch.stack(rewards, dim=0)  # [T, N, 1]
        mean_r = rewards_tensor.mean()
        std_r = rewards_tensor.std() + 1e-8
        rewards_tensor = (rewards_tensor - mean_r) / std_r
        # обратно в список шагов
        rewards = [rewards_tensor[t] for t in range(num_steps)]
        '''


        returns, deltas = compute_gae(next_value, rewards, masks, values)

        # Какая-то постобработка данных
        rewards   = torch.cat(rewards).detach()
        deltas    = torch.cat(deltas).detach()
        returns   = torch.cat(returns)
        log_probs = torch.cat(log_probs)
        values    = torch.cat(values)
        states    = torch.cat(states)
        actions   = torch.cat(actions)
        advantage = returns - values
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8) # Нормализуем advantage
         # Нормализуем returns
        rts_rms.update(returns) # Обновление rms для returns
        returns_norm = rts_rms.normalize(returns)
        #print("log_probs requires_grad:", log_probs.requires_grad)
        #print("values requires_grad:", values.requires_grad)
        #print("advantages requires_grad:", advantage.requires_grad)

        #print("rewards mean/std:", rewards.mean().item(), rewards.std().item())
        #print("returns mean/std:", returns.mean().item(), returns.std().item())
        #print("values mean/std:", values.mean().item(), values.std().item())
        
        actor_loss, critic_loss, entropy, loss, ratio_mean, learning_rate = ppo_update(model, 
                   optimizer, 
                   ppo_epochs, 
                   mini_batch_size, 
                   states, 
                   actions, 
                   log_probs, 
                   returns,
                   advantage,
                   values,
                   clip_param = clip_param,
                   value_clip=0.2, # 0.2
                   entropy_loss_scale=0.01, # 0.01
                   value_loss_scale=1.0, # 1.0
                   kl_threshold=0.01) # 0.01
        
        writer.add_histogram('Info/' + 'rewards', rewards, step_idx)
        writer.add_histogram('Info/' + 'values', values, step_idx)
        writer.add_histogram('Info/' + 'returns', returns, step_idx)
        writer.add_histogram('Info/' + 'advantage', advantage, step_idx)
        writer.add_histogram('Info/' + 'deltas', deltas, step_idx)
        writer.add_histogram('Info/' + 'log_probs', log_probs, step_idx)
        writer.add_scalar('Loss/' + 'actor_loss', actor_loss, step_idx)
        writer.add_scalar('Loss/' + 'critic_loss', critic_loss, step_idx)
        writer.add_scalar('Loss/' + 'entropy', entropy, step_idx)
        writer.add_scalar('Loss/' + 'loss', loss, step_idx)
        writer.add_scalar('Loss/' + 'ratio_mean', ratio_mean, step_idx)
        writer.add_scalar('Loss/' + 'learning_rate', learning_rate, step_idx)

        torch.save({"state_dict": model.state_dict(), # А надо ли сохранять веса критика тоже?
                                "optimizer": optimizer.state_dict(), 
                                "hidden_sizes": hidden_sizes},
                                #"T": T}, 
                                os.path.join(log_dir, 'checkpoints', 'agent_' + str(step_idx) + '.pth'))

    print('----------------------------')
    print('Complete')

    #writer.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
