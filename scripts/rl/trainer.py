"""Общий цикл PPO: среда, rollout, GAE, обновление, логирование. Без ветвления по типу агента."""

import os
import random
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch

from .logging import MLflowLogger
from .ppo import compute_gae, ppo_update


@dataclass
class TrainConfig:
    """Конфигурация одного прогона PPO (поля из Hydra ``configs/train.yaml``)."""

    task: str
    num_envs: int = 256
    seed: int = 1
    log_dir: str = "logs"
    device: str = "cuda"

    lr: float = 1e-4
    num_steps: int = 32
    mini_batch_size: int = 192
    ppo_epochs: int = 5
    clip_param: float = 0.2
    max_steps: int = 50000
    gamma: float = 0.99
    tau: float = 0.95

    use_mlflow: bool = True
    run_name: str | None = None
    experiment_name: str | None = None
    log_interval: int | None = None

    save_checkpoints: bool = False
    checkpoint_interval: int = 5000
    checkpoint_path: str | None = None

    video: bool = False
    video_length: int = 200
    video_interval: int = 2000


class PPOTrainer:
    """
    Один прогон обучения PPO.

    Агент (ANN / SNN / hybrid) передаётся адаптером: trainer не содержит
    ``if agent == "snn"``.
    """

    def __init__(self, env_cfg, agent, config: TrainConfig):
        """
        Параметры:
            env_cfg: конфиг среды Isaac Lab (после ``parse_env_cfg``).
            agent: адаптер агента; trainer не ветвится по типу сети.
            config: гиперпараметры одного прогона.
        """
        self.env_cfg = env_cfg
        self.agent = agent
        self.config = config
        if str(config.device).startswith("cuda") and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(config.device)

    def run(self):
        """
        Создаёт среду и модель, выполняет цикл, закрывает среду.

        Возвращает:
            dict с ``mean_reward`` и ``final_step``.
        """
        cfg = self.config
        agent = self.agent
        seed = cfg.seed

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        os.makedirs(cfg.log_dir, exist_ok=True)
        os.makedirs(os.path.join(cfg.log_dir, "checkpoints"), exist_ok=True)

        log_params = {
            "task": cfg.task,
            "num_envs": cfg.num_envs,
            "seed": seed,
            "lr": cfg.lr,
            "num_steps": cfg.num_steps,
            "mini_batch_size": cfg.mini_batch_size,
            "ppo_epochs": cfg.ppo_epochs,
            "clip_param": cfg.clip_param,
            "max_steps": cfg.max_steps,
            **agent.extra_log_params(),
        }
        logger = MLflowLogger(
            enabled=cfg.use_mlflow,
            log_dir=cfg.log_dir,
            run_name=cfg.run_name,
            experiment_name=cfg.experiment_name,
            params=log_params,
        )

        self.env_cfg.seed = seed
        envs = None
        step_idx = 0
        mean_reward = 0.0
        log_interval = (
            cfg.log_interval if cfg.log_interval is not None else cfg.num_steps * 10
        )

        try:
            envs = gym.make(
                cfg.task,
                cfg=self.env_cfg,
                render_mode="rgb_array" if cfg.video else None,
            )
            if cfg.video:
                video_kwargs = {
                    "video_folder": os.path.join(cfg.log_dir, "videos"),
                    "step_trigger": lambda step: step % cfg.video_interval == 0,
                    "video_length": cfg.video_length,
                    "disable_logger": True,
                }
                print("[INFO] Recording videos during training.")
                envs = gym.wrappers.RecordVideo(envs, **video_kwargs)

            num_inputs = envs.observation_space.spaces["policy"].shape[1]
            num_outputs = envs.action_space.shape[1]
            print("State Num: %d, Action Num: %d" % (num_inputs, num_outputs))

            agent.build(num_inputs, num_outputs, self.device, cfg.lr)

            if cfg.checkpoint_path and os.path.isfile(cfg.checkpoint_path):
                step_idx = agent.load_checkpoint(cfg.checkpoint_path, self.device)
                print(
                    "[INFO] Продолжение с чекпоинта: %s, шаг %d"
                    % (cfg.checkpoint_path, step_idx)
                )

            hidden = agent.init_hidden(cfg.num_envs, num_inputs, self.device)
            state, _ = envs.reset()
            state = state["policy"]

            recent_rewards = []
            while step_idx < cfg.max_steps:
                log_probs = []
                values = []
                states = []
                actions = []
                rewards = []
                masks = []
                total_reward = 0
                rollout_hidden = []
                agent.on_rollout_start(step_idx, cfg.use_mlflow)

                for rollout_step in range(cfg.num_steps):
                    cached_sums = {}
                    for term_name, buf in envs.env.reward_manager._episode_sums.items():
                        cached_sums[term_name] = buf.clone()

                    rollout_hidden.append(agent.snapshot(hidden))
                    dist, value, hidden = agent.act(state, hidden, rollout_step)
                    action = dist.sample()
                    agent.after_action(action, rollout_step)
                    next_state, reward, terminated, truncated, _ = envs.step(action)
                    total_reward = total_reward + torch.sum(reward).item()

                    done = torch.logical_or(terminated, truncated)
                    agent.reset_on_done(hidden, done)
                    log_prob = dist.log_prob(action)
                    log_probs.append(log_prob)
                    values.append(value)
                    rewards.append(torch.unsqueeze(reward.detach().clone(), 1))
                    masks.append(torch.unsqueeze(torch.logical_not(done), 1))
                    states.append(state)
                    actions.append(action)

                    state = next_state["policy"]
                    step_idx += 1

                    if terminated.any():
                        env_ids = terminated.nonzero(as_tuple=False).squeeze(-1)
                        for i in env_ids:
                            env_i = i.item()
                            for term_name in cached_sums:
                                episodic_value = (
                                    cached_sums[term_name][env_i]
                                    / envs.env.max_episode_length_s
                                )
                                logger.log_metric(
                                    "Episode_Reward/%s" % term_name,
                                    episodic_value.item(),
                                    step=step_idx,
                                )

                agent.on_rollout_end(step_idx, logger)

                mean_total_reward = total_reward / (cfg.num_envs * cfg.num_steps)
                recent_rewards.append(mean_total_reward)
                if step_idx % log_interval == 0:
                    print(
                        "Step: %d, Mean total reward: %.4f"
                        % (step_idx, mean_total_reward)
                    )
                    logger.log_metric(
                        "Reward/mean_total_reward", mean_total_reward, step=step_idx
                    )

                next_value = agent.value(next_state["policy"], hidden)
                returns = compute_gae(
                    next_value, rewards, masks, values, gamma=cfg.gamma, tau=cfg.tau
                )

                returns = torch.cat(returns).detach()
                log_probs = torch.cat(log_probs).detach()
                values = torch.cat(values).detach()
                states = torch.cat(states)
                actions = torch.cat(actions)
                advantage = returns - values

                actor_flat, critic_flat = agent.stack_snapshots(rollout_hidden)
                actor_loss, critic_loss, loss, entropy = ppo_update(
                    agent.model,
                    agent.optimizer,
                    cfg.ppo_epochs,
                    cfg.mini_batch_size,
                    states,
                    actions,
                    log_probs,
                    returns,
                    advantage,
                    actor_flat,
                    critic_flat,
                    clip_param=cfg.clip_param,
                )

                logger.log_metrics(
                    {
                        "Loss/actor_loss": actor_loss,
                        "Loss/critic_loss": critic_loss,
                        "Loss/entropy": entropy,
                        "Loss/loss": loss,
                    },
                    step=step_idx,
                )

                if (
                    cfg.save_checkpoints
                    and step_idx % cfg.checkpoint_interval == 0
                ):
                    ckpt_path = os.path.join(
                        cfg.log_dir, "checkpoints", "agent_%d.pth" % step_idx
                    )
                    agent.save_checkpoint(ckpt_path, step_idx)

            mean_reward = float(np.mean(recent_rewards)) if recent_rewards else 0.0
        finally:
            if envs is not None:
                envs.close()
            logger.finish(mean_reward)

        return {"mean_reward": mean_reward, "final_step": step_idx}
