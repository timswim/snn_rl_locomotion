"""
Script to train ANN RL agent with torch.
Supports optional MLFlow logging and Optuna hyperparameter tuning.
"""
"mlflow ui --backend-store-uri ./mlruns --port 5000"
"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an SNN RL agent with torch.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Unitree-A1-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the agent.",
)
# Hyperparameter overrides (optional)
parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
parser.add_argument("--num_steps", type=int, default=None, help="PPO rollout steps per update.")
parser.add_argument("--mini_batch_size", type=int, default=None, help="PPO mini-batch size.")
parser.add_argument("--ppo_epochs", type=int, default=None, help="PPO epochs per update.")
parser.add_argument("--clip_param", type=float, default=None, help="PPO clip parameter.")
parser.add_argument("--max_steps", type=int, default=None, help="Total training steps.")
parser.add_argument(
    "--hidden_sizes",
    type=int,
    nargs="*",
    default=None,
    help="Hidden layer sizes (e.g. 512 256 128). Default: 512 256 128.",
)
parser.add_argument("--T", type=int, default=10, help="Time steps.")
# MLFlow and Optuna
parser.add_argument("--use_mlflow", action="store_true", default=True, help="Log to local MLFlow.")
parser.add_argument("--optuna", action="store_true", default=False, help="Run Optuna hyperparameter study.")
parser.add_argument("--optuna_n_trials", type=int, default=20, help="Number of Optuna trials when --optuna is set.")
parser.add_argument("--run_name", type=str, default=None, help="MLFlow run name (default: train_ann_<timestamp> when run alone).")
parser.add_argument("--experiment", type=str, default="test", help="MLFlow experiment name (default: use MLFlow's default).")
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

import os
import random
from dataclasses import dataclass, field
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)

from models.SNN_PPO import (
    ActorCritic,
    compute_gae,
    detach_state,
    initial_zero_hidden,
    ppo_update,
    reset_state_batch_indices,
    stack_rollout_states,
)

# CUDA при наличии
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Default hyperparameters
DEFAULT_HIDDEN_SIZES = [512, 256, 128]
DEFAULT_LR = 1e-4
DEFAULT_NUM_STEPS = 32
DEFAULT_MINI_BATCH_SIZE = 192
DEFAULT_PPO_EPOCHS = 5
DEFAULT_CLIP_PARAM = 0.2
DEFAULT_MAX_STEPS = 50000
DEFAULT_T = 10

MODEL_TEST_FREQ = DEFAULT_NUM_STEPS*10 # Логгируем каждые 10 rollout
CHECKPOINT_INTERVAL = 5000


@dataclass
class Hyperparams:
    """Hyperparameter container for PPO training."""

    hidden_sizes: list = field(default_factory=lambda: list(DEFAULT_HIDDEN_SIZES))
    lr: float = DEFAULT_LR
    num_steps: int = DEFAULT_NUM_STEPS
    mini_batch_size: int = DEFAULT_MINI_BATCH_SIZE
    ppo_epochs: int = DEFAULT_PPO_EPOCHS
    clip_param: float = DEFAULT_CLIP_PARAM
    max_steps: int = DEFAULT_MAX_STEPS
    T: int = DEFAULT_T

    def to_dict(self):
        return {
            "hidden_sizes": self.hidden_sizes,
            "lr": self.lr,
            "num_steps": self.num_steps,
            "mini_batch_size": self.mini_batch_size,
            "ppo_epochs": self.ppo_epochs,
            "clip_param": self.clip_param,
            "max_steps": self.max_steps,
            "T": self.T,
        }


def _build_hyperparams_from_cli():
    """Build Hyperparams from defaults and CLI overrides."""
    hp = Hyperparams()
    if args_cli.lr is not None:
        hp.lr = args_cli.lr
    if args_cli.num_steps is not None:
        hp.num_steps = args_cli.num_steps
    if args_cli.mini_batch_size is not None:
        hp.mini_batch_size = args_cli.mini_batch_size
    if args_cli.ppo_epochs is not None:
        hp.ppo_epochs = args_cli.ppo_epochs
    if args_cli.clip_param is not None:
        hp.clip_param = args_cli.clip_param
    if args_cli.max_steps is not None:
        hp.max_steps = args_cli.max_steps
    if args_cli.hidden_sizes is not None and len(args_cli.hidden_sizes) > 0:
        hp.hidden_sizes = list(args_cli.hidden_sizes)
    if args_cli.max_iterations is not None:
        hp.max_steps = args_cli.max_iterations
    if args_cli.T is not None:
        hp.T = args_cli.T
    return hp


def train_one_run(
    env_cfg,
    hyperparams: Hyperparams,
    log_dir: str,
    seed: int,
    use_mlflow: bool = False,
    run_name: str = None,
    checkpoint_path: str = None,
    task: str = None,
    num_envs: int = 32,
    video: bool = False,
    video_length: int = 200,
    video_interval: int = 2000,
    experiment_name: str = None,
):
    """
    Один прогон обучения PPO: создаёт среду, модель, цикл обучения, закрывает среду.
    Возвращает словарь метрик для Optuna/MLFlow (например mean_reward, final_step).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)

    writer = SummaryWriter(logdir=log_dir)

    if use_mlflow:
        import mlflow
        # Use env var so subprocess drivers (optuna_tune_ann.py) can point to the same store as the UI
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or ("file://" + os.path.abspath("mlruns"))
        mlflow.set_tracking_uri(tracking_uri)
        if experiment_name is not None:
            mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name)
        mlflow.log_params({
            "task": task,
            "num_envs": num_envs,
            "seed": seed,
            **{k: str(v) if isinstance(v, list) else v for k, v in hyperparams.to_dict().items()},
        })

    # Set seed on env config so environment creation is deterministic (avoids Isaac Lab warning).
    env_cfg.seed = seed

    envs = None
    envs = gym.make(
        task,
        cfg=env_cfg,
        render_mode="rgb_array" if video else None,
    )

    if video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % video_interval == 0,
            "video_length": video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        envs = gym.wrappers.RecordVideo(envs, **video_kwargs)

    num_inputs = envs.observation_space.spaces["policy"].shape[1]
    num_outputs = envs.action_space.shape[1]
    print("State Num: %d, Action Num: %d" % (num_inputs, num_outputs))

    model = ActorCritic(
        num_inputs,
        num_outputs,
        hyperparams.hidden_sizes,
        T=hyperparams.T,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hyperparams.lr)

    step_idx = 0
    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "step_idx" in ckpt:
            step_idx = ckpt["step_idx"]
        print("[INFO] Продолжение с чекпоинта: %s, шаг %d" % (checkpoint_path, step_idx))

    current_actor_state, current_critic_state = initial_zero_hidden(
        model, num_envs, num_inputs, device
    )

    state, _ = envs.reset()
    state = state["policy"]

    recent_rewards = []
    mean_reward = 0.0
    try:
        while step_idx < hyperparams.max_steps:
            log_probs = []
            values = []
            states = []
            actions = []
            rewards = []
            masks = []
            total_reward = 0
            rollout_actor_states = []
            rollout_critic_states = []

            for _ in range(hyperparams.num_steps):
                cached_sums = {}
                for term_name, buf in envs.env.reward_manager._episode_sums.items():
                    cached_sums[term_name] = buf.clone()

                rollout_actor_states.append(detach_state(current_actor_state))
                rollout_critic_states.append(detach_state(current_critic_state))
                dist, value, current_actor_state, current_critic_state = model(
                    state, current_actor_state, current_critic_state
                )
                current_actor_state = detach_state(current_actor_state)
                current_critic_state = detach_state(current_critic_state)
                action = dist.sample()
                next_state, reward, terminated, truncated, _ = envs.step(action)
                total_reward = total_reward + torch.sum(reward).item()

                done = torch.logical_or(terminated, truncated)
                # Среда перезапускает эпизод: обнуляем скрытое состояние SNN для этих env,
                # иначе память LIF переносится на новый rollout и искажает PPO.
                if done.any():
                    env_ids = torch.where(done)[0]
                    reset_state_batch_indices(current_actor_state, env_ids)
                    reset_state_batch_indices(current_critic_state, env_ids)
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
                            episodic_value = cached_sums[term_name][env_i] / envs.env.max_episode_length_s
                            writer.add_scalar("Episode_Reward/%s" % term_name, episodic_value.item(), step_idx)
                            if use_mlflow:
                                mlflow.log_metric("Episode_Reward/%s" % term_name, episodic_value.item(), step=step_idx)

            # После полного rollout: средняя награда за шаг на одно env
            mean_total_reward = total_reward / (num_envs * hyperparams.num_steps)
            recent_rewards.append(mean_total_reward)
            if step_idx % (hyperparams.num_steps*10) == 0: # Логгируем каждые 10 rollout TODO: убрать магическое число 10
                print("Step: %d, Mean total reward: %.2f" % (step_idx, mean_total_reward))
                writer.add_scalar("Reward/mean_total_reward", mean_total_reward, step_idx)
                if use_mlflow:
                    mlflow.log_metric("Reward/mean_total_reward", mean_total_reward, step=step_idx)

            next_val_state = next_state["policy"]
            _, next_value, _, _ = model(
                next_val_state, current_actor_state, current_critic_state
            )
            returns = compute_gae(next_value, rewards, masks, values)

            rewards = torch.cat(rewards).detach()
            returns = torch.cat(returns).detach()
            log_probs = torch.cat(log_probs).detach()
            values = torch.cat(values).detach()
            states = torch.cat(states)
            actions = torch.cat(actions)
            advantage = returns - values

            actor_states_flat = stack_rollout_states(rollout_actor_states)
            critic_states_flat = stack_rollout_states(rollout_critic_states)

            actor_loss, critic_loss, loss, entropy = ppo_update(
                model,
                optimizer,
                hyperparams.ppo_epochs,
                hyperparams.mini_batch_size,
                states,
                actions,
                log_probs,
                returns,
                advantage,
                actor_states_flat,
                critic_states_flat,
                clip_param=hyperparams.clip_param,
            )

            writer.add_histogram("Info/rewards", rewards, step_idx)
            writer.add_histogram("Info/values", values, step_idx)
            writer.add_histogram("Info/returns", returns, step_idx)
            writer.add_histogram("Info/advantage", advantage, step_idx)
            writer.add_histogram("Info/log_probs", log_probs, step_idx)
            writer.add_scalar("Loss/actor_loss", actor_loss, step_idx)
            writer.add_scalar("Loss/critic_loss", critic_loss, step_idx)
            writer.add_scalar("Loss/entropy", entropy, step_idx)
            writer.add_scalar("Loss/loss", loss, step_idx)

            if use_mlflow:
                mlflow.log_metrics({
                    "Loss/actor_loss": actor_loss,
                    "Loss/critic_loss": critic_loss,
                    "Loss/entropy": entropy,
                    "Loss/loss": loss,
                }, step=step_idx)

            if step_idx % CHECKPOINT_INTERVAL == 0: # Пока не будем сохранять модели
                ckpt_path = os.path.join(log_dir, "checkpoints", "agent_%d.pth" % step_idx)
                #torch.save({
                #    "state_dict": model.state_dict(),
                #    "optimizer": optimizer.state_dict(),
                #    "hidden_sizes": hyperparams.hidden_sizes,
                #    "step_idx": step_idx,
               # }, ckpt_path)

        mean_reward = float(np.mean(recent_rewards)) if recent_rewards else 0.0
    finally:
        if envs is not None:
            envs.close()
        writer.close()
        if use_mlflow:
            mlflow.log_metric("mean_reward", mean_reward)
            mlflow.log_artifacts(log_dir, artifact_path="run")
            mlflow.end_run()

    return {"mean_reward": mean_reward, "final_step": step_idx}


def main():
    seed = args_cli.seed if args_cli.seed is not None else 1

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    if args_cli.optuna:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            hp = Hyperparams(
                lr=trial.suggest_float("lr", 1e-5, 1e-3, log=True),
                num_steps=trial.suggest_int("num_steps", 16, 64),
                mini_batch_size=trial.suggest_int("mini_batch_size", 64, 512),
                ppo_epochs=trial.suggest_int("ppo_epochs", 3, 15),
                clip_param=trial.suggest_float("clip_param", 0.1, 0.3),
                hidden_sizes=[512, 256, 128],
                max_steps=DEFAULT_MAX_STEPS,
            )
            if args_cli.max_iterations is not None:
                hp.max_steps = args_cli.max_iterations
            log_dir = os.path.join(
                "logs",
                args_cli.task,
                "optuna",
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                "trial_%d" % trial.number,
            )
            trial_seed = seed + trial.number
            metrics = train_one_run(
                env_cfg=env_cfg,
                hyperparams=hp,
                log_dir=log_dir,
                seed=trial_seed,
                use_mlflow=args_cli.use_mlflow,
                run_name="trial_%d" % trial.number,
                checkpoint_path=None,
                task=args_cli.task,
                num_envs=args_cli.num_envs,
                video=args_cli.video,
                video_length=args_cli.video_length,
                video_interval=args_cli.video_interval,
                experiment_name=args_cli.experiment,
            )
            return metrics["mean_reward"]

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args_cli.optuna_n_trials)
        print("Best trial: value=%.4f, params=%s" % (study.best_value, study.best_params))
        return

    hp = _build_hyperparams_from_cli()
    if args_cli.max_iterations is not None:
        hp.max_steps = args_cli.max_iterations

    log_dir = os.path.join(
        "logs",
        args_cli.task,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )

    metric_file = os.environ.get("OPTUNA_METRIC_FILE")
    BAD_METRIC_SENTINEL = -1e9  # written on crash so Optuna driver can continue

    try:
        run_name = args_cli.run_name or ("train_snn_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        metrics = train_one_run(
            env_cfg=env_cfg,
            hyperparams=hp,
            log_dir=log_dir,
            seed=seed,
            use_mlflow=args_cli.use_mlflow,
            run_name=run_name,
            checkpoint_path=args_cli.checkpoint,
            task=args_cli.task,
            num_envs=args_cli.num_envs,
            video=args_cli.video,
            video_length=args_cli.video_length,
            video_interval=args_cli.video_interval,
            experiment_name=args_cli.experiment,
        )
        print("----------------------------")
        print("Complete. mean_reward=%.4f, final_step=%d" % (metrics["mean_reward"], metrics["final_step"]))
        if metric_file:
            with open(metric_file, "w") as f:
                f.write("%.6f\n" % metrics["mean_reward"])
    except Exception:
        if metric_file:
            try:
                with open(metric_file, "w") as f:
                    f.write("%.6f\n" % BAD_METRIC_SENTINEL)
            except Exception:
                pass
        raise


if __name__ == "__main__":
    main()
    simulation_app.close()
