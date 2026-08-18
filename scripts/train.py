"""
Единая точка входа обучения PPO (ANN / SNN / hybrid).

Isaac Sim стартует до остальных импортов (AppLauncher), затем Hydra
сливает ``configs/train.yaml`` + ``configs/agent/*.yaml`` и CLI-overrides.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Обучение PPO (ANN / SNN / hybrid) с Hydra-конфигом."
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()


def _hydra_bool_override(hydra_argv: list[str], key: str) -> bool | None:
    """Читает ``key=true/false`` из Hydra-overrides до загрузки конфига."""
    for arg in hydra_argv:
        if "=" not in arg:
            continue
        left, _, right = arg.partition("=")
        left = left.lstrip("+")
        if left != key:
            continue
        return right.lower() in ("true", "1", "yes")
    return None


# Камеры нужны до запуска симулятора, если video=true в overrides.
_video_override = _hydra_bool_override(hydra_args, "video")
if _video_override:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Остальное — после старта Isaac Sim."""

import hydra
from omegaconf import DictConfig, OmegaConf

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# Сейчас закомментирован — используется официальная среда Isaac Lab.
#import locomotion.tasks  # noqa: F401

# Пакет ``rl`` лежит рядом со скриптом.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from rl import PPOTrainer, TrainConfig, make_agent  # noqa: E402

BAD_METRIC_SENTINEL = -1e9


def _build_train_config(
    cfg: DictConfig, log_dir: str, run_name: str | None
) -> TrainConfig:
    """Собирает ``TrainConfig`` из Hydra-конфига."""
    ppo = cfg.ppo
    log_interval = cfg.get("log_interval", None)
    if log_interval is not None:
        log_interval = int(log_interval)

    checkpoint_path = cfg.get("checkpoint_path", None)
    if checkpoint_path is not None:
        checkpoint_path = str(checkpoint_path)

    experiment_name = cfg.get("experiment_name", None)
    if experiment_name is not None:
        experiment_name = str(experiment_name)

    return TrainConfig(
        task=str(cfg.task),
        num_envs=int(cfg.num_envs),
        seed=int(cfg.seed),
        log_dir=log_dir,
        device=str(args_cli.device),
        lr=float(ppo.lr),
        num_steps=int(ppo.num_steps),
        mini_batch_size=int(ppo.mini_batch_size),
        ppo_epochs=int(ppo.ppo_epochs),
        clip_param=float(ppo.clip_param),
        max_steps=int(ppo.max_steps),
        gamma=float(ppo.gamma),
        tau=float(ppo.tau),
        use_mlflow=bool(cfg.use_mlflow),
        run_name=run_name,
        experiment_name=experiment_name,
        log_interval=log_interval,
        save_checkpoints=bool(cfg.save_checkpoints),
        checkpoint_interval=int(cfg.checkpoint_interval),
        checkpoint_path=checkpoint_path,
        video=bool(cfg.video),
        video_length=int(cfg.video_length),
        video_interval=int(cfg.video_interval),
    )


def _make_agent_from_cfg(cfg: DictConfig):
    """Фабрика агента: ``cfg.agent.name`` + остальные поля YAML."""
    agent_dict = OmegaConf.to_container(cfg.agent, resolve=True)
    if not isinstance(agent_dict, dict):
        raise TypeError("cfg.agent должен быть словарём, получено: %r" % type(agent_dict))
    name = agent_dict.pop("name", None)
    if not name:
        raise ValueError("В configs/agent/*.yaml нужно поле name (ann / snn / hybrid).")
    return make_agent(str(name), **agent_dict)


# config_path относительно этого файла (scripts/ → ../configs).
@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Создаёт среду, агента и запускает ``PPOTrainer``."""
    # YAML по умолчанию video=false; overrides уже учтены в cfg.
    if bool(cfg.video) and not getattr(args_cli, "enable_cameras", False):
        print(
            "[WARN] video=true, но камеры не включены при старте AppLauncher. "
            "Передайте video=true в CLI (override), чтобы включить enable_cameras."
        )

    agent = _make_agent_from_cfg(cfg)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("logs", str(cfg.task), timestamp)

    run_name = cfg.get("run_name", None)
    if run_name is None:
        run_name = "train_%s_%s" % (
            agent.name,
            datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
    else:
        run_name = str(run_name)

    train_cfg = _build_train_config(cfg, log_dir=log_dir, run_name=run_name)

    env_cfg = parse_env_cfg(
        train_cfg.task,
        device=args_cli.device,
        num_envs=train_cfg.num_envs,
        use_fabric=not bool(cfg.disable_fabric),
    )

    metric_file = os.environ.get("OPTUNA_METRIC_FILE")
    try:
        print("[INFO] Hydra config:\n%s" % OmegaConf.to_yaml(cfg))
        trainer = PPOTrainer(env_cfg=env_cfg, agent=agent, config=train_cfg)
        metrics = trainer.run()
        print("----------------------------")
        print(
            "Complete. mean_reward=%.4f, final_step=%d"
            % (metrics["mean_reward"], metrics["final_step"])
        )
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
    try:
        main()
    finally:
        simulation_app.close()
