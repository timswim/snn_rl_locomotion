"""Пакет обучения PPO: модели, адаптеры агентов, trainer и логирование."""

from .agents import make_agent, register_agent
from .trainer import PPOTrainer, TrainConfig

__all__ = [
    "make_agent",
    "register_agent",
    "PPOTrainer",
    "TrainConfig",
]
