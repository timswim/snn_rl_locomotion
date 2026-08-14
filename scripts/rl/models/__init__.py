"""Модели актор-критик: ANN, SNN и гибрид. Единый контракт ``forward``."""

from .ann import ActorCritic as AnnActorCritic
from .ann import build_mlp
from .hybrid import ActorCritic as HybridActorCritic
from .snn import ActorCritic as SnnActorCritic
from .snn import build_snn_sequential

__all__ = [
    "AnnActorCritic",
    "SnnActorCritic",
    "HybridActorCritic",
    "build_mlp",
    "build_snn_sequential",
]
