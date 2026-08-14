"""
Адаптеры агентов: единый интерфейс для trainer.

SNN-специфика (reset hidden на done, mu-trace, spike plots) живёт здесь,
а не в цикле обучения. Новый метод: класс адаптера + ``register_agent``.
"""

from abc import ABC, abstractmethod

import torch
import torch.optim as optim

from .logging import log_mu_trace_plot, log_spike_activity_plots
from .models.ann import ActorCritic as AnnActorCritic
from .models.hybrid import ActorCritic as HybridActorCritic
from .models.snn import ActorCritic as SnnActorCritic
from .snn_state import (
    detach_state,
    reset_state_batch_indices,
    stack_rollout_states,
    structural_zeros_like,
)

DEFAULT_HIDDEN_SIZES = [512, 256, 128]


class AgentAdapter(ABC):
    """Базовый адаптер: модель, скрытое состояние, диагностика."""

    name: str = "base"

    def __init__(self, hidden_sizes=None):
        self.hidden_sizes = list(hidden_sizes or DEFAULT_HIDDEN_SIZES)
        self.model = None
        self.optimizer = None

    @abstractmethod
    def build_model(self, num_inputs, num_outputs):
        """Создаёт nn.Module с контрактом ``forward(x, actor_state, critic_state)``."""

    def extra_log_params(self):
        """Дополнительные параметры агента для MLflow."""
        return {"hidden_sizes": self.hidden_sizes}

    def checkpoint_extra(self):
        """Дополнительные поля чекпоинта."""
        return {}

    def build(self, num_inputs, num_outputs, device, lr):
        """Создаёт модель и оптимизатор на устройстве."""
        self.model = self.build_model(num_inputs, num_outputs).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        return self.model, self.optimizer

    def load_checkpoint(self, path, device):
        """Загружает веса и оптимизатор. Возвращает step_idx."""
        ckpt = torch.load(path, map_location=device)
        self.model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt and self.optimizer is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        return int(ckpt.get("step_idx", 0))

    def save_checkpoint(self, path, step_idx):
        """Сохраняет веса, оптимизатор и метаданные."""
        payload = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "hidden_sizes": self.hidden_sizes,
            "step_idx": step_idx,
        }
        payload.update(self.checkpoint_extra())
        torch.save(payload, path)

    def init_hidden(self, num_envs, num_inputs, device):
        """Начальное скрытое состояние (None, если сеть без состояния)."""
        return None

    def on_rollout_start(self, step_idx, use_mlflow):
        """Подготовка к rollout (например, решить, писать ли mu-trace)."""
        pass

    def snapshot(self, hidden):
        """Снимок состояния на вход шага (для PPO replay)."""
        return None

    def act(self, obs, hidden, rollout_step):
        """
        Один шаг политики.

        Returns:
            dist, value, new_hidden
        """
        dist, value, _, _ = self.model(obs, None, None)
        return dist, value, None

    def after_action(self, action, rollout_step):
        """Опциональный захват действия для графиков."""
        pass

    def reset_on_done(self, hidden, done):
        """Обнуляет скрытое состояние для завершённых env (in-place)."""
        pass

    def on_rollout_end(self, step_idx, logger):
        """Логирование диагностики после rollout."""
        pass

    def stack_snapshots(self, snapshots):
        """Склеивает снимки состояний по шагам rollout → (actor_flat, critic_flat)."""
        return None, None

    def value(self, obs, hidden):
        """V(s) для bootstrap GAE."""
        _, next_value, _, _ = self.model(obs, None, None)
        return next_value


class AnnAgent(AgentAdapter):
    """Адаптер полносвязного актор-критика."""

    name = "ann"

    def build_model(self, num_inputs, num_outputs):
        return AnnActorCritic(num_inputs, num_outputs, self.hidden_sizes)


class SnnAgent(AgentAdapter):
    """Адаптер полного SNN: reset hidden, mu-trace и spike-графики."""

    name = "snn"

    def __init__(
        self,
        hidden_sizes=None,
        T=10,
        alpha=0.5,
        lif_v_th=0.4,
        dt=0.01,
        mu_plot_steps=10,
        mu_plot_env=0,
        mu_plot_interval=960,
    ):
        super().__init__(hidden_sizes=hidden_sizes)
        self.T = T
        self.alpha = alpha
        self.lif_v_th = lif_v_th
        self.dt = dt
        self.mu_plot_steps = mu_plot_steps
        self.mu_plot_env = mu_plot_env
        self.mu_plot_interval = mu_plot_interval
        self._trace_this_rollout = False
        self._mu_trace_parts = []
        self._action_parts = []
        self._spike_activity_parts = []

    def extra_log_params(self):
        return {
            "hidden_sizes": self.hidden_sizes,
            "T": self.T,
            "alpha": self.alpha,
            "lif_v_th": self.lif_v_th,
            "coding_type": "ConstantCurrentLIFEncoder + tanh",
            "mu_plot_steps": self.mu_plot_steps,
            "mu_plot_env": self.mu_plot_env,
            "mu_plot_interval": self.mu_plot_interval,
        }

    def build_model(self, num_inputs, num_outputs):
        return SnnActorCritic(
            num_inputs,
            num_outputs,
            self.hidden_sizes,
            T=self.T,
            alpha=self.alpha,
            lif_v_th=self.lif_v_th,
            dt=self.dt,
        )

    def init_hidden(self, num_envs, num_inputs, device):
        with torch.no_grad():
            z = torch.zeros(num_envs, num_inputs, device=device)
            _, _, actor_state, critic_state = self.model(z, None, None)
        return (
            detach_state(structural_zeros_like(actor_state)),
            detach_state(structural_zeros_like(critic_state)),
        )

    def on_rollout_start(self, step_idx, use_mlflow):
        self._trace_this_rollout = bool(use_mlflow) and (
            step_idx % self.mu_plot_interval == 0
        )
        self._mu_trace_parts = []
        self._action_parts = []
        self._spike_activity_parts = []

    def snapshot(self, hidden):
        actor_state, critic_state = hidden
        return detach_state(actor_state), detach_state(critic_state)

    def act(self, obs, hidden, rollout_step):
        actor_state, critic_state = hidden
        capture = self._trace_this_rollout and rollout_step < self.mu_plot_steps
        if capture:
            dist, value, actor_state, critic_state, mu_trace, spike_activity = self.model(
                obs,
                actor_state,
                critic_state,
                return_mu_trace=True,
                return_spike_activity=True,
            )
            self._mu_trace_parts.append(mu_trace[:, self.mu_plot_env, :].detach().cpu())
            self._spike_activity_parts.append(spike_activity)
        else:
            dist, value, actor_state, critic_state = self.model(
                obs, actor_state, critic_state
            )
        actor_state = detach_state(actor_state)
        critic_state = detach_state(critic_state)
        return dist, value, (actor_state, critic_state)

    def after_action(self, action, rollout_step):
        if self._trace_this_rollout and rollout_step < self.mu_plot_steps:
            self._action_parts.append(action[self.mu_plot_env].detach().cpu())

    def reset_on_done(self, hidden, done):
        if hidden is None or not done.any():
            return
        env_ids = torch.where(done)[0]
        actor_state, critic_state = hidden
        reset_state_batch_indices(actor_state, env_ids)
        reset_state_batch_indices(critic_state, env_ids)

    def on_rollout_end(self, step_idx, logger):
        if self._trace_this_rollout and self._mu_trace_parts:
            log_mu_trace_plot(
                self._mu_trace_parts,
                self._action_parts,
                self.mu_plot_env,
                step_idx,
                logger,
            )
        if self._trace_this_rollout and self._spike_activity_parts:
            log_spike_activity_plots(
                self._spike_activity_parts,
                self.hidden_sizes,
                step_idx,
                logger,
            )

    def stack_snapshots(self, snapshots):
        actor_list = [a for a, _ in snapshots]
        critic_list = [c for _, c in snapshots]
        return stack_rollout_states(actor_list), stack_rollout_states(critic_list)

    def value(self, obs, hidden):
        actor_state, critic_state = hidden
        _, next_value, _, _ = self.model(obs, actor_state, critic_state)
        return next_value


class HybridAgent(AgentAdapter):
    """Адаптер гибрида: SNN-актор + ANN-критик (графики актора — вне скоупа)."""

    name = "hybrid"

    def __init__(
        self,
        hidden_sizes=None,
        T=10,
        alpha=0.5,
        lif_v_th=0.4,
        dt=0.01,
    ):
        super().__init__(hidden_sizes=hidden_sizes)
        self.T = T
        self.alpha = alpha
        self.lif_v_th = lif_v_th
        self.dt = dt

    def extra_log_params(self):
        return {
            "hidden_sizes": self.hidden_sizes,
            "T": self.T,
            "alpha": self.alpha,
            "lif_v_th": self.lif_v_th,
            "dt": self.dt,
            "model_type": "snn_actor_ann_critic",
        }

    def checkpoint_extra(self):
        return {"model_type": "snn_actor_ann_critic"}

    def build_model(self, num_inputs, num_outputs):
        return HybridActorCritic(
            num_inputs,
            num_outputs,
            self.hidden_sizes,
            T=self.T,
            alpha=self.alpha,
            lif_v_th=self.lif_v_th,
            dt=self.dt,
        )

    def init_hidden(self, num_envs, num_inputs, device):
        with torch.no_grad():
            z = torch.zeros(num_envs, num_inputs, device=device)
            _, _, actor_state, _ = self.model(z, None, None)
        return detach_state(structural_zeros_like(actor_state))

    def snapshot(self, hidden):
        return detach_state(hidden)

    def act(self, obs, hidden, rollout_step):
        dist, value, actor_state, _ = self.model(obs, hidden, None)
        actor_state = detach_state(actor_state)
        return dist, value, actor_state

    def reset_on_done(self, hidden, done):
        if hidden is None or not done.any():
            return
        env_ids = torch.where(done)[0]
        reset_state_batch_indices(hidden, env_ids)

    def stack_snapshots(self, snapshots):
        return stack_rollout_states(snapshots), None

    def value(self, obs, hidden):
        _, next_value, _, _ = self.model(obs, hidden, None)
        return next_value


AGENT_REGISTRY = {
    "ann": AnnAgent,
    "snn": SnnAgent,
    "hybrid": HybridAgent,
}


def register_agent(name, cls):
    """Регистрирует адаптер в фабрике (новый метод = класс + имя)."""
    AGENT_REGISTRY[name] = cls


def make_agent(name, **kwargs):
    """Создаёт адаптер по имени (``ann`` / ``snn`` / ``hybrid``)."""
    key = name.lower()
    if key not in AGENT_REGISTRY:
        raise ValueError(
            "Неизвестный агент %r. Доступны: %s"
            % (name, ", ".join(sorted(AGENT_REGISTRY)))
        )
    return AGENT_REGISTRY[key](**kwargs)
