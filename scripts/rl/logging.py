"""Логирование обучения: MLflow (без TensorBoard) и диагностические графики SNN."""

import os

import numpy as np
import torch


class MLflowLogger:
    """
    Обёртка над MLflow. При ``enabled=False`` все вызовы — no-op
    (в консоль пишет сам trainer).
    """

    def __init__(
        self,
        enabled,
        log_dir,
        run_name=None,
        experiment_name=None,
        params=None,
    ):
        """
        Параметры:
            enabled: если False, все вызовы — no-op.
            log_dir: каталог прогона; в конце кладётся в артефакты MLflow.
            run_name: имя run; ``None`` — имя по умолчанию MLflow.
            experiment_name: имя experiment; ``None`` — текущий experiment.
            params: словарь параметров для ``log_params``.
        """
        self.enabled = bool(enabled)
        self.log_dir = log_dir
        self._mlflow = None
        if not self.enabled:
            return
        import mlflow

        self._mlflow = mlflow
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or (
            "file://" + os.path.abspath("mlruns")
        )
        mlflow.set_tracking_uri(tracking_uri)
        if experiment_name is not None:
            mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name)
        if params:
            mlflow.log_params(
                {k: str(v) if isinstance(v, list) else v for k, v in params.items()}
            )

    def log_metric(self, key, value, step=None):
        """Пишет одну скалярную метрику."""
        if not self.enabled:
            return
        self._mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics, step=None):
        """Пишет набор скалярных метрик."""
        if not self.enabled:
            return
        self._mlflow.log_metrics(metrics, step=step)

    def log_figure(self, fig, artifact_file):
        """Сохраняет matplotlib-фигуру как артефакт."""
        if not self.enabled:
            return
        self._mlflow.log_figure(fig, artifact_file)

    def finish(self, mean_reward):
        """Закрывает run: итоговая метрика и артефакты каталога прогона."""
        if not self.enabled:
            return
        self._mlflow.log_metric("mean_reward", mean_reward)
        self._mlflow.log_artifacts(self.log_dir, artifact_path="run")
        self._mlflow.end_run()


def log_mu_trace_plot(mu_trace_parts, action_parts, env_idx, global_step, logger):
    """
    Строит график mu по внутренним шагам SNN (num_env_steps * T точек) и логирует в MLflow.

    mu_trace_parts: список тензоров формы (T, num_outputs) для одной среды.
    action_parts: список тензоров формы (num_outputs,) — сэмплированные действия на каждом env step.
    """
    if not mu_trace_parts or logger is None or not logger.enabled:
        return
    import matplotlib.pyplot as plt

    series = torch.cat(mu_trace_parts, dim=0).numpy()
    num_points = series.shape[0]
    mu_min, mu_max = float(series.min()), float(series.max())
    mu_abs_max = float(np.abs(series).max())

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)

    ax = axes[0]
    for dim in range(series.shape[1]):
        ax.plot(range(num_points), series[:, dim], label="mu[%d]" % dim, alpha=0.8)
    ax.set_xlabel("Internal timestep (env_step * T + t)")
    ax.set_ylabel("mu (actor mean)")
    ax.set_title(
        "Actor mu trace (env %d, %d points, min=%.2e, max=%.2e)"
        % (env_idx, num_points, mu_min, mu_max)
    )
    if mu_abs_max > 0:
        ax.legend(loc="upper right", ncol=min(4, series.shape[1]), fontsize=8)
    else:
        ax.text(
            0.5,
            0.5,
            "mu ≈ 0: check LIF v_th and readout path",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    actions = None
    if action_parts:
        actions = torch.stack(action_parts, dim=0).numpy()
        env_step_x = np.arange(actions.shape[0])
        for dim in range(actions.shape[1]):
            ax.plot(env_step_x, actions[:, dim], label="action[%d]" % dim, alpha=0.8)
        ax.set_xlabel("Env step (within trace window)")
        ax.set_ylabel("sampled action")
        ax.set_title(
            "Sampled actions (env %d, std=%.3f)" % (env_idx, float(np.std(actions)))
        )
        ax.legend(loc="upper right", ncol=min(4, actions.shape[1]), fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    plot_name = "mu_trace_env%d_step%d.png" % (env_idx, global_step)
    logger.log_figure(fig, "mu_trace/%s" % plot_name)
    logger.log_metrics(
        {
            "Diagnostics/mu_abs_max": mu_abs_max,
            "Diagnostics/mu_mean": float(series.mean()),
            "Diagnostics/action_abs_mean": float(np.abs(actions).mean()) if action_parts else 0.0,
        },
        step=global_step,
    )
    plt.close(fig)


def log_spike_activity_plots(spike_activity_parts, hidden_sizes, global_step, logger):
    """
    Строит по одному графику на каждый LIF-слой актора: доля спайкующих нейронов (%)
    усреднённая по T микрошагам SNN, по env steps в окне трассировки.

    spike_activity_parts: список длины num_env_steps; каждый элемент — список из L float (%).
    """
    if not spike_activity_parts or logger is None or not logger.enabled:
        return
    import matplotlib.pyplot as plt

    activity = np.asarray(spike_activity_parts, dtype=np.float64)
    num_env_steps = activity.shape[0]
    env_step_x = np.arange(num_env_steps)

    for layer_idx, hidden_size in enumerate(hidden_sizes):
        layer_series = activity[:, layer_idx]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(env_step_x, layer_series, color="C%d" % layer_idx, linewidth=1.5)
        ax.set_xlabel("Env step (within trace window)")
        ax.set_ylabel("Spiking neurons (%)")
        ax.set_ylim(0, 100)
        ax.set_title(
            "Actor LIF layer %d (%d neurons): mean spike fraction over T micro-steps"
            % (layer_idx + 1, hidden_size)
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        plot_name = "spike_activity_layer%d_step%d.png" % (layer_idx + 1, global_step)
        logger.log_figure(fig, "spike_activity/%s" % plot_name)
        logger.log_metric(
            "SpikeActivity/actor_layer%d_pct" % (layer_idx + 1),
            float(layer_series.mean()),
            step=global_step,
        )
        plt.close(fig)
