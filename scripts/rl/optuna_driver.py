"""
Драйвер Optuna: search space из YAML, trial в subprocess ``train.py``.

Isaac Sim здесь не запускается. Метрика читается из файла
``OPTUNA_METRIC_FILE`` (его пишет ``scripts/train.py``).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

BAD_METRIC_SENTINEL = -1e9

_RESERVED_OVERRIDE_KEYS = frozenset(
    {
        "agent",
        "task",
        "seed",
        "run_name",
        "use_mlflow",
        "ppo.max_steps",
        "num_envs",
    }
)


def load_optuna_config(path: str | Path) -> dict[str, Any]:
    """Читает YAML с полями ``search_space`` и опционально ``agent``, ``max_steps``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Нет файла search space Optuna: %s" % path)

    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "Для Optuna YAML нужен PyYAML (ставится вместе с Hydra)."
        ) from exc

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise TypeError("%s: ожидался словарь, получено %r" % (path, type(data)))
    search_space = data.get("search_space")
    if not isinstance(search_space, dict) or not search_space:
        raise ValueError("%s: нужно непустое поле search_space" % path)
    reserved = sorted(k for k in search_space if k in _RESERVED_OVERRIDE_KEYS)
    if reserved:
        raise ValueError(
            "%s: ключи search_space пересекаются с фиксированными overrides: %s"
            % (path, ", ".join(reserved))
        )
    data["search_space"] = search_space
    return data


def suggest_one(trial, name: str, spec: dict[str, Any]) -> Any:
    """Вызывает ``trial.suggest_*`` по спецификации из YAML."""
    if not isinstance(spec, dict) or "type" not in spec:
        raise ValueError("search_space[%r]: нужна спецификация с полем type" % name)

    kind = str(spec["type"]).lower()
    if kind in ("float", "suggest_float"):
        if "low" not in spec or "high" not in spec:
            raise ValueError("search_space[%r]: для float нужны low и high" % name)
        kwargs: dict[str, Any] = {}
        if spec.get("log"):
            kwargs["log"] = True
        if "step" in spec:
            kwargs["step"] = spec["step"]
        return trial.suggest_float(name, spec["low"], spec["high"], **kwargs)

    if kind in ("int", "suggest_int"):
        if "low" not in spec or "high" not in spec:
            raise ValueError("search_space[%r]: для int нужны low и high" % name)
        kwargs = {}
        if spec.get("log"):
            kwargs["log"] = True
        if "step" in spec:
            kwargs["step"] = spec["step"]
        return trial.suggest_int(name, spec["low"], spec["high"], **kwargs)

    if kind in ("categorical", "suggest_categorical"):
        choices = spec.get("choices")
        if not choices:
            raise ValueError("search_space[%r]: для categorical нужны choices" % name)
        return trial.suggest_categorical(name, list(choices))

    raise ValueError(
        "search_space[%r]: неизвестный type=%r (float / int / categorical)"
        % (name, spec["type"])
    )


def suggest_params(trial, search_space: dict[str, Any]) -> dict[str, Any]:
    """Сэмплирует все параметры search space. Ключи — Hydra-overrides."""
    return {name: suggest_one(trial, name, spec) for name, spec in search_space.items()}


def _format_override(key: str, value: Any) -> str:
    """Собирает один Hydra-override ``key=value``."""
    if hasattr(value, "item") and not isinstance(value, (bytes, str)):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, bool):
        rendered = "true" if value else "false"
    else:
        rendered = str(value)
    if any(ch in rendered for ch in ' \t"'):
        rendered = '"%s"' % rendered.replace('"', '\\"')
    return "%s=%s" % (key, rendered)


def build_train_argv(
    python_exe: str,
    train_script: str,
    agent: str,
    trial_params: dict[str, Any],
    *,
    task: str,
    seed: int,
    run_name: str,
    max_steps: int | None,
    num_envs: int | None,
    use_mlflow: bool,
    headless: bool = True,
    extra_overrides: list[str] | None = None,
) -> list[str]:
    """Аргументы subprocess: AppLauncher (``--headless``) + Hydra-overrides."""
    argv = [python_exe, train_script, _format_override("agent", agent)]
    if headless:
        argv.append("--headless")
    argv.append(_format_override("task", task))
    argv.append(_format_override("seed", seed))
    argv.append(_format_override("run_name", run_name))
    if max_steps is not None:
        argv.append(_format_override("ppo.max_steps", max_steps))
    if num_envs is not None:
        argv.append(_format_override("num_envs", num_envs))
    argv.append(_format_override("use_mlflow", use_mlflow))
    if extra_overrides:
        argv.extend(extra_overrides)
    for key, value in trial_params.items():
        argv.append(_format_override(key, value))
    return argv


def _read_metric(metric_file: str) -> float | None:
    """Читает mean_reward из файла trial; ``None`` если файла нет или он пустой."""
    if not os.path.isfile(metric_file):
        return None
    try:
        with open(metric_file, encoding="utf-8") as f:
            line = f.read().strip()
        if line:
            return float(line)
    except (ValueError, OSError):
        return None
    return None


def run_trial(
    trial_params: dict[str, Any],
    *,
    agent: str,
    task: str,
    seed: int,
    run_name: str,
    max_steps: int | None,
    num_envs: int | None,
    use_mlflow: bool,
    script_dir: str,
    python_exe: str,
    train_script: str | None = None,
    headless: bool = True,
    extra_overrides: list[str] | None = None,
) -> float:
    """
    Запускает ``train.py`` в отдельном процессе; stdout/stderr идут в терминал.

    Метрика — из ``OPTUNA_METRIC_FILE``. При падении trial возвращает
    ``BAD_METRIC_SENTINEL``, study продолжается.
    """
    if train_script is None:
        train_script = os.path.join(script_dir, "train.py")
    if not os.path.isfile(train_script):
        raise FileNotFoundError("Нет скрипта обучения: %s" % train_script)

    argv = build_train_argv(
        python_exe,
        train_script,
        agent,
        trial_params,
        task=task,
        seed=seed,
        run_name=run_name,
        max_steps=max_steps,
        num_envs=num_envs,
        use_mlflow=use_mlflow,
        headless=headless,
        extra_overrides=extra_overrides,
    )
    print("[optuna_tune] %s" % " ".join(argv))

    fd, metric_file = tempfile.mkstemp(suffix=".optuna_metric", prefix="trial_")
    os.close(fd)
    try:
        env = os.environ.copy()
        env["OPTUNA_METRIC_FILE"] = metric_file
        env["PYTHONUNBUFFERED"] = "1"
        mlruns_dir = os.path.abspath(os.path.join(script_dir, "..", "mlruns"))
        env["MLFLOW_TRACKING_URI"] = "file://" + mlruns_dir
        result = subprocess.run(
            argv,
            cwd=script_dir,
            env=env,
            timeout=None,
        )
        metric = _read_metric(metric_file)
        if metric is not None:
            return metric
        if result.returncode != 0:
            print(
                "[optuna_tune] Trial subprocess exited with code %d, returning bad metric %.2f"
                % (result.returncode, BAD_METRIC_SENTINEL),
                file=sys.stderr,
            )
        else:
            print(
                "[optuna_tune] OPTUNA_METRIC_FILE missing or empty, returning bad metric %.2f"
                % BAD_METRIC_SENTINEL,
                file=sys.stderr,
            )
        return BAD_METRIC_SENTINEL
    finally:
        if os.path.isfile(metric_file):
            os.remove(metric_file)


def run_study(
    *,
    agent: str,
    search_space: dict[str, Any],
    n_trials: int,
    task: str,
    seed: int,
    max_steps: int | None,
    num_envs: int | None,
    use_mlflow: bool,
    study_name: str | None,
    script_dir: str,
    python_exe: str,
    train_script: str | None = None,
    headless: bool = True,
    extra_overrides: list[str] | None = None,
):
    """Создаёт study (maximize mean_reward) и запускает ``n_trials`` subprocess-trial."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.INFO)

    def objective(trial):
        trial_params = suggest_params(trial, search_space)
        return run_trial(
            trial_params,
            agent=agent,
            task=task,
            seed=seed + trial.number,
            run_name="trial_%d" % trial.number,
            max_steps=max_steps,
            num_envs=num_envs,
            use_mlflow=use_mlflow,
            script_dir=script_dir,
            python_exe=python_exe,
            train_script=train_script,
            headless=headless,
            extra_overrides=extra_overrides,
        )

    study = optuna.create_study(direction="maximize", study_name=study_name)
    study.optimize(objective, n_trials=n_trials)
    print("Best value: %.6f" % study.best_value)
    print("Best params: %s" % study.best_params)
    return study
