"""
Optuna: один trial — один процесс ``train.py`` (отдельная сессия Isaac Sim).

Драйвер сам симулятор не запускает. Search space берётся из
``configs/optuna/<agent>.yaml``; trial передаёт Hydra-overrides в CLI.
Метрика — файл ``OPTUNA_METRIC_FILE``. Из каталога ``scripts/``:

    python optuna_tune.py --agent ann --n_trials=20
    python optuna_tune.py --agent snn --n_trials=20
    python optuna_tune.py --agent hybrid --n_trials=20 --max_steps=10000 --use_mlflow
    python optuna_tune.py --agent snn --n_trials=5 ppo.gamma=0.99
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from rl.optuna_driver import load_optuna_config, run_study  # noqa: E402


def _available_optuna_agents() -> list[str]:
    """Имена YAML в ``configs/optuna/`` (без расширения)."""
    optuna_dir = _REPO_ROOT / "configs" / "optuna"
    if not optuna_dir.is_dir():
        return []
    return sorted(p.stem for p in optuna_dir.glob("*.yaml"))


def parse_args():
    """CLI драйвера (не Hydra): агент, число trial и общие overrides."""
    p = argparse.ArgumentParser(
        description="Optuna-тюнинг train.py: один trial = один subprocess."
    )
    p.add_argument(
        "--agent",
        type=str,
        default="ann",
        help="Имя агента: configs/optuna/<agent>.yaml и Hydra agent=<agent>.",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Путь к YAML search space (по умолчанию configs/optuna/<agent>.yaml).",
    )
    p.add_argument(
        "--task",
        type=str,
        default="Isaac-Velocity-Flat-Unitree-A1-v0",
        help="Имя gym-задачи.",
    )
    p.add_argument("--n_trials", type=int, default=20, help="Число trial Optuna.")
    p.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Базовый seed (seed trial = seed + trial.number).",
    )
    p.add_argument(
        "--max_steps",
        "--max_iterations",
        dest="max_steps",
        type=int,
        default=None,
        help="ppo.max_steps на trial (иначе значение из YAML search space).",
    )
    p.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="Число сред (иначе дефолт configs/train.yaml).",
    )
    p.add_argument(
        "--use_mlflow",
        action="store_true",
        help="Включить MLflow в каждом trial (иначе use_mlflow=false).",
    )
    p.add_argument("--study_name", type=str, default=None, help="Имя Optuna study.")
    p.add_argument(
        "overrides",
        nargs="*",
        metavar="OVERRIDE",
        help="Доп. Hydra-overrides на каждый trial, например ppo.gamma=0.99.",
    )
    return p.parse_args()


def _resolve_config_path(args) -> Path:
    """Путь к YAML search space; при отсутствии файла перечисляет доступных агентов."""
    agent = str(args.agent)
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = (_REPO_ROOT / config_path).resolve()
    else:
        config_path = _REPO_ROOT / "configs" / "optuna" / ("%s.yaml" % agent)

    if not config_path.is_file():
        available = ", ".join(_available_optuna_agents()) or "(нет yaml)"
        raise FileNotFoundError(
            "Нет search space Optuna: %s. Доступные --agent: %s" % (config_path, available)
        )
    return config_path


def main():
    """Загружает YAML search space и оптимизирует mean_reward."""
    args = parse_args()
    agent = str(args.agent)
    config_path = _resolve_config_path(args)

    optuna_cfg = load_optuna_config(config_path)
    yaml_agent = optuna_cfg.get("agent")
    if yaml_agent is not None and str(yaml_agent) != agent:
        raise ValueError(
            "%s указывает agent=%s, CLI --agent=%s"
            % (config_path, yaml_agent, agent)
        )

    max_steps = args.max_steps
    if max_steps is None and optuna_cfg.get("max_steps") is not None:
        max_steps = int(optuna_cfg["max_steps"])

    print("[optuna_tune] search space: %s" % config_path)
    return run_study(
        agent=agent,
        search_space=optuna_cfg["search_space"],
        n_trials=args.n_trials,
        task=args.task,
        seed=args.seed,
        max_steps=max_steps,
        num_envs=args.num_envs,
        use_mlflow=bool(args.use_mlflow),
        study_name=args.study_name,
        script_dir=str(_SCRIPTS_DIR),
        python_exe=sys.executable,
        extra_overrides=list(args.overrides),
    )


if __name__ == "__main__":
    main()
