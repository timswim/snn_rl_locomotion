"""
Optuna hyperparameter search for train_ann.py — one trial per process.

Runs each Optuna trial in a separate subprocess (separate Isaac Sim session).
Stdout/stderr are not captured so all prints and MLflow logs go to your terminal.
Metric is passed via file (OPTUNA_METRIC_FILE). Run from the scripts/ directory:

    cd scripts && python optuna_tune_ann.py --task=Isaac-Velocity-Flat-Unitree-A1-v0 --n_trials=20
"""

import argparse
import os
import subprocess
import sys
import tempfile


def parse_args():
    p = argparse.ArgumentParser(
        description="Optuna tuning for train_ann.py (one trial per subprocess)."
    )
    p.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Unitree-A1-v0", help="Task name.")
    p.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials.")
    p.add_argument("--seed", type=int, default=1, help="Base seed (trial seed = seed + trial.number).")
    p.add_argument("--max_iterations", type=int, default=50000, help="Max training steps per trial.")
    p.add_argument("--num_envs", type=int, default=None, help="Number of envs (default from train_ann).")
    p.add_argument("--use_mlflow", action="store_true", help="Pass --use_mlflow to each trial.")
    p.add_argument("--study_name", type=str, default=None, help="Optuna study name for storage.")
    return p.parse_args()


def run_trial(
    trial_params: dict,
    task: str,
    seed: int,
    run_name: str,
    max_iterations: int | None,
    num_envs: int | None,
    use_mlflow: bool,
    script_dir: str,
    python_exe: str,
) -> float:
    """Run train_ann.py in a subprocess; stdout/stderr go to terminal. Metric is read from file."""
    argv = [
        python_exe,
        os.path.join(script_dir, "train_ann.py"),
        "--task", task,
        "--seed", str(seed),
        "--run_name", run_name,
        "--lr", str(trial_params["lr"]),
        "--num_steps", str(trial_params["num_steps"]),
        "--mini_batch_size", str(trial_params["mini_batch_size"]),
        "--ppo_epochs", str(trial_params["ppo_epochs"]),
        "--clip_param", str(trial_params["clip_param"]),
        "--headless", # Запускаем в headless режиме
    ]
    if max_iterations is not None:
        argv.extend(["--max_iterations", str(max_iterations)])
    if num_envs is not None:
        argv.extend(["--num_envs", str(num_envs)])
    if use_mlflow:
        argv.append("--use_mlflow")

    fd, metric_file = tempfile.mkstemp(suffix=".optuna_metric", prefix="trial_")
    os.close(fd)
    try:
        env = os.environ.copy()
        env["OPTUNA_METRIC_FILE"] = metric_file
        # Same MLflow store as when running mlflow ui from project root (subprocess cwd is script_dir)
        mlruns_dir = os.path.abspath(os.path.join(script_dir, "..", "mlruns"))
        env["MLFLOW_TRACKING_URI"] = "file://" + mlruns_dir
        result = subprocess.run(
            argv,
            cwd=script_dir,
            env=env,
            timeout=None,
        )
        # On crash, train_ann may write BAD_METRIC_SENTINEL to the file; still return it so Optuna continues
        bad_metric = -1e9
        if os.path.isfile(metric_file):
            try:
                with open(metric_file, "r") as f:
                    line = f.read().strip()
                if line:
                    return float(line)
            except (ValueError, OSError):
                pass
        if result.returncode != 0:
            print("[optuna_tune_ann] Trial subprocess exited with code %d, returning bad metric %.2f" % (result.returncode, bad_metric), file=sys.stderr)
        else:
            print("[optuna_tune_ann] OPTUNA_METRIC_FILE missing or empty, returning bad metric %.2f" % bad_metric, file=sys.stderr)
        return bad_metric
    finally:
        if os.path.isfile(metric_file):
            os.remove(metric_file)


def main():
    args = parse_args()
    import optuna
    optuna.logging.set_verbosity(optuna.logging.INFO)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    python_exe = sys.executable

    def objective(trial):
        trial_params = {
            "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            "num_steps": trial.suggest_int("num_steps", 16, 64),
            "mini_batch_size": trial.suggest_int("mini_batch_size", 64, 512),
            "ppo_epochs": trial.suggest_int("ppo_epochs", 3, 15),
            "clip_param": trial.suggest_float("clip_param", 0.1, 0.3),
        }
        seed = args.seed + trial.number
        return run_trial(
            trial_params=trial_params,
            task=args.task,
            seed=seed,
            run_name="trial_%d" % trial.number,
            max_iterations=args.max_iterations,
            num_envs=args.num_envs,
            use_mlflow=args.use_mlflow,
            script_dir=script_dir,
            python_exe=python_exe,
        )

    study = optuna.create_study(direction="maximize", study_name=args.study_name)
    study.optimize(objective, n_trials=args.n_trials)
    print("Best value: %.6f" % study.best_value)
    print("Best params: %s" % study.best_params)
    return study


if __name__ == "__main__":
    main()
