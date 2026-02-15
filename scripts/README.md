# Scripts

## train_ann.py — ANN RL training with optional MLFlow and Optuna

Trains a PPO agent (custom ANN) with torch. Hyperparameters can be overridden from the CLI. Optional local MLFlow logging and Optuna hyperparameter search are supported.

**Run from the `scripts` directory** (so that `models.PPO` resolves), or from the project root with `scripts` on `PYTHONPATH`:

```bash
cd scripts && python train_ann.py --task=...
# or from repo root: PYTHONPATH=scripts python scripts/train_ann.py --task=...
```

### Single training run

From the `scripts` directory:

```bash
# use 'PATH_TO_isaaclab.sh -p' instead of 'python' if Isaac Lab is not in your venv
python train_ann.py --task=Isaac-Velocity-Flat-Unitree-A1-v0
```

Override hyperparameters:

```bash
python train_ann.py --task=Isaac-Velocity-Flat-Unitree-A1-v0 --lr=5e-5 --ppo_epochs=10 --max_steps=50000 --hidden_sizes 256 128
```

### MLFlow (local tracking)

Use `--use_mlflow` to log params, metrics, and artifacts to a local MLFlow store under `./mlruns` (relative to the current working directory).

```bash
python train_ann.py --task=Isaac-Velocity-Flat-Unitree-A1-v0 --use_mlflow
```

View runs in the MLFlow UI:

```bash
mlflow ui
```

Use the same tracking URI (default is `./mlruns` in the directory where you run `mlflow ui`), or set `MLFLOW_TRACKING_URI=file:///path/to/your/mlruns` if needed.

### Optuna hyperparameter search

Use `--optuna` to run an Optuna study (multiple trials in the same process, one Isaac Sim session). Each trial gets its own log directory and, if `--use_mlflow` is set, its own MLFlow run.

```bash
python train_ann.py --task=Isaac-Velocity-Flat-Unitree-A1-v0 --optuna --optuna_n_trials=20
```

With MLFlow:

```bash
python train_ann.py --task=Isaac-Velocity-Flat-Unitree-A1-v0 --optuna --optuna_n_trials=20 --use_mlflow
```

Trials reuse the same Isaac Sim process; `env.close()` is called after each trial. If you see GPU memory growth over many trials, consider a subprocess-per-trial workflow (e.g. a driver script that invokes this script once per trial with different CLI args and reads the objective from stdout or a file).

### Dependencies

MLFlow and Optuna are listed in the locomotion package `install_requires`. Install the package in editable mode:

```bash
pip install -e source/locomotion
```
