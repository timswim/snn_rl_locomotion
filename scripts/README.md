# Scripts

Обучение PPO для ANN, полного SNN и гибрида (SNN-актор + ANN-критик) идёт через один скрипт и Hydra-конфиги. Тип агента выбирается YAML-ом, а не отдельным `train_*.py`.

Запускайте из каталога `scripts/` — пакет `rl` лежит рядом со скриптами. Из корня репозитория добавьте `PYTHONPATH=scripts`.

Каталог `scripts/old/` — архив шаблона Isaac Lab (skrl, dummy-агенты). Его не используем для обучения.

## train.py — один прогон PPO

Isaac Sim стартует до Hydra: флаги AppLauncher (`--headless`, `--device`, …) отделяются от Hydra-overrides.

```bash
cd scripts

# ANN (дефолт configs/train.yaml → agent/ann.yaml)
python train.py --headless

# полный SNN
python train.py agent=snn --headless

# гибрид: SNN-актор + ANN-критик
python train.py agent=hybrid --headless
```

Из корня репозитория:

```bash
PYTHONPATH=scripts python scripts/train.py agent=snn --headless
```

Если Isaac Lab не в том же venv, вызывайте через `PATH_TO_isaaclab.sh -p` вместо `python`.

### Hydra: композиция и CLI-overrides

`configs/train.yaml` задаёт задачу, PPO, MLflow, видео и чекпоинты; `defaults: - agent: ann` подмешивает `configs/agent/ann.yaml`. Смена агента — `agent=snn` или `agent=hybrid` (подхватывается `configs/agent/<имя>.yaml`). Новый агент = новый YAML, без правок парсера.

Вложенные ключи переопределяются из CLI без argparse на каждый гиперпараметр:

```bash
python train.py agent=snn ppo.lr=1e-4 agent.T=5 ppo.max_steps=10000 --headless
```

Полезные ключи: `task`, `num_envs`, `seed`, `ppo.lr`, `ppo.num_steps`, `ppo.mini_batch_size`, `ppo.ppo_epochs`, `ppo.clip_param`, `ppo.max_steps`, `agent.hidden_sizes`, для SNN/hybrid — `agent.T`, `agent.alpha`, `agent.lif_v_th`.

### MLflow

В YAML `use_mlflow: true`. Параметры, метрики (награда, лоссы) и для полного SNN — графики `mu_trace` / spike activity. TensorBoard не используется.

Выключить трекинг (остаётся print в консоль):

```bash
python train.py agent=ann use_mlflow=false --headless
```

Каталог `mlruns/` создаётся относительно текущей рабочей директории (при запуске из `scripts/` — `scripts/mlruns`). UI:

```bash
cd scripts && mlflow ui
```

Или задайте `MLFLOW_TRACKING_URI=file:///path/to/mlruns`.

### Чекпоинты и видео

Сохранение выключено по умолчанию (`save_checkpoints: false`). Включить:

```bash
python train.py agent=ann save_checkpoints=true checkpoint_interval=5000 --headless
```

Продолжить с файла: `checkpoint_path=/path/to/agent_5000.pth`.

Видео: `video=true` (камеры включаются до старта симулятора, поэтому `video=true` должен быть в CLI-overrides).

## optuna_tune.py — тюнинг гиперпараметров

Драйвер **не** запускает Isaac Sim. Каждый trial — subprocess `train.py` (новая сессия симулятора). Search space берётся из `configs/optuna/<agent>.yaml`; trial передаёт те же Hydra-overrides, что и ручной запуск. `hydra-optuna-sweeper` не используется.

```bash
cd scripts
python optuna_tune.py --agent ann --n_trials=20
python optuna_tune.py --agent snn --n_trials=20
python optuna_tune.py --agent hybrid --n_trials=20 --max_steps=10000 --use_mlflow
python optuna_tune.py --agent snn --n_trials=5 ppo.gamma=0.99
```

Аргументы: `--agent`, `--task`, `--n_trials`, `--seed`, `--max_steps` (алиас `--max_iterations`), `--num_envs`, `--use_mlflow`, `--study_name`, плюс произвольные Hydra-overrides. По умолчанию MLflow в trial выключен; `--use_mlflow` ставит `use_mlflow=true`.

Метрика trial — среднее вознаграждение из файла `OPTUNA_METRIC_FILE` (его пишет `train.py`). При падении subprocess study продолжается (сентинел `-1e9`).

При `--use_mlflow` драйвер задаёт `MLFLOW_TRACKING_URI` на **корень репозитория** (`../mlruns` относительно `scripts/`). UI тогда из корня:

```bash
cd /path/to/SNN_RL_locomotion && mlflow ui
```

Юнит-тесты драйвера (без симулятора):

```bash
cd scripts && python test_optuna_driver.py
```

## Новый метод агента

Цикл обучения и Optuna не ветвятся по `if agent == "snn"`. Контракт модели:

```python
forward(x, actor_state=None, critic_state=None, **kwargs)
# -> dist, value, actor_state, critic_state
```

ANN возвращает `None, None`. Hybrid — состояние только актора. Reset hidden, mu-trace и spike-графики живут в адаптере, не в `train.py`.

1. Класс в `scripts/rl/models/`
2. Адаптер + `register_agent("my_snn", MyAgent)` в `scripts/rl/agents.py`
3. YAML `configs/agent/my_snn.yaml` (`name: my_snn` и поля конструктора)
4. При тюнинге — `configs/optuna/my_snn.yaml`

Запуск: `python train.py agent=my_snn`.

## Зависимости

MLflow и Optuna указаны в `install_requires` пакета locomotion. Hydra/OmegaConf обычно уже есть в окружении Isaac Lab.

```bash
pip install -e source/locomotion
```
