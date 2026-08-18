# SNN RL locomotion

Обучение политики ходьбы (velocity tracking) для **Unitree A1** в [Isaac Lab](https://isaac-sim.github.io/IsaacLab/): PPO с тремя архитектурами актор-критика.

- **ANN** — полносвязный актор и критик
- **SNN** — спайковая сеть на [Norse](https://github.com/norse/norse) (LIF, ConstantCurrent-кодирование)
- **hybrid** — SNN-актор + ANN-критик

Среды — расширение `source/locomotion` (gym-задачи flat / rough). Алгоритм PPO, адаптеры агентов и цикл обучения лежат в `scripts/rl/`, конфиги — Hydra в `configs/`. Трекинг — MLflow; подбор гиперпараметров — Optuna (каждый trial в отдельном процессе Isaac Sim).

Команды запуска обучения и тюнинга: **[scripts/README.md](scripts/README.md)**.

## Установка

**Isaac Lab ставится отдельно.** Для этого репозитория используется **тот же venv**, что и у Isaac Lab (Python 3.11). Новый интерпретатор создавать не нужно: в нём уже есть Isaac Sim, `isaaclab` и `isaaclab_tasks`. Сюда же доставляются пакеты проекта.

1. Установите Isaac Lab по [официальной инструкции](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). Репозиторий клонируйте **рядом** с `IsaacLab/`, не внутрь него.

2. Активируйте venv Isaac Lab и в нём поставьте зависимости этого проекта:

   ```bash
   source /path/to/IsaacLab/.venv/bin/activate   # путь к вашему venv Isaac Lab
   pip install -r requirements.txt
   pip install -e source/locomotion
   ```

   В `requirements.txt`: MLflow, Norse, Optuna. Пакет `locomotion` регистрирует gym-среды.

После этого запускайте скрипты тем же `python` из venv Isaac Lab. Подробности CLI, Hydra-overrides, MLflow и Optuna — в [scripts/README.md](scripts/README.md).

---

# Шаблон проектов Isaac Lab

Ниже — заметки из исходного шаблона репозитория (IDE, расширение Omniverse, pre-commit).

## Обзор

Этот репозиторий сделан по шаблону внешних проектов Isaac Lab: разработка идёт **вне** основного дерева Isaac Lab, код можно подключать как расширение Omniverse.

**Особенности:**

- `Isolation` — работа вне ядра Isaac Lab, изменения остаются в этом репозитории.
- `Flexibility` — проект можно включить как расширение Omniverse.

**Ключевые слова:** extension, template, isaaclab

## Установка (шаблон)

- Установите Isaac Lab по [инструкции](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  Удобнее conda или uv: проще вызывать Python-скрипты из терминала.

- Клонируйте или скопируйте этот репозиторий отдельно от установки Isaac Lab (то есть **вне** каталога `IsaacLab`).

- Тем интерпретатором, в котором уже стоит Isaac Lab, поставьте библиотеку в editable-режиме:

    ```bash
    # вместо python используйте 'PATH_TO_isaaclab.sh|bat -p', если Isaac Lab не в вашем venv/conda
    python -m pip install -e source/locomotion
    ```

- Проверьте, что расширение установлено:

    - Список задач:

        Если имя задачи изменилось, может понадобиться обновить шаблон поиска `"Template-"`
        (в файле `scripts/old/list_envs.py`), чтобы она попала в список.

        ```bash
        # вместо python — 'FULL_PATH_TO_isaaclab.sh|bat -p', если Isaac Lab не в venv/conda
        python scripts/old/list_envs.py
        ```

    - Запуск задачи:

        ```bash
        # вместо python — 'FULL_PATH_TO_isaaclab.sh|bat -p', если Isaac Lab не в venv/conda
        python scripts/train.py --task=<TASK_NAME>
        ```

    - Запуск с dummy-агентами:

        Агенты с нулевым или случайным действием. Нужны, чтобы проверить, что среды собраны правильно.

        - Агент с нулевым действием

            ```bash
            # вместо python — 'FULL_PATH_TO_isaaclab.sh|bat -p', если Isaac Lab не в venv/conda
            python scripts/old/zero_agent.py --task=<TASK_NAME>
            ```
        - Агент со случайным действием

            ```bash
            # вместо python — 'FULL_PATH_TO_isaaclab.sh|bat -p', если Isaac Lab не в venv/conda
            python scripts/old/random_agent.py --task=<TASK_NAME>
            ```

### Настройка IDE (необязательно)

Чтобы настроить IDE:

- Запустите VSCode Tasks: `Ctrl+Shift+P` → `Tasks: Run Task` → в списке выберите `setup_python_env`.
  Задача спросит абсолютный путь к установке Isaac Sim.

Если всё прошло успешно, в каталоге `.vscode` появится файл `.python.env`.
В нём — python-пути ко всем расширениям Isaac Sim и Omniverse.
Это нужно для индексирования модулей и подсказок в редакторе.

### Подключение как расширение Omniverse (необязательно)

Пример UI-расширения, которое загружается при включении вашего расширения, лежит в `source/locomotion/locomotion/ui_extension_example.py`.

Чтобы включить расширение:

1. **Добавьте путь поиска этого репозитория** в менеджер расширений:
    - Откройте менеджер: `Window` → `Extensions`.
    - Нажмите **иконку-гамбургер**, затем `Settings`.
    - В `Extension Search Paths` укажите абсолютный путь к каталогу `source` этого репозитория.
    - Если его ещё нет, в те же `Extension Search Paths` добавьте путь к расширениям Isaac Lab (`IsaacLab/source`).
    - Снова **гамбургер** → `Refresh`.

2. **Найдите и включите расширение**:
    - Оно в категории `Third Party`.
    - Включите переключателем.

## Форматирование кода

В репозитории есть шаблон pre-commit для автоформатирования.
Установка:

```bash
pip install pre-commit
```

Запуск:

```bash
pre-commit run --all-files
```

## Устранение неполадок

### Pylance не индексирует расширения

В части версий VS Code индексирование расширений неполное.
Добавьте путь к расширению в `.vscode/settings.json` в ключ `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/locomotion"
    ]
}
```

### Падение Pylance

Если `pylance` падает, скорее всего проиндексировано слишком много файлов и не хватает памяти.
Можно исключить неиспользуемые пакеты Omniverse: в `.vscode/settings.json` закомментируйте лишние пути в `"python.analysis.extraPaths"`.
Примеры того, что обычно можно убрать:

```
"<path-to-isaac-sim>/extscache/omni.anim.*"         // пакеты анимации
"<path-to-isaac-sim>/extscache/omni.kit.*"          // UI-инструменты Kit
"<path-to-isaac-sim>/extscache/omni.graph.*"        // UI-инструменты Graph
"<path-to-isaac-sim>/extscache/omni.services.*"     // сервисы
...
```
