"""Пакет обучения PPO: модели, адаптеры агентов, trainer и логирование."""

__all__ = [
    "make_agent",
    "register_agent",
    "PPOTrainer",
    "TrainConfig",
]

# Ленивый импорт: ``optuna_tune.py`` берёт только ``rl.optuna_driver``
# и не должен тянуть torch / norse / gymnasium.
_EXPORTS = {
    "make_agent": (".agents", "make_agent"),
    "register_agent": (".agents", "register_agent"),
    "PPOTrainer": (".trainer", "PPOTrainer"),
    "TrainConfig": (".trainer", "TrainConfig"),
}


def __getattr__(name):
    """Подгружает адаптеры и trainer только при обращении к имени."""
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    from importlib import import_module

    module_name, attr = target
    value = getattr(import_module(module_name, __name__), attr)
    globals()[name] = value
    return value


def __dir__():
    """Список имён модуля, включая ленивые экспорты из ``__all__``."""
    return sorted(set(globals().keys()) | set(__all__))
