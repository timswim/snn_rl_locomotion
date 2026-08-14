"""
Утилиты для скрытого состояния Norse (дерево list / named tuple / тензор).

BPTT / detach:
- Внутри одного forward() градиенты идут по микрошагам t=0..T-1 (динамика мембраны).
- Между шагами среды скрытое состояние отсоединяем (detach), чтобы граф не тянулся
  на всю длину rollout (экономия памяти). Градиенты по параметрам считаются по текущему
  наблюдению и отсоединённым потенциалам на входе (нет dL/dv между шагами среды).
- PPO пересчитывает каждый переход с сохранённым скрытым состоянием, чтобы шаг
  оптимизации видел то же представление, что и при сборе данных.
"""

import numpy as np
import torch


def structural_zeros_like(state):
    """Рекурсивно строит дерево нулевых тензоров той же структуры, что и состояние Norse."""
    if state is None:
        return None
    if isinstance(state, list):
        return [structural_zeros_like(s) for s in state]
    if hasattr(state, "_fields"):
        return type(state)(
            *(torch.zeros_like(t) if isinstance(t, torch.Tensor) else t for t in state)
        )
    if isinstance(state, torch.Tensor):
        # Запасной вариант: в типичном дереве Norse сюда не заходим.
        return torch.zeros_like(state)
    return state


def reset_state_batch_indices(state, indices):
    """
    Обнуляет скрытое состояние Norse по размерности батча (dim=0) для указанных индексов сред.

    Вызывать при завершении эпизода (terminated / truncated / time limit), когда среда
    автоматически перезапускается, чтобы память SNN не переносилась между эпизодами.
    """
    if state is None:
        return
    if not isinstance(indices, torch.Tensor) or indices.numel() == 0:
        return
    t0 = next(_state_tensors(state), None)
    if t0 is not None:
        indices = indices.to(device=t0.device, dtype=torch.long)
    if isinstance(state, list):
        for s in state:
            reset_state_batch_indices(s, indices)
        return
    if hasattr(state, "_fields"):
        for t in state:
            if isinstance(t, torch.Tensor):
                t[indices] = 0
        return
    if isinstance(state, torch.Tensor):
        state[indices] = 0


def detach_state(state):
    """Отсоединяет тензоры состояния (градиент не идёт через предыдущий шаг)."""
    if state is None:
        return None
    if isinstance(state, list):
        return [detach_state(s) for s in state]
    if hasattr(state, "_fields"):
        return type(state)(*(t.detach() if isinstance(t, torch.Tensor) else t for t in state))
    if isinstance(state, torch.Tensor):
        return state.detach()
    return state


def stack_rollout_states(states_per_step):
    """
    Склеивает список состояний по шагам (батч = num_envs) в одно состояние
    с батчем num_steps * num_envs (тот же порядок, что у torch.cat по списку наблюдений).
    """
    if not states_per_step:
        return None
    if states_per_step[0] is None:
        return None
    if isinstance(states_per_step[0], list):
        return [
            stack_rollout_states([s[i] for s in states_per_step])
            for i in range(len(states_per_step[0]))
        ]
    if hasattr(states_per_step[0], "_fields"):
        fields = states_per_step[0]._fields
        out = []
        for f in fields:
            parts = [getattr(s, f) for s in states_per_step]
            if parts[0] is None:
                out.append(None)
            else:
                out.append(torch.cat(parts, dim=0))
        return type(states_per_step[0])(*out)
    if isinstance(states_per_step[0], torch.Tensor):
        return torch.cat(states_per_step, dim=0)
    return states_per_step[0]


def index_state_batch(state, indices, device=None):
    """Индексация по размерности 0 у всех тензоров в state (indices: long tensor или ndarray)."""
    if state is None:
        return None
    if isinstance(indices, np.ndarray):
        dev = device
        if dev is None:
            t0 = next(_state_tensors(state), None)
            dev = t0.device if t0 is not None else torch.device("cpu")
        indices = torch.as_tensor(indices, dtype=torch.long, device=dev)
    if isinstance(state, list):
        return [index_state_batch(s, indices) for s in state]
    if hasattr(state, "_fields"):
        return type(state)(
            *(t[indices] if isinstance(t, torch.Tensor) else t for t in state)
        )
    if isinstance(state, torch.Tensor):
        return state[indices]
    return state


def _state_tensors(state):
    """Обход листьев-тензоров (для устройства и индексации)."""
    if state is None:
        return
    if isinstance(state, list):
        for s in state:
            yield from _state_tensors(s)
    elif hasattr(state, "_fields"):
        for t in state:
            if isinstance(t, torch.Tensor):
                yield t
    elif isinstance(state, torch.Tensor):
        yield state
