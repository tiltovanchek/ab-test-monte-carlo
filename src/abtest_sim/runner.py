"""Цикл Монте-Карло: режем работу на чанки, считаем параллельно, собираем одну таблицу.

Воспроизводимость:
- у каждой точки сетки свой SeedSequence из общего seed и ключа точки,
  поэтому новая точка в конфиге не меняет результаты старых;
- каждый чанк получает свой дочерний поток через spawn;
- размер чанка не зависит от числа процессов, поэтому результат одинаковый
  и при 1, и при 4 процессах.

Частая ошибка - один seed во всех процессах: тогда "разные" эксперименты
в процессах просто повторяют друг друга.
"""

import os
import zlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

COLUMNS = [
    "scenario",
    "metric",
    "test",
    "n_per_group",
    "uplift",
    "param",
    "rep",
    "p_value",
    "reject",
    "effect",
    "ci_covers",
]


@dataclass(frozen=True)
class Task:
    """Одна точка сетки параметров."""

    scenario: str
    metric: str
    n_per_group: int
    uplift: float
    param: str = ""
    n_reps: int = 10_000
    alpha: float = 0.05
    chunk_size: int | None = None  # если задан, используется как есть
    kwargs: dict = field(default_factory=dict, hash=False, compare=False)

    @property
    def key(self):
        return f"{self.scenario}|{self.metric}|{self.n_per_group}|{self.uplift}|{self.param}"


@dataclass
class Block:
    """Результат одного критерия на чанке: массивы длины n_reps.
    metric и uplift берутся из задачи, если не заданы."""

    test: str
    p_value: np.ndarray
    reject: np.ndarray | None = None
    effect: np.ndarray | None = None
    ci_covers: np.ndarray | None = None
    metric: str | None = None
    uplift: float | None = None


def task_seed(seed, task):
    return np.random.SeedSequence([seed, zlib.crc32(task.key.encode())])


def _available_gb():
    """Доступная память из /proc/meminfo. Не на Linux вернёт None."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return None


def safe_n_jobs(n_jobs, gb_per_job=1.0):
    """Не больше ядер и не больше одного процесса на гигабайт доступной памяти."""
    cpus = os.cpu_count() or 1
    n_jobs = cpus if n_jobs < 1 else min(n_jobs, cpus)
    available = _available_gb()
    if available is not None:
        n_jobs = min(n_jobs, int(available / gb_per_job))
    return max(1, n_jobs)


def chunk_for(task, chunk_size, max_cells):
    """Размер чанка: не больше chunk_size повторений и не больше max_cells чисел в массиве."""
    if task.chunk_size:
        return task.chunk_size
    return max(1, min(chunk_size, max_cells // task.n_per_group))


def _to_frame(task, blocks, rep_offset):
    frames = []
    for b in blocks:
        p = np.asarray(b.p_value, dtype=float)
        frames.append(
            pd.DataFrame(
                {
                    "scenario": task.scenario,
                    "metric": b.metric or task.metric,
                    "test": b.test,
                    "n_per_group": task.n_per_group,
                    "uplift": task.uplift if b.uplift is None else b.uplift,
                    "param": task.param,
                    "rep": np.arange(rep_offset, rep_offset + len(p)),
                    "p_value": p,
                    "reject": p < task.alpha if b.reject is None else b.reject,
                    "effect": np.nan if b.effect is None else np.asarray(b.effect, dtype=float),
                    "ci_covers": pd.NA if b.ci_covers is None else b.ci_covers,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _run_chunk(simulate, task, n_reps, rep_offset, seed_seq):
    rng = np.random.default_rng(seed_seq)
    return _to_frame(task, simulate(task, n_reps, rng), rep_offset)


def run(tasks, simulate, seed, n_jobs=4, chunk_size=1000, max_cells=1_000_000):
    """Прогоняет все задачи и возвращает длинную таблицу результатов.
    simulate(task, n_reps, rng) должна вернуть список Block."""
    jobs = []
    for task in tasks:
        chunk = chunk_for(task, chunk_size, max_cells)
        n_chunks = -(-task.n_reps // chunk)
        for i, seed_seq in enumerate(task_seed(seed, task).spawn(n_chunks)):
            size = min(chunk, task.n_reps - i * chunk)
            jobs.append(delayed(_run_chunk)(simulate, task, size, i * chunk, seed_seq))

    n_jobs = safe_n_jobs(n_jobs)
    if n_jobs == 1:
        frames = [f(*args, **kw) for f, args, kw in jobs]
    else:
        frames = Parallel(n_jobs=n_jobs)(jobs)

    df = pd.concat(frames, ignore_index=True)[COLUMNS]
    df["reject"] = df["reject"].astype(bool)
    df["ci_covers"] = df["ci_covers"].astype("boolean")
    order = ["metric", "test", "n_per_group", "uplift", "param", "rep"]
    return df.sort_values(order, kind="stable").reset_index(drop=True)


def save(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
