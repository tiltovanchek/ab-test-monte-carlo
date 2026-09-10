"""Бенчмарк: одни и те же A/A-тесты (t-тест на выручке) тремя способами:
цикл Python по экспериментам, векторизация NumPy и векторизация плюс процессы.
Размеры задачи берутся из config.yaml."""

import time
from functools import partial

import numpy as np
import pandas as pd

from . import generators as gen
from . import runner
from . import stat_tests as st
from .config import load_config, path_in_root


def _simulate(task, n_reps, rng, cfg):
    r = cfg["metrics"]["revenue"]
    a, b = gen.revenue(task.n_per_group, r["p"], r["mu"], r["sigma"], 0.0, rng, reps=n_reps)
    return [runner.Block("welch_t", st.welch_t(a, b).p_value)]


def python_loop(n, reps, cfg, seed):
    r = cfg["metrics"]["revenue"]
    rng = np.random.default_rng(seed)
    p = np.empty(reps)
    for i in range(reps):
        a, b = gen.revenue(n, r["p"], r["mu"], r["sigma"], 0.0, rng)
        p[i] = st.welch_t(a, b).p_value
    return p


def cpu_name():
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return "unknown"


def _seconds(fn, *args, **kwargs):
    start = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - start


def run_benchmark(n_jobs=4):
    cfg = load_config()
    simulate = partial(_simulate, cfg=cfg)
    kw = {"seed": cfg["seed"], "chunk_size": cfg["chunk_size"], "max_cells": cfg["max_chunk_cells"]}
    jobs = runner.safe_n_jobs(n_jobs)

    rows = []
    for case in cfg["benchmark"]["cases"]:
        n, reps = case["n"], case["n_reps"]
        task = runner.Task("benchmark", "revenue", n, 0.0, n_reps=reps)
        loop = _seconds(python_loop, n, reps, cfg, cfg["seed"])
        vec = _seconds(runner.run, [task], simulate, n_jobs=1, **kw)
        par = _seconds(runner.run, [task], simulate, n_jobs=jobs, **kw)
        rows.append((n, reps, "Цикл Python по повторениям", loop))
        rows.append((n, reps, "Векторизация NumPy", vec))
        rows.append((n, reps, f"Векторизация + {jobs} процесса", par))

    df = pd.DataFrame(rows, columns=["n", "reps", "method", "seconds"])
    df["speedup"] = df.groupby("n")["seconds"].transform("first") / df["seconds"]
    df["cpu"] = cpu_name()
    df.to_csv(path_in_root(cfg["results_dir"]) / "benchmark.csv", index=False)
    return df
