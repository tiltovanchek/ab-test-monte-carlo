"""Сценарии 1-8. В каждом модуле есть NAME, build_tasks(cfg) и simulate(task, n_reps, rng, cfg)."""

from functools import partial

from .. import runner
from ..config import load_config, path_in_root
from . import aa, cuped, heavy_tails, multiple, peeking, power, ratio, srm

SCENARIOS = {m.NAME: m for m in (aa, power, peeking, heavy_tails, ratio, cuped, multiple, srm)}


def run_scenario(name, cfg=None, n_jobs=None, save=True):
    """Считает сценарий и сохраняет results/<name>.parquet."""
    cfg = cfg or load_config()
    module = SCENARIOS[name]
    df = runner.run(
        module.build_tasks(cfg),
        partial(module.simulate, cfg=cfg),
        seed=cfg["seed"],
        n_jobs=cfg["n_jobs"] if n_jobs is None else n_jobs,
        chunk_size=cfg["chunk_size"],
        max_cells=cfg["max_chunk_cells"],
    )
    if save:
        runner.save(df, path_in_root(cfg["results_dir"]) / f"{name}.parquet")
    return df
