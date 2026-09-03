"""Цикл Монте-Карло: воспроизводимость и дымовой A/A-тест."""

from functools import partial

import numpy as np
import pandas as pd
import pytest

from abtest_sim import runner
from abtest_sim.config import load_config
from abtest_sim.scenarios import SCENARIOS, power


def _power_tasks(n_reps):
    return [
        runner.Task("power", "conversion", n, u, n_reps=n_reps)
        for n in (1000, 5000)
        for u in (0.0, 0.1)
    ]


def _simulate():
    return partial(power.simulate, cfg=load_config())


def test_same_result_with_1_and_4_processes():
    tasks = _power_tasks(2000)
    one = runner.run(tasks, _simulate(), seed=7, n_jobs=1, chunk_size=300)
    four = runner.run(tasks, _simulate(), seed=7, n_jobs=4, chunk_size=300)
    pd.testing.assert_frame_equal(one, four)


def test_different_seed_gives_different_result():
    tasks = _power_tasks(500)
    a = runner.run(tasks, _simulate(), seed=1, n_jobs=1)
    b = runner.run(tasks, _simulate(), seed=2, n_jobs=1)
    assert not np.allclose(a["p_value"], b["p_value"])


def test_chunks_are_independent():
    # с одним seed на все чанки повторения 0..299 и 300..599 совпали бы
    df = runner.run(_power_tasks(600)[:1], _simulate(), seed=3, n_jobs=1, chunk_size=300)
    p = df["p_value"].to_numpy()
    assert not np.allclose(p[:300], p[300:])


def test_smoke_aa_fpr_within_bounds():
    # при R = 5000 FPR z-теста должен быть в [4%; 6%]
    task = runner.Task("power", "conversion", 5000, 0.0, n_reps=5000)
    df = runner.run([task], _simulate(), seed=11, n_jobs=1)
    assert 0.04 <= df["reject"].mean() <= 0.06


def test_schema():
    df = runner.run(_power_tasks(100), _simulate(), seed=0, n_jobs=1)
    assert list(df.columns) == runner.COLUMNS
    assert df["rep"].between(0, 99).all()
    assert df["reject"].dtype == bool


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_every_scenario_builds_tasks_with_unique_keys(name):
    tasks = SCENARIOS[name].build_tasks(load_config())
    assert tasks
    keys = [t.key for t in tasks]
    assert len(keys) == len(set(keys))
