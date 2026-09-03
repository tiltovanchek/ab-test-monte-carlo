"""SQL-макросы DuckDB и запросы из sql/."""

import duckdb
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from abtest_sim import db
from abtest_sim.stat_tests import wilson_ci


@pytest.fixture
def con():
    c = duckdb.connect()
    c.execute((db.SQL_DIR / "macros.sql").read_text(encoding="utf-8"))
    return c


@pytest.mark.parametrize("z", [0.0, 0.5, 1.96, 2.5, 3.3])
def test_norm_sf_macro(con, z):
    got = con.sql(f"SELECT norm_sf({z})").fetchone()[0]
    assert got == pytest.approx(stats.norm.sf(z), abs=2e-7)


@pytest.mark.parametrize("chi2", [1.0, 3.84, 10.0])
def test_chi2_macro(con, chi2):
    got = con.sql(f"SELECT chi2_df1_p({chi2})").fetchone()[0]
    assert got == pytest.approx(stats.chi2.sf(chi2, df=1), abs=5e-7)


def test_wilson_macro(con):
    lo, hi = con.sql("SELECT wilson_lo(500, 10000), wilson_hi(500, 10000)").fetchone()
    ref_lo, ref_hi = wilson_ci(500, 10000)
    assert lo == pytest.approx(ref_lo, abs=1e-6)
    assert hi == pytest.approx(ref_hi, abs=1e-6)


def test_srm_query_on_fake_data(con):
    rng = np.random.default_rng(0)
    fake = pd.DataFrame(
        {
            "userid": np.arange(1000),
            "version": np.where(np.arange(1000) < 520, "gate_30", "gate_40"),
            "sum_gamerounds": rng.integers(0, 100, 1000),
            "retention_1": rng.random(1000) < 0.4,
            "retention_7": rng.random(1000) < 0.2,
        }
    )
    con.register("cookie_cats", fake)
    res = db.query("cookie_cats/01_group_sizes_srm", con)
    ref = stats.chisquare([520, 480])
    assert res["chi2"].iloc[0] == pytest.approx(ref.statistic, abs=1e-4)
    assert res["p_value"].iloc[0] == pytest.approx(ref.pvalue, abs=1e-6)
