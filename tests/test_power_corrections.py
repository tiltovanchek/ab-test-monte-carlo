"""Формулы мощности, поправки на множественные сравнения и границы peeking."""

import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

from abtest_sim import corrections as cr
from abtest_sim.power import power_two_proportions, required_sample_size


def test_required_sample_size_control_point():
    # p = 0.10 и uplift +10% дают около 14 700 на группу (сценарий 2)
    assert required_sample_size(0.10, 0.10) == pytest.approx(14_700, rel=0.01)


def test_half_effect_needs_four_times_sample():
    ratio = required_sample_size(0.10, 0.05) / required_sample_size(0.10, 0.10)
    assert ratio == pytest.approx(4, rel=0.05)


def test_power_at_required_n_is_target():
    n = required_sample_size(0.10, 0.10, power=0.8)
    assert power_two_proportions(0.10, 0.10, n) == pytest.approx(0.8, abs=0.002)


@pytest.mark.parametrize(
    ("name", "method"), [("bonferroni", "bonferroni"), ("holm", "holm"), ("bh", "fdr_bh")]
)
def test_corrections_match_statsmodels(name, method):
    rng = np.random.default_rng(0)
    p = rng.uniform(size=(50, 10)) ** 2
    ours = cr.CORRECTIONS[name](p)
    for i in range(p.shape[0]):
        _, ref, _, _ = multipletests(p[i], method=method)
        np.testing.assert_allclose(ours[i], ref, atol=1e-8)


@pytest.mark.parametrize(("k", "table"), [(2, 2.178), (5, 2.413), (10, 2.555)])
def test_pocock_matches_table(k, table):
    assert cr.pocock_boundary(k) == pytest.approx(table, abs=0.005)


@pytest.mark.parametrize(("k", "table"), [(2, 1.977), (5, 2.040)])
def test_obrien_fleming_matches_table(k, table):
    assert cr.obrien_fleming_constant(k) == pytest.approx(table, abs=0.005)


def test_naive_peeking_inflates_alpha():
    assert cr.crossing_probability(np.full(10, 1.96)) == pytest.approx(0.193, abs=0.005)
    assert cr.crossing_probability(np.array([1.96])) == pytest.approx(0.05, abs=0.001)
