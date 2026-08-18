from pathlib import Path

import pytest

import analysis
import evaluate


ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def candidates():
    measurements = evaluate.load_measurements(ROOT / "data" / "measurements.csv")
    runs = evaluate.load_runs(ROOT / "data" / "runs.json")
    return analysis.candidate_results(measurements, runs, ROOT / "specs")


def test_initial_near_ridge_claim_is_reproducible(candidates):
    table = analysis.initial_hard_max_regime_table(candidates, ROOT / "specs")
    near_ridge = {
        row["sku"]: row for row in table if row["initial_regime"] == "near_ridge"
    }
    assert near_ridge["v5e"]["n"] == 250
    assert near_ridge["v5e"]["geomean_ratio"] == pytest.approx(0.9471368666)
    assert near_ridge["v6e"]["n"] == 191
    assert near_ridge["v6e"]["geomean_ratio"] == pytest.approx(0.9020891652)


def test_tile_boundary_penalty_claims_are_reproducible():
    measurements = evaluate.load_measurements(ROOT / "data" / "measurements.csv")
    table = {
        row["boundary"]: row for row in analysis.tile_boundary_penalties(measurements)
    }
    assert table["M: 2048 to 2049"]["v5e"] == pytest.approx(0.01857, abs=5e-6)
    assert table["K: 1024 to 1025"]["v5e"] == pytest.approx(0.11232, abs=5e-6)
    assert table["K: 2048 to 2049"]["v6e"] == pytest.approx(0.09089, abs=5e-6)


def test_worst_decile_composition_is_reproducible(candidates):
    table = analysis.worst_decile_family_counts(candidates)
    v5e = {row["family"]: row["count"] for row in table if row["sku"] == "v5e"}
    assert v5e == {
        "random": 2,
        "ridge_band": 7,
        "skinny": 22,
        "square": 0,
        "tile_probe": 14,
    }
