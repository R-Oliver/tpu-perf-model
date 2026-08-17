import csv
from pathlib import Path

import pytest

import evaluate
from hardware import load_hardware
import model


ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def results():
    return evaluate.load_results()


def test_evaluator_uses_the_public_prediction_api(results):
    overheads = evaluate.launch_overheads(evaluate.load_runs(ROOT / "data" / "runs.json"))
    for row in results[::37]:
        problem = model.GemmProblem(row["M"], row["N"], row["K"], row["dtype"])
        expected = model.predict(
            problem,
            load_hardware(row["sku"]),
            launch_overhead_s=overheads[row["sku"]],
        )
        assert row["predicted_s"] == expected


def test_empirical_accuracy_bounds_over_committed_measurements(results):
    assert len(results) == 900
    expected = {
        "v5e": (0.074, 0.234),
        "v6e": (0.056, 0.138),
    }
    for sku, (median_bound, p90_bound) in expected.items():
        score = evaluate.metrics(row for row in results if row["sku"] == sku)
        assert score["n"] == 450
        assert score["median_abs_rel"] <= median_bound
        assert score["p90_abs_rel"] <= p90_bound


def test_measurement_schema_is_deliberately_narrow(tmp_path):
    path = tmp_path / "legacy.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow((*evaluate.MEASUREMENT_COLUMNS, "a_dtype"))
        writer.writerow(("v5e", "square", 128, 128, 128, "bf16", 1e-4, 1e-6, 20, "bf16"))
    with pytest.raises(ValueError, match="schema"):
        evaluate.load_measurements(path)


def test_results_tables_cover_each_sku_and_family(results):
    overall = evaluate.result_table(results, ("sku",))
    families = evaluate.result_table(results, ("sku", "family"))
    assert [row["sku"] for row in overall] == ["v5e", "v6e"]
    assert len(families) == 10
