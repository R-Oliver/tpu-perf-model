"""Load committed measurements and compute final-model result tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from hardware import load_hardware
import model


ROOT = Path(__file__).parent
MEASUREMENT_COLUMNS = (
    "sku", "family", "M", "N", "K", "dtype", "median_s", "iqr_s", "n_iters"
)


def load_measurements(path: Path) -> list[dict]:
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MEASUREMENT_COLUMNS:
            raise ValueError(
                f"{path} schema is {reader.fieldnames}; expected {list(MEASUREMENT_COLUMNS)}"
            )
        rows = []
        for line, raw in enumerate(reader, start=2):
            try:
                row = {
                    "sku": raw["sku"],
                    "family": raw["family"],
                    "M": int(raw["M"]),
                    "N": int(raw["N"]),
                    "K": int(raw["K"]),
                    "dtype": raw["dtype"],
                    "median_s": float(raw["median_s"]),
                    "iqr_s": float(raw["iqr_s"]),
                    "n_iters": int(raw["n_iters"]),
                }
                model.GemmProblem(row["M"], row["N"], row["K"], row["dtype"])
                if row["median_s"] <= 0 or row["iqr_s"] < 0 or row["n_iters"] <= 0:
                    raise ValueError("timings and iteration count are out of range")
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid measurement at {path}:{line}: {error}") from error
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no measurements")
    keys = [(r["sku"], r["M"], r["N"], r["K"], r["dtype"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path} contains duplicate problem keys")
    return rows


def load_runs(path: Path) -> list[dict]:
    runs = json.loads(Path(path).read_text())
    if not isinstance(runs, list) or not runs:
        raise ValueError(f"{path} must contain a non-empty JSON list")
    return runs


def launch_overheads(runs: Iterable[dict]) -> dict[str, float]:
    """Use the newest recorded run for each SKU."""

    selected = {}
    for run in sorted(runs, key=lambda item: item["started_utc"]):
        overhead = float(run["overhead_floor_s"])
        if overhead < 0 or not math.isfinite(overhead):
            raise ValueError(f"invalid launch overhead for {run['sku']}")
        selected[run["sku"]] = overhead
    return selected


def load_results(
    measurements_path: Path = ROOT / "data" / "measurements.csv",
    specs_dir: Path = ROOT / "specs",
    runs_path: Path | None = None,
) -> list[dict]:
    """Evaluator output consumed by the tables, tests, and plotting script."""

    measurements_path = Path(measurements_path)
    runs_path = Path(runs_path) if runs_path else measurements_path.with_name("runs.json")
    overheads = launch_overheads(load_runs(runs_path))
    hardware_by_sku = {}
    results = []
    for row in load_measurements(measurements_path):
        sku = row["sku"]
        if sku not in hardware_by_sku:
            hardware_by_sku[sku] = load_hardware(sku, Path(specs_dir))
        hardware = hardware_by_sku[sku]
        if sku not in overheads:
            raise ValueError(f"no launch-overhead metadata for {sku}")
        problem = model.GemmProblem(row["M"], row["N"], row["K"], row["dtype"])
        work = model.estimate(problem, hardware)
        results.append(
            {
                **row,
                "predicted_s": model.predict(
                    problem, hardware, launch_overhead_s=overheads[sku]
                ),
                "regime": model.regime(problem, hardware),
                "modeled_flops": work.flops,
                "modeled_bytes": work.bytes_moved,
                "n_blocks": work.n_blocks,
            }
        )
    return results


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def metrics(rows: Iterable[dict], prediction: str = "predicted_s") -> dict:
    sample = list(rows)
    if not sample:
        raise ValueError("metrics require at least one result")
    ratios = [row[prediction] / row["median_s"] for row in sample]
    errors = [abs(ratio - 1.0) for ratio in ratios]
    return {
        "n": len(sample),
        "median_abs_rel": statistics.median(errors),
        "p90_abs_rel": _percentile(errors, 90),
        "geomean_ratio": math.exp(statistics.fmean(math.log(ratio) for ratio in ratios)),
    }


def result_table(rows: Iterable[dict], group_by: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_by)].append(row)
    table = []
    for keys in sorted(groups):
        table.append(
            {
                **dict(zip(group_by, keys, strict=True)),
                **metrics(groups[keys]),
            }
        )
    return table


def markdown_table(rows: list[dict]) -> str:
    if not rows:
        return ""
    columns = list(rows[0])

    def render(column: str, value) -> str:
        if column in {"median_abs_rel", "p90_abs_rel"}:
            return f"{value:.1%}"
        if column == "geomean_ratio":
            return f"{value:.3f}x"
        return str(value)

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines += [
        "| " + " | ".join(render(column, row[column]) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, default=ROOT / "data" / "measurements.csv")
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--specs", type=Path, default=ROOT / "specs")
    args = parser.parse_args(argv)
    rows = load_results(args.measurements, args.specs, args.runs)
    for title, fields in (
        ("Overall", ("sku",)),
        ("By family", ("sku", "family")),
        ("By regime", ("sku", "regime")),
    ):
        print(f"\n## {title}\n")
        print(markdown_table(result_table(rows, fields)))


if __name__ == "__main__":
    main()
