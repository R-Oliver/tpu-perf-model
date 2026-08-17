"""Historical baselines, diagnostic calibration, and final-model ablations."""

from __future__ import annotations

import dataclasses
import hashlib
import math
from pathlib import Path

import numpy as np

import evaluate
from hardware import HardwareSpec, ITEMSIZE_BYTES, load_hardware
import model


@dataclasses.dataclass(frozen=True)
class DiagnosticCalibration:
    eta_compute: float = 1.0
    eta_memory: float = 1.0
    launch_overhead_s: float = 0.0
    sharpness: float = math.inf


def textbook_roofline(
    problem: model.GemmProblem,
    hardware: HardwareSpec,
    launch_overhead_s: float,
) -> float:
    """Useful work and unpadded compulsory traffic: the textbook baseline."""

    flops = 2.0 * problem.M * problem.N * problem.K
    moved = (
        ITEMSIZE_BYTES[problem.dtype] * (problem.M * problem.K + problem.K * problem.N)
        + ITEMSIZE_BYTES[problem.output_dtype] * problem.M * problem.N
    )
    return launch_overhead_s + max(
        flops / hardware.peak_flops[problem.dtype],
        moved / hardware.hbm_bandwidth_bytes,
    )


def _all_dimension_flops(
    problem: model.GemmProblem, hardware: HardwareSpec
) -> float:
    granularity = max(
        hardware.mxu_dim, *hardware.tile_multiple[problem.dtype]
    )
    return float(
        2
        * model.round_up(problem.M, granularity)
        * model.round_up(problem.N, granularity)
        * model.round_up(problem.K, granularity)
    )


def _legacy_traffic(
    problem: model.GemmProblem, hardware: HardwareSpec
) -> float:
    """Original total-working-set branch, retained only for reproduction."""

    a = model.layout_bytes(problem.M, problem.K, problem.dtype, hardware)
    b = model.layout_bytes(problem.K, problem.N, problem.dtype, hardware)
    c = model.layout_bytes(problem.M, problem.N, problem.output_dtype, hardware)
    if a + b + c <= hardware.vmem_bytes:
        return a + b + c
    block = min(
        model.vmem_block_dim(problem.dtype, hardware), problem.M, problem.N
    )
    reread = (2.0 * problem.M * problem.N / block) / (problem.M + problem.N)
    return reread * (a + b) + c


def accounted_hard_max(
    problem: model.GemmProblem,
    hardware: HardwareSpec,
    launch_overhead_s: float,
) -> float:
    flops = _all_dimension_flops(problem, hardware)
    moved = _legacy_traffic(problem, hardware)
    return launch_overhead_s + max(
        flops / hardware.peak_flops[problem.dtype],
        moved / hardware.hbm_bandwidth_bytes,
    )


def padding_ablation(
    problem: model.GemmProblem,
    hardware: HardwareSpec,
    launch_overhead_s: float,
) -> float:
    """Final traffic/pipeline model with the falsified all-dimension FLOP bill."""

    compute_s = _all_dimension_flops(problem, hardware) / hardware.peak_flops[
        problem.dtype
    ]
    memory_s = model.traffic_bytes(problem, hardware) / hardware.hbm_bandwidth_bytes
    return launch_overhead_s + model.pipeline_time(
        compute_s, memory_s, model.grid_blocks(problem, hardware)
    )


def _blend(compute_s, memory_s, sharpness: float):
    large = np.maximum(compute_s, memory_s)
    small = np.minimum(compute_s, memory_s)
    if not np.isfinite(sharpness):
        return large
    ratio = np.divide(small, large, out=np.zeros_like(large), where=large > 0)
    return large * (1.0 + ratio**sharpness) ** (1.0 / sharpness)


def _diagnostic_predict(flops, moved, peaks, bandwidth, calibration):
    return calibration.launch_overhead_s + _blend(
        flops / (peaks * calibration.eta_compute),
        moved / (bandwidth * calibration.eta_memory),
        calibration.sharpness,
    )


def split_half(row: dict, seed: int = 0) -> int:
    key = f"{row['M']}x{row['N']}x{row['K']}:{row['dtype']}"
    digest = hashlib.blake2b(f"{seed}|{key}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % 2


_SHARPNESS = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0, math.inf]


def fit_diagnostic(
    rows: list[dict],
    hardware: HardwareSpec,
    launch_overhead_s: float,
    seed: int = 0,
) -> DiagnosticCalibration:
    """Reproduce the original four-constant fit on the stable development half."""

    train = [row for row in rows if split_half(row, seed) == 0]
    problems = [
        model.GemmProblem(row["M"], row["N"], row["K"], row["dtype"])
        for row in train
    ]
    flops = np.array([_all_dimension_flops(p, hardware) for p in problems])
    moved = np.array([_legacy_traffic(p, hardware) for p in problems])
    peaks = np.array([hardware.peak_flops[p.dtype] for p in problems])
    actual = np.array([row["median_s"] for row in train])

    def loss(calibration: DiagnosticCalibration) -> float:
        predicted = _diagnostic_predict(
            flops, moved, peaks, hardware.hbm_bandwidth_bytes, calibration
        )
        return float(np.median(np.abs(np.log(predicted / actual))))

    def scan(calibration, field, values):
        best, best_loss = calibration, loss(calibration)
        for value in values:
            candidate = dataclasses.replace(calibration, **{field: float(value)})
            candidate_loss = loss(candidate)
            if candidate_loss < best_loss:
                best, best_loss = candidate, candidate_loss
        return best

    best_calibration = None
    best_loss = math.inf
    coarse = np.linspace(0.05, 1.5, 30)
    for eta_compute in coarse:
        for eta_memory in coarse:
            for sharpness in _SHARPNESS:
                candidate = DiagnosticCalibration(
                    float(eta_compute),
                    float(eta_memory),
                    launch_overhead_s,
                    sharpness,
                )
                candidate_loss = loss(candidate)
                if candidate_loss < best_loss:
                    best_calibration, best_loss = candidate, candidate_loss

    span_eta = coarse[1] - coarse[0]
    span_overhead = max(launch_overhead_s, 5e-5) * 4
    for _ in range(8):
        for field, span, lower, upper in (
            ("eta_compute", span_eta, 0.02, 1.5),
            ("eta_memory", span_eta, 0.02, 1.5),
            ("launch_overhead_s", span_overhead, 0.0, 1e-3),
        ):
            current = getattr(best_calibration, field)
            values = np.clip(
                np.linspace(current - span, current + span, 21), lower, upper
            )
            best_calibration = scan(best_calibration, field, values)
        best_calibration = scan(best_calibration, "sharpness", _SHARPNESS)
        span_eta *= 0.5
        span_overhead *= 0.5
    return best_calibration


def diagnostic_prediction(
    problem: model.GemmProblem,
    hardware: HardwareSpec,
    calibration: DiagnosticCalibration,
) -> float:
    return float(
        _diagnostic_predict(
            np.array([_all_dimension_flops(problem, hardware)]),
            np.array([_legacy_traffic(problem, hardware)]),
            np.array([hardware.peak_flops[problem.dtype]]),
            hardware.hbm_bandwidth_bytes,
            calibration,
        )[0]
    )


def candidate_results(rows: list[dict], runs: list[dict], specs_dir: Path) -> list[dict]:
    overheads = evaluate.launch_overheads(runs)
    hardware_by_sku = {
        sku: load_hardware(sku, specs_dir) for sku in {row["sku"] for row in rows}
    }
    output = []
    for row in rows:
        hardware = hardware_by_sku[row["sku"]]
        problem = model.GemmProblem(row["M"], row["N"], row["K"], row["dtype"])
        overhead = overheads[row["sku"]]
        output.append(
            {
                **row,
                "textbook": textbook_roofline(problem, hardware, overhead),
                "accounted": accounted_hard_max(problem, hardware, overhead),
                "all_dimensions": padding_ablation(problem, hardware, overhead),
                "final": model.predict(
                    problem, hardware, launch_overhead_s=overhead
                ),
            }
        )
    return output


def main() -> None:
    root = Path(__file__).parent
    measurements = evaluate.load_measurements(root / "data" / "measurements.csv")
    runs = evaluate.load_runs(root / "data" / "runs.json")
    candidates = candidate_results(measurements, runs, root / "specs")
    for sku in sorted({row["sku"] for row in candidates}):
        sample = [row for row in candidates if row["sku"] == sku]
        print(f"\n## {sku} ablations\n")
        for candidate in ("textbook", "accounted", "all_dimensions", "final"):
            score = evaluate.metrics(sample, candidate)
            print(
                f"{candidate:15s} median={score['median_abs_rel']:.1%} "
                f"p90={score['p90_abs_rel']:.1%} "
                f"bias={score['geomean_ratio']:.3f}x"
            )

        hardware = load_hardware(sku)
        overhead = evaluate.launch_overheads(runs)[sku]
        calibration = fit_diagnostic(sample, hardware, overhead)
        for row in sample:
            problem = model.GemmProblem(row["M"], row["N"], row["K"], row["dtype"])
            row["diagnostic"] = diagnostic_prediction(problem, hardware, calibration)
        print(f"diagnostic fit: {calibration}")
        for split, label in ((0, "fit"), (1, "holdout")):
            selected = [row for row in sample if split_half(row) == split]
            initial = evaluate.metrics(selected, "accounted")
            diagnostic = evaluate.metrics(selected, "diagnostic")
            print(
                f"  {label:7s} n={diagnostic['n']} "
                f"initial={initial['median_abs_rel']:.1%}/{initial['p90_abs_rel']:.1%} "
                f"diagnostic={diagnostic['median_abs_rel']:.1%}/"
                f"{diagnostic['p90_abs_rel']:.1%} "
                f"bias={diagnostic['geomean_ratio']:.3f}x"
            )


if __name__ == "__main__":
    main()
