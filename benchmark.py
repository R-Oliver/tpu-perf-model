"""Resumable TPU benchmark for the deterministic GEMM shape families."""

from __future__ import annotations

import csv
import json
import os
import platform
import socket
import statistics
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import evaluate
from hardware import HardwareSpec
import shapes


DEVICE_ALIASES = {
    "v5e": {"TPU v5 lite", "TPU v5e"},
    "v6e": {"TPU v6 lite", "TPU v6e"},
}
JAX_DTYPE = {
    "bf16": jnp.bfloat16,
    "int8": jnp.int8,
    "f32": jnp.float32,
    "int32": jnp.int32,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def validate_device(sku: str) -> str:
    """Catch accidental SKU mislabeling without depending on private JAX APIs."""

    devices = jax.devices()
    if len(devices) != 1:
        raise RuntimeError(f"expected one JAX device, found {len(devices)}")
    kind = devices[0].device_kind
    aliases = DEVICE_ALIASES.get(sku)
    if aliases is None:
        raise ValueError(f"no benchmark device aliases configured for {sku!r}")
    if kind not in aliases:
        raise RuntimeError(
            f"attached device {kind!r} does not match {sku!r}: {sorted(aliases)}"
        )
    return kind


def _operands(problem):
    key_a, key_b = jax.random.split(jax.random.key(0))
    if problem.dtype == "int8":
        a = jax.random.randint(
            key_a, (problem.M, problem.K), -8, 8, dtype=jnp.int32
        ).astype(jnp.int8)
        b = jax.random.randint(
            key_b, (problem.K, problem.N), -8, 8, dtype=jnp.int32
        ).astype(jnp.int8)
    else:
        a = jax.random.normal(
            key_a, (problem.M, problem.K), dtype=JAX_DTYPE[problem.dtype]
        )
        b = jax.random.normal(
            key_b, (problem.K, problem.N), dtype=JAX_DTYPE[problem.dtype]
        )
    return jax.block_until_ready(a), jax.block_until_ready(b)


def _compiled_dot(output_dtype: str):
    return jax.jit(
        lambda a, b: jnp.dot(
            a, b, preferred_element_type=JAX_DTYPE[output_dtype]
        )
    )


def time_dot(problem, warmup: int, iterations: int) -> tuple[float, float]:
    """Exclude compilation, synchronize every sample, and return median/IQR."""

    if warmup < 1 or iterations < 1:
        raise ValueError("warmup and iterations must be positive")
    a, b = _operands(problem)
    operation = _compiled_dot(problem.output_dtype)
    for _ in range(warmup):
        jax.block_until_ready(operation(a, b))
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        jax.block_until_ready(operation(a, b))
        samples.append(time.perf_counter() - started)
    q1, q3 = np.percentile(samples, [25, 75])
    return statistics.median(samples), float(q3 - q1)


def measure_launch_overhead(warmup: int, iterations: int) -> float:
    from model import GemmProblem

    median_s, _ = time_dot(GemmProblem(8, 8, 8, "bf16"), warmup, iterations)
    return median_s


def _read_measurements(path: Path) -> list[dict]:
    return evaluate.load_measurements(path) if path.exists() else []


def _write_measurements(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=evaluate.MEASUREMENT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_runs(path: Path, runs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(runs, indent=2) + "\n")
    temporary.replace(path)


def run_benchmark(
    sku: str,
    hardware: HardwareSpec,
    output: Path,
    runs_path: Path,
    *,
    seed: int = 0,
    warmup: int = 5,
    iterations: int = 20,
    families: set[str] | None = None,
) -> None:
    if hardware.name != sku:
        raise ValueError(f"spec name {hardware.name!r} does not match requested {sku!r}")
    device_kind = validate_device(sku)
    plan = shapes.build_sweep(hardware, seed)
    if families:
        unknown = families - set(shapes.FAMILIES)
        if unknown:
            raise ValueError(f"unknown shape families: {sorted(unknown)}")
        plan = [case for case in plan if case.family in families]

    rows = _read_measurements(output)
    completed = {
        (row["sku"], row["M"], row["N"], row["K"], row["dtype"])
        for row in rows
    }
    pending = [
        case
        for case in plan
        if (
            sku,
            case.problem.M,
            case.problem.N,
            case.problem.K,
            case.problem.dtype,
        )
        not in completed
    ]
    print(f"{sku}: {len(pending)} pending, {len(plan) - len(pending)} resumed")

    overhead = measure_launch_overhead(warmup, iterations)
    runs = evaluate.load_runs(runs_path) if runs_path.exists() else []
    run = {
        "sku": sku,
        "started_utc": _utc_now(),
        "finished_utc": None,
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jaxlib": _version("jaxlib"),
        "libtpu": _version("libtpu"),
        "device_kind": device_kind,
        "n_devices": len(jax.devices()),
        "seed": seed,
        "warmup": warmup,
        "iters": iterations,
        "overhead_floor_s": overhead,
    }
    runs.append(run)
    _write_runs(runs_path, runs)
    print(f"launch overhead: {overhead * 1e6:.1f} us")

    for index, case in enumerate(pending, start=1):
        median_s, iqr_s = time_dot(case.problem, warmup, iterations)
        rows.append(
            {
                "sku": sku,
                "family": case.family,
                "M": case.problem.M,
                "N": case.problem.N,
                "K": case.problem.K,
                "dtype": case.problem.dtype,
                "median_s": median_s,
                "iqr_s": iqr_s,
                "n_iters": iterations,
            }
        )
        rows.sort(
            key=lambda row: (
                row["sku"], row["family"], row["M"], row["N"], row["K"], row["dtype"]
            )
        )
        _write_measurements(output, rows)
        if index % 25 == 0 or index == len(pending):
            print(f"[{index}/{len(pending)}] {case.family}: {median_s * 1e3:.3f} ms")

    run["finished_utc"] = _utc_now()
    _write_runs(runs_path, runs)
