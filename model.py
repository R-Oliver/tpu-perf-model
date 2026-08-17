"""Final analytical model for a single TPU GEMM.

All values use seconds, bytes, and FLOPs. A multiply-add counts as two FLOPs.
The public surface is deliberately small: construct a :class:`GemmProblem`
and call :func:`predict` with a :class:`hardware.HardwareSpec`.
"""

from __future__ import annotations

import dataclasses
import math

from hardware import HardwareSpec, INPUT_DTYPES, ITEMSIZE_BYTES, OUTPUT_DTYPE


@dataclasses.dataclass(frozen=True)
class GemmProblem:
    M: int
    N: int
    K: int
    dtype: str = "bf16"

    def __post_init__(self) -> None:
        for name in ("M", "N", "K"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.dtype not in INPUT_DTYPES:
            raise ValueError(
                f"unsupported input dtype {self.dtype!r}; expected one of {INPUT_DTYPES}"
            )

    @property
    def output_dtype(self) -> str:
        return OUTPUT_DTYPE[self.dtype]


@dataclasses.dataclass(frozen=True)
class WorkEstimate:
    """Inspectable work and schedule terms behind one prediction."""

    flops: float
    bytes_moved: float
    n_blocks: int
    compute_s: float
    memory_s: float


def round_up(value: int, multiple: int) -> int:
    if value <= 0 or multiple <= 0:
        raise ValueError("value and multiple must be positive")
    return -(-value // multiple) * multiple


def layout_bytes(rows: int, cols: int, dtype: str, hardware: HardwareSpec) -> float:
    """HBM footprint after TPU physical-layout padding."""

    if rows <= 0 or cols <= 0:
        raise ValueError("array dimensions must be positive")
    if dtype not in ITEMSIZE_BYTES:
        raise ValueError(f"unsupported storage dtype {dtype!r}")
    tile_rows, tile_cols = hardware.tile_multiple[dtype]
    return float(
        round_up(rows, tile_rows)
        * round_up(cols, tile_cols)
        * ITEMSIZE_BYTES[dtype]
    )


def arithmetic_flops(problem: GemmProblem, hardware: HardwareSpec) -> float:
    """MXU work: output remainders are cheap; contraction remainders are billed."""

    padded_k = round_up(problem.K, hardware.mxu_dim)
    return float(2 * problem.M * problem.N * padded_k)


def compulsory_bytes(problem: GemmProblem, hardware: HardwareSpec) -> float:
    return (
        layout_bytes(problem.M, problem.K, problem.dtype, hardware)
        + layout_bytes(problem.K, problem.N, problem.dtype, hardware)
        + layout_bytes(problem.M, problem.N, problem.output_dtype, hardware)
    )


def _block_working_set(edge: int, dtype: str, hardware: HardwareSpec) -> float:
    accumulator = ITEMSIZE_BYTES[OUTPUT_DTYPE[dtype]] * edge * edge
    # One current and one prefetched MXU-deep panel of each input.
    panels = 4 * edge * hardware.mxu_dim * ITEMSIZE_BYTES[dtype]
    return float(accumulator + panels)


def vmem_block_dim(dtype: str, hardware: HardwareSpec) -> int:
    """Largest square output block that fits, rounded to the MXU width."""

    edge = hardware.mxu_dim
    if _block_working_set(edge, dtype, hardware) > hardware.vmem_bytes:
        raise ValueError("vmem_bytes cannot hold one MXU-sized working set")
    while _block_working_set(edge + hardware.mxu_dim, dtype, hardware) <= hardware.vmem_bytes:
        edge += hardware.mxu_dim
    return edge


def accumulator_resident_bytes(problem: GemmProblem, hardware: HardwareSpec) -> float:
    """VMEM needed for C plus double-buffered, MXU-deep A and B panels."""

    accumulator = (
        ITEMSIZE_BYTES[problem.output_dtype] * problem.M * problem.N
    )
    panels = 2 * hardware.mxu_dim * ITEMSIZE_BYTES[problem.dtype] * (
        problem.M + problem.N
    )
    return float(accumulator + panels)


def traffic_bytes(problem: GemmProblem, hardware: HardwareSpec) -> float:
    """HBM traffic including input re-reads forced by output blocking."""

    a = layout_bytes(problem.M, problem.K, problem.dtype, hardware)
    b = layout_bytes(problem.K, problem.N, problem.dtype, hardware)
    c = layout_bytes(problem.M, problem.N, problem.output_dtype, hardware)
    compulsory = a + b + c

    if compulsory <= hardware.vmem_bytes:
        return compulsory
    if accumulator_resident_bytes(problem, hardware) <= hardware.vmem_bytes:
        return compulsory

    block = min(
        vmem_block_dim(problem.dtype, hardware), problem.M, problem.N
    )
    reread_factor = (2.0 * problem.M * problem.N / block) / (
        problem.M + problem.N
    )
    return reread_factor * (a + b) + c


def grid_blocks(problem: GemmProblem, hardware: HardwareSpec) -> int:
    """Depth of the double-buffered output-block pipeline."""

    padded_m = round_up(problem.M, hardware.mxu_dim)
    padded_n = round_up(problem.N, hardware.mxu_dim)
    block = min(
        vmem_block_dim(problem.dtype, hardware), padded_m, padded_n
    )
    return math.ceil(padded_m / block) * math.ceil(padded_n / block)


def estimate(problem: GemmProblem, hardware: HardwareSpec) -> WorkEstimate:
    flops = arithmetic_flops(problem, hardware)
    moved = traffic_bytes(problem, hardware)
    compute_s = flops / hardware.peak_flops[problem.dtype]
    memory_s = moved / hardware.hbm_bandwidth_bytes
    return WorkEstimate(
        flops=flops,
        bytes_moved=moved,
        n_blocks=grid_blocks(problem, hardware),
        compute_s=compute_s,
        memory_s=memory_s,
    )


def pipeline_time(compute_s: float, memory_s: float, n_blocks: int) -> float:
    """Double-buffered pipeline, including one unhidden block at either end."""

    if compute_s < 0 or memory_s < 0:
        raise ValueError("pipeline times must be non-negative")
    if not isinstance(n_blocks, int) or isinstance(n_blocks, bool) or n_blocks <= 0:
        raise ValueError("n_blocks must be a positive integer")
    return max(compute_s, memory_s) + min(compute_s, memory_s) / n_blocks


def predict(
    problem: GemmProblem,
    hardware: HardwareSpec,
    *,
    launch_overhead_s: float = 0.0,
) -> float:
    """Predict warm runtime in seconds.

    ``launch_overhead_s`` is optional run calibration, not a hardware property.
    The direct ``predict(problem, hardware)`` path is therefore spec-only.
    """

    if launch_overhead_s < 0 or not math.isfinite(launch_overhead_s):
        raise ValueError("launch_overhead_s must be finite and non-negative")
    work = estimate(problem, hardware)
    return launch_overhead_s + pipeline_time(
        work.compute_s, work.memory_s, work.n_blocks
    )


def arithmetic_intensity(problem: GemmProblem, hardware: HardwareSpec) -> float:
    return arithmetic_flops(problem, hardware) / traffic_bytes(problem, hardware)


def regime(problem: GemmProblem, hardware: HardwareSpec) -> str:
    intensity = arithmetic_intensity(problem, hardware)
    ridge = hardware.ridge_point[problem.dtype]
    if ridge / 2 <= intensity <= ridge * 2:
        return "near_ridge"
    return "compute_bound" if intensity > ridge else "memory_bound"
