"""Seeded, deterministic shape families used by the benchmark."""

from __future__ import annotations

import dataclasses

import numpy as np

import model
from hardware import HardwareSpec


FAMILIES = ("square", "random", "skinny", "tile_probe", "ridge_band")
MAX_DIM = 16384
MIN_DIM = 128


@dataclasses.dataclass(frozen=True)
class ShapeCase:
    family: str
    problem: model.GemmProblem


def _cases(family: str, triples, dtype: str) -> list[ShapeCase]:
    return [
        ShapeCase(family, model.GemmProblem(int(M), int(N), int(K), dtype))
        for M, N, K in triples
    ]


def _square() -> list[ShapeCase]:
    sizes = [
        128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144,
        8192, 12288, 16384,
    ]
    return sum(
        (_cases("square", [(size, size, size) for size in sizes], dtype)
         for dtype in ("bf16", "int8")),
        [],
    )


def _random(rng: np.random.Generator) -> list[ShapeCase]:
    """Log-uniform and only 8-aligned, so padding failures remain visible."""

    low, high = np.log(MIN_DIM), np.log(MAX_DIM)
    dimensions = np.exp(rng.uniform(low, high, size=(160, 3)))
    triples = [
        tuple(model.round_up(int(dimension), 8) for dimension in row)
        for row in dimensions
    ]
    int8_indices = sorted(rng.permutation(160)[:80])
    return _cases("random", triples, "bf16") + _cases(
        "random", [triples[index] for index in int8_indices], "int8"
    )


def _skinny() -> list[ShapeCase]:
    tiny = [1, 8, 16, 32, 64, 128]
    decode_nk = [
        (4096, 4096), (8192, 4096), (11008, 4096),
        (16384, 8192), (16384, 16384),
    ]
    big_mn = [(4096, 4096), (8192, 4096), (4096, 8192), (16384, 4096)]
    big_mk = [(4096, 4096), (8192, 4096), (4096, 8192), (16384, 4096)]
    triples = (
        [(M, N, K) for M in tiny for N, K in decode_nk]
        + [(M, N, K) for K in tiny for M, N in big_mn]
        + [(M, N, K) for N in tiny for M, K in big_mk]
    )
    return _cases("skinny", triples, "bf16")


def _tile_probe() -> list[ShapeCase]:
    """Separate probes of output and contraction dimensions at boundaries."""

    boundary = [
        255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049,
    ]
    odd_128 = [384, 640, 1152, 1920]
    dimensions = boundary + odd_128
    cases: list[ShapeCase] = []
    for dtype in ("bf16", "int8"):
        cases += _cases(
            "tile_probe", [(dimension, 4096, 4096) for dimension in dimensions], dtype
        )
        cases += _cases(
            "tile_probe", [(4096, 4096, dimension) for dimension in dimensions], dtype
        )
    return cases


def _ridge_band(
    rng: np.random.Generator, hardware: HardwareSpec
) -> list[ShapeCase]:
    """Forty bf16 shapes spanning 0.3x to 3x this SKU's modeled ridge."""

    ridge = hardware.ridge_point["bf16"]
    accepted: list[tuple[int, int, int]] = []
    attempts = 0
    while len(accepted) < 40 and attempts < 20_000:
        attempts += 1
        target = ridge * np.exp(rng.uniform(np.log(0.3), np.log(3.0)))
        aspect = np.exp(rng.uniform(np.log(1.5), np.log(12.0)))
        M = model.round_up(int(2 * target * aspect), 8)
        K = model.round_up(int(M / (aspect - 1.0)), 8)
        if not (MIN_DIM <= M <= MAX_DIM and MIN_DIM <= K <= MAX_DIM):
            continue
        problem = model.GemmProblem(M, M, K, "bf16")
        intensity = model.arithmetic_intensity(problem, hardware)
        if ridge * 0.3 <= intensity <= ridge * 3.0:
            accepted.append((M, M, K))
    if len(accepted) != 40:
        raise RuntimeError("could not construct the ridge-band family")
    return _cases("ridge_band", accepted, "bf16")


def build_sweep(hardware: HardwareSpec, seed: int = 0) -> list[ShapeCase]:
    """Build the same ordered set on every machine for a given seed and SKU."""

    rng = np.random.default_rng(seed)
    candidates = (
        _square() + _random(rng) + _skinny() + _tile_probe()
        + _ridge_band(rng, hardware)
    )
    unique: dict[tuple[int, int, int, str], ShapeCase] = {}
    for case in candidates:
        problem = case.problem
        key = (problem.M, problem.N, problem.K, problem.dtype)
        unique.setdefault(key, case)
    return sorted(
        unique.values(),
        key=lambda case: (
            case.family,
            case.problem.M,
            case.problem.N,
            case.problem.K,
            case.problem.dtype,
        ),
    )
