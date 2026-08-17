"""Hardware inputs consumed by the GEMM runtime model."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml


ITEMSIZE_BYTES = {"bf16": 2, "int8": 1, "f32": 4, "int32": 4}
OUTPUT_DTYPE = {"bf16": "f32", "int8": "int32"}
INPUT_DTYPES = tuple(OUTPUT_DTYPE)


@dataclasses.dataclass(frozen=True)
class HardwareSpec:
    """The small, prediction-only description of one accelerator chip."""

    name: str
    peak_flops: dict[str, float]
    hbm_bandwidth_bytes: float
    mxu_dim: int
    vmem_bytes: int
    tile_multiple: dict[str, tuple[int, int]]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("hardware name must not be empty")
        if self.hbm_bandwidth_bytes <= 0:
            raise ValueError("hbm_bandwidth_bytes must be positive")
        if not isinstance(self.mxu_dim, int) or isinstance(self.mxu_dim, bool) or self.mxu_dim <= 0:
            raise ValueError("mxu_dim must be a positive integer")
        if not isinstance(self.vmem_bytes, int) or isinstance(self.vmem_bytes, bool) or self.vmem_bytes <= 0:
            raise ValueError("vmem_bytes must be a positive integer")

        for dtype in INPUT_DTYPES:
            if dtype not in self.peak_flops or self.peak_flops[dtype] <= 0:
                raise ValueError(f"peak_flops[{dtype!r}] must be positive")
        for dtype in ITEMSIZE_BYTES:
            tile = self.tile_multiple.get(dtype)
            if (
                tile is None
                or len(tile) != 2
                or any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in tile)
            ):
                raise ValueError(f"tile_multiple[{dtype!r}] must contain two positive integers")

    @property
    def ridge_point(self) -> dict[str, float]:
        return {
            dtype: peak / self.hbm_bandwidth_bytes
            for dtype, peak in self.peak_flops.items()
        }


def load_hardware(name_or_path: str | Path, specs_dir: Path | None = None) -> HardwareSpec:
    """Load ``specs/<name>.yaml`` or an explicit YAML path."""

    candidate = Path(name_or_path)
    if candidate.suffix not in {".yaml", ".yml"}:
        root = specs_dir or Path(__file__).parent / "specs"
        candidate = Path(root) / f"{candidate}.yaml"
    if not candidate.exists():
        raise FileNotFoundError(f"hardware spec not found: {candidate}")

    raw = yaml.safe_load(candidate.read_text())
    required = {
        "name",
        "peak_flops",
        "hbm_bandwidth_bytes",
        "mxu_dim",
        "vmem_bytes",
        "tile_multiple",
    }
    missing = required - set(raw or {})
    if missing:
        raise ValueError(f"{candidate} is missing: {', '.join(sorted(missing))}")
    return HardwareSpec(
        name=str(raw["name"]),
        peak_flops={key: float(value) for key, value in raw["peak_flops"].items()},
        hbm_bandwidth_bytes=float(raw["hbm_bandwidth_bytes"]),
        mxu_dim=int(raw["mxu_dim"]),
        vmem_bytes=int(raw["vmem_bytes"]),
        tile_multiple={
            key: tuple(int(value) for value in values)
            for key, values in raw["tile_multiple"].items()
        },
    )
