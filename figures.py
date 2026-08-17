"""Plot evaluator output; prediction and metric logic live elsewhere."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

import evaluate


ROOT = Path(__file__).parent
OUT = ROOT / "figures"
FAMILIES = ("random", "ridge_band", "skinny", "square", "tile_probe")
LABELS = {
    "random": "Random",
    "ridge_band": "Ridge band",
    "skinny": "Skinny",
    "square": "Square",
    "tile_probe": "Tile probe",
}
COLORS = {
    "random": "#4477AA",
    "ridge_band": "#EE6677",
    "skinny": "#228833",
    "square": "#CCBB44",
    "tile_probe": "#AA3377",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
        }
    )


def prediction_scatter(rows: list[dict]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    values = [row[key] * 1e3 for row in rows for key in ("median_s", "predicted_s")]
    low, high = min(values) * 0.8, max(values) * 1.25

    for axis, sku in zip(axes, ("v5e", "v6e"), strict=True):
        sample = [row for row in rows if row["sku"] == sku]
        for family in FAMILIES:
            family_rows = [row for row in sample if row["family"] == family]
            axis.scatter(
                [row["median_s"] * 1e3 for row in family_rows],
                [row["predicted_s"] * 1e3 for row in family_rows],
                s=18,
                color=COLORS[family],
                alpha=0.7,
                edgecolors="none",
                label=LABELS[family],
            )
        axis.fill_between(
            [low, high],
            [low * 0.75, high * 0.75],
            [low * 1.25, high * 1.25],
            color="#BBBBBB",
            alpha=0.16,
            linewidth=0,
        )
        axis.plot([low, high], [low, high], color="#222222", linewidth=1.2)
        axis.set(xscale="log", yscale="log", xlim=(low, high), ylim=(low, high))
        axis.set_title(f"TPU {sku}", loc="left", fontweight="bold")
        axis.set_xlabel("Measured runtime (ms)")
    axes[0].set_ylabel("Predicted runtime (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(top=0.80, bottom=0.13, wspace=0.20)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=5)
    fig.suptitle("Predicted versus measured GEMM runtime", y=0.99, fontweight="bold")
    axes[1].text(
        0.98, 0.03, "Gray band: within 25%", transform=axes[1].transAxes,
        ha="right", color="#555555", fontsize=9,
    )
    path = OUT / "prediction_scatter.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def tile_boundaries(rows: list[dict]) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    dimensions = [255, 256, 257, 384, 511, 512, 513, 640, 1023, 1024, 1025,
                  1152, 1920, 2047, 2048, 2049]  # fmt: skip

    for row_index, sku in enumerate(("v5e", "v6e")):
        for column_index, varied in enumerate(("M", "K")):
            axis = axes[row_index][column_index]
            sample = []
            for dimension in dimensions:
                matches = [
                    row
                    for row in rows
                    if row["sku"] == sku
                    and row["family"] == "tile_probe"
                    and row["dtype"] == "bf16"
                    and row[varied] == dimension
                    and row["N"] == 4096
                    and row["K" if varied == "M" else "M"] == 4096
                ]
                if matches:
                    sample.append(matches[0])
            positions = range(len(sample))
            useful_flops = [2 * row["M"] * row["N"] * row["K"] for row in sample]
            measured = [flops / row["median_s"] / 1e12 for flops, row in zip(useful_flops, sample, strict=True)]
            predicted = [flops / row["predicted_s"] / 1e12 for flops, row in zip(useful_flops, sample, strict=True)]
            axis.plot(positions, measured, color="#007C83", marker="o", markersize=3.5, label="Measured")
            axis.plot(positions, predicted, color="#D55E00", marker="s", markersize=3.2, linestyle="--", label="Predicted")
            axis.set_title(f"TPU {sku}: vary {varied}", loc="left", fontweight="bold")
            axis.set_ylabel("Useful TFLOP/s")
            axis.set_xticks(list(positions), [str(row[varied]) for row in sample], rotation=55, ha="right", fontsize=8)
            for position in (2.5, 6.5, 10.5, 13.5):
                axis.axvline(position, color="#888888", linewidth=0.6, alpha=0.35)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.subplots_adjust(top=0.84, bottom=0.18, hspace=0.38, wspace=0.20)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=2)
    fig.suptitle(
        "Tile-boundary probes: output versus contraction dimension",
        y=0.99,
        fontweight="bold",
    )
    fig.supxlabel("Varied dimension (groups surround MXU-relevant boundaries)", y=0.03)
    path = OUT / "tile_boundaries.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def error_distributions(rows: list[dict]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    for axis, sku in zip(axes, ("v5e", "v6e"), strict=True):
        sample = [row for row in rows if row["sku"] == sku]
        for family in FAMILIES:
            errors = sorted(
                abs(row["predicted_s"] / row["median_s"] - 1) * 100
                for row in sample
                if row["family"] == family
            )
            cumulative = [(index + 1) / len(errors) * 100 for index in range(len(errors))]
            axis.step(errors, cumulative, where="post", color=COLORS[family], linewidth=1.8, label=LABELS[family])
        axis.axvline(25, color="#555555", linewidth=1, linestyle=":")
        axis.set_xlim(0, 75)
        axis.set_ylim(0, 100)
        axis.set_title(f"TPU {sku}", loc="left", fontweight="bold")
        axis.set_xlabel("Absolute relative error (%)")
    axes[0].set_ylabel("Observations at or below error (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(top=0.78, bottom=0.14, wspace=0.20)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.90), ncol=5)
    fig.suptitle("Error distributions by workload family", y=0.99, fontweight="bold")
    axes[1].text(
        0.98, 0.03, "Dotted line: 25% error", transform=axes[1].transAxes,
        ha="right", color="#555555", fontsize=9,
    )
    path = OUT / "error_distributions.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(exist_ok=True)
    _style()
    rows = evaluate.load_results(ROOT / "data" / "measurements.csv", ROOT / "specs")
    for path in (tile_boundaries(rows), prediction_scatter(rows), error_distributions(rows)):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
