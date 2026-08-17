"""Thin command-line interface for prediction, collection, and reproduction."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import analysis
import evaluate
from hardware import INPUT_DTYPES, load_hardware
import model
import shapes


ROOT = Path(__file__).parent


def predict_command(args) -> None:
    hardware = load_hardware(args.sku)
    problem = model.GemmProblem(args.M, args.N, args.K, args.dtype)
    work = model.estimate(problem, hardware)
    seconds = model.predict(
        problem, hardware, launch_overhead_s=args.launch_overhead_us * 1e-6
    )
    useful_flops = 2.0 * problem.M * problem.N * problem.K
    print(f"{hardware.name} {problem.M}x{problem.N}x{problem.K} {problem.dtype}")
    print(f"predicted runtime: {seconds * 1e3:.4f} ms")
    print(f"useful throughput: {useful_flops / seconds / 1e12:.1f} TFLOP/s")
    print(
        f"work: {work.flops:.4g} FLOPs, {work.bytes_moved:.4g} bytes, "
        f"{work.n_blocks} pipeline blocks"
    )


def shapes_command(args) -> None:
    hardware = load_hardware(args.sku)
    cases = shapes.build_sweep(hardware, args.seed)
    counts = {
        family: sum(case.family == family for case in cases)
        for family in shapes.FAMILIES
    }
    print(f"{hardware.name}: {len(cases)} deterministic cases")
    for family, count in counts.items():
        print(f"  {family:12s} {count}")
    if args.out:
        with args.out.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("family", "M", "N", "K", "dtype"))
            for case in cases:
                p = case.problem
                writer.writerow((case.family, p.M, p.N, p.K, p.dtype))
        print(f"wrote {args.out}")


def benchmark_command(args) -> None:
    import benchmark

    hardware = load_hardware(args.sku)
    families = set(args.families.split(",")) if args.families else None
    benchmark.run_benchmark(
        args.sku,
        hardware,
        args.output,
        args.runs,
        seed=args.seed,
        warmup=args.warmup,
        iterations=args.iterations,
        families=families,
    )


def evaluate_command(args) -> None:
    argv = ["--measurements", str(args.measurements), "--specs", str(args.specs)]
    if args.runs:
        argv += ["--runs", str(args.runs)]
    evaluate.main(argv)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="perfmodel", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    predict_parser = commands.add_parser("predict", help="predict one GEMM")
    predict_parser.add_argument("--sku", required=True)
    predict_parser.add_argument("-M", type=int, required=True)
    predict_parser.add_argument("-N", type=int, required=True)
    predict_parser.add_argument("-K", type=int, required=True)
    predict_parser.add_argument("--dtype", choices=INPUT_DTYPES, default="bf16")
    predict_parser.add_argument(
        "--launch-overhead-us",
        type=float,
        default=0.0,
        help="optional run calibration in microseconds",
    )
    predict_parser.set_defaults(function=predict_command)

    shapes_parser = commands.add_parser("shapes", help="inspect the generated sweep")
    shapes_parser.add_argument("--sku", required=True)
    shapes_parser.add_argument("--seed", type=int, default=0)
    shapes_parser.add_argument("--out", type=Path)
    shapes_parser.set_defaults(function=shapes_command)

    bench_parser = commands.add_parser("benchmark", help="measure on an attached TPU")
    bench_parser.add_argument("--sku", required=True)
    bench_parser.add_argument(
        "--output", type=Path, default=ROOT / "data" / "measurements.csv"
    )
    bench_parser.add_argument("--runs", type=Path, default=ROOT / "data" / "runs.json")
    bench_parser.add_argument("--seed", type=int, default=0)
    bench_parser.add_argument("--warmup", type=int, default=5)
    bench_parser.add_argument("--iterations", type=int, default=20)
    bench_parser.add_argument("--families", help="comma-separated subset")
    bench_parser.set_defaults(function=benchmark_command)

    eval_parser = commands.add_parser("evaluate", help="print final-model tables")
    eval_parser.add_argument(
        "--measurements", type=Path, default=ROOT / "data" / "measurements.csv"
    )
    eval_parser.add_argument("--runs", type=Path)
    eval_parser.add_argument("--specs", type=Path, default=ROOT / "specs")
    eval_parser.set_defaults(function=evaluate_command)

    analysis_parser = commands.add_parser(
        "ablate", help="reproduce baselines, calibration, and ablations"
    )
    analysis_parser.set_defaults(function=lambda args: analysis.main())

    figures_parser = commands.add_parser("figures", help="regenerate report figures")
    figures_parser.set_defaults(
        function=lambda args: __import__("figures").main()
    )

    args = parser.parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
