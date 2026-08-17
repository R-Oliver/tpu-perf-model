import dataclasses

import pytest

from hardware import HardwareSpec, load_hardware
import model
import shapes


V5E = load_hardware("v5e")
V6E = load_hardware("v6e")


def test_hardware_contract_contains_only_prediction_inputs():
    assert {field.name for field in dataclasses.fields(HardwareSpec)} == {
        "name",
        "peak_flops",
        "hbm_bandwidth_bytes",
        "mxu_dim",
        "vmem_bytes",
        "tile_multiple",
    }


def test_layout_padding_uses_dtype_specific_physical_tiles():
    assert model.layout_bytes(33, 128, "bf16", V5E) == 40 * 128 * 2
    assert model.layout_bytes(33, 128, "int8", V5E) == 64 * 128
    assert model.layout_bytes(4096, 130, "bf16", V5E) == 4096 * 256 * 2


def test_arithmetic_rounds_only_the_contraction_dimension():
    problem = model.GemmProblem(129, 257, 129, "bf16")
    assert model.arithmetic_flops(problem, V5E) == 2 * 129 * 257 * 256
    assert model.arithmetic_flops(problem, V6E) == 2 * 129 * 257 * 256


def test_accumulator_residency_streams_large_inputs_once():
    problem = model.GemmProblem(8, 4096, 16384, "bf16")
    compulsory = model.compulsory_bytes(problem, V5E)
    assert compulsory > V5E.vmem_bytes
    assert model.accumulator_resident_bytes(problem, V5E) < V5E.vmem_bytes
    assert model.traffic_bytes(problem, V5E) == compulsory


def test_nonresident_large_gemm_rereads_inputs_but_writes_output_once():
    problem = model.GemmProblem(16384, 16384, 16384, "bf16")
    compulsory = model.compulsory_bytes(problem, V5E)
    assert model.accumulator_resident_bytes(problem, V5E) > V5E.vmem_bytes
    assert model.traffic_bytes(problem, V5E) > compulsory


def test_pipeline_endpoints_and_intermediate_depth():
    assert model.pipeline_time(1.0, 0.5, 1) == pytest.approx(1.5)
    assert model.pipeline_time(1.0, 0.5, 4) == pytest.approx(1.125)
    assert model.pipeline_time(1.0, 0.5, 10_000) == pytest.approx(1.0, rel=1e-4)


@pytest.mark.parametrize(
    "arguments",
    [
        (0, 128, 128, "bf16"),
        (128, -1, 128, "bf16"),
        (128, 128, True, "bf16"),
        (128, 128, 128, "f32"),
    ],
)
def test_invalid_problems_are_rejected(arguments):
    with pytest.raises(ValueError):
        model.GemmProblem(*arguments)


def test_invalid_hardware_and_runtime_inputs_are_rejected():
    values = dataclasses.asdict(V5E)
    values["hbm_bandwidth_bytes"] = 0
    with pytest.raises(ValueError):
        HardwareSpec(**values)
    with pytest.raises(ValueError):
        model.predict(model.GemmProblem(128, 128, 128), V5E, launch_overhead_s=-1)
    with pytest.raises(ValueError):
        model.pipeline_time(1, 1, 0)


def test_shape_generator_is_deterministic_and_preserves_experimental_families():
    first = shapes.build_sweep(V5E, seed=0)
    second = shapes.build_sweep(V5E, seed=0)
    assert first == second
    assert len(first) == 450
    assert {case.family for case in first} == set(shapes.FAMILIES)
