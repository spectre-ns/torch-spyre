# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal compare_with_cpu helper for standalone coarse-tile scripts.

Extracted from tests/inductor/utils_inductor.py — only the functions
needed by the standalone scripts are included.
"""

import os

import torch
from torch._inductor.utils import run_and_get_code

DEVICE = torch.device("spyre")


def _to_cpu(result, device):
    if isinstance(result, torch.Tensor):
        assert result.device.type == device.type, (
            f"Output not on expected device. Expected {device}, got {result.device}"
        )
        return result.cpu()
    elif isinstance(result, (tuple, list)):
        cpu_items = [_to_cpu(r, device) for r in result]
        return type(result)(cpu_items)
    else:
        return result


def _compile_and_run(
    fn,
    args,
    device,
    backend="inductor",
    needs_device=False,
    compile=True,
    source_check=None,
):
    torch._dynamo.reset_code_caches()
    torch._inductor.codecache.FxGraphCache.clear()
    device = torch.device(device) if isinstance(device, str) else device
    device_args = [
        arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args
    ]
    device_kwargs = {"device": device} if needs_device else {}

    if compile:
        comp_func = torch.compile(fn, backend=backend)
        if source_check is not None and device == "spyre":
            result, source_codes = run_and_get_code(
                comp_func, *device_args, **device_kwargs
            )
            if len(source_codes) > 0:
                source_check(source_codes[0])
        else:
            result = comp_func(*device_args, **device_kwargs)
    else:
        result = fn(*device_args, **device_kwargs)

    if isinstance(result, (int, float)):
        return result

    return _to_cpu(result, device)


def _assert_results_close(actual, expected, atol, rtol, comparison_name):
    if isinstance(actual, (tuple, list)):
        assert isinstance(actual, type(expected)) and len(actual) == len(expected), (
            f"{comparison_name} mismatch: result types or lengths differ "
            f"(actual: {type(actual).__name__}[{len(actual)}], "
            f"expected: {type(expected).__name__}[{len(expected)}])"
        )
        for i, (a, e) in enumerate(zip(actual, expected)):
            _assert_results_close(a, e, atol, rtol, f"{comparison_name}[{i}]")
    else:
        torch.testing.assert_close(
            actual,
            expected,
            equal_nan=True,
            atol=atol,
            rtol=rtol,
            msg=lambda msg: f"{comparison_name} mismatch\n\n{msg}\n",
        )


def compare_with_cpu(
    fn,
    *args,
    atol=0.1,
    rtol=0.1,
    needs_device=False,
    cpu_compile=None,
    target=None,
    run_eager=True,
    run_compile=True,
    source_check=None,
    clone_inputs=False,
):
    """Compare Spyre execution against CPU reference.

    ``run_compile`` and ``run_eager`` select which Spyre paths to exercise.
    At least one must be True.  Set ``cpu_compile=True`` (or env var
    TEST_COMPARE_CPU_COMPILE) to also compare against a compiled CPU run.
    """
    if cpu_compile is None:
        cpu_compile = bool(os.getenv("TEST_COMPARE_CPU_COMPILE"))

    def get_args():
        if not clone_inputs:
            return args
        return [arg.clone() if isinstance(arg, torch.Tensor) else arg for arg in args]

    cpu_result = fn(*get_args())

    modes = tuple(
        compiled
        for compiled, enabled in ((True, run_compile), (False, run_eager))
        if enabled
    )
    if not modes:
        raise ValueError("At least one of run_compile or run_eager must be True")

    for compiled in modes:
        mode = "compiled" if compiled else "eager"
        spyre_result = (
            target
            if target is not None
            else _compile_and_run(
                fn,
                get_args(),
                DEVICE,
                needs_device=needs_device,
                compile=compiled,
                source_check=source_check,
            )
        )

        _assert_results_close(
            spyre_result, cpu_result, atol, rtol, f"{mode} spyre <-> cpu"
        )

        if cpu_compile:
            cpu_other_result = _compile_and_run(
                fn, get_args(), "cpu", needs_device=needs_device, compile=True
            )
            _assert_results_close(
                spyre_result,
                cpu_other_result,
                atol,
                rtol,
                f"{mode} spyre <-> {mode} cpu",
            )
