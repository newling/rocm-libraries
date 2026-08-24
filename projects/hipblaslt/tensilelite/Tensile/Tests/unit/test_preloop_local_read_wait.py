# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os

import pytest

from rocisa.instruction import SWaitCnt
from Tensile.KernelWriter import KernelWriter
from config_harness import emit_kernels_from_config


pytestmark = pytest.mark.unit

_ARCH = "gfx950"
_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "..",
    "common",
    "gemm",
    _ARCH,
    "preloop_local_read_wait.yaml",
)


@pytest.mark.parametrize(
    "use_custom_schedule, force_unroll_sub_iter, num_iters_plr, wait_already_emitted",
    [
        (False, True, 1, False),
        (True, False, 1, False),
        (True, True, 0, False),
        (True, True, 1, True),
    ],
)
def test_preloop_local_read_wait_skips_unaffected_paths(
    use_custom_schedule, force_unroll_sub_iter, num_iters_plr,
    wait_already_emitted,
):
    kernel = {
        "UseCustomMainLoopSchedule": use_custom_schedule,
        "ForceUnrollSubIter": force_unroll_sub_iter,
    }

    module = KernelWriter.preLoopLocalReadWait(
        kernel, num_iters_plr, wait_already_emitted
    )

    assert module.flatitems() == []


def test_preloop_local_read_wait_drains_unpacked_cms_prefetch():
    kernel = {
        "UseCustomMainLoopSchedule": True,
        "ForceUnrollSubIter": True,
    }

    module = KernelWriter.preLoopLocalReadWait(kernel, 1, False)
    instructions = module.flatitems()

    assert len(instructions) == 1
    assert isinstance(instructions[0], SWaitCnt)
    assert instructions[0].dscnt == 0
    assert instructions[0].vlcnt == -1
    assert instructions[0].vscnt == -1
    assert instructions[0].comment == "Wait for pre-loop local reads to complete"


def test_forced_subiteration_preloop_reads_are_drained_before_loop_entry():
    """Drain all 16 regression-kernel reads before any loop-entry branch."""
    results = emit_kernels_from_config(_CONFIG, limit=1, arch=_ARCH)
    assert len(results) == 1
    _basename, source, error = results[0]
    assert error == 0

    loop_entry = source.index("label_openLoopL:")
    preceding_barrier = source.rfind("s_barrier", 0, loop_entry)
    first_branch = source.index("s_cbranch", loop_entry)
    first_mfma = source.index("v_mfma", loop_entry)
    preloop = source[preceding_barrier:loop_entry]

    preloop_reads = [line for line in preloop.splitlines() if "ds_read_b128" in line]
    assert len(preloop_reads) == 16
    assert preloop.count("s_waitcnt lgkmcnt(0)") == 1
    assert preloop.rfind("ds_read_b128") < preloop.index("s_waitcnt lgkmcnt(0)")
    assert loop_entry < first_branch < first_mfma
