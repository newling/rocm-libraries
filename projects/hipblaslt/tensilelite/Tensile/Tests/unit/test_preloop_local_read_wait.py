# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
from unittest.mock import MagicMock

import pytest

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
    "use_custom_schedule, force_subiteration, drain_emitted, expected_waits",
    [
        (True, True, False, 1),
        (True, True, True, 0),
        (True, False, False, 0),
        (False, True, False, 0),
    ],
)
def test_preloop_local_read_wait_predicate(
    use_custom_schedule, force_subiteration, drain_emitted, expected_waits
):
    """An undrained forced-subiteration custom schedule needs a fallback."""
    kernel = {
        "UseCustomMainLoopSchedule": use_custom_schedule,
        "ForceUnrollSubIter": force_subiteration,
    }

    module = KernelWriter.preLoopLocalReadWait(MagicMock(), kernel, drain_emitted)

    assert str(module).count("s_waitcnt lgkmcnt(0)") == expected_waits


def test_forced_subiteration_preloop_reads_are_drained_before_loop_entry():
    """Drain the gfx950 regression kernel's 16 reads before any branch."""
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
