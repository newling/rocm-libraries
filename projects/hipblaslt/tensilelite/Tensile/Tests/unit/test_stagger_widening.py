# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Check the signedness of both stagger products for each supported tensor.

An ordinary increment of 0x80000000 represents +2 GiB; a mirrored increment
of 0xffffffe0 represents -32 bytes. The same unsigned widening cannot serve
both. Mirroring a free dimension must not change the unroll increment's sign.
"""

import re

import pytest

from rocisa import rocIsa
from rocisa.enum import RegisterType
from rocisa.register import RegisterPool

from Tensile.Common.Types import DebugConfig
from Tensile.KernelWriterAssembly import KernelWriterAssembly
from Tensile.Toolchain.Validators import validateToolchain


pytestmark = pytest.mark.unit

_TENSORS = ("A", "B", "Metadata", "MXSA", "MXSB")


@pytest.fixture
def writer():
    # Exercise the real emitter and register allocator, without unrelated
    # whole-kernel setup. The unit conftest restores rocisa state afterwards.
    ri = rocIsa.getInstance()
    ri.init((9, 4, 2), validateToolchain("amdclang++"))
    ri.setKernel((9, 4, 2), 64)
    result = KernelWriterAssembly(None, DebugConfig())
    result.states.version = (9, 4, 2)
    result.states.unrollIdx = 0
    result.states.asmCaps = ri.getAsmCaps()
    result.states.regCaps = ri.getRegCaps()
    result.states.use64bShadowLimit = True
    result.states.use64bShadowLimitMX = True
    result.states.staggerUCode = True
    result.sgprPool = RegisterPool(0, RegisterType.Sgpr, defaultPreventOverflow=False)
    result.vgprPool = RegisterPool(0, RegisterType.Vgpr, defaultPreventOverflow=False)
    result.sgprPool.add(0, 32, "stagger test")
    result.vgprPool.add(0, 8, "stagger test")
    return result


def _kernel(tensor, mirror_dims, sparse):
    problem = {"IndicesSummation": [3], "Sparse": sparse}
    problem.update({f"MirrorDims{tc}": [] for tc in _TENSORS})
    problem[f"MirrorDims{tensor}"] = mirror_dims
    if sparse:
        # Solution derivation copies the sparse operand's mirror list.
        problem["MirrorDimsMetadata"] = list(mirror_dims)
    return {
        "ProblemType": problem,
        "BufferLoad": True,
        "PrefetchGlobalRead": 1,
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "DirectToVgprSparseMetadata": bool(sparse),
        "enableTDMMetadata": False,
        "GlobalSplitU": 0,
        "DepthU": 32,
    }


@pytest.mark.parametrize("tensor,sparse", [(tc, 0) for tc in _TENSORS] + [("A", 1), ("B", 2)])
@pytest.mark.parametrize(
    "mirror_dims,expected_opcode",
    [([], "s_mul_hi_u32"), ([3], "s_mul_hi_i32"), ([0], "s_mul_hi_u32")],
    ids=["forward", "mirrored-unroll", "mirrored-free-dimension"],
)
def test_stagger_widening_follows_unroll_direction(
    writer, tensor, sparse, mirror_dims, expected_opcode
):
    kernel = _kernel(tensor, mirror_dims, sparse)
    writer.states.kernel = kernel
    params = {"tensorChar": tensor, "isA": tensor == "A", "isB": tensor == "B"}
    source = str(writer.calculateStagger(kernel, params))

    # Require both the offset and loop-span products, per tensor, so removing
    # either multiply cannot leave a vacuous signedness check.
    offsets = [
        line for line in source.splitlines()
        if "stagger byte offset" in line and "s_mul_hi_" in line
    ]
    spans = [
        line for line in source.splitlines()
        if "Number of bytes accessed by the unroll loop" in line and "s_mul_hi_" in line
    ]
    expected_tensors = [tensor, "Metadata"] if sparse else [tensor]
    assert len(offsets) == len(spans) == len(expected_tensors)
    for line in offsets + spans:
        assert line.split()[0] == expected_opcode, line
    for tc, line in zip(expected_tensors, spans):
        assert re.search(rf"s\[sgprWrapU{tc}\+1\]", line), line

    # Both register pools must release the temporary multiplication registers.
    writer.sgprPool.checkFinalState()
    writer.vgprPool.checkFinalState()
