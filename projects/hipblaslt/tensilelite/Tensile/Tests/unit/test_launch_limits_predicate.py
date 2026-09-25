# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest

from Tensile.Contractions import TaskPredicate


@pytest.mark.unit
@pytest.mark.parametrize("stream_k", [None, 0, 1, 2, 3, 4, 5])
def test_launch_limits_attached_to_every_kernel(stream_k):
    state = {} if stream_k is None else {"StreamK": stream_k}
    predicate = TaskPredicate.FromOriginalState(state, None)
    assert predicate.tag == "LaunchLimits"
