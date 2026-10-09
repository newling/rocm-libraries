# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from component_ci import detect_changed_components


def test_moved_solution_uniqueness_test_triggers_its_ci_gate():
    changed = {
        "projects/hipblaslt/tensilelite/tensilelite/Tests/unit/"
        "test_solution_uid_uniqueness.py"
    }

    assert detect_changed_components(changed)["hipblaslt_library_uniqueness"]
