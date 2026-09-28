# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import copy
import json
import zlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import msgpack
import pytest
import yaml

from Tensile import LibraryIO
from Tensile.SolutionFingerprint import PREFIX, SolutionFingerprints, solution_fingerprint

pytestmark = pytest.mark.unit


@pytest.fixture
def solution():
    return {
        "index": 123,
        "name": "same_solution_name",
        "kernelName": "same_kernel_name",
        "sizeMapping": {"workGroupMapping": 1, "workGroupMappingXCC": 2, "globalSplitU": 1},
        "problemPredicate": {"type": "TruePred"},
    }


def fingerprint(solution, arch="gfx942", main="main", helpers=("helper",)):
    return solution_fingerprint(solution, arch, main, helpers)


@pytest.mark.parametrize(
    "setting", ["workGroupMapping", "workGroupMappingXCC", "globalSplitU", "futureLaunchSetting"]
)
def test_same_names_changed_launch_defaults_are_different(solution, setting):
    changed = copy.deepcopy(solution)
    changed["sizeMapping"][setting] = 8
    assert fingerprint(changed) != fingerprint(solution)


def test_identity_is_independent_of_index_ranking_and_dictionary_order(solution):
    changed = dict(reversed(list(solution.items())))
    changed.update(
        index=999,
        libraryLogicIndex=42,
        ideals={0: 123.0},
        linearModel={"slope": 4},
        fingerprint="old",
    )
    assert fingerprint(changed) == fingerprint(solution)
    assert fingerprint(solution).startswith(PREFIX)
    assert len(fingerprint(solution)) == len(PREFIX) + 64


def test_device_code_and_target_are_part_of_identity(solution):
    original = fingerprint(solution)
    assert fingerprint(solution, main="changed") != original
    assert fingerprint(solution, helpers=("changed",)) != original
    assert fingerprint(solution, arch="gfx1250v0") != fingerprint(solution, arch="gfx1250v1")
    assert fingerprint(solution, helpers=("a", "b")) == fingerprint(solution, helpers=("b", "a"))


def make_library(solution, bundle):
    # Exercise the exact serialized document passed to LibraryIO; originalSolution
    # is generator-only information used to locate the code-object bundle.
    original = {} if bundle is None else {"codeObjectFile": bundle}
    return SimpleNamespace(
        solutions={solution["index"]: SimpleNamespace(originalSolution=original)},
        state=lambda: {"solutions": [copy.deepcopy(solution)], "library": {"type": "Matching"}},
    )


@pytest.mark.parametrize("bundle", [None, "TensileLibrary_lazy_gfx942_shard"])
@pytest.mark.parametrize("format", ["yaml", "json", "msgpack", "msgpack-indexed"])
def test_generated_files_are_hashed_once_then_serialized(
    tmp_path, solution, bundle, format, monkeypatch
):
    directory = tmp_path / "gfx942"
    directory.mkdir()
    main = directory / ((bundle or "TensileLibrary_gfx942") + ".co")
    helper = directory / "Kernels.so-000-gfx942.hsaco"
    main.write_bytes(b"compiled main bundle")
    helper.write_bytes(b"compiled helper bundle")
    library = make_library(solution, bundle)
    identities = SolutionFingerprints([main, helper, main])
    assert len(identities.digests) == 2
    document = identities.library_state(library, "gfx942", directory)
    value = document["solutions"][0]["fingerprint"]

    # Serialization does no binary I/O, even across many solutions/shards.
    with monkeypatch.context() as patch:
        patch.setattr(type(main), "open", lambda *a, **kw: pytest.fail("binary reread"))
        assert identities.library_state(library, "gfx942", directory) == document

    LibraryIO.write(str(tmp_path / "library"), document, format)
    if format == "yaml":
        round_trip = yaml.safe_load((tmp_path / "library.yaml").read_text())["solutions"][0]
    elif format == "json":
        round_trip = json.loads((tmp_path / "library.json").read_text())["solutions"][0]
    else:
        packed = msgpack.unpackb(
            zlib.decompress((tmp_path / "library.dat.zlib").read_bytes()), raw=False
        )
        round_trip = (
            msgpack.unpackb(packed["solutions_blob"], raw=False)
            if format == "msgpack-indexed"
            else packed["solutions"][0]
        )
    assert round_trip["fingerprint"] == value
    assert "fingerprint" not in solution

    main.write_bytes(b"modified code, unchanged names")
    rebuilt = SolutionFingerprints([main, helper]).library_state(library, "gfx942", directory)
    assert rebuilt["solutions"][0]["fingerprint"] != value
    main.write_bytes(b"compiled main bundle")
    helper.write_bytes(b"modified helper only")
    rebuilt = SolutionFingerprints([main, helper]).library_state(library, "gfx942", directory)
    assert rebuilt["solutions"][0]["fingerprint"] != value


def test_missing_outputs_fail_instead_of_using_stale_files(tmp_path, solution):
    main = tmp_path / "TensileLibrary_gfx942.co"
    helper = tmp_path / "Kernels.so-000-gfx942.hsaco"
    main.write_bytes(b"stale main")
    helper.write_bytes(b"stale helper")
    library = make_library(solution, None)
    for outputs in ([], [main], [helper]):
        with pytest.raises(RuntimeError, match="missing generated"):
            SolutionFingerprints(outputs).library_state(library, "gfx942", tmp_path)


def test_install_location_does_not_affect_identity(tmp_path, solution):
    values = []
    for install in ("first", "second"):
        directory = tmp_path / install / "gfx942"
        directory.mkdir(parents=True)
        main = directory / "TensileLibrary_gfx942.co"
        helper = directory / "Kernels.hsaco"
        main.write_bytes(b"same main")
        helper.write_bytes(b"same helper")
        values.append(
            SolutionFingerprints([main, helper]).library_state(
                make_library(solution, None), "gfx942", directory
            )["solutions"][0]["fingerprint"]
        )
    assert values[0] == values[1]


def test_kernel_writer_reports_exact_generated_artifacts(tmp_path, monkeypatch):
    from Tensile.TensileCreateLibrary import Run

    main, helper = tmp_path / "main.co", tmp_path / "helper.hsaco"
    monkeypatch.setattr(Run, "buildAssemblyCodeObjectFiles", lambda *a, **kw: [main])
    monkeypatch.setattr(Run, "buildSourceCodeObjectFiles", lambda *a, **kw: [str(helper)])
    monkeypatch.setattr(Run, "ParallelMap2", lambda *a, **kw: [])
    monkeypatch.setattr(Run, "writeHelpers", lambda *a, **kw: None)
    monkeypatch.setattr(Run, "rocisa", MagicMock())
    outputs = []
    Run.writeSolutionsAndKernelsTCL(
        tmp_path,
        MagicMock(),
        MagicMock(),
        [],
        [],
        [],
        MagicMock(),
        ["gfx942"],
        generatedCodeObjects=outputs,
    )
    assert outputs == [main, str(helper)]
