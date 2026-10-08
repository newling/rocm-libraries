# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Verify SolutionUID uniqueness across shipped logic YAML files."""

from __future__ import annotations

import importlib.util
import multiprocessing as mp
import os
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Tuple

import pytest
import yaml

try:
    from yaml import CSafeLoader as yamlLoader
except ImportError:
    from yaml import SafeLoader as yamlLoader


def _load_solution_id_gen() -> ModuleType:
    """Load SolutionIdGen without importing ``Tensile.Common``.

    The ``library-uniqueness`` tox env skips installing Tensile and rocisa.
    ``from Tensile.Common.SolutionIdGen import ...`` still executes
    ``Tensile.Common.__init__``, which imports rocisa.

    Args:
        None.

    Returns:
        The loaded ``SolutionIdGen`` module.

    Raises:
        FileNotFoundError: If ``SolutionIdGen.py`` is missing.
        ImportError: If the module spec or loader cannot be created.
    """
    module_path = Path(__file__).resolve().parents[2] / "Common" / "SolutionIdGen.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"SolutionIdGen.py not found: {module_path}")
    spec = importlib.util.spec_from_file_location("tensile_solution_id_gen", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load SolutionIdGen from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_solution_id_gen = _load_solution_id_gen()
decode_solution_uid = _solution_id_gen.decode_solution_uid
encode_solution_uid = _solution_id_gen.encode_solution_uid

pytestmark = pytest.mark.unit

DEFAULT_LOGIC_ROOT = (
    Path(__file__).resolve().parents[4]
    / "library"
    / "src"
    / "amd_detail"
    / "rocblaslt"
    / "src"
    / "Tensile"
    / "Logic"
)

# The shipped logic corpus lives in the hipBLASLt product tree, not in
# tensilelite, so it is absent whenever this file is run from installed
# tensilelite test artifacts. The enforcement point is the library-uniqueness
# component job, which checks out the source tree; this copy is a
# convenience/local-dev signal otherwise -- hence ``skipif`` (an unmet
# precondition), not ``xfail`` (an expected failure).
_LOGIC_ROOT = Path(os.environ.get("HIPBLASLT_LOGIC_ROOT") or DEFAULT_LOGIC_ROOT)

_needs_logic_dir = pytest.mark.skipif(
    not _LOGIC_ROOT.is_dir(),
    reason="Logic files not found: https://github.com/ROCm/rocm-libraries/issues/7481",
)

STRICT_ENV = "HIPBLASLT_REQUIRE_SOLUTION_UID"


def _collect_yaml_files(logic_root: Path) -> List[Path]:
    """Return logic YAML files that contain solution definitions.

    Args:
        logic_root: Root directory containing arch/datatype logic trees.

    Returns:
        Sorted list of YAML file paths.
    """
    asm_full = logic_root / "asm_full"
    if asm_full.is_dir():
        return sorted(asm_full.rglob("*.yaml"))
    return sorted(logic_root.rglob("*.yaml"))


def test_collect_yaml_files_includes_all_yaml_names(tmp_path: Path) -> None:
    """Verify discovery does not require the ``_UserArgs.yaml`` suffix.

    Args:
        tmp_path: Pytest temporary-directory fixture.

    Returns:
        None.

    Raises:
        AssertionError: If any YAML file under ``asm_full`` is omitted.
    """
    asm_full = tmp_path / "asm_full"
    asm_full.mkdir()
    user_args = asm_full / "logic_UserArgs.yaml"
    conventional = asm_full / "logic.yaml"
    ignored = asm_full / "README.txt"
    user_args.touch()
    conventional.touch()
    ignored.touch()

    assert _collect_yaml_files(tmp_path) == [conventional, user_args]


def _compose_uid_node(loader, anchors: dict, scope: str | None) -> yaml.Node | None:
    """Compose identity fields without retaining the rest of a logic file.

    Consumes one node. ``scope`` selects the logic root, its solutions list,
    one solution mapping, all fields, or no fields (``None``). Returns the
    retained YAML node, or ``None`` for a discarded node; ``anchors`` retains
    nodes that later aliases may reference.

    The YAML parser handles quoting, comments and flow collections. Anchored
    nodes and merge values are retained in full so aliases and inherited fields
    have the same meaning as in a normal safe load. Unanchored tuning tables
    and kernel parameters are consumed without building their object trees.
    """
    event = loader.get_event()
    if isinstance(event, yaml.AliasEvent):
        if event.anchor not in anchors:
            raise ValueError(f"Undefined YAML alias: {event.anchor}")
        return anchors[event.anchor]

    if event.anchor is not None:
        if event.anchor in anchors:
            raise ValueError(f"Duplicate YAML anchor: {event.anchor}")
        scope = "all"

    node = None
    if isinstance(event, yaml.ScalarEvent):
        if scope is not None:
            tag = event.tag or loader.resolve(yaml.ScalarNode, event.value, event.implicit)
            node = yaml.ScalarNode(tag, event.value, event.start_mark, event.end_mark)
        if event.anchor is not None:
            anchors[event.anchor] = node
        return node

    mapping = isinstance(event, yaml.MappingStartEvent)
    node_type = yaml.MappingNode if mapping else yaml.SequenceNode
    end_event = yaml.MappingEndEvent if mapping else yaml.SequenceEndEvent
    if scope is not None:
        tag = event.tag or loader.resolve(node_type, None, event.implicit)
        node = node_type(tag, [], event.start_mark, event.end_mark)
    if event.anchor is not None:
        anchors[event.anchor] = node

    index = 0
    while not loader.check_event(end_event):
        key = _compose_uid_node(loader, anchors, "all" if scope else None) if mapping else None
        child_scope = None
        if scope == "all":
            child_scope = "all"
        elif mapping and key is not None:
            if key.tag == "tag:yaml.org,2002:merge":
                child_scope = "all"
            elif key.tag == "tag:yaml.org,2002:str":
                if scope == "logic" and key.value == "Solutions":
                    child_scope = "solutions"
                elif scope == "solution" and key.value in {"SolutionIndex", "SolutionUID"}:
                    child_scope = "all"
        elif not mapping:
            if scope == "logic" and index == 5:
                child_scope = "solutions"
            elif scope == "solutions":
                child_scope = "solution"

        child = _compose_uid_node(loader, anchors, child_scope)
        if node is not None:
            if mapping:
                if child_scope is not None:
                    node.value.append((key, child))
            else:
                # Preserve positions in legacy list-format logic.
                node.value.append(
                    child if child_scope is not None else yaml.ScalarNode("tag:yaml.org,2002:null", "")
                )
        index += 1

    node_end = loader.get_event()
    if node is not None:
        node.end_mark = node_end.end_mark
    return node


def _scan_yaml_file(yaml_path: Path) -> Tuple[List[Tuple[int, str, int]], int]:
    """Extract SolutionUID entries from one logic YAML.

    Args:
        yaml_path: Path to a logic YAML file.

    Returns:
        Tuple of (entries, missing_count) where each entry is
        ``(SolutionUID, yaml_path, SolutionIndex)`` and
        ``missing_count`` counts solutions without the field.

    Raises:
        TypeError: If a UID value is not a string.
        ValueError: If a UID is malformed, zero, out of range, or an index
            value is not an integer.
    """
    with yaml_path.open(encoding="utf-8") as handle:
        loader = yamlLoader(handle)
        try:
            loader.get_event()  # StreamStartEvent
            if loader.check_event(yaml.StreamEndEvent):
                return [], 0
            loader.get_event()  # DocumentStartEvent
            node = _compose_uid_node(loader, {}, "logic")
            loader.get_event()  # DocumentEndEvent
            if not loader.check_event(yaml.StreamEndEvent):
                raise ValueError(f"Expected one YAML document: {yaml_path}")
            data = loader.construct_document(node)
        finally:
            loader.dispose()

    if isinstance(data, dict):
        solutions = data.get("Solutions", [])
    elif isinstance(data, list) and len(data) > 5:
        solutions = data[5]
    else:
        solutions = []
    if not isinstance(solutions, list):
        raise ValueError(f"Expected a Solutions list: {yaml_path}")

    entries: List[Tuple[int, str, int]] = []
    missing = 0
    for solution in solutions:
        if not isinstance(solution, dict) or "SolutionIndex" not in solution:
            raise ValueError(f"Expected a solution mapping with SolutionIndex: {yaml_path}")
        local_index = solution["SolutionIndex"]
        if type(local_index) is not int:
            raise ValueError(f"SolutionIndex must be an integer: {yaml_path}")
        if "SolutionUID" not in solution:
            missing += 1
            continue
        try:
            uid = decode_solution_uid(solution["SolutionUID"])
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"{yaml_path} (SolutionIndex={local_index}): {exc}") from exc
        if uid == 0:
            raise ValueError(
                f"Stored SolutionUID must not be zero: {yaml_path} "
                f"(SolutionIndex={local_index})"
            )
        entries.append((uid, str(yaml_path), local_index))
    return entries, missing


def collect_solution_uids(
    logic_root: Path,
    *,
    process_count: int,
) -> Tuple[List[Tuple[int, str, int]], int]:
    """Collect SolutionUID values using multiple processes.

    Args:
        logic_root: Root directory containing logic YAML files.
        process_count: Number of worker processes to use.

    Returns:
        Tuple of all ``(uid, yaml_path, local_index)`` entries and total missing count.
    """
    yaml_files = _collect_yaml_files(logic_root)
    if not yaml_files:
        return [], 0

    process_count = max(1, min(process_count, len(yaml_files)))
    ctx = mp.get_context("spawn")
    all_entries: List[Tuple[int, str, int]] = []
    total_missing = 0
    with ctx.Pool(process_count) as pool:
        for entries, missing in pool.imap(_scan_yaml_file, yaml_files):
            all_entries.extend(entries)
            total_missing += missing
    return all_entries, total_missing


def find_duplicate_uids(
    entries: List[Tuple[int, str, int]],
) -> Dict[int, List[Tuple[str, int]]]:
    """Find duplicate SolutionUID values.

    Args:
        entries: Output from :func:`collect_solution_uids`.

    Returns:
        Mapping from duplicate UID to list of ``(yaml_path, local SolutionIndex)``.
    """
    seen: Dict[int, List[Tuple[str, int]]] = {}
    for uid, yaml_path, local_index in entries:
        seen.setdefault(uid, []).append((yaml_path, local_index))
    return {uid: locations for uid, locations in seen.items() if len(locations) > 1}


@pytest.mark.parametrize(
    "solution_yaml",
    [
        "- SolutionIndex: 7\n  SolutionUID: 0u1\n",
        "- SolutionUID: 0u1\n  SolutionIndex: 7\n",
        "- {SolutionIndex: 7, SolutionUID: 0u1}\n",
        "- SolutionIndex: 7 # local index\n  SolutionUID: 0U1 # persistent ID\n",
        "- SolutionIndex: 7\n  SolutionUID: '0u1'\n",
        '- SolutionIndex: 7\n  SolutionUID: "0u1"\n',
    ],
)
def test_scan_yaml_file_reads_solution_mappings(tmp_path: Path, solution_yaml: str) -> None:
    """Equivalent YAML mappings must produce the same numeric identity."""
    path = tmp_path / "logic.yaml"
    path.write_text("Solutions:\n" + solution_yaml, encoding="utf-8")
    assert _scan_yaml_file(path) == ([(1, str(path), 7)], 0)


@pytest.mark.parametrize("uid_first", [False, True])
@pytest.mark.parametrize("uid_yaml", ["123", "null", "0u0", "0u01", "0u_", "0uLygHa16AHYG"])
def test_scan_yaml_file_rejects_invalid_stored_uids(
    tmp_path: Path, uid_yaml: str, uid_first: bool
) -> None:
    """Field order must not hide an invalid UID."""
    fields = ["SolutionIndex: 7", f"SolutionUID: {uid_yaml}"]
    if uid_first:
        fields.reverse()
    path = tmp_path / "logic.yaml"
    path.write_text("Solutions:\n- " + "\n  ".join(fields) + "\n", encoding="utf-8")
    with pytest.raises((TypeError, ValueError)) as error:
        _scan_yaml_file(path)
    assert str(path) in str(error.value)


def test_scan_yaml_file_ignores_fields_outside_solution_mapping(tmp_path: Path) -> None:
    """Text and nested metadata are not solution definitions."""
    path = tmp_path / "logic.yaml"
    path.write_text(
        "Notes: |\n  SolutionIndex: 2\n  SolutionUID: 0u2\n"
        "Solutions:\n- SolutionIndex: 7\n  SolutionUID: 0u1\n"
        "  Metadata:\n    SolutionUID: 0u3\n",
        encoding="utf-8",
    )
    assert _scan_yaml_file(path) == ([(1, str(path), 7)], 0)


def test_scan_yaml_file_reads_legacy_solutions(tmp_path: Path) -> None:
    """Legacy list-format logic stores solutions at element five."""
    path = tmp_path / "logic.yaml"
    path.write_text(
        "- {MinimumRequiredVersion: 5.0.0}\n- schedule\n- gfx908\n- []\n- {}\n"
        "- - SolutionUID: 0u1\n    SolutionIndex: 7\n",
        encoding="utf-8",
    )
    assert _scan_yaml_file(path) == ([(1, str(path), 7)], 0)


@pytest.mark.parametrize(
    "logic_yaml",
    [
        "UID: &uid 0u1\nSolutions:\n- SolutionIndex: 7\n  SolutionUID: *uid\n",
        "Defaults: &defaults {SolutionUID: 0u1}\n"
        "Solutions:\n- <<: *defaults\n  SolutionIndex: 7\n",
        "<<:\n  Solutions:\n  - {SolutionIndex: 7, SolutionUID: 0u1}\n",
        "Template: &solution {SolutionIndex: 7, SolutionUID: 0u1}\n"
        "Solutions: [*solution]\n",
        '"Soluti\\u006fns":\n- {SolutionIndex: 7, "SolutionU\\u0049D": 0u1}\n',
    ],
)
def test_scan_yaml_file_resolves_yaml_identity_fields(tmp_path: Path, logic_yaml: str) -> None:
    """Aliases, merges and escaped keys retain the loader's YAML semantics."""
    path = tmp_path / "logic.yaml"
    path.write_text(logic_yaml, encoding="utf-8")
    assert _scan_yaml_file(path) == ([(1, str(path), 7)], 0)


@pytest.mark.parametrize("index_yaml", ["null", "true", "7.5", "'7'"])
def test_scan_yaml_file_requires_integer_index(tmp_path: Path, index_yaml: str) -> None:
    path = tmp_path / "logic.yaml"
    path.write_text(
        f"Solutions:\n- SolutionIndex: {index_yaml}\n  SolutionUID: 0u1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="SolutionIndex must be an integer"):
        _scan_yaml_file(path)


def test_scan_yaml_file_rejects_multiple_documents(tmp_path: Path) -> None:
    path = tmp_path / "logic.yaml"
    path.write_text(
        "---\nSolutions: [{SolutionIndex: 0, SolutionUID: 0u1}]\n"
        "---\nSolutions: [{SolutionIndex: 1, SolutionUID: 0u2}]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Expected one YAML document"):
        _scan_yaml_file(path)


def test_collect_solution_uids_finds_duplicates_across_layouts(tmp_path: Path) -> None:
    """The corpus gate must see duplicates despite layout and prefix differences."""
    asm_full = tmp_path / "asm_full"
    asm_full.mkdir()
    first = asm_full / "a.yaml"
    second = asm_full / "b.yaml"
    first.write_text("Solutions:\n- SolutionIndex: 0\n  SolutionUID: 0u1\n", encoding="utf-8")
    second.write_text(
        "Solutions:\n- {SolutionUID: 0U1, SolutionIndex: 3}\n- SolutionIndex: 4\n",
        encoding="utf-8",
    )
    entries, missing = collect_solution_uids(tmp_path, process_count=2)
    assert missing == 1
    assert find_duplicate_uids(entries) == {1: [(str(first), 0), (str(second), 3)]}


@pytest.fixture(name="logic_root")
def fixture_logic_root() -> Path:
    """Resolve the logic YAML root, overridable via env var.

    Returns:
        Path to ``Logic/`` under the hipBLASLt library tree.
    """
    return _LOGIC_ROOT


@_needs_logic_dir
def test_solution_uid_unique_across_logic_files(logic_root: Path) -> None:
    """All present SolutionUID values must be unique repo-wide."""
    process_count = int(os.environ.get("HIPBLASLT_UID_TEST_PROCESSES", "8"))
    entries, missing = collect_solution_uids(logic_root, process_count=process_count)
    if not entries and missing == 0:
        pytest.fail(f"No solution definitions found under {logic_root}")

    duplicates = find_duplicate_uids(entries)
    if duplicates:
        lines = ["Duplicate SolutionUID values detected:"]
        for uid, locations in sorted(duplicates.items()):
            lines.append(f"  uid {encode_solution_uid(uid)}:")
            for yaml_path, local_index in locations:
                lines.append(f"    {yaml_path} (SolutionIndex={local_index})")
        pytest.fail("\n".join(lines))

    strict = os.environ.get(STRICT_ENV, "").lower() in {"1", "true", "yes"}
    if strict and missing > 0:
        pytest.fail(
            f"{missing} solution(s) under {logic_root} are missing SolutionUID "
            f"(set {STRICT_ENV}=0 to warn only during migration)"
        )
