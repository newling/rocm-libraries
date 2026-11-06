# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Regression checks for TensileLite's distributable package metadata."""

import configparser
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def test_wheel_metadata_does_not_require_unpublished_rocisa(tmp_path, monkeypatch):
    """rocisa is currently provisioned from source, not resolved by pip."""
    monkeypatch.setenv("ROCM_VERSION", "7.2.4")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-build-isolation",
            "--no-deps",
            "--wheel-dir",
            str(tmp_path),
            ".",
        ],
        cwd=_PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    wheel = next(tmp_path.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        metadata = archive.read(
            next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        ).decode("utf-8")
        entry_points = archive.read(
            next(name for name in archive.namelist() if name.endswith(".dist-info/entry_points.txt"))
        ).decode("utf-8")
        installed = tmp_path / "installed"
        archive.extractall(installed)

    component_version = re.search(
        r'^__version__ = "([^"]+)"$',
        (_PROJECT_ROOT / "tensilelite/__init__.py").read_text(encoding="utf-8"),
        re.MULTILINE,
    ).group(1)
    assert f"Version: {component_version}+rocm7.2.4" in metadata
    assert "Requires-Dist: rocisa" not in metadata

    result = subprocess.run(
        [sys.executable, "-S", "-m", "tensilelite", "--version"],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(installed)),
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == f"{component_version}+rocm7.2.4"

    parsed_entry_points = configparser.ConfigParser()
    parsed_entry_points.optionxform = str
    parsed_entry_points.read_string(entry_points)
    assert dict(parsed_entry_points["console_scripts"]) == {
        "tensilelite": "tensilelite.cli:main",
        "Tensile": "tensilelite.tensilelite:main",
        "TensileCreateLibrary": "tensilelite.create_library:main",
        "TensileVerifyStinkyElfText": "tensilelite.verify_stinky_comment_vs_elf_text:main",
        "TensileGetPath": "tensilelite:PrintTensileRoot",
        "TensileBenchmarkCluster": "tensilelite.benchmark_cluster:main",
        "TensileRetuneLibrary": "tensilelite.retune_library:main",
    }


def test_codegen_runs_filtered_package_commands_and_checks_missing_modules(tmp_path):
    cmake = shutil.which("cmake")
    if cmake is None or shutil.which("ninja") is None:
        pytest.skip("CMake and Ninja are required for the source-only codegen test")

    source = tmp_path / "source"
    codegen = source / "codegen"
    for relative_path in (
        "tensilelite/__init__.py",
        "tensilelite/__main__.py",
        "tensilelite/cli.py",
        "tensilelite/logic/run.py",
        "tensilelite/create_library/run.py",
        "tensilelite/logic/known_bugs.yaml",
    ):
        path = codegen / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    # Record what the generated build actually invokes, without compiling kernels.
    (codegen / "tensilelite/__main__.py").write_text(
        "import json, sys\n"
        "with open('calls.jsonl', 'a') as output:\n"
        "    output.write(json.dumps(sys.argv[1:]) + '\\n')\n",
        encoding="utf-8",
    )
    logic = source / "logic with spaces"
    logic.mkdir()
    (logic / "selected.yaml").write_text("{}\n", encoding="utf-8")
    (logic / "unselected.yaml").write_text("{}\n", encoding="utf-8")
    (source / "CMakeLists.txt").write_text(
        textwrap.dedent("""\
            cmake_minimum_required(VERSION 3.25)
            project(CodegenCommands NONE)
            include("${CODEGEN_MODULE}")
            create_device_library(
                TARGET smoke
                CODEGEN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/codegen"
                LOGIC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/logic with spaces"
                OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/output"
                PYTHON_EXECUTABLE "${PYTHON_EXECUTABLE}"
                CXX_COMPILER unused-compiler
                ARCHES gfx942
                LOGIC_FILTER "selected*")
            """),
        encoding="utf-8",
    )
    build = tmp_path / "build"
    configure = [
        cmake, "-S", str(source), "-B", str(build), "-G", "Ninja",
        f"-DCODEGEN_MODULE={_PROJECT_ROOT / 'cmake/hipblaslt_codegen.cmake'}",
        f"-DPYTHON_EXECUTABLE={sys.executable}",
    ]
    subprocess.run(configure, check=True, capture_output=True, text=True)
    subprocess.run(
        [cmake, "--build", str(build), "--parallel", "1"],
        check=True, capture_output=True, text=True,
    )
    validation, generation = [
        json.loads(line) for line in (build / "calls.jsonl").read_text().splitlines()
    ]
    assert validation[:2] == ["logic", str(logic)]
    assert "--check-all" in validation
    assert "--logic-filter=**/selected*.yaml" in validation
    assert generation[0] == "create-library"
    assert generation[-3:] == [str(logic), str(build / "output"), "HIP"]
    assert "--logic-filter=selected*" in generation

    missing = codegen / "tensilelite/cli.py"
    missing.unlink()
    result = subprocess.run(configure, capture_output=True, text=True)
    assert result.returncode != 0
    assert "required codegen resource not found" in result.stderr
    assert missing.as_posix() in result.stderr


def test_removed_source_autobuild_option_is_not_referenced():
    paths = (
        _PROJECT_ROOT.parent / "CMakeLists.txt",
        _PROJECT_ROOT.parent / "README.md",
        _PROJECT_ROOT / "README.md",
        _PROJECT_ROOT / "AGENTS_reference.md",
    )

    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "TENSILELITE_ENABLE_AUTOBUILD" not in text, path
        assert "--rebuild-rocisa" not in text, path

    launcher_docs = (_PROJECT_ROOT / "README.md", _PROJECT_ROOT / "AGENTS_reference.md")
    for path in launcher_docs:
        text = path.read_text(encoding="utf-8")
        for retired_name in ("TENSILE_BIN", "DEVELOP_MODE", "Tensile.sh", "Tensile.bat"):
            assert retired_name not in text, path

    rocisa_docs = (
        _PROJECT_ROOT / "rocisa/README.md",
        _PROJECT_ROOT / "rocisa/pyproject.toml",
    )
    for path in rocisa_docs:
        text = path.read_text(encoding="utf-8")
        assert "--no-rebuild-on-import" not in text, path
        assert "--rebuild-on-import" not in text, path
        assert "SKBUILD_EDITABLE_REBUILD" not in text, path
        assert "editable.rebuild" not in text, path
