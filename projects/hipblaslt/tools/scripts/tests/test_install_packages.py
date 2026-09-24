# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Check install arguments without running a package manager or a GPU build."""

import ast
import contextlib
import importlib.util
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("hipblaslt_tasks", PROJECT / "tasks.py")
TASKS = importlib.util.module_from_spec(SPEC)
with patch.object(sys, "dont_write_bytecode", True):
    SPEC.loader.exec_module(TASKS)


class InstallPackages(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hipblaslt-install-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "build with spaces"
        self.root.mkdir()

    def install_command(self, distro, static):
        # Run the actual final install block, with build/elevation recorded.
        tree = ast.parse((PROJECT / "tasks.py").read_text())
        block = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
            and node.test.id == "install_pkg"
        )
        commands = []

        class Context:
            def cd(self, path):
                return contextlib.nullcontext()

            def run(self, command):
                pass

        namespace = dict(vars(TASKS), install_pkg=True, c=Context(),
                         build_subdir=self.root, distro=distro, static=static,
                         _elevate=lambda c, command: commands.append(command))
        exec(compile(ast.Module(body=[block], type_ignores=[]), "tasks.py", "exec"), namespace)
        self.assertEqual(len(commands), 1)
        return shlex.split(commands[0])

    def test_static_install_uses_only_existing_packages(self):
        for distro, extension in [("ubuntu", "deb"), ("rhel", "rpm"),
                                  ("fedora", "rpm"), ("sles", "rpm")]:
            with self.subTest(distro=distro):
                package = self.root / f"hipblaslt-static-devel-1.5.0.{extension}"
                package.touch()
                command = self.install_command(distro, static=True)
                self.assertEqual([arg for arg in command if arg.endswith("." + extension)],
                                 [str(package)])

    def test_shared_install_includes_host_package(self):
        runtime = self.root / "hipblaslt_1.5.0.deb"
        host = self.root / "tensilelite-host_1.5.0.deb"
        runtime.touch()
        host.touch()
        command = self.install_command("ubuntu", static=False)
        self.assertEqual(command, ["dpkg", "-i", str(runtime), str(host)])

    def test_shared_install_rejects_missing_host_package(self):
        (self.root / "hipblaslt_1.5.0.deb").touch()
        with self.assertRaisesRegex(FileNotFoundError, "tensilelite-host"):
            self.install_command("ubuntu", static=False)

    def test_install_rejects_missing_hipblaslt_packages(self):
        with self.assertRaisesRegex(FileNotFoundError, "hipblaslt"):
            self.install_command("ubuntu", static=True)


if __name__ == "__main__":
    unittest.main()
