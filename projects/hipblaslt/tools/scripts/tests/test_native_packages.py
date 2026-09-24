# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Exercise the native packaging rules without building GPU kernels.

Run with Python's unittest runner. CMake, Ninja, Clang, dpkg-deb and readelf are required.
Set TMPDIR to an out-of-tree build directory. ROCM_CMAKE_DIR can select an
existing rocm-cmake module directory; otherwise use the project's normal fetch.
The fixture compiles two small libraries and executes the production packaging
and library-install blocks, then inspects actual CPack output.
"""

import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[3]


class NativePackages(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hipblaslt-packaging-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = os.environ.copy()
        for name in ("ROCM_LIBPATCH_VERSION", "CPACK_DEBIAN_PACKAGE_RELEASE"):
            self.env.pop(name, None)

    def run_command(self, *args):
        result = subprocess.run(
            [str(arg) for arg in args], env=self.env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        self.assertEqual(result.returncode, 0, f"{args}\n{result.stdout}")
        return result.stdout.strip()

    def build_packages(self, *options):
        top = (PROJECT / "CMakeLists.txt").read_text()
        start = top.index("if(TARGET tensilelite-host AND NOT ENABLE_ASAN_PACKAGING)")
        end = top.index("\nif(ROCM_LIBS_SUPERBUILD OR NOT HIPBLASLT_IS_SUBPROJECT)", start)
        packaging = top[start:end]
        tensile = (PROJECT / "tensilelite/CMakeLists.txt").read_text()
        start = tensile.index("    set(_tensilelite_host_component tensilelite)")
        end = tensile.index("        FILE_SET", start)
        # Header file sets are irrelevant to shared-library package ownership.
        install = tensile[start:end] + ")\n"
        fixture = self.root / "src"
        fixture.mkdir()
        (fixture / "host.c").write_text("int tensilelite_probe(void) { return 1; }\n")
        (fixture / "hipblaslt.c").write_text(
            "extern int tensilelite_probe(void);\n"
            "int hipblaslt_probe(void) { return tensilelite_probe(); }\n"
        )
        (fixture / "CMakeLists.txt").write_text('''
cmake_minimum_required(VERSION 3.25.2)
project(hipblaslt VERSION 1.5.0 LANGUAGES C)
list(APPEND CMAKE_MODULE_PATH "${HIPBLASLT_SOURCE_DIR}/cmake")
include(fetch_rocm_cmake)
set(PROJECT_VERSION_TWEAK "42-git")
set(ROCM_USE_DEV_COMPONENT ON)
set(CPACK_GENERATOR DEB)
set(CPACK_THREADS 2)
set(CPACK_RESOURCE_FILE_LICENSE "${HIPBLASLT_SOURCE_DIR}/LICENSE.md")
option(BUILD_SHARED_LIBS "Shared hipBLASLt" ON)
option(TENSILELITE_BUILD_SHARED_LIBS "Shared TensileLite" ON)
if(TENSILELITE_BUILD_SHARED_LIBS)
    add_library(tensilelite-host SHARED host.c)
else()
    add_library(tensilelite-host STATIC host.c)
endif()
set(TEST_HOST_SOVERSION "1.0" CACHE STRING "Fixture host library version")
rocm_set_soversion(tensilelite-host "${TEST_HOST_SOVERSION}")
add_library(hipblaslt hipblaslt.c)
target_link_libraries(hipblaslt PRIVATE tensilelite-host)
install(TARGETS hipblaslt LIBRARY DESTINATION lib COMPONENT runtime
        ARCHIVE DESTINATION lib COMPONENT devel)
rocm_package_add_dependencies(DEPENDS "roctracer >= 1.0.0")
''' + install + packaging + '''
rocm_create_package(NAME hipblaslt DESCRIPTION "Native package test"
                    MAINTAINER "Package maintainer")
''')
        build = self.root / "build"
        cmake_options = []
        if "ROCM_CMAKE_DIR" in self.env:
            cmake_options.append("-DROCmCMakeBuildTools_DIR=" + self.env["ROCM_CMAKE_DIR"])
        self.run_command(
            "cmake", "-S", fixture, "-B", build, "-G", "Ninja",
            "-DCMAKE_C_COMPILER=clang", "-DCMAKE_INSTALL_PREFIX=/opt/rocm",
            "-DHIPBLASLT_SOURCE_DIR=" + str(PROJECT), *cmake_options, *options,
        )
        self.run_command("cmake", "--build", build, "--target", "package", "-j2")
        return build

    def check_dependency(self, build):
        runtime = next(build.glob("hipblaslt_*.deb"))
        host = next(build.glob("tensilelite-host_*.deb"))
        version = self.run_command("dpkg-deb", "-f", host, "Version")
        depends = self.run_command("dpkg-deb", "-f", runtime, "Depends")
        required = re.search(r"tensilelite-host \(= ([^)]+)\)", depends)
        self.assertIsNotNone(required, depends)
        self.run_command("dpkg", "--compare-versions", version, "eq", required[1])
        self.assertIn("roctracer (>= 1.0.0)", depends)
        self.assertIn("rocm-core", depends)
        return version

    def test_debian_dependency_includes_sanitized_git_release(self):
        version = self.check_dependency(self.build_packages())
        self.assertEqual(version, "1.5.0-42~git")

    def test_debian_dependency_includes_release_override_patch_and_epoch(self):
        self.env["CPACK_DEBIAN_PACKAGE_RELEASE"] = "99"
        self.env["ROCM_LIBPATCH_VERSION"] = "70002"
        version = self.check_dependency(self.build_packages("-DCPACK_DEBIAN_PACKAGE_EPOCH=2"))
        self.assertEqual(version, "2:1.5.0.70002-99")

    def test_asan_runtime_contains_both_libraries(self):
        build = self.build_packages("-DENABLE_ASAN_PACKAGING=ON")
        packages = list(build.glob("*.deb"))
        self.assertEqual(len(packages), 1)
        self.assertEqual(self.run_command("dpkg-deb", "-f", packages[0], "Package"), "hipblaslt-asan")
        contents = self.run_command("dpkg-deb", "-c", packages[0])
        self.assertIn("libhipblaslt.so", contents)
        self.assertIn("libtensilelite-host.so.1.0", contents)
        depends = self.run_command("dpkg-deb", "-f", packages[0], "Depends")
        self.assertIn("rocm-core-asan", depends)
        self.assertNotIn("tensilelite-host", depends)

    def test_shared_host_library_has_its_own_package(self):
        build = self.build_packages()
        runtime = next(build.glob("hipblaslt_*.deb"))
        host = next(build.glob("tensilelite-host_*.deb"))
        self.assertNotIn("libtensilelite-host.so", self.run_command("dpkg-deb", "-c", runtime))
        self.assertIn("libtensilelite-host.so.1.0", self.run_command("dpkg-deb", "-c", host))

    def test_rpm_requirement_follows_host_soversion(self):
        build = self.build_packages("-DTEST_HOST_SOVERSION=2.7")
        dynamic = self.run_command("readelf", "-d", build / "libtensilelite-host.so.2.7")
        soname = re.search(r"\(SONAME\).*\[([^]]+)\]", dynamic)
        self.assertIsNotNone(soname, dynamic)
        config = (build / "CPackConfig.cmake").read_text()
        requires = re.search(r'^set\(CPACK_RPM_RUNTIME_PACKAGE_REQUIRES "([^"]*)"\)',
                             config, re.MULTILINE)
        self.assertIsNotNone(requires, config)
        self.assertIn(f"{soname[1]}()(64bit)", requires[1].split(", "))


if __name__ == "__main__":
    unittest.main()
