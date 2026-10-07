# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Use the monorepo baseline, also used by hipBLASLt and rocRoller.
# CMake 3.25.2 was released on 2023-01-19.
cmake_minimum_required(VERSION 3.25.2)
project(HostNumerics VERSION 0.1.0 LANGUAGES CXX)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(HOSTNUMERICS_STANDALONE ON)
else()
    set(HOSTNUMERICS_STANDALONE OFF)
endif()

option(HOSTNUMERICS_BUILD_TESTING "Build HostNumerics tests." ${HOSTNUMERICS_STANDALONE})
option(HOSTNUMERICS_BUILD_PYTHON "Build the HostNumerics Python module."
       ${HOSTNUMERICS_STANDALONE})

add_library(hostnumerics-core INTERFACE)
add_library(roc::hostnumerics-core ALIAS hostnumerics-core)
target_compile_features(hostnumerics-core INTERFACE cxx_std_20)
target_include_directories(hostnumerics-core
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_sources(hostnumerics-core
    INTERFACE
        FILE_SET HEADERS
        BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include"
        FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/hostnumerics/version.hpp"
)

add_library(hostnumerics INTERFACE)
add_library(roc::hostnumerics ALIAS hostnumerics)
target_link_libraries(hostnumerics INTERFACE roc::hostnumerics-core)

if(HOSTNUMERICS_BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
if(HOSTNUMERICS_BUILD_PYTHON)
    add_subdirectory(python)
endif()

install(TARGETS hostnumerics-core hostnumerics
    EXPORT hostnumerics-targets
    FILE_SET HEADERS DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
install(EXPORT hostnumerics-targets
    FILE HostNumericsTargets.cmake
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/HostNumerics"
    NAMESPACE roc::
)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/HostNumericsConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/HostNumericsConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/HostNumerics"
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/HostNumericsConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/HostNumericsConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/HostNumericsConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/HostNumerics"
)
install(FILES LICENSE.md
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/HostNumerics"
)
