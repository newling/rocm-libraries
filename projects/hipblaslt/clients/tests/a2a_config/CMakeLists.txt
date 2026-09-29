# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)

# This directory can also be configured directly on a machine without ROCm.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(hipblaslt_a2a_config_tests LANGUAGES CXX)
    enable_testing()
    find_package(GTest CONFIG REQUIRED)
endif()

add_executable(hipblaslt-a2a-config-test a2a_config_gtest.cpp)
target_compile_features(hipblaslt-a2a-config-test PRIVATE cxx_std_17)
target_include_directories(hipblaslt-a2a-config-test PRIVATE ../../common/include)
target_link_libraries(hipblaslt-a2a-config-test PRIVATE GTest::gtest_main)

add_test(NAME hipblaslt-a2a-config COMMAND hipblaslt-a2a-config-test)
set_tests_properties(hipblaslt-a2a-config PROPERTIES LABELS "host;smoke")
