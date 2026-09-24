# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include(CMakeFindDependencyMacro)

if(TARGET hip::host)
  get_target_property(_hipblas_hip_definitions hip::host INTERFACE_COMPILE_DEFINITIONS)
  if("${_hipblas_hip_definitions}" MATCHES "__HIP_PLATFORM_AMD__")
    message(FATAL_ERROR "CUDA hipBLAS cannot use an AMD hip::host target")
  endif()
  unset(_hipblas_hip_definitions)
else()
  set(HIP_PLATFORM nvidia)
  find_dependency(hip CONFIG)
endif()
