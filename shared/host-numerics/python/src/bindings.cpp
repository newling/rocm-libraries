// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Build a compiled extension so the smoke tests cover C++/Python bindings and
// installation. nanobind supplies NB_MODULE and module.attr() below.
#include <nanobind/nanobind.h>

#include <roc/host_numerics/version.hpp>

NB_MODULE(_roc_host_numerics, module) {
    module.attr("__version__") = roc::host_numerics::version().data();
}
