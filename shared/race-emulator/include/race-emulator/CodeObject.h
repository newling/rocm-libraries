// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "Parsing.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>

namespace raceemulator {

/// Parse kernel metadata from an AMDGPU code object ELF.
using CodeObjectResult = std::variant<KernelMetadata, std::string>;
CodeObjectResult parseCodeObjectMetadata(const uint8_t *elfData, size_t elfSize,
                                         const std::string &kernelName = "");

} // namespace raceemulator
