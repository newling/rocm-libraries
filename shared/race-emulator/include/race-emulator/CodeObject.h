// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "Parsing.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace raceemulator {

/// Parse kernel metadata from an AMDGPU code object ELF.
/// Extracts kernel name, args, wavefront size, register counts, and LDS size
/// from the kernel descriptor (.rodata) and msgpack metadata (.note section).
/// Populates the metadata fields of a ParsedAsm (name, wavefrontSize, args,
/// kernargSegmentSize, amdhsa, initialRegisterAllocation, kernargPreload*).
///
/// Returns an error string on failure.
using CodeObjectResult = std::variant<ParsedAsm, std::string>;
CodeObjectResult parseCodeObjectMetadata(const uint8_t *elfData, size_t elfSize,
                                         const std::string &kernelName = "");

} // namespace raceemulator
