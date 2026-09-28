// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <string_view>

namespace TensileLite
{
    // No runtime hashing: this is the versioned value generated with the library.
    inline bool validSolutionFingerprint(std::string_view fingerprint)
    {
        constexpr std::string_view prefix = "sha256-v1:";
        return fingerprint.size() == prefix.size() + 64
               && fingerprint.substr(0, prefix.size()) == prefix
               && std::all_of(fingerprint.begin() + prefix.size(), fingerprint.end(), [](char c) {
                      return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
                  });
    }

    inline bool solutionFingerprintMatches(std::string_view recorded, std::string_view current)
    {
        return validSolutionFingerprint(recorded) && recorded == current;
    }
}
