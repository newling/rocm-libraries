// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <unordered_set>
#include <vector>

namespace hipblaslt_bench
{
    struct CandidatePolicy
    {
        bool   exhaustive = false;
        size_t limit      = 0; // Zero means all supported candidates.
        int    baseline   = -1;
    };

    // Providers return durable indexes. accept resolves the index and checks support
    // and workspace before it consumes a slot. The baseline consumes a slot too.
    template <typename Ranked, typename All, typename Accept>
    std::vector<int> select_candidate_indexes(const CandidatePolicy& policy,
                                              Ranked&&               ranked,
                                              All&&                  all,
                                              Accept&&               accept)
    {
        std::vector<int>        result;
        std::unordered_set<int> seen;
        auto full   = [&] { return policy.limit != 0 && result.size() >= policy.limit; };
        auto append = [&](int index) {
            if(!full() && index >= 0 && seen.insert(index).second && accept(index))
                result.push_back(index);
        };
        if(policy.baseline >= 0)
            append(policy.baseline);
        if(!policy.exhaustive && !full())
            for(int index : ranked())
                append(index);
        if(!full())
            for(int index : all())
            {
                append(index);
                if(full())
                    break;
            }
        return result;
    }
}
