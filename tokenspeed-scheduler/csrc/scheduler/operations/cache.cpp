// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "scheduler/operations/cache.h"

#include <algorithm>

#include "scheduler/types.h"

namespace tokenspeed {

bool PrefillRangeCutsUnsplittableSpan(std::int32_t first_pos, std::int32_t chunk_size,
                                      std::span<const UnsplittableSpan> unsplittable_spans) {
    _assert(first_pos >= 0 && chunk_size >= 0, "prefill positions must be non-negative");
    const std::int32_t end = first_pos + chunk_size;
    for (const auto& [start, stop] : unsplittable_spans) {
        _assert(start >= 0 && stop > start, "unsplittable span must be a non-empty half-open range");
        if (first_pos < stop && end > start && end < stop) {
            return true;
        }
    }
    return false;
}

std::int32_t AdjustPrefillChunkForUnsplittableSpans(std::int32_t first_pos, std::int32_t chunk_size,
                                                    std::int32_t unscheduled, std::int32_t token_budget,
                                                    std::span<const UnsplittableSpan> unsplittable_spans) {
    _assert(first_pos >= 0 && chunk_size >= 0 && unscheduled >= 0 && token_budget >= 0,
            "prefill positions must be non-negative");
    const std::int32_t end = first_pos + chunk_size;
    // Spans are sorted and disjoint (validated at submit), so the first span
    // the candidate chunk fails to finish decides the outcome; a chunk that
    // finishes every span it touches is kept as-is.
    for (const auto& [start, stop] : unsplittable_spans) {
        _assert(start >= 0 && stop > start, "unsplittable span must be a non-empty half-open range");
        if (stop <= first_pos || end >= stop) {
            continue;  // entirely behind the chunk, or fully covered by it
        }
        if (end <= start && first_pos < start) {
            break;  // the chunk ends before this span; later spans start later still
        }
        // The chunk would end strictly inside [start, stop): either it enters
        // the span without finishing it, or it starts inside and stops short.
        const std::int32_t take_all = stop - first_pos;
        if (take_all <= unscheduled && take_all <= token_budget) {
            return take_all;
        }
        // Already inside the span, nothing shorter is legal: wait for budget.
        // Not yet inside: stop right before it.
        return first_pos >= start ? 0 : start - first_pos;
    }
    return chunk_size;
}

std::int32_t ClampPrefixHitToUnsplittableSpans(std::int32_t hit_tokens,
                                               std::span<const UnsplittableSpan> unsplittable_spans,
                                               std::int32_t prefix_granularity) {
    _assert(hit_tokens >= 0, "prefix hit must be non-negative");
    _assert(prefix_granularity > 0, "prefix_granularity must be > 0");
    std::int32_t clamped = hit_tokens;
    for (const auto& [start, stop] : unsplittable_spans) {
        _assert(start >= 0 && stop > start, "unsplittable span must be a non-empty half-open range");
        if (start < clamped && clamped < stop) {
            clamped = start;
        }
    }
    return (clamped / prefix_granularity) * prefix_granularity;
}

std::int32_t AlignPrefillChunk(std::int32_t first_pos, std::int32_t unscheduled, std::int32_t token_budget,
                               std::int32_t prefix_granularity, std::int32_t promotion_boundary_tokens,
                               std::span<const UnsplittableSpan> unsplittable_spans) {
    _assert(first_pos >= 0 && unscheduled >= 0 && token_budget >= 0, "prefill positions must be non-negative");
    _assert(prefix_granularity > 0, "prefix_granularity must be > 0");
    std::int32_t chunk_size = std::min(unscheduled, token_budget);
    if (promotion_boundary_tokens > first_pos) {
        chunk_size = std::min(chunk_size, promotion_boundary_tokens - first_pos);
    }
    if (chunk_size == unscheduled) {
        return AdjustPrefillChunkForUnsplittableSpans(first_pos, chunk_size, unscheduled, token_budget,
                                                      unsplittable_spans);
    }

    const std::int32_t prefix_page_offset = first_pos % prefix_granularity;
    if (prefix_page_offset != 0) {
        const std::int32_t tokens_to_boundary = prefix_granularity - prefix_page_offset;
        chunk_size = token_budget >= tokens_to_boundary ? tokens_to_boundary : 0;
    } else {
        chunk_size = chunk_size - chunk_size % prefix_granularity;
    }
    return AdjustPrefillChunkForUnsplittableSpans(first_pos, chunk_size, unscheduled, token_budget, unsplittable_spans);
}

std::optional<std::int32_t> FinalAlignedTailTokens(std::int32_t first_pos, std::int32_t unscheduled,
                                                   std::int32_t token_budget, std::int32_t prefix_granularity,
                                                   std::int32_t promotion_boundary_tokens,
                                                   std::span<const UnsplittableSpan> unsplittable_spans) {
    _assert(first_pos >= 0 && unscheduled >= 0 && token_budget >= 0, "prefill positions must be non-negative");
    _assert(prefix_granularity > 0, "prefix_granularity must be > 0");
    std::int32_t chunk_size = std::min(unscheduled, token_budget);
    if (promotion_boundary_tokens > first_pos) {
        chunk_size = std::min(chunk_size, promotion_boundary_tokens - first_pos);
    }
    if (chunk_size != unscheduled) {
        return std::nullopt;
    }

    const std::int32_t tail_tokens = (first_pos + chunk_size) % prefix_granularity;
    if (tail_tokens == 0 || chunk_size - tail_tokens <= 0) {
        return std::nullopt;
    }
    if (PrefillRangeCutsUnsplittableSpan(first_pos, chunk_size - tail_tokens, unsplittable_spans)) {
        return std::nullopt;
    }
    return tail_tokens;
}

std::vector<CacheGroupSpec> MakeSpecsFromConfig(const SchedulerConfig& config) {
    std::vector<CacheGroupSpec> specs;
    specs.reserve(config.cache_groups.size());
    for (const CacheGroupConfig& group : config.cache_groups) {
        if (group.IsSnapshotStateGroup()) {
            specs.push_back(CacheGroupSpec{
                .kind = AttnKind::kMambaState,
                .sliding_window = 0,
                .cache_blocks_per_lcm_block = group.cache_blocks_per_lcm_block,
                .block_granularity = group.BlockGranularity(),
            });
            continue;
        }
        // family=State also covers linear-attention groups with a trailing
        // window; those translate like any other sliding group.
        const bool is_swa = group.retention == CacheGroupConfig::Retention::SlidingWindow;
        specs.push_back(CacheGroupSpec{
            .kind = is_swa ? AttnKind::kSlidingWindow : AttnKind::kFull,
            .sliding_window = is_swa ? *group.sliding_window_tokens : 0,
            .cache_blocks_per_lcm_block = group.cache_blocks_per_lcm_block,
            .block_granularity = group.BlockGranularity(),
        });
    }
    return specs;
}

void FreeRequest(CacheCoordinator& coordinator, std::vector<BlockTable>& tables) {
    if (tables.empty()) {
        return;  // request never got tables, or a failure path already released them
    }
    coordinator.Free(tables);
}

std::map<std::string, std::vector<std::int32_t>> BuildBlockTables(const CacheCoordinator& coordinator,
                                                                  const std::vector<BlockTable>& tables,
                                                                  std::span<const std::string> group_ids) {
    _assert(tables.size() == group_ids.size(), "BuildBlockTables: tables/group_ids size mismatch");
    _assert(tables.size() == static_cast<std::size_t>(coordinator.NumGroups()),
            "BuildBlockTables: tables/coordinator size mismatch");
    std::map<std::string, std::vector<std::int32_t>> out;
    for (std::size_t i = 0; i < tables.size(); ++i) {
        out.emplace(group_ids[i], coordinator.Allocator(static_cast<std::int32_t>(i)).BlockTablePageIds(tables[i]));
    }
    return out;
}

}  // namespace tokenspeed
