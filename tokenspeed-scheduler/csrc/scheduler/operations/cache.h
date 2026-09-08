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

#pragma once

#include <map>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "cache/core/cache_types.h"
#include "cache/coordinator/cache_coordinator.h"

namespace tokenspeed {

struct SchedulerConfig;

// Half-open [start, end) token range that must stay in one prefill chunk.
// A prefix-cache hit may end at `start` or at `end`, but not strictly inside.
using UnsplittableSpan = std::pair<std::int32_t, std::int32_t>;

// One CacheGroupSpec per config cache_group (group_id = index); all groups share config.prefix_granularity.
// Pure translation: the caller must have accepted `config` through
// SchedulerConfig::Validate() first, which is what makes every field read here
// (packing, block granularity, a sliding group's window) well-formed.
std::vector<CacheGroupSpec> MakeSpecsFromConfig(const SchedulerConfig& config);

// True when [first_pos, first_pos + chunk_size) overlaps a span but ends
// strictly inside it.
bool PrefillRangeCutsUnsplittableSpan(std::int32_t first_pos, std::int32_t chunk_size,
                                      std::span<const UnsplittableSpan> unsplittable_spans);

// Keep a candidate chunk from cutting an unsplittable span. A chunk that
// finishes every span it touches is returned unchanged. Otherwise the first
// span it would end inside decides: take up to that span's end when the
// budget covers it, stop right before the span when the chunk has not entered
// it yet, or return 0 when the chunk already starts inside the span and the
// budget cannot finish it this round.
std::int32_t AdjustPrefillChunkForUnsplittableSpans(std::int32_t first_pos, std::int32_t chunk_size,
                                                    std::int32_t unscheduled, std::int32_t token_budget,
                                                    std::span<const UnsplittableSpan> unsplittable_spans);

// Truncate a page-aligned prefix hit so it does not end strictly inside a
// span. The result is floored to `prefix_granularity`.
std::int32_t ClampPrefixHitToUnsplittableSpans(std::int32_t hit_tokens,
                                               std::span<const UnsplittableSpan> unsplittable_spans,
                                               std::int32_t prefix_granularity);

std::int32_t AlignPrefillChunk(std::int32_t first_pos, std::int32_t unscheduled, std::int32_t token_budget,
                               std::int32_t prefix_granularity, std::int32_t promotion_boundary_tokens,
                               std::span<const UnsplittableSpan> unsplittable_spans);

std::optional<std::int32_t> FinalAlignedTailTokens(std::int32_t first_pos, std::int32_t unscheduled,
                                                   std::int32_t token_budget, std::int32_t prefix_granularity,
                                                   std::int32_t promotion_boundary_tokens,
                                                   std::span<const UnsplittableSpan> unsplittable_spans);

void FreeRequest(CacheCoordinator& coordinator, std::vector<BlockTable>& tables);

// One row per config group_id. Each group allocator resolves the LCM placement
// to the kernel-visible page id.
std::map<std::string, std::vector<std::int32_t>> BuildBlockTables(const CacheCoordinator& coordinator,
                                                                  const std::vector<BlockTable>& tables,
                                                                  std::span<const std::string> group_ids);

}  // namespace tokenspeed
