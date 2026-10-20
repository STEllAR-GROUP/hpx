//  Copyright (c) 2026 Arivoli Ramamoorthy
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Blockwise in-place distribution engine shared by IPS^2Ra (radix sort) and
// IPS^4o (samplesort), after Axtmann, Witt, Ferizovic and Sanders,
// "Engineering In-place (Shared-memory) Sorting Algorithms", ACM TOPC 2022.
//
// Given a range of n elements and a classifier mapping each element to one of
// K buckets, ips_distribute reorders the range so that bucket k occupies
// [bucket_start[k], bucket_start[k + 1]), using O(K * B) auxiliary memory
// where B is the block size. The phases are:
//
//   1. Local classification: elements are moved into per-bucket buffers of B
//      elements; every full buffer is flushed back to the front of the range
//      as a homogeneous block. Afterwards [0, W) holds full blocks and the
//      buffers hold fewer than B elements per bucket.
//   2. Prefix sums: bucket sizes become bucket boundaries. Each bucket is
//      assigned the block-aligned region [align_up(start), align_up(end)).
//   3. Block permutation: every block is swapped into the aligned region of
//      its bucket. Per bucket, a write pointer w marks the end of correctly
//      placed blocks and a read pointer r marks the last unprocessed block.
//   4. Cleanup: aligned regions do not coincide with exact bucket boundaries,
//      so the partial head and tail of every bucket are filled from the
//      bucket buffer and from blocks that spilled into the next bucket.
//
// The current implementation is sequential. The data layout (stripe-local
// classification, per-bucket read/write pointers) is the one the parallel
// version will use.

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/modules/type_support.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <utility>
#include <vector>

namespace hpx::parallel::detail {

    // Default block size: 2 KiB worth of elements, as in the reference
    // implementation, but at least one element.
    HPX_CXX_CORE_EXPORT template <typename T>
    inline constexpr std::size_t ips_default_block_size =
        sizeof(T) >= 2048 ? 1 : 2048 / sizeof(T);

    // Auxiliary state for one (sequential) distribution. It is allocated once
    // and reused across recursion levels; every buffer is empty between calls
    // to ips_distribute.
    //
    // Buffers are vectors with reserved capacity rather than raw arrays so that
    // the element type only needs to be move constructible and move
    // assignable.
    HPX_CXX_CORE_EXPORT template <typename T>
    struct ips_distribution_buffers
    {
        ips_distribution_buffers(std::size_t num_buckets, std::size_t block)
          : block_size(block)
          , buckets(num_buckets)
          , bucket_start(num_buckets + 1)
          , write(num_buckets)
          , read(num_buckets)
        {
            HPX_ASSERT(block_size != 0);
            for (auto& b : buckets)
                b.reserve(block_size);
            swap[0].reserve(block_size);
            swap[1].reserve(block_size);
            overflow.reserve(block_size);
        }

        std::size_t num_buckets() const noexcept
        {
            return buckets.size();
        }

        std::size_t block_size;

        // Partial blocks produced by local classification, one per bucket.
        std::vector<std::vector<T>> buckets;

        // Carry buffers for block permutation.
        std::vector<T> swap[2];

        // Receives the single block whose aligned slot extends past the end of
        // the range, if any.
        std::vector<T> overflow;

        // Output: bucket boundaries relative to the start of the range.
        std::vector<std::size_t> bucket_start;

        // Block permutation pointers, relative to the start of the range.
        // read[k] < write[k] means bucket k has no unprocessed blocks left.
        std::vector<std::size_t> write;
        std::vector<std::ptrdiff_t> read;
    };

    namespace ips {

        template <typename T, typename RandomIt>
        void load_block(std::vector<T>& buf, RandomIt src, std::size_t block)
        {
            buf.clear();
            std::move(src, src + static_cast<std::ptrdiff_t>(block),
                std::back_inserter(buf));
        }

        template <typename T, typename RandomIt>
        void store_block(std::vector<T>& buf, RandomIt dest)
        {
            std::move(buf.begin(), buf.end(), dest);
            buf.clear();
        }

        constexpr std::size_t align_up(std::size_t v, std::size_t b) noexcept
        {
            return (v + b - 1) / b * b;
        }

        ///////////////////////////////////////////////////////////////////////
        // Phase 1: local classification. Returns W, the end of the full-block
        // prefix. Bucket sizes are accumulated into buf.bucket_start[0..K).
        template <typename RandomIt, typename T, typename Classifier>
        std::size_t classify_locally(RandomIt first, std::size_t n,
            Classifier& classify, ips_distribution_buffers<T>& buf)
        {
            std::size_t const block = buf.block_size;
            std::fill(buf.bucket_start.begin(), buf.bucket_start.end(), 0);

            std::size_t written = 0;
            for (std::size_t i = 0; i != n; ++i)
            {
                auto it = first + static_cast<std::ptrdiff_t>(i);
                std::size_t const b = classify(*it);
                HPX_ASSERT(b < buf.num_buckets());

                ++buf.bucket_start[b];
                auto& bucket = buf.buckets[b];
                bucket.push_back(HPX_MOVE(*it));

                // We have read i + 1 elements and written `written`, and the
                // buffers hold the difference, so written + block <= i + 1:
                // the flush only overwrites slots that were already consumed.
                if (bucket.size() == block)
                {
                    store_block(
                        bucket, first + static_cast<std::ptrdiff_t>(written));
                    written += block;
                }
            }
            return written;
        }

        ///////////////////////////////////////////////////////////////////////
        // Phase 2: exclusive prefix sum over the bucket sizes, and initial
        // read/write pointers for block permutation.
        template <typename T>
        void compute_boundaries(
            std::size_t full_end, ips_distribution_buffers<T>& buf)
        {
            std::size_t const block = buf.block_size;
            std::size_t const k = buf.num_buckets();

            std::size_t sum = 0;
            for (std::size_t b = 0; b != k; ++b)
            {
                std::size_t const cnt = buf.bucket_start[b];
                buf.bucket_start[b] = sum;
                sum += cnt;
            }
            buf.bucket_start[k] = sum;

            for (std::size_t b = 0; b != k; ++b)
            {
                std::size_t const astart = align_up(buf.bucket_start[b], block);
                std::size_t const aend =
                    align_up(buf.bucket_start[b + 1], block);

                // The last full block inside this bucket's aligned region, if
                // any; blocks only exist in [0, full_end).
                buf.write[b] = astart;
                buf.read[b] =
                    static_cast<std::ptrdiff_t>((std::min) (aend, full_end)) -
                    static_cast<std::ptrdiff_t>(block);
            }
        }

        ///////////////////////////////////////////////////////////////////////
        // Phase 3: block permutation.
        //
        // Pick up the last unprocessed block of a bucket and carry it to the
        // write pointer of its destination bucket. If that slot holds a block
        // that already belongs there, skip it; if it holds an unprocessed block
        // of another bucket, swap and continue carrying the evicted block; if
        // the slot is free, drop the block and pick up the next one.
        template <typename RandomIt, typename T, typename Classifier>
        void permute_blocks(RandomIt first, std::size_t n, Classifier& classify,
            ips_distribution_buffers<T>& buf)
        {
            std::size_t const block = buf.block_size;
            std::size_t const k = buf.num_buckets();
            auto at = [first](std::size_t pos) {
                return first + static_cast<std::ptrdiff_t>(pos);
            };

            for (std::size_t src = 0; src != k; ++src)
            {
                while (buf.read[src] >=
                    static_cast<std::ptrdiff_t>(buf.write[src]))
                {
                    std::size_t const r =
                        static_cast<std::size_t>(buf.read[src]);
                    buf.read[src] -= static_cast<std::ptrdiff_t>(block);

                    int cur = 0;
                    load_block(buf.swap[cur], at(r), block);
                    std::size_t dest = classify(buf.swap[cur].front());

                    for (;;)
                    {
                        std::size_t const w = buf.write[dest];
                        buf.write[dest] += block;

                        if (static_cast<std::ptrdiff_t>(w) > buf.read[dest])
                        {
                            // Free slot. Only the slot straddling the end of
                            // the range can extend past n, and at most one
                            // block is ever written there.
                            if (w + block > n)
                            {
                                HPX_ASSERT(buf.overflow.empty());
                                buf.overflow.swap(buf.swap[cur]);
                            }
                            else
                            {
                                store_block(buf.swap[cur], at(w));
                            }
                            break;
                        }

                        std::size_t const evicted = classify(*at(w));
                        if (evicted == dest)
                            continue;    // already in place

                        load_block(buf.swap[1 - cur], at(w), block);
                        store_block(buf.swap[cur], at(w));
                        cur = 1 - cur;
                        dest = evicted;
                    }
                }
            }
        }

        ///////////////////////////////////////////////////////////////////////
        // Phase 4: cleanup.
        //
        // After permutation, the blocks of bucket b cover [astart, w) where
        // astart = align_up(start). Relative to the exact range [start, end):
        //  - the head [start, astart) is free (it lies in the aligned region of
        //    bucket b - 1, whose spill-over has already been moved out because
        //    buckets are processed in order),
        //  - the tail [w, end) is free if w < end,
        //  - otherwise the last block spilled w - end elements into the head of
        //    bucket b + 1 (or into the overflow buffer if w > n).
        // The free slots are filled from the spilled elements and from the
        // bucket buffer; the counts always match.
        template <typename RandomIt, typename T>
        void cleanup(
            RandomIt first, std::size_t n, ips_distribution_buffers<T>& buf)
        {
            std::size_t const block = buf.block_size;
            std::size_t const k = buf.num_buckets();
            auto at = [first](std::size_t pos) {
                return first + static_cast<std::ptrdiff_t>(pos);
            };

            for (std::size_t b = 0; b != k; ++b)
            {
                std::size_t const start = buf.bucket_start[b];
                std::size_t const end = buf.bucket_start[b + 1];
                std::size_t const astart = align_up(start, block);
                std::size_t const w = buf.write[b];
                bool const has_blocks = w > astart;

                // The overflow block belongs at [w - block, w); put the part
                // that fits into the range there. Those slots are past the
                // full-block prefix and hence free.
                std::size_t overflow_pos = 0;
                if (has_blocks && w > n)
                {
                    HPX_ASSERT(buf.overflow.size() == block);
                    overflow_pos = n - (w - block);
                    std::move(buf.overflow.begin(),
                        buf.overflow.begin() +
                            static_cast<std::ptrdiff_t>(overflow_pos),
                        at(w - block));
                }

                // Free slots in [start, end), in order: the head, then the
                // tail after the blocks.
                std::size_t fill = start;
                std::size_t const head_end = has_blocks ? astart : end;
                auto put = [&](T&& v) {
                    // Named outside HPX_ASSERT as well: contract assertions do
                    // not capture variables implicitly.
                    HPX_UNUSED(end);
                    if (fill == head_end && has_blocks)
                        fill = w;
                    HPX_ASSERT(fill < end);
                    *at(fill++) = HPX_MOVE(v);
                };

                if (has_blocks && w > end)
                {
                    for (std::size_t i = end; i < (std::min) (w, n); ++i)
                        put(HPX_MOVE(*at(i)));
                    if (w > n)
                    {
                        for (std::size_t i = overflow_pos; i != block; ++i)
                            put(HPX_MOVE(buf.overflow[i]));
                        buf.overflow.clear();
                    }
                }

                for (auto& v : buf.buckets[b])
                    put(HPX_MOVE(v));
                buf.buckets[b].clear();

                // Either the free slots end at the bucket end, or the blocks
                // reach past it and only the head needed filling.
                HPX_ASSERT(fill == ((has_blocks && w >= end) ? astart : end));
            }
        }
    }    // namespace ips

    ///////////////////////////////////////////////////////////////////////////
    // Distribute [first, first + n) into buf.num_buckets() buckets according to
    // `classify`, which maps an element to a bucket index. On return,
    // buf.bucket_start[k] is the offset of bucket k and buf.bucket_start[K]
    // equals n. The order within a bucket is unspecified.
    //
    // Exception safety: if `classify` throws, elements held in the buffers are
    // lost. Classifiers used by the sorting algorithms are non-throwing.
    HPX_CXX_CORE_EXPORT template <typename RandomIt, typename T,
        typename Classifier>
    void ips_distribute(RandomIt first, std::size_t n, Classifier&& classify,
        ips_distribution_buffers<T>& buf)
    {
        std::size_t const full_end =
            ips::classify_locally(first, n, classify, buf);

        // Bucket sizes are in bucket_start[0..K) at this point. If one bucket
        // received everything, the blocks are already in place.
        std::size_t single = buf.num_buckets();
        for (std::size_t b = 0; b != buf.num_buckets(); ++b)
        {
            if (buf.bucket_start[b] == n)
            {
                single = b;
                break;
            }
        }

        ips::compute_boundaries(full_end, buf);

        if (single != buf.num_buckets())
            buf.write[single] = full_end;
        else
            ips::permute_blocks(first, n, classify, buf);

        ips::cleanup(first, n, buf);
    }
}    // namespace hpx::parallel::detail
