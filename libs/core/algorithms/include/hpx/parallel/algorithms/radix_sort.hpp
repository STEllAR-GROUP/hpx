//  Copyright (c) 2026 Arivoli Ramamoorthy
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \page hpx::radix_sort
/// \headerfile hpx/algorithm.hpp
///
/// In-place MSD radix sort over fixed-width integral keys, structured along
/// the IPS^2Ra pipeline of Axtmann et al. (ACM TOPC 2022): classify into
/// digit buckets, compute boundaries via prefix sums, permute in place, then
/// recurse or fall back to a sequential base case for small partitions.
///
/// Large partitions are distributed with the blockwise engine from
/// detail/ips_distribution.hpp; smaller ones use an element-wise in-place
/// permutation, and the smallest ones are handed to std::sort. The
/// implementation is still sequential; the CPO already accepts an execution
/// policy so the parallel version can land behind the same surface.

#pragma once

#if defined(DOXYGEN)

namespace hpx {

    /// Sort the range [first, last) of integral keys (or values whose
    /// projection yields an integral key) in ascending order, in place,
    /// using radix sort.
    template <typename RandomIt, typename Proj = hpx::identity>
    void radix_sort(RandomIt first, RandomIt last, Proj proj = Proj());

    /// Execution-policy overload. The current implementation dispatches to
    /// the sequential path regardless of policy; parallelization is a
    /// follow-up milestone.
    template <typename ExPolicy, typename RandomIt,
        typename Proj = hpx::identity>
    hpx::parallel::util::detail::algorithm_result_t<ExPolicy> radix_sort(
        ExPolicy&& policy, RandomIt first, RandomIt last, Proj proj = Proj());
}    // namespace hpx

#else

#include <hpx/config.hpp>
#include <hpx/algorithms/traits/projected.hpp>
#include <hpx/assert.hpp>
#include <hpx/modules/concepts.hpp>
#include <hpx/modules/execution.hpp>
#include <hpx/modules/functional.hpp>
#include <hpx/modules/iterator_support.hpp>
#include <hpx/modules/type_support.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/ips_distribution.hpp>
#include <hpx/parallel/algorithms/detail/tag_dispatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

namespace hpx::parallel::detail {

    ///////////////////////////////////////////////////////////////////////////
    // Key extraction.
    //
    // We sort by an unsigned "lifted" key derived from the projected value.
    // Signed integers are mapped by flipping the sign bit so that ascending
    // unsigned order matches ascending signed order.

    HPX_CXX_CORE_EXPORT template <typename T>
    struct radix_key_traits
    {
        static_assert(std::is_integral_v<T> && !std::is_same_v<T, bool>,
            "hpx::radix_sort currently supports integral projected keys "
            "only.");

        using unsigned_t = std::make_unsigned_t<T>;
        static constexpr int bits = std::numeric_limits<unsigned_t>::digits;

        static constexpr unsigned_t lift(T v) noexcept
        {
            if constexpr (std::is_signed_v<T>)
            {
                constexpr unsigned_t sign_bit = unsigned_t{1} << (bits - 1);
                return static_cast<unsigned_t>(v) ^ sign_bit;
            }
            else
            {
                return v;
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // Pipeline parameters.
    //
    // Radix width is fixed at 8 bits per pass (256 buckets) for the serial
    // implementation. The proposal calls out adaptive radix width as a
    // follow-up exploration.

    HPX_CXX_CORE_EXPORT inline constexpr int radix_sort_bits_per_pass = 8;
    HPX_CXX_CORE_EXPORT inline constexpr std::size_t radix_sort_buckets =
        std::size_t{1} << radix_sort_bits_per_pass;
    HPX_CXX_CORE_EXPORT inline constexpr std::size_t radix_sort_bucket_mask =
        radix_sort_buckets - 1;

    // Below this many elements, fall back to std::sort. Chosen so that the
    // overhead of histogramming + permutation does not dominate.
    HPX_CXX_CORE_EXPORT inline constexpr std::size_t radix_sort_base_case = 128;

    // At or above this many elements, use the blockwise distribution engine
    // instead of element-wise permutation. Below it the engine's buffers
    // (one block per bucket) would be mostly empty. Initial guess, to be
    // tuned with benchmarks.
    HPX_CXX_CORE_EXPORT template <typename T>
    inline constexpr std::size_t radix_sort_blockwise_threshold =
        radix_sort_buckets * ips_default_block_size<T>;

    HPX_CXX_CORE_EXPORT template <typename UInt>
    constexpr std::size_t radix_sort_digit(UInt key, int shift) noexcept
    {
        return static_cast<std::size_t>(key >> shift) & radix_sort_bucket_mask;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Stage 1+2: build a per-bucket histogram, then turn it into prefix-sum
    // boundaries. `bucket_begin[k]` ends up holding the offset of bucket k's
    // start; `bucket_end[k]` holds the offset of its end (== begin of k+1).

    HPX_CXX_CORE_EXPORT template <typename RandomIt, typename Proj>
    void radix_sort_classify(RandomIt first, RandomIt last, int shift,
        Proj& proj, std::array<std::size_t, radix_sort_buckets>& bucket_begin,
        std::array<std::size_t, radix_sort_buckets>& bucket_end)
    {
        bucket_begin.fill(0);

        using value_type = typename std::iterator_traits<RandomIt>::value_type;
        using key_t = std::decay_t<decltype(HPX_INVOKE(
            proj, std::declval<value_type const&>()))>;
        using traits = radix_key_traits<key_t>;

        for (RandomIt it = first; it != last; ++it)
        {
            auto const k = traits::lift(HPX_INVOKE(proj, *it));
            ++bucket_begin[radix_sort_digit(k, shift)];
        }

        std::size_t sum = 0;
        for (std::size_t b = 0; b < radix_sort_buckets; ++b)
        {
            std::size_t const cnt = bucket_begin[b];
            bucket_begin[b] = sum;
            sum += cnt;
            bucket_end[b] = sum;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Stage 3 (small partitions): in-place permutation by cycle walking.
    //
    // For each bucket, walk a cursor through its segment. Any element whose
    // current digit places it in a different bucket gets swapped into that
    // bucket's write head; we keep swapping until the slot under the cursor
    // is correct, then advance. This is O(n) swaps per pass and uses only
    // the 256 head/tail pointers as auxiliary state.
    //
    // Every swap touches a random bucket, so for large partitions this is
    // dominated by cache misses; those go through ips_distribute instead.

    HPX_CXX_CORE_EXPORT template <typename RandomIt, typename Proj>
    void radix_sort_permute(RandomIt first, int shift, Proj& proj,
        std::array<std::size_t, radix_sort_buckets> const& bucket_begin,
        std::array<std::size_t, radix_sort_buckets> const& bucket_end)
    {
        using value_type = typename std::iterator_traits<RandomIt>::value_type;
        using key_t = std::decay_t<decltype(HPX_INVOKE(
            proj, std::declval<value_type const&>()))>;
        using traits = radix_key_traits<key_t>;

        std::array<std::size_t, radix_sort_buckets> head = bucket_begin;

        for (std::size_t b = 0; b < radix_sort_buckets; ++b)
        {
            while (head[b] < bucket_end[b])
            {
                value_type& slot = first[static_cast<std::ptrdiff_t>(head[b])];
                std::size_t const target = radix_sort_digit(
                    traits::lift(HPX_INVOKE(proj, slot)), shift);

                if (target == b)
                {
                    ++head[b];
                }
                else
                {
                    using std::swap;
                    swap(
                        slot, first[static_cast<std::ptrdiff_t>(head[target])]);
                    ++head[target];
                }
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Stage 4: recurse on each non-trivial bucket at the next-lower digit.
    // Buckets of size <= radix_sort_base_case go through std::sort with a
    // projection-aware comparator built on the lifted key.
    //
    // `buf` holds the blockwise engine's buffers. It is allocated on first
    // use and reused by every recursive call, since the buffers are empty
    // again once ips_distribute returns.

    HPX_CXX_CORE_EXPORT template <typename RandomIt, typename Proj>
    void radix_sort_serial(RandomIt first, RandomIt last, int shift, Proj& proj,
        std::unique_ptr<ips_distribution_buffers<
            typename std::iterator_traits<RandomIt>::value_type>>& buf)
    {
        using value_type = typename std::iterator_traits<RandomIt>::value_type;
        using key_t = std::decay_t<decltype(HPX_INVOKE(
            proj, std::declval<value_type const&>()))>;
        using traits = radix_key_traits<key_t>;

        std::ptrdiff_t const n = last - first;
        if (n < 2)
            return;

        if (shift < 0 || static_cast<std::size_t>(n) <= radix_sort_base_case)
        {
            std::sort(
                first, last, [&proj](value_type const& a, value_type const& b) {
                    return traits::lift(HPX_INVOKE(proj, a)) <
                        traits::lift(HPX_INVOKE(proj, b));
                });
            return;
        }

        std::array<std::size_t, radix_sort_buckets> bucket_begin{};
        std::array<std::size_t, radix_sort_buckets> bucket_end{};

        if (static_cast<std::size_t>(n) >=
            radix_sort_blockwise_threshold<value_type>)
        {
            if (!buf)
            {
                buf = std::make_unique<ips_distribution_buffers<value_type>>(
                    radix_sort_buckets, ips_default_block_size<value_type>);
            }

            ips_distribute(
                first, static_cast<std::size_t>(n),
                [&proj, shift](value_type const& v) {
                    return radix_sort_digit(
                        traits::lift(HPX_INVOKE(proj, v)), shift);
                },
                *buf);

            for (std::size_t b = 0; b < radix_sort_buckets; ++b)
            {
                bucket_begin[b] = buf->bucket_start[b];
                bucket_end[b] = buf->bucket_start[b + 1];
            }
        }
        else
        {
            radix_sort_classify(
                first, last, shift, proj, bucket_begin, bucket_end);

            // Lazy pass skipping (proposal section 5): if every element fell
            // in the same bucket this pass would do no useful separation.
            bool single_bucket = false;
            for (std::size_t b = 0; b < radix_sort_buckets; ++b)
            {
                if (bucket_end[b] - bucket_begin[b] ==
                    static_cast<std::size_t>(n))
                {
                    single_bucket = true;
                    break;
                }
            }

            if (!single_bucket)
            {
                radix_sort_permute(
                    first, shift, proj, bucket_begin, bucket_end);
            }
        }

        int const next_shift = shift - radix_sort_bits_per_pass;
        for (std::size_t b = 0; b < radix_sort_buckets; ++b)
        {
            std::size_t const sz = bucket_end[b] - bucket_begin[b];
            if (sz < 2)
                continue;
            radix_sort_serial(
                first + static_cast<std::ptrdiff_t>(bucket_begin[b]),
                first + static_cast<std::ptrdiff_t>(bucket_end[b]), next_shift,
                proj, buf);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Algorithm entry struct. Both sequential() and parallel() currently go
    // through the serial pipeline; the parallel branch will be filled in by
    // later milestones (parallel histogram + blockwise permutation).

    HPX_CXX_CORE_EXPORT template <typename RandomIt>
    struct radix_sort_algo
      : public algorithm<radix_sort_algo<RandomIt>, RandomIt>
    {
        constexpr radix_sort_algo() noexcept
          : algorithm<radix_sort_algo, RandomIt>("radix_sort")
        {
        }

        template <typename ExPolicy, typename Sent, typename Proj>
        static RandomIt sequential(
            ExPolicy, RandomIt first, Sent last, Proj proj)
        {
            RandomIt last_iter = first;
            std::advance(last_iter, static_cast<std::ptrdiff_t>(last - first));

            using value_type =
                typename std::iterator_traits<RandomIt>::value_type;
            using key_t = std::decay_t<decltype(HPX_INVOKE(
                proj, std::declval<value_type const&>()))>;
            using traits = radix_key_traits<key_t>;

            int const start_shift = traits::bits - radix_sort_bits_per_pass;
            std::unique_ptr<ips_distribution_buffers<value_type>> buf;
            radix_sort_serial(first, last_iter, start_shift, proj, buf);
            return last_iter;
        }

        template <typename ExPolicy, typename Sent, typename Proj>
        static util::detail::algorithm_result_t<ExPolicy, RandomIt> parallel(
            ExPolicy&& policy, RandomIt first, Sent last, Proj proj)
        {
            // TODO(GSoC milestone Weeks 8-10): parallel histogram + blockwise
            // in-place permutation. Until then, run the serial pipeline on
            // the executor so the policy still routes work off-thread for
            // task policies.
            using algorithm_result =
                util::detail::algorithm_result<ExPolicy, RandomIt>;
            return algorithm_result::get(
                execution::async_execute(policy.executor(),
                    [first, last, proj = HPX_MOVE(proj)]() mutable {
                        return radix_sort_algo::sequential(
                            hpx::execution::seq, first, last, HPX_MOVE(proj));
                    }));
        }
    };

}    // namespace hpx::parallel::detail

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::radix_sort.
    HPX_CXX_CORE_EXPORT inline constexpr struct radix_sort_t final
      : hpx::detail::tag_dispatch<radix_sort_t,
            hpx::detail::tag_parallel_algorithm<radix_sort_t>>
    {
    private:
        template <typename Iter, typename Proj>
        using projected_key_t = std::decay_t<decltype(HPX_INVOKE(
            std::declval<Proj&>(), *std::declval<Iter>()))>;

        template <typename Iter, typename Proj>
        static constexpr bool is_radix_key_v =
            std::is_integral_v<projected_key_t<Iter, Proj>> &&
            !std::is_same_v<projected_key_t<Iter, Proj>, bool>;

    public:
        template <typename RandomIt, typename Proj = hpx::identity>
            requires(hpx::traits::is_iterator_v<RandomIt> &&
                is_radix_key_v<RandomIt, Proj>)
        static void invoke_default(
            RandomIt first, RandomIt last, Proj proj = Proj())
        {
            static_assert(std::random_access_iterator<RandomIt>,
                "hpx::radix_sort requires a random access iterator.");

            hpx::parallel::detail::radix_sort_algo<RandomIt>().call(
                hpx::execution::seq, first, last, HPX_MOVE(proj));
        }

        template <typename ExPolicy, typename RandomIt,
            typename Proj = hpx::identity>
            requires(hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_iterator_v<RandomIt> &&
                is_radix_key_v<RandomIt, Proj>)
        static parallel::util::detail::algorithm_result_t<ExPolicy>
        invoke_default(ExPolicy&& policy, RandomIt first, RandomIt last,
            Proj proj = Proj())
        {
            static_assert(std::random_access_iterator<RandomIt>,
                "hpx::radix_sort requires a random access iterator.");

            using result_type =
                typename hpx::parallel::util::detail::algorithm_result<
                    ExPolicy>::type;

            return hpx::util::void_guard<result_type>(),
                   hpx::parallel::detail::radix_sort_algo<RandomIt>().call(
                       HPX_FORWARD(ExPolicy, policy), first, last,
                       HPX_MOVE(proj));
        }
    } radix_sort{};
}    // namespace hpx

#endif    // DOXYGEN
