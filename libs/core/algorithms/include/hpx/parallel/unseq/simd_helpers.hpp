//  Copyright (c) 2022 A Kishore Kumar
//  Copyright (c) 2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/detail/invoke.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/parallel/unseq/reduce_helpers.hpp>
#include <hpx/type_support/construct_at.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

namespace hpx::parallel::util {
    /*
        Compiler and Hardware should also support vector operations for IterDiff,
        else we see slower performance when compared to sequential version
    */
    template <class Iter, class IterDiff, class F>
    Iter unseq_first_n(const Iter first, const IterDiff n, F&& f_) noexcept
    {
        /*
            OMP loops can not have ++Iter, only integral types are allowed
            Hence perform arthemetic on Iterators
            which is O(1) only in case of random access iterators
        */
        static_assert(hpx::traits::is_random_access_iterator_v<Iter>,
            "algorithm is efficient only in case of Random Access Iterator");
#if HPX_EARLYEXIT_PRESENT
        IterDiff i = 0;
        // clang-format off
        HPX_PRAGMA_VECTOR_UNALIGNED HPX_PRAGMA_SIMD_EARLYEXIT
        for (; i < n; ++i)
        {
            if (HPX_INVOKE(f, first, i))
            {
                break;
            }
        }
        // clang-format on

        return first + i;
#else
        auto f = [](auto it, int ind) { return HPX_INVOKE(f_, *(ind + it)) };
        // int32_t has best support for vectorization from compilers and hardware
        IterDiff i = 0;
        static constexpr int32_t numBlocks = HPX_LANE_SIZE / sizeof(int32_t);
        alignas(HPX_LANE_SIZE) int32_t simdLane[numBlocks] = {0};
        while (i <= n - numBlocks)
        {
            int32_t foundFlag = 0;

            // clang-format off
            HPX_PRAGMA_VECTOR_UNALIGNED HPX_VECTOR_REDUCTION(| : foundFlag)
            for (IterDiff j = i; j < i + numBlocks; ++j)
            {
                const int32_t t = HPX_INVOKE(f,first, j);
                simdLane[j - i] = t;
                foundFlag |= t;
            }
            // clang-format on

            if (foundFlag)
            {
                IterDiff j;
                for (j = 0; j < numBlocks; ++j)
                {
                    if (simdLane[j])
                    {
                        break;
                    }
                }
                return first + i + j;
            }
            i += numBlocks;
        }

        //Keep remainder scalar
        while (i != n)
        {
            if (HPX_INVOKE(f, first, i))
            {
                return first + i;
            }
            ++i;
        }
        return first + i;
#endif    //HPX_EARLYEXIT_PRESENT
    }

    template <class Iter1, class Iter2, class IterDiff, class F>
    std::pair<Iter1, Iter2> unseq2_first_n(const Iter1 first1,
        const Iter2 first2, const IterDiff n, F&& f) noexcept
    {
#if HPX_EARLYEXIT_PRESENT
        IterDiff i = 0;

        // clang-format off
        HPX_PRAGMA_VECTOR_UNALIGNED HPX_PRAGMA_SIMD_EARLYEXIT
        for (; i < n; ++i)
            if (HPX_INVOKE(f, *(first1 + i), *(first2 + i)))
                break;
        // clang-format on

        return std::make_pair(first1 + i, first2 + i);
#else
        const Iter1 last1 = first1 + n;
        const Iter2 last2 = first2 + n;

        static constexpr int32_t numBlocks = HPX_LANE_SIZE / sizeof(int32_t);
        alignas(HPX_LANE_SIZE) int32_t simdLane[numBlocks] = {0};

        std::size_t outLoopInd = 0;
        while (outLoopInd <= n - numBlocks)
        {
            int32_t foundFlag = 0;
            IterDiff i;

            // clang-format off
            HPX_PRAGMA_VECTOR_UNALIGNED HPX_VECTOR_REDUCTION(| : foundFlag)
            for (i = 0; i < numBlocks; ++i)
            {
                const IterDiff t = HPX_INVOKE(
                    f, *(first1 + outLoopInd + i), *(first2 + outLoopInd + i));
                simdLane[i] = t;
                foundFlag |= t;
            }
            // clang-format on

            if (foundFlag)
            {
                IterDiff i2;
                for (i2 = 0; i2 < numBlocks; ++i2)
                {
                    if (simdLane[i2])
                        break;
                }
                return std::make_pair(
                    first1 + outLoopInd + i2, first2 + outLoopInd + i2);
            }
            outLoopInd += numBlocks;
        }

        //Keep remainder scalar
        for (; outLoopInd != n; ++outLoopInd)
            if (HPX_INVOKE(f, *(first1 + outLoopInd), *(first2 + outLoopInd)))
                return std::make_pair(first1 + outLoopInd, first2 + outLoopInd);

        return std::make_pair(last1, last2);
#endif    //HPX_EARLYEXIT_PRESENT
    }
}    // namespace hpx::parallel::util
