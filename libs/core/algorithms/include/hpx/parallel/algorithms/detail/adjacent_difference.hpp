//  Copyright (c) 2021 Srinivas Yadav
//  Copyright (c) 2021 Karame M.shokooh
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/parallel/util/loop.hpp>

#include <type_traits>
#include <utility>

namespace hpx::parallel::detail {

    template <typename ExPolicy>
    struct sequential_adjacent_difference_t
      : hpx::functional::detail::tag_fallback<
            sequential_adjacent_difference_t<ExPolicy>>
    {
    private:
        template <typename InIter, typename Sent, typename OutIter, typename Op>
        friend constexpr inline OutIter tag_fallback_invoke(
            sequential_adjacent_difference_t<ExPolicy>, InIter first, Sent last,
            OutIter dest, Op&& op)
        {
            if (first == last)
                return dest;

            using value_t = typename std::iterator_traits<InIter>::value_type;
            value_t acc = *first;
            *dest = acc;
            while (++first != last)
            {
                value_t val = *first;
                *++dest = HPX_INVOKE(op, val, HPX_MOVE(acc));
                acc = HPX_MOVE(val);
            }
            return ++dest;
        }
    };

#if !defined(HPX_COMPUTE_DEVICE_CODE)
    template <typename ExPolicy>
    inline constexpr sequential_adjacent_difference_t<ExPolicy>
        sequential_adjacent_difference =
            sequential_adjacent_difference_t<ExPolicy>{};
#else
    template <typename ExPolicy, typename InIter, typename Sent,
        typename OutIter, typename Op>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr OutIter
    sequential_adjacent_difference(
        InIter first, Sent last, OutIter dest, Op&& op)
    {
        return sequential_adjacent_difference_t<ExPolicy>{}(
            first, last, dest, HPX_FORWARD(Op, op));
    }
#endif

}    // namespace hpx::parallel::detail
