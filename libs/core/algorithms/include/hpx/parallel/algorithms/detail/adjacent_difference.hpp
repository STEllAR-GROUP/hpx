//  Copyright (c) 2021 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution/traits/is_execution_policy.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/functional/tag_fallback_dispatch.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <functional>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1 { namespace detail {

    template <typename ExPolicy>
    struct sequential_adjacent_difference_t
      : hpx::functional::tag_fallback<
            sequential_adjacent_difference_t<ExPolicy>>
    {
    private:
        template <typename InIter, typename OutIter, typename Op>
        friend inline OutIter tag_fallback_dispatch(
            sequential_adjacent_difference_t<ExPolicy>, InIter first,
            InIter last, OutIter dest, Op&& op)
        {
            return std::adjacent_difference(
                first, last, dest, std::forward<Op>(op));
        }
    };

#if !defined(HPX_COMPUTE_DEVICE_CODE)
    template <typename ExPolicy>
    HPX_INLINE_CONSTEXPR_VARIABLE sequential_adjacent_difference_t<ExPolicy>
        sequential_adjacent_difference =
            sequential_adjacent_difference_t<ExPolicy>{};
#else
    template <typename ExPolicy, typename InIter, typename OutIter, typename Op>
    HPX_HOST_DEVICE HPX_FORCEINLINE OutIter sequential_adjacent_difference(
        InIter first, InIter last, OutIter dest, Op&& op)
    {
        return sequential_adjacent_difference_t<ExPolicy>{}(
            first, last, dest, std::forward<Op>(op));
    }
#endif

}}}}    // namespace hpx::parallel::v1::detail
