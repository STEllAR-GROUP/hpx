//  Copyright (c) 2019-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/invoke.hpp>

#include <functional>

namespace hpx::parallel::detail {

    // provide implementation of std::accumulate supporting iterators/sentinels
    template <typename Iter, typename Sent, typename T, typename F>
    constexpr T accumulate(Iter first, Sent last, T value, F&& reduce_op)
    {
        for (/**/; first != last; ++first)
        {
            value = HPX_INVOKE(reduce_op, HPX_MOVE(value), *first);
        }
        return value;
    }

    template <typename Iter, typename Sent, typename T>
    constexpr T accumulate(Iter first, Sent last, T value)
    {
        return accumulate(first, last, HPX_MOVE(value), std::plus<T>());
    }
}    // namespace hpx::parallel::detail
