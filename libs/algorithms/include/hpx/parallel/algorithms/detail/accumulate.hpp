//  Copyright (c) 2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/invoke.hpp>

#include <functional>

namespace hpx { namespace parallel { inline namespace v1 { namespace detail {
    // provide implementation of std::accumulate supporting iterators/sentinels
    template <typename InIterB, typename InIterE, typename T, typename F>
    inline constexpr T accumulate(
        InIterB first, InIterE last, T value, F reduce_op)
    {
        for (/**/; first != last; ++first)
        {
            value = hpx::util::invoke(reduce_op, value, *first);
        }
        return value;
    }

    template <typename InIterB, typename InIterE, typename T>
    inline constexpr T accumulate(InIterB first, InIterE last, T value)
    {
        return accumulate(first, last, value, std::plus<T>());
    }
}}}}    // namespace hpx::parallel::v1::detail
