//  Copyright (c) 2020-2022 STE||AR Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/parallel/algorithms/move.hpp>

namespace hpx::experimental { namespace ranges {
    template <typename Iter, typename T>
    struct in_value_result
    {
        [[no_unique_address]] Iter in;
        [[no_unique_address]] T value;

        template <class I2, class T2>
        constexpr operator in_value_result<I2, T2>() const&
        {
            return {in, value};
        }

        template <class I2, class T2>
        constexpr operator in_value_result<I2, T2>() &&
        {
            return {HPX_MOVE(in), HPX_MOVE(value)};
        }
    };
}}    // namespace hpx::experimental::ranges
