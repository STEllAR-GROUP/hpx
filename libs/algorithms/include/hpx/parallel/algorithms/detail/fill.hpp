//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

namespace hpx { namespace parallel { inline namespace v1 { namespace detail {

    template <typename Iter, typename Sent, typename T>
    constexpr Iter sequential_fill(Iter first, Sent last, T const& value)
    {
        for (; first != last; ++first)
        {
            *first = value;
        }
        return first;
    }
}}}}    // namespace hpx::parallel::v1::detail
