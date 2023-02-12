//  Copyright (c) 2020-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/iterator_support/traits/is_sentinel_for.hpp>

#include <iterator>
#include <type_traits>

namespace hpx::parallel::detail {

    // Generic implementation for advancing a given iterator to its sentinel
    template <typename Iter, typename Sent>
    constexpr Iter advance_to_sentinel(Iter first, Sent last)
    {
        if constexpr (std::is_same_v<Iter, Sent>)
        {
            return last;
        }
        else if constexpr (hpx::traits::is_sized_sentinel_for_v<Sent, Iter>)
        {
            std::advance(first, last - first);
            return first;
        }
        else
        {
            for (/**/; first != last; ++first)
            {
                /**/;
            }
            return first;
        }
    }
}    // namespace hpx::parallel::detail
