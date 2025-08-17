//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/type_support/is_contiguous_iterator.hpp>

#include <memory>

namespace hpx::util {

    template <typename Iter>
    HPX_FORCEINLINE auto get_unwrapped(Iter it)
    {
        // is_contiguous_iterator_v is true for pointers
        if constexpr (hpx::traits::is_contiguous_iterator_v<Iter>)
        {
            return std::to_address(it);
        }
        else
        {
            return it;
        }
    }
}    // namespace hpx::util
