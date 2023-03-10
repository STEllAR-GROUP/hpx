//  Copyright (c) 2019 Hartmut Kaiser
//  Copyright (c) 2021 Akhil J Nair
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/iterator_support/traits/is_sentinel_for.hpp>

#include <iterator>
#include <type_traits>

namespace hpx::parallel::detail {

    // provide implementation of std::distance supporting iterators/sentinels
    template <typename InIterB, typename InIterE>
    constexpr typename std::iterator_traits<InIterB>::difference_type distance(
        InIterB first, InIterE last)
    {
        // we add this since passing in random access iterators
        // as begin and end might not pass the sized sentinel check
        if constexpr (std::is_same_v<InIterB, InIterE> &&
            hpx::traits::is_random_access_iterator_v<InIterB>)
        {
            return last - first;
        }

        if constexpr (hpx::traits::is_sized_sentinel_for_v<InIterE, InIterB>)
        {
            return last - first;
        }
        else
        {
            typename std::iterator_traits<InIterB>::difference_type offset = 0;
            for (/**/; first != last; ++first)
            {
                ++offset;
            }
            return offset;
        }
    }
}    // namespace hpx::parallel::detail
