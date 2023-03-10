//  Copyright (c) 2019-2023 Hartmut Kaiser
//  Copyright (c) 2021 Akhil J Nair
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/iterator_support/traits/is_sentinel_for.hpp>

#include <iterator>
#include <type_traits>

namespace hpx::parallel::detail {

    // helper facility to both advance the iterator to the sentinel and return the
    // distance
    template <typename Iter, typename Sent>
    constexpr typename std::iterator_traits<Iter>::difference_type
    advance_and_get_distance(Iter& first, Sent last)
    {
        using difference_type =
            typename std::iterator_traits<Iter>::difference_type;

        // we add this since passing in random access iterators
        // as begin and end might not pass the sized sentinel check
        if constexpr (std::is_same_v<Iter, Sent>)
        {
            if constexpr (hpx::traits::is_random_access_iterator_v<Iter>)
            {
                difference_type offset = last - first;
                first = last;
                return offset;
            }
            else
            {
                difference_type offset = detail::distance(first, last);
                first = last;
                return offset;
            }
        }

        if constexpr (hpx::traits::is_sized_sentinel_for_v<Sent, Iter>)
        {
            difference_type offset = last - first;
            std::advance(first, offset);
            return offset;
        }
        else
        {
            difference_type offset = 0;
            for (/**/; first != last; ++first)
            {
                ++offset;
            }
            return offset;
        }
    }
}    // namespace hpx::parallel::detail
