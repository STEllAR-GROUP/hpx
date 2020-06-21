//  Copyright (c) 2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/iterator_support/traits/is_sentinel_for.hpp>

#include <iterator>
#include <type_traits>

namespace hpx { namespace parallel { inline namespace v1 { namespace detail {
    // provide implementation of std::distance supporting iterators/sentinels
    template <typename InIterB, typename InIterE>
    constexpr inline typename std::iterator_traits<InIterB>::difference_type
    distance(InIterB first, InIterE last, std::false_type)
    {
        typename std::iterator_traits<InIterB>::difference_type offset = 0;
        for (/**/; first != last; ++first)
        {
            ++offset;
        }
        return offset;
    }

    template <typename RanIterB, typename RanIterE>
    constexpr inline typename std::iterator_traits<RanIterB>::difference_type
    distance(RanIterB first, RanIterE last, std::true_type)
    {
        return last - first;
    }

    template <typename InIterB, typename InIterE>
    constexpr inline typename std::iterator_traits<InIterB>::difference_type
    distance(InIterB first, InIterE last)
    {
        return distance(first, last,
            typename hpx::traits::is_sized_sentinel_for<InIterE,
                InIterB>::type{});
    }
}}}}    // namespace hpx::parallel::v1::detail
