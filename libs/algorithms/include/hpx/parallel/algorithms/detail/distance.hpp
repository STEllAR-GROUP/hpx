//  Copyright (c) 2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_DETAIL_DISTANCE_2019_FEB_02_0508PM)
#define HPX_PARALLEL_DETAIL_DISTANCE_2019_FEB_02_0508PM

#include <hpx/config.hpp>

#include <iterator>

namespace hpx { namespace parallel { inline namespace v1 { namespace detail {
    // provide implementation of std::distance supporting iterators/sentinels
    template <typename InIterB, typename InIterE>
    constexpr inline typename std::iterator_traits<InIterB>::difference_type
    distance(InIterB first, InIterE last, std::input_iterator_tag)
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
    distance(RanIterB first, RanIterE last, std::random_access_iterator_tag)
    {
        return last - first;
    }

    template <typename InIterB, typename InIterE>
    constexpr inline typename std::iterator_traits<InIterB>::difference_type
    distance(InIterB first, InIterE last)
    {
        return distance(first, last,
            typename std::iterator_traits<InIterB>::iterator_category{});
    }
}}}}    // namespace hpx::parallel::v1::detail

#endif
