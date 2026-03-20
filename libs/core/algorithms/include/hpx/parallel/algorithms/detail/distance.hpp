//  Copyright (c) 2019-2025 Hartmut Kaiser
//  Copyright (c) 2021 Akhil J Nair
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>

#include <iterator>
#include <type_traits>

namespace hpx { namespace parallel { namespace detail {

    // provide implementation of std::distance supporting iterators/sentinels
    template <typename InIterB, typename InIterE>
    constexpr typename std::iterator_traits<InIterB>::difference_type distance(
        InIterB first, InIterE last)
    {
#if defined(HPX_HAVE_CXX20_STD_CONCEPTS)
        if constexpr (std::is_same_v<InIterB, InIterE> &&
            std::random_access_iterator<InIterB>)
        {
            return last - first;
        }
        else if constexpr (std::sized_sentinel_for<InIterE, InIterB>)
        {
            return last - first;
        }
#else
        if constexpr (std::is_same<InIterB, InIterE>::value &&
            std::is_base_of<std::random_access_iterator_tag,
                typename std::iterator_traits<InIterB>::iterator_category>::
                value)
        {
            return last - first;
        }
#endif
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
}}}    // namespace hpx::parallel::detail
