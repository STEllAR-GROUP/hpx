//  Copyright (c) 2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <iterator>
#include <type_traits>

namespace hpx { namespace parallel { inline namespace v1 { namespace detail {

    // provide implementation of std::distance supporting iterators/sentinels
    // std::is_sorted is not available on all supported platforms yet
    template <typename Iter, typename Sent, typename Compare>
    inline bool is_sorted_sequential(Iter first, Sent last, Compare const& comp)
    {
        bool sorted = true;
        if (first != last)
        {
            for (Iter it1 = first, it2 = first + 1;
                 it2 != last && (sorted = !comp(*it2, *it1)); it1 = it2++)
            {
                /**/
            }
        }
        return sorted;
    }

}}}}    // namespace hpx::parallel::v1::detail
