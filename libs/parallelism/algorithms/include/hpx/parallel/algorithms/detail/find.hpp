//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/invoke.hpp>

namespace hpx { namespace parallel { inline namespace v1 { namespace detail {

    // provide implementation of std::find supporting iterators/sentinels
    template <typename Iterator, typename Sentinel, typename T>
    inline constexpr Iterator sequential_find(
        Iterator first, Sentinel last, T const& value)
    {
        for (; first != last; ++first)
        {
            if (*first == value)
            {
                return first;
            }
        }
        return last;
    }

    // provide implementation of std::find_if supporting iterators/sentinels
    template <typename Iterator, typename Sentinel, typename Pred>
    inline constexpr Iterator sequential_find_if(
        Iterator first, Sentinel last, Pred pred)
    {
        for (; first != last; ++first)
        {
            if (hpx::util::invoke(pred, *first))
            {
                return first;
            }
        }
        return last;
    }

    // provide implementation of std::find_if supporting iterators/sentinels
    template <typename Iterator, typename Sentinel, typename Pred>
    inline constexpr Iterator sequential_find_if_not(
        Iterator first, Sentinel last, Pred pred)
    {
        for (; first != last; ++first)
        {
            if (!hpx::util::invoke(pred, *first))
            {
                return first;
            }
        }
        return last;
    }
}}}}    // namespace hpx::parallel::v1::detail
