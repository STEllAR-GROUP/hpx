//  Copyright (c) 2020 Hartmut Kaiser
//  Copyright (c) 2021 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

namespace hpx { namespace parallel { inline namespace v1 { namespace detail {

    // provide implementation of std::find supporting iterators/sentinels
    template <typename Iterator, typename Sentinel, typename T,
        typename Proj = util::projection_identity>
    inline constexpr Iterator sequential_find(
        Iterator first, Sentinel last, T const& value, Proj proj = Proj())
    {
        for (; first != last; ++first)
        {
            if (hpx::util::invoke(proj, *first) == value)
            {
                return first;
            }
        }
        return first;
    }

    // provide implementation of std::find_if supporting iterators/sentinels
    template <typename Iterator, typename Sentinel, typename Pred,
        typename Proj = util::projection_identity>
    inline constexpr Iterator sequential_find_if(
        Iterator first, Sentinel last, Pred pred, Proj proj = Proj())
    {
        for (; first != last; ++first)
        {
            if (hpx::util::invoke(pred, hpx::util::invoke(proj, *first)))
            {
                return first;
            }
        }
        return first;
    }

    // provide implementation of std::find_if supporting iterators/sentinels
    template <typename Iterator, typename Sentinel, typename Pred,
        typename Proj = util::projection_identity>
    inline constexpr Iterator sequential_find_if_not(
        Iterator first, Sentinel last, Pred pred, Proj proj = Proj())
    {
        for (; first != last; ++first)
        {
            if (!hpx::util::invoke(pred, hpx::util::invoke(proj, *first)))
            {
                return first;
            }
        }
        return first;
    }
}}}}    // namespace hpx::parallel::v1::detail
