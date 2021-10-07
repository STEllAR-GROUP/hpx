//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <iterator>
#include <type_traits>

namespace hpx { namespace parallel { inline namespace v1 { namespace detail {

    template <typename Iter, typename Sent, typename Compare,
        typename Proj = hpx::parallel::util::projection_identity>
    inline constexpr bool is_sorted_sequential(
        Iter first, Sent last, Compare&& comp, Proj&& proj = Proj())
    {
        bool sorted = true;
        if (first != last)
        {
            for (Iter it1 = first, it2 = ++first; it2 != last &&
                 (sorted = !HPX_INVOKE(
                      comp, HPX_INVOKE(proj, *it2), HPX_INVOKE(proj, *it1)));
                 it1 = it2++)
            {
                /**/
            }
        }
        return sorted;
    }

    template <typename Iter, typename Sent, typename Compare,
        typename Proj = hpx::parallel::util::projection_identity>
    inline constexpr Iter is_sorted_until_sequential(
        Iter first, Sent last, Compare&& comp, Proj&& proj = Proj())
    {
        if (first != last)
        {
            Iter it1 = first;
            Iter it2 = ++first;
            for (; it2 != last &&
                 !HPX_INVOKE(
                     comp, HPX_INVOKE(proj, *it2), HPX_INVOKE(proj, *it1));
                 it1 = it2++)
            {
                /**/
            }

            return it2;
        }
        else
        {
            return first;
        }
    }

}}}}    // namespace hpx::parallel::v1::detail
