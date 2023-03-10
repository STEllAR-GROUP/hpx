//  Copyright (c) 2020-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/parallel/algorithms/detail/distance.hpp>

#include <iterator>

namespace hpx::parallel::detail {

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename Sent, typename T, typename F,
        typename Proj, typename CancelToken>
    Iter lower_bound(
        Iter first, Sent last, T&& value, F&& f, Proj&& proj, CancelToken& tok)
    {
        using difference_type =
            typename std::iterator_traits<Iter>::difference_type;

        difference_type count = detail::distance(first, last);
        while (count > 0)
        {
            if (tok.was_cancelled())
                break;

            difference_type step = count / 2;
            Iter it = std::next(first, step);

            if (HPX_INVOKE(f, HPX_INVOKE(proj, *it), value))
            {
                first = ++it;
                count -= step + 1;
            }
            else
            {
                count = step;
            }
        }
        return first;
    }

    template <typename Iter, typename Sent, typename T, typename F,
        typename Proj>
    constexpr Iter lower_bound(
        Iter first, Sent last, T&& value, F&& f, Proj&& proj)
    {
        using difference_type =
            typename std::iterator_traits<Iter>::difference_type;

        difference_type count = detail::distance(first, last);
        while (count > 0)
        {
            difference_type step = count / 2;
            Iter it = std::next(first, step);

            if (HPX_INVOKE(f, HPX_INVOKE(proj, *it), value))
            {
                first = ++it;
                count -= step + 1;
            }
            else
            {
                count = step;
            }
        }
        return first;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename Sent, typename T, typename F,
        typename Proj, typename CancelToken>
    Iter upper_bound(
        Iter first, Sent last, T&& value, F&& f, Proj&& proj, CancelToken& tok)
    {
        using difference_type =
            typename std::iterator_traits<Iter>::difference_type;

        difference_type count = detail::distance(first, last);
        while (count > 0)
        {
            if (tok.was_cancelled())
                break;

            difference_type step = count / 2;
            Iter it = std::next(first, step);

            if (!HPX_INVOKE(f, value, HPX_INVOKE(proj, *it)))
            {
                first = ++it;
                count -= step + 1;
            }
            else
            {
                count = step;
            }
        }
        return first;
    }

    template <typename Iter, typename Sent, typename T, typename F,
        typename Proj>
    constexpr Iter upper_bound(
        Iter first, Sent last, T&& value, F&& f, Proj&& proj)
    {
        using difference_type =
            typename std::iterator_traits<Iter>::difference_type;

        difference_type count = detail::distance(first, last);
        while (count > 0)
        {
            difference_type step = count / 2;
            Iter it = std::next(first, step);

            if (!HPX_INVOKE(f, value, HPX_INVOKE(proj, *it)))
            {
                first = ++it;
                count -= step + 1;
            }
            else
            {
                count = step;
            }
        }
        return first;
    }
}    // namespace hpx::parallel::detail
