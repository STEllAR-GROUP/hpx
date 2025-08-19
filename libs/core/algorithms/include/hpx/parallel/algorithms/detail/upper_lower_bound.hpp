//  Copyright (c) 2020-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/iterator_support/unwrap_iterator.hpp>
#include <hpx/parallel/algorithms/detail/distance.hpp>
#include <hpx/type_support/identity.hpp>

#include <algorithm>
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

    template <typename Iter, typename T, typename F, typename Proj>
    constexpr Iter lower_bound_n(Iter first,
        typename std::iterator_traits<Iter>::difference_type count, T&& value,
        F&& f, Proj&& proj)
    {
        using difference_type =
            typename std::iterator_traits<Iter>::difference_type;

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

    template <typename Iter, typename T, typename F>
    constexpr Iter lower_bound_n(Iter first,
        typename std::iterator_traits<Iter>::difference_type count, T&& value,
        F&& f, hpx::identity)
    {
        using difference_type =
            typename std::iterator_traits<Iter>::difference_type;

        if constexpr (hpx::traits::is_random_access_iterator_v<Iter>)
        {
            // This code enables the compiler to generate conditional mov
            // operations instead of branches

            auto begin = hpx::util::get_unwrapped(first);
            auto it = begin;
            while (count > 0)
            {
                difference_type const step = count / 2;
                auto const mid = std::next(it, step);

                auto next_it = it;
                auto next_count = step;

                if (HPX_INVOKE(f, *mid, value))
                {
                    next_it = std::next(mid);
                    next_count = count - (step + 1);
                }

                it = next_it;
                count = next_count;
            }
            return std::next(first, detail::distance(begin, it));
        }
        else
        {
            while (count > 0)
            {
                difference_type const step = count / 2;
                Iter const mid = std::next(first, step);

                if (HPX_INVOKE(f, *mid, value))
                {
                    first = std::next(mid);
                    count -= step + 1;
                }
                else
                {
                    count = step;
                }
            }
            return first;
        }
    }

    template <typename Iter, typename Sent, typename T, typename F,
        typename Proj>
    constexpr Iter lower_bound(
        Iter first, Sent last, T&& value, F&& f, Proj&& proj)
    {
        using difference_type =
            typename std::iterator_traits<Iter>::difference_type;

        difference_type count = detail::distance(first, last);

        return lower_bound_n(first, count, HPX_FORWARD(T, value),
            HPX_FORWARD(F, f), HPX_FORWARD(Proj, proj));
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

    template <typename Iter, typename T, typename F, typename Proj>
    constexpr Iter upper_bound_n(Iter first,
        typename std::iterator_traits<Iter>::difference_type count, T&& value,
        F&& f, Proj&& proj)
    {
        using difference_type =
            typename std::iterator_traits<Iter>::difference_type;

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

    template <typename Iter, typename T, typename F>
    constexpr Iter upper_bound_n(Iter first,
        typename std::iterator_traits<Iter>::difference_type count, T&& value,
        F&& f, hpx::identity)
    {
        using difference_type =
            typename std::iterator_traits<Iter>::difference_type;

        if constexpr (hpx::traits::is_random_access_iterator_v<Iter>)
        {
            // This code enables the compiler to generate conditional mov
            // operations instead of branches

            auto begin = hpx::util::get_unwrapped(first);
            auto it = begin;
            while (count > 0)
            {
                difference_type const step = count / 2;
                auto const mid = std::next(it, step);

                auto next_it = it;
                auto next_count = step;

                if (!HPX_INVOKE(f, value, *mid))
                {
                    next_it = std::next(mid);
                    next_count = count - (step + 1);
                }

                it = next_it;
                count = next_count;
            }
            return std::next(first, detail::distance(begin, it));
        }
        else
        {
            while (count > 0)
            {
                difference_type const step = count / 2;
                Iter const mid = std::next(first, step);

                if (!HPX_INVOKE(f, value, *mid))
                {
                    first = std::next(mid);
                    count -= step + 1;
                }
                else
                {
                    count = step;
                }
            }
            return first;
        }
    }

    template <typename Iter, typename Sent, typename T, typename F,
        typename Proj>
    constexpr Iter upper_bound(
        Iter first, Sent last, T&& value, F&& f, Proj&& proj)
    {
        using difference_type =
            typename std::iterator_traits<Iter>::difference_type;

        difference_type count = detail::distance(first, last);

        return upper_bound_n(first, count, HPX_FORWARD(T, value),
            HPX_FORWARD(F, f), HPX_FORWARD(Proj, proj));
    }
}    // namespace hpx::parallel::detail
