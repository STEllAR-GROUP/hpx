//  Copyright (c) 2017 Denis Blank
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/functional/deferred_call.hpp>
#include <hpx/futures/traits/acquire_future.hpp>
#include <hpx/futures/traits/acquire_shared_state.hpp>
#include <hpx/futures/traits/detail/future_traits.hpp>
#include <hpx/futures/traits/is_future.hpp>
#include <hpx/util/detail/reserve.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx::lcos::detail {

    // Returns true when the given future is ready,
    // the future is deferred executed if possible first.
    template <typename T,
        std::enable_if_t<traits::is_future_v<std::decay_t<T>>>* = nullptr>
    bool async_visit_future(T&& current)
    {
        // Check for state right away as the element might not be able to
        // produce a shared state (even if it's ready).
        auto const& state =
            traits::detail::get_shared_state(HPX_FORWARD(T, current));

        if (!state || state->is_ready(std::memory_order_relaxed))
        {
            return true;
        }

        // Execute_deferred might make the future ready
        state->execute_deferred();

        // Detach the context if the future isn't ready
        return state->is_ready();
    }

    // Attach the continuation next to the given future
    template <typename T, typename N,
        std::enable_if_t<traits::is_future_v<std::decay_t<T>>>* = nullptr>
    void async_detach_future(T&& current, N&& next)
    {
        auto const& state =
            traits::detail::get_shared_state(HPX_FORWARD(T, current));

        // Attach a continuation to this future which will
        // re-evaluate it and continue to the next argument (if any).
        state->set_on_completed(util::deferred_call(HPX_FORWARD(N, next)));
    }

    // Acquire a future range from the given begin and end iterator
    template <typename Iterator,
        typename Container = std::vector<future_iterator_traits_t<Iterator>>>
    Container acquire_future_iterators(Iterator begin, Iterator end)
    {
        Container lazy_values;

        auto difference = std::distance(begin, end);
        if (difference > 0)
        {
            traits::detail::reserve_if_reservable(
                lazy_values, static_cast<std::size_t>(difference));
        }

        std::transform(begin, end, std::back_inserter(lazy_values),
            traits::acquire_future_disp());

        return lazy_values;
    }

    // Acquire a future range from the given begin iterator and count
    template <typename Iterator,
        typename Container = std::vector<future_iterator_traits_t<Iterator>>>
    Container acquire_future_n(Iterator begin, std::size_t count)
    {
        Container values;
        traits::detail::reserve_if_reservable(values, count);

        traits::acquire_future_disp func;
        for (std::size_t i = 0; i != count; ++i)
        {
            values.push_back(func(*begin++));
        }

        return values;
    }
}    // namespace hpx::lcos::detail
