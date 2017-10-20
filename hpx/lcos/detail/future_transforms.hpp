//  Copyright (c) 2017 Denis Blank
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_DETAIL_FUTURE_TRANSFORMS_HPP
#define HPX_LCOS_DETAIL_FUTURE_TRANSFORMS_HPP

#include <hpx/lcos/future.hpp>
#include <hpx/traits/acquire_future.hpp>
#include <hpx/traits/acquire_shared_state.hpp>
#include <hpx/traits/detail/reserve.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/util/deferred_call.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx {
namespace lcos {
    namespace detail {
        /// Returns true when the given future is ready,
        /// the future is deferred executed if possible first.
        template <typename T,
            typename std::enable_if<traits::is_future<
                typename std::decay<T>::type>::value>::type* = nullptr>
        bool async_visit_future(T&& current)
        {
            auto state =
                traits::detail::get_shared_state(std::forward<T>(current));

            if ((state.get() == nullptr) || state->is_ready())
            {
                return true;
            }

            // Execute_deferred might have made the future ready
            state->execute_deferred();

            // Detach the context if the future isn't ready
            return state->is_ready();
        }

        /// Attach the continuation next to the given future
        template <typename T, typename N,
            typename std::enable_if<traits::is_future<
                typename std::decay<T>::type>::value>::type* = nullptr>
        void async_detach_future(T&& current, N&& next)
        {
            auto state =
                traits::detail::get_shared_state(std::forward<T>(current));

            // Attach a continuation to this future which will
            // re-evaluate it and continue to the next argument (if any).
            state->set_on_completed(util::deferred_call(std::forward<N>(next)));
        }

        /// Acquire a future range from the given begin and end iterator
        template <typename Iterator,
            typename Container =
                std::vector<typename future_iterator_traits<Iterator>::type>>
        Container acquire_future_iterators(Iterator begin, Iterator end)
        {
            Container lazy_values;

            auto difference = std::distance(begin, end);
            if (difference > 0)
                traits::detail::reserve_if_reservable(
                    lazy_values, static_cast<std::size_t>(difference));

            std::transform(begin, end, std::back_inserter(lazy_values),
                traits::acquire_future_disp());

            return lazy_values;    // Should be optimized by RVO
        }

        /// Acquire a future range from the given
        /// begin iterator and count
        template <typename Iterator,
            typename Container =
                std::vector<typename future_iterator_traits<Iterator>::type>>
        Container acquire_future_n(Iterator begin, std::size_t count)
        {
            Container values;
            traits::detail::reserve_if_reservable(values, count);

            traits::acquire_future_disp func;
            for (std::size_t i = 0; i != count; ++i)
                values.push_back(func(*begin++));

            return values;    // Should be optimized by RVO
        }
    }    // end namespace detail
}    // end namespace lcos
}    // end namespace hpx

#endif    // HPX_LCOS_DETAIL_FUTURE_TRANSFORMS_HPP
