//  Copyright (c) 2007-2022 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//  Copyright (c) 2016 Lukas Troska
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file when_each.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx {
    /// The function \a when_each is an operator allowing to join on the results
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the event of all those futures
    /// having finished executing. It also calls the supplied callback
    /// for each of the futures which becomes ready.
    ///
    /// \param f        The function which will be called for each of the
    ///                 input futures once the future has become ready.
    ///
    /// \param futures  A vector holding an arbitrary amount of \a future or
    ///                 \a shared_future objects for which \a wait_each should
    ///                 wait.
    ///
    /// \note This function consumes the futures as they are passed on to the
    ///       supplied function. The callback should take one or two parameters,
    ///       namely either a \a future to be processed or a type that
    ///       \a std::size_t is implicitly convertible to as the
    ///       first parameter and the \a future as the second
    ///       parameter. The first parameter will correspond to the
    ///       index of the current \a future in the collection.
    ///
    /// \return   Returns a future representing the event of all input futures
    ///           being ready.
    ///
    template <typename F, typename Future>
    future<void> when_each(F&& f, std::vector<Future>&& futures);

    /// The function \a when_each is an operator allowing to join on the results
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the event of all those futures
    /// having finished executing. It also calls the supplied callback
    /// for each of the futures which becomes ready.
    ///
    /// \param f        The function which will be called for each of the
    ///                 input futures once the future has become ready.
    /// \param begin    The iterator pointing to the first element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a wait_each should wait.
    /// \param end      The iterator pointing to the last element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a wait_each should wait.
    ///
    /// \note This function consumes the futures as they are passed on to the
    ///       supplied function. The callback should take one or two parameters,
    ///       namely either a \a future to be processed or a type that
    ///       \a std::size_t is implicitly convertible to as the
    ///       first parameter and the \a future as the second
    ///       parameter. The first parameter will correspond to the
    ///       index of the current \a future in the collection.
    ///
    /// \return   Returns a future representing the event of all input futures
    ///           being ready.
    ///
    template <typename F, typename Iterator>
    future<Iterator> when_each(F&& f, Iterator begin, Iterator end);

    /// The function \a when_each is an operator allowing to join on the results
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the event of all those futures
    /// having finished executing. It also calls the supplied callback
    /// for each of the futures which becomes ready.
    ///
    /// \param f        The function which will be called for each of the
    ///                 input futures once the future has become ready.
    /// \param futures  An arbitrary number of \a future or \a shared_future
    ///                 objects, possibly holding different types for which
    ///                 \a wait_each should wait.
    ///
    /// \note This function consumes the futures as they are passed on to the
    ///       supplied function. The callback should take one or two parameters,
    ///       namely either a \a future to be processed or a type that
    ///       \a std::size_t is implicitly convertible to as the
    ///       first parameter and the \a future as the second
    ///       parameter. The first parameter will correspond to the
    ///       index of the current \a future in the collection.
    ///
    /// \return   Returns a future representing the event of all input futures
    ///           being ready.
    ///
    template <typename F, typename... Ts>
    future<void> when_each(F&& f, Ts&&... futures);

    /// The function \a when_each is an operator allowing to join on the results
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the event of all those futures
    /// having finished executing. It also calls the supplied callback
    /// for each of the futures which becomes ready.
    ///
    /// \param f        The function which will be called for each of the
    ///                 input futures once the future has become ready.
    /// \param begin    The iterator pointing to the first element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a wait_each_n should wait.
    /// \param count    The number of elements in the sequence starting at
    ///                 \a first.
    ///
    /// \note This function consumes the futures as they are passed on to the
    ///       supplied function. The callback should take one or two parameters,
    ///       namely either a \a future to be processed or a type that
    ///       \a std::size_t is implicitly convertible to as the
    ///       first parameter and the \a future as the second
    ///       parameter. The first parameter will correspond to the
    ///       index of the current \a future in the collection.
    ///
    /// \return   Returns a future holding the iterator pointing to the first
    ///           element after the last one.
    ///
    template <typename F, typename Iterator>
    future<Iterator> when_each_n(F&& f, Iterator begin, std::size_t count);
}    // namespace hpx

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/futures/traits/acquire_future.hpp>
#include <hpx/futures/traits/detail/future_traits.hpp>
#include <hpx/futures/traits/future_access.hpp>
#include <hpx/futures/traits/future_traits.hpp>
#include <hpx/futures/traits/is_future.hpp>
#include <hpx/futures/traits/is_future_range.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/type_support/decay.hpp>
#include <hpx/type_support/unwrap_ref.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::lcos::detail {

    template <typename Tuple, typename F>
    struct when_each_frame    //-V690
      : lcos::detail::future_data<void>
    {
        using type = hpx::future<void>;

    private:
        when_each_frame(when_each_frame const&) = delete;
        when_each_frame(when_each_frame&&) = delete;

        when_each_frame& operator=(when_each_frame const&) = delete;
        when_each_frame& operator=(when_each_frame&&) = delete;

        template <std::size_t I>
        struct is_end
          : std::integral_constant<bool, hpx::tuple_size<Tuple>::value == I>
        {
        };

        template <std::size_t I>
        static constexpr bool is_end_v = is_end<I>::value;

    public:
        template <typename Tuple_, typename F_>
        when_each_frame(Tuple_&& t, F_&& f, std::size_t needed_count)
          : t_(HPX_FORWARD(Tuple_, t))
          , f_(HPX_FORWARD(F_, f))
          , count_(0)
          , needed_count_(needed_count)
        {
        }

    public:
        template <std::size_t I>
        HPX_FORCEINLINE void do_await()
        {
            if constexpr (is_end_v<I>)
            {
                this->set_data(util::unused);
            }
            else
            {
                using future_type = hpx::util::decay_unwrap_t<
                    typename hpx::tuple_element<I, Tuple>::type>;

                if constexpr (hpx::traits::is_future_v<future_type> ||
                    hpx::traits::is_ref_wrapped_future_v<future_type>)
                {
                    await_future<I>();
                }
                else
                {
                    static_assert(hpx::traits::is_future_range_v<future_type> ||
                            hpx::traits::is_ref_wrapped_future_range_v<
                                future_type>,
                        "element must be future or range of futures");

                    auto&& curr = hpx::util::unwrap_ref(hpx::get<I>(t_));
                    await_range<I>(
                        hpx::util::begin(curr), hpx::util::end(curr));
                }
            }
        }

    protected:
        // Current element is a range (vector) of futures
        template <std::size_t I, typename Iter>
        void await_range(Iter&& next, Iter&& end)
        {
            using future_type = typename std::iterator_traits<Iter>::value_type;

            hpx::intrusive_ptr<when_each_frame> this_(this);
            for (/**/; next != end; ++next)
            {
                auto next_future_data = traits::detail::get_shared_state(*next);

                if (next_future_data &&
                    !next_future_data->is_ready(std::memory_order_relaxed))
                {
                    next_future_data->execute_deferred();

                    // execute_deferred might have made the future ready
                    if (!next_future_data->is_ready(std::memory_order_relaxed))
                    {
                        // Attach a continuation to this future which will
                        // re-evaluate it and continue to the next argument
                        // (if any).
                        next_future_data->set_on_completed(
                            [this_ = HPX_MOVE(this_), next = HPX_MOVE(next),
                                end = HPX_MOVE(end)]() mutable -> void {
                                this_->template await_range<I>(
                                    HPX_MOVE(next), HPX_MOVE(end));
                            });

                        // explicitly destruct iterators as those might
                        // become dangling after we make ourselves ready
                        next = std::decay_t<Iter>{};
                        end = std::decay_t<Iter>{};
                        return;
                    }
                }

                // call supplied callback with or without index
                if constexpr (hpx::is_invocable_v<F, std::size_t, future_type>)
                {
                    f_(count_, HPX_MOVE(*next));
                }
                else
                {
                    f_(HPX_MOVE(*next));
                }

                if (++count_ == needed_count_)
                {
                    this->set_data(util::unused);

                    // explicitly destruct iterators as those might
                    // become dangling after we make ourselves ready
                    next = std::decay_t<Iter>{};
                    end = std::decay_t<Iter>{};
                    return;
                }
            }

            do_await<I + 1>();
        }

        // Current element is a simple future
        template <std::size_t I>
        HPX_FORCEINLINE void await_future()
        {
            using future_type = hpx::util::decay_unwrap_t<
                typename hpx::tuple_element<I, Tuple>::type>;

            hpx::intrusive_ptr<when_each_frame> this_(this);

            future_type& fut = hpx::get<I>(t_);
            auto next_future_data = traits::detail::get_shared_state(fut);
            if (next_future_data &&
                !next_future_data->is_ready(std::memory_order_relaxed))
            {
                next_future_data->execute_deferred();

                // execute_deferred might have made the future ready
                if (!next_future_data->is_ready(std::memory_order_relaxed))
                {
                    // Attach a continuation to this future which will
                    // re-evaluate it and continue to the next argument
                    // (if any).
                    next_future_data->set_on_completed(
                        [this_ = HPX_MOVE(this_)]() -> void {
                            this_->template await_future<I>();
                        });

                    return;
                }
            }

            // call supplied callback with or without index
            if constexpr (hpx::is_invocable_v<F, std::size_t, future_type>)
            {
                f_(count_, HPX_MOVE(fut));
            }
            else
            {
                f_(HPX_MOVE(fut));
            }

            if (++count_ == needed_count_)
            {
                this->set_data(util::unused);
                return;
            }

            do_await<I + 1>();
        }

    private:
        Tuple t_;
        F f_;
        std::size_t count_;
        std::size_t needed_count_;
    };
}    // namespace hpx::lcos::detail

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    inline constexpr struct when_each_t final
      : hpx::functional::tag<when_each_t>
    {
    private:
        template <typename F, typename Future,
            typename Enable =
                std::enable_if_t<hpx::traits::is_future_v<Future>>>
        friend decltype(auto) tag_invoke(
            when_each_t, F&& func, std::vector<Future>& lazy_values)
        {
            using argument_type = hpx::tuple<std::vector<Future>>;
            using frame_type =
                lcos::detail::when_each_frame<argument_type, std::decay_t<F>>;

            std::vector<Future> values;
            values.reserve(lazy_values.size());

            std::transform(lazy_values.begin(), lazy_values.end(),
                std::back_inserter(values), traits::acquire_future_disp());

            hpx::intrusive_ptr<frame_type> p(
                new frame_type(hpx::forward_as_tuple(HPX_MOVE(values)),
                    HPX_FORWARD(F, func), values.size()));

            p->template do_await<0>();

            return hpx::traits::future_access<
                typename frame_type::type>::create(HPX_MOVE(p));
        }

        template <typename F, typename Future>
        friend decltype(auto) tag_invoke(
            when_each_t, F&& f, std::vector<Future>&& values)
        {
            return tag_invoke(when_each_t{}, HPX_FORWARD(F, f), values);
        }

        template <typename F, typename Iterator,
            typename Enable =
                std::enable_if_t<hpx::traits::is_iterator_v<Iterator>>>
        friend decltype(auto) tag_invoke(
            when_each_t, F&& f, Iterator begin, Iterator end)
        {
            using future_type =
                lcos::detail::future_iterator_traits_t<Iterator>;

            std::vector<future_type> values;
            traits::detail::reserve_if_random_access_by_range(
                values, begin, end);

            std::transform(begin, end, std::back_inserter(values),
                traits::acquire_future_disp());

            return tag_invoke(when_each_t{}, HPX_FORWARD(F, f), values)
                .then(hpx::launch::sync,
                    [end = HPX_MOVE(end)](hpx::future<void> fut) -> Iterator {
                        fut.get();    // rethrow exceptions, if any
                        return end;
                    });
        }

        template <typename F>
        friend decltype(auto) tag_invoke(when_each_t, F&&)
        {
            return hpx::make_ready_future();
        }

        template <typename F, typename... Ts,
            typename Enable =
                std::enable_if_t<!hpx::traits::is_future_v<std::decay_t<F>> &&
                    hpx::util::all_of_v<hpx::traits::is_future<Ts>...>>>
        friend decltype(auto) tag_invoke(when_each_t, F&& f, Ts&&... ts)
        {
            using argument_type = hpx::tuple<traits::acquire_future_t<Ts>...>;
            using frame_type =
                lcos::detail::when_each_frame<argument_type, std::decay_t<F>>;

            traits::acquire_future_disp func;
            argument_type values(func(HPX_FORWARD(Ts, ts))...);

            hpx::intrusive_ptr<frame_type> p(new frame_type(
                HPX_MOVE(values), HPX_FORWARD(F, f), sizeof...(Ts)));

            p->template do_await<0>();

            return hpx::traits::future_access<
                typename frame_type::type>::create(HPX_MOVE(p));
        }
    } when_each{};

    ///////////////////////////////////////////////////////////////////////////
    inline constexpr struct when_each_n_t final
      : hpx::functional::tag<when_each_n_t>
    {
    private:
        template <typename F, typename Iterator,
            typename Enable =
                std::enable_if_t<hpx::traits::is_iterator_v<Iterator>>>
        friend decltype(auto) tag_invoke(
            when_each_n_t, F&& f, Iterator begin, std::size_t count)
        {
            using future_type =
                lcos::detail::future_iterator_traits_t<Iterator>;

            std::vector<future_type> values;
            values.reserve(count);

            traits::acquire_future_disp func;
            while (count-- != 0)
            {
                values.push_back(func(*begin++));
            }

            return hpx::when_each(HPX_FORWARD(F, f), values)
                .then(hpx::launch::sync,
                    [begin = HPX_MOVE(begin)](auto&& fut) -> Iterator {
                        fut.get();    // rethrow exceptions, if any
                        return begin;
                    });
        }
    } when_each_n{};
}    // namespace hpx

namespace hpx::lcos {

    template <typename F, typename... Ts>
    HPX_DEPRECATED_V(
        1, 8, "hpx::lcos::when_each is deprecated. Use hpx::when_each instead.")
    auto when_each(F&& f, Ts&&... ts)
    {
        return hpx::when_each(HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
    }

    template <typename F, typename Iterator,
        typename Enable =
            std::enable_if_t<hpx::traits::is_iterator_v<Iterator>>>
    HPX_DEPRECATED_V(1, 8,
        "hpx::lcos::when_each_n is deprecated. Use hpx::when_each_n instead.")
    hpx::future<Iterator> when_each_n(F&& f, Iterator begin, std::size_t count)
    {
        return hpx::when_each_n(HPX_FORWARD(F, f), begin, count);
    }
}    // namespace hpx::lcos

#endif    // DOXYGEN
