//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file lcos/wait_all.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx {
    /// The function \a wait_all is an operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns after they finished executing.
    ///
    /// \param first    The iterator pointing to the first element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a wait_all should wait.
    /// \param last     The iterator pointing to the last element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a wait_all should wait.
    ///
    /// \note The function \a wait_all returns after all futures have become
    ///       ready. All input futures are still valid after \a wait_all
    ///       returns. Exceptional futures will not cause wait_all to throw an
    ///       exception.
    ///
    template <typename InputIter>
    void wait_all(InputIter first, InputIter last);

    /// The function \a wait_all is an operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns after they finished executing.
    ///
    /// \param futures  A vector or array holding an arbitrary amount of
    ///                 \a future or \a shared_future objects for which
    ///                 \a wait_all should wait.
    ///
    /// \note The function \a wait_all returns after all futures have become
    ///       ready. All input futures are still valid after \a wait_all
    ///       returns. Exceptional futures will not cause wait_all to throw an
    ///       exception.
    ///
    template <typename R>
    void wait_all(std::vector<future<R>>&& futures);

    /// The function \a wait_all is an operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns after they finished executing.
    ///
    /// \param futures  A vector or array holding an arbitrary amount of
    ///                 \a future or \a shared_future objects for which
    ///                 \a wait_all should wait.
    ///
    /// \note The function \a wait_all returns after all futures have become
    ///       ready. All input futures are still valid after \a wait_all
    ///       returns. Exceptional futures will not cause wait_all to throw an
    ///       exception.
    ///
    template <typename R, std::size_t N>
    void wait_all(std::array<future<R>, N>&& futures);

    /// The function \a wait_all is an operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns after they finished executing.
    ///
    /// \param futures  An arbitrary number of \a future or \a shared_future
    ///                 objects, possibly holding different types for which
    ///                 \a wait_all should wait.
    ///
    /// \note The function \a wait_all returns after all futures have become
    ///       ready. All input futures are still valid after \a wait_all
    ///       returns. Exceptional futures will not cause wait_all to throw an
    ///       exception.
    ///
    template <typename... T>
    void wait_all(T&&... futures);

    /// The function \a wait_all_n is an operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns after they finished executing.
    ///
    /// \param begin    The iterator pointing to the first element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a wait_all_n should wait.
    /// \param count    The number of elements in the sequence starting at
    ///                 \a first.
    ///
    /// \return         The function \a wait_all_n will return an iterator
    ///                 referring to the first element in the input sequence
    ///                 after the last processed element.
    ///
    /// \note The function \a wait_all_n returns after all futures have become
    ///       ready. All input futures are still valid after \a wait_all_n
    ///       returns. Exceptional futures will not cause wait_all to throw an
    ///       exception.
    ///
    template <typename InputIter>
    InputIter wait_all_n(InputIter begin, std::size_t count);
}    // namespace hpx

#else    // DOXYGEN

#include <hpx/local/config.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/futures/detail/future_data.hpp>
#include <hpx/futures/traits/acquire_shared_state.hpp>
#include <hpx/futures/traits/detail/future_traits.hpp>
#include <hpx/futures/traits/future_access.hpp>
#include <hpx/futures/traits/is_future.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/type_support/always_void.hpp>
#include <hpx/type_support/decay.hpp>
#include <hpx/type_support/unwrap_ref.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos {

    // forward declare wait_all()
    template <typename Future>
    void wait_all(std::vector<Future>&& values);

    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        template <typename Future, typename Enable = void>
        struct is_future_or_shared_state : traits::is_future<Future>
        {
        };

        template <typename R>
        struct is_future_or_shared_state<
            hpx::intrusive_ptr<future_data_base<R>>> : std::true_type
        {
        };

        template <typename R>
        struct is_future_or_shared_state<std::reference_wrapper<R>>
          : is_future_or_shared_state<R>
        {
        };

        template <typename R>
        using is_future_or_shared_state_t =
            typename is_future_or_shared_state<R>::type;

        ///////////////////////////////////////////////////////////////////////
        template <typename Range, typename Enable = void>
        struct is_future_or_shared_state_range : std::false_type
        {
        };

        template <typename T>
        struct is_future_or_shared_state_range<std::vector<T>>
          : is_future_or_shared_state<T>
        {
        };

        template <typename T, std::size_t N>
        struct is_future_or_shared_state_range<std::array<T, N>>
          : is_future_or_shared_state<T>
        {
        };

        template <typename R>
        using is_future_or_shared_state_range_t =
            typename is_future_or_shared_state_range<R>::type;

        ///////////////////////////////////////////////////////////////////////
        template <typename Future, typename Enable = void>
        struct future_or_shared_state_result;

        template <typename Future>
        struct future_or_shared_state_result<Future,
            std::enable_if_t<traits::is_future_v<Future>>>
          : traits::future_traits<Future>
        {
        };

        template <typename R>
        struct future_or_shared_state_result<
            hpx::intrusive_ptr<future_data_base<R>>>
        {
            using type = R;
        };

        template <typename R>
        using future_or_shared_state_result_t =
            typename future_or_shared_state_result<R>::type;

        ///////////////////////////////////////////////////////////////////////
        template <typename Tuple>
        struct wait_all_frame    //-V690
          : hpx::lcos::detail::future_data<void>
        {
        private:
            using base_type = hpx::lcos::detail::future_data<void>;

            // workaround gcc regression wrongly instantiating constructors
            wait_all_frame();
            wait_all_frame(wait_all_frame const&);

            template <std::size_t I>
            struct is_end
              : std::integral_constant<bool, hpx::tuple_size<Tuple>::value == I>
            {
            };

        public:
            using init_no_addref = typename base_type::init_no_addref;

            wait_all_frame(Tuple const& t)
              : t_(t)
            {
            }

            wait_all_frame(init_no_addref no_addref, Tuple const& t)
              : base_type(no_addref)
              , t_(t)
            {
            }

        protected:
            // End of the tuple is reached
            template <std::size_t I>
            HPX_FORCEINLINE void do_await(std::true_type)
            {
                this->set_value(util::unused);    // simply make ourself ready
            }

            // Current element is a range (vector or array) of futures
            template <std::size_t I, typename Iter>
            void await_range(Iter&& next, Iter&& end)
            {
                using future_type =
                    typename std::iterator_traits<Iter>::value_type;
                using future_result_type =
                    detail::future_or_shared_state_result_t<future_type>;

                hpx::intrusive_ptr<wait_all_frame> this_(this);

                for (/**/; next != end; ++next)
                {
                    traits::detail::shared_state_ptr_t<future_result_type>
                        next_future_data =
                            traits::detail::get_shared_state(*next);

                    if (next_future_data.get() != nullptr &&
                        !next_future_data->is_ready())
                    {
                        next_future_data->execute_deferred();

                        // execute_deferred might have made the future ready
                        if (!next_future_data->is_ready())
                        {
                            // Attach a continuation to this future which will
                            // re-evaluate it and continue to the next element
                            // in the sequence (if any).
                            next_future_data->set_on_completed(
                                [this_ = std::move(this_),
                                    next = std::forward<Iter>(next),
                                    end = std::forward<Iter>(
                                        end)]() mutable -> void {
                                    return this_->template await_range<I>(
                                        std::move(next), std::move(end));
                                });

                            // explicitly destruct iterators as those might
                            // become dangling after we make ourselves ready
                            next = std::decay_t<Iter>{};
                            end = std::decay_t<Iter>{};
                            return;
                        }
                    }
                }

                // explicitly destruct iterators as those might become dangling
                // after we make ourselves ready
                next = std::decay_t<Iter>{};
                end = std::decay_t<Iter>{};

                // All elements of the sequence are ready now, proceed to the
                // next argument.
                do_await<I + 1>(is_end<I + 1>());
            }

            template <std::size_t I>
            HPX_FORCEINLINE void await_next(std::false_type, std::true_type)
            {
                await_range<I>(util::begin(util::unwrap_ref(hpx::get<I>(t_))),
                    util::end(util::unwrap_ref(hpx::get<I>(t_))));
            }

            // Current element is a simple future
            template <std::size_t I>
            HPX_FORCEINLINE void await_next(std::true_type, std::false_type)
            {
                using future_type = util::decay_unwrap_t<
                    typename hpx::tuple_element<I, Tuple>::type>;
                using future_result_type =
                    detail::future_or_shared_state_result_t<future_type>;

                hpx::intrusive_ptr<wait_all_frame> this_(this);
                traits::detail::shared_state_ptr_t<future_result_type>
                    next_future_data =
                        traits::detail::get_shared_state(hpx::get<I>(t_));

                if (next_future_data.get() != nullptr &&
                    !next_future_data->is_ready())
                {
                    next_future_data->execute_deferred();

                    // execute_deferred might have made the future ready
                    if (!next_future_data->is_ready())
                    {
                        // Attach a continuation to this future which will
                        // re-evaluate it and continue to the next argument
                        // (if any).
                        next_future_data->set_on_completed(
                            [this_ = std::move(this_)]() -> void {
                                return this_->template await_next<I>(
                                    std::true_type(), std::false_type());
                            });

                        return;
                    }
                }

                do_await<I + 1>(is_end<I + 1>());
            }

            template <std::size_t I>
            HPX_FORCEINLINE void do_await(std::false_type)
            {
                using future_type = util::decay_unwrap_t<
                    typename hpx::tuple_element<I, Tuple>::type>;

                using is_future =
                    detail::is_future_or_shared_state_t<future_type>;
                using is_range =
                    detail::is_future_or_shared_state_range_t<future_type>;

                await_next<I>(is_future(), is_range());
            }

        public:
            void wait_all()
            {
                do_await<0>(is_end<0>());

                // If there are still futures which are not ready, suspend and
                // wait.
                if (!this->is_ready())
                {
                    this->wait();
                }
            }

        private:
            Tuple const& t_;
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future>
    void wait_all(std::vector<Future> const& values)
    {
        using result_type = hpx::tuple<std::vector<Future> const&>;
        using frame_type = detail::wait_all_frame<result_type>;
        using init_no_addref = typename frame_type::init_no_addref;

        result_type data(values);
        hpx::intrusive_ptr<frame_type> frame(
            new frame_type(init_no_addref{}, data), false);
        frame->wait_all();
    }

    template <typename Future>
    HPX_FORCEINLINE void wait_all(std::vector<Future>& values)
    {
        lcos::wait_all(const_cast<std::vector<Future> const&>(values));
    }

    template <typename Future>
#if !defined(HPX_INTEL_VERSION)
    HPX_FORCEINLINE
#endif
        void
        wait_all(std::vector<Future>&& values)
    {
        lcos::wait_all(const_cast<std::vector<Future> const&>(values));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future, std::size_t N>
    void wait_all(std::array<Future, N> const& values)
    {
        using result_type = hpx::tuple<std::array<Future, N> const&>;
        using frame_type = detail::wait_all_frame<result_type>;
        using init_no_addref = typename frame_type::init_no_addref;

        result_type data(values);
        hpx::intrusive_ptr<frame_type> frame(
            new frame_type(init_no_addref{}, data), false);
        frame->wait_all();
    }

    template <typename Future, std::size_t N>
    HPX_FORCEINLINE void wait_all(std::array<Future, N>& values)
    {
        lcos::wait_all(const_cast<std::array<Future, N> const&>(values));
    }

    template <typename Future, std::size_t N>
    HPX_FORCEINLINE void wait_all(std::array<Future, N>&& values)
    {
        lcos::wait_all(const_cast<std::array<Future, N> const&>(values));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterator>
    typename util::always_void<
        lcos::detail::future_iterator_traits_t<Iterator>>::type
    wait_all(Iterator begin, Iterator end)
    {
        using future_type = lcos::detail::future_iterator_traits_t<Iterator>;
        using shared_state_ptr =
            traits::detail::shared_state_ptr_for_t<future_type>;
        using result_type = std::vector<shared_state_ptr>;

        result_type values;
        std::transform(begin, end, std::back_inserter(values),
            traits::detail::wait_get_shared_state<future_type>());

        lcos::wait_all(values);
    }

    template <typename Iterator>
    Iterator wait_all_n(Iterator begin, std::size_t count)
    {
        using future_type = lcos::detail::future_iterator_traits_t<Iterator>;
        using shared_state_ptr =
            traits::detail::shared_state_ptr_for_t<future_type>;
        using result_type = std::vector<shared_state_ptr>;

        result_type values;
        values.reserve(count);

        traits::detail::wait_get_shared_state<future_type> func;
        for (std::size_t i = 0; i != count; ++i)
        {
            values.push_back(func(*begin++));
        }

        lcos::wait_all(values);

        return begin;
    }

    inline void wait_all() {}

    ///////////////////////////////////////////////////////////////////////////
    template <typename... Ts>
    void wait_all(Ts&&... ts)
    {
        using result_type =
            hpx::tuple<traits::detail::shared_state_ptr_for_t<Ts>...>;
        using frame_type = detail::wait_all_frame<result_type>;
        using init_no_addref = typename frame_type::init_no_addref;

        result_type values =
            result_type(traits::detail::get_shared_state(ts)...);

        hpx::intrusive_ptr<frame_type> frame(
            new frame_type(init_no_addref{}, values), false);
        frame->wait_all();
    }
}}    // namespace hpx::lcos

namespace hpx {
    using lcos::wait_all;
    using lcos::wait_all_n;
}    // namespace hpx

#endif    // DOXYGEN
