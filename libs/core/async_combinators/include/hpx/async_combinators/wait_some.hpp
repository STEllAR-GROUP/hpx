//  Copyright (c) 2007-2023 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file wait_some.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx {
    /// The function \a wait_some is an operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the same list of futures
    /// after n of them finished executing.
    ///
    /// \param n        [in] The number of futures out of the arguments which
    ///                 have to become ready in order for the function to return.
    /// \param first    [in] The iterator pointing to the first element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a when_all should wait.
    /// \param last     [in] The iterator pointing to the last element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a when_all should wait.
    ///
    /// \note The function \a wait_some returns after \a n futures have become
    ///       ready. All input futures are still valid after \a wait_some
    ///       returns.
    ///
    /// \note           The function wait_some will rethrow any exceptions
    ///                 captured by the futures while becoming ready. If this
    ///                 behavior is undesirable, use \a wait_some_nothrow
    ///                 instead.
    ///
    template <typename InputIter>
    void wait_some(std::size_t n, InputIter first, InputIter last);

    /// The function \a wait_some is an operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the same list of futures
    /// after n of them finished executing.
    ///
    /// \param n        [in] The number of futures out of the arguments which
    ///                 have to become ready in order for the returned future
    ///                 to get ready.
    /// \param futures  [in] A vector holding an arbitrary amount of \a future
    ///                 or \a shared_future objects for which \a wait_some
    ///                 should wait.
    ///
    /// \note The function \a wait_some returns after \a n futures have become
    ///       ready. All input futures are still valid after \a wait_some
    ///       returns.
    ///
    /// \note           The function wait_some will rethrow any exceptions
    ///                 captured by the futures while becoming ready. If this
    ///                 behavior is undesirable, use \a wait_some_nothrow
    ///                 instead.
    ///
    template <typename R>
    void wait_some(std::size_t n, std::vector<future<R>>&& futures);

    /// The function \a wait_some is an operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the same list of futures
    /// after n of them finished executing.
    ///
    /// \param n        [in] The number of futures out of the arguments which
    ///                 have to become ready in order for the returned future
    ///                 to get ready.
    /// \param futures  [in] An array holding an arbitrary amount of \a future
    ///                 or \a shared_future objects for which \a wait_some
    ///                 should wait.
    ///
    /// \note The function \a wait_some returns after \a n futures have become
    ///       ready. All input futures are still valid after \a wait_some
    ///       returns.
    ///
    /// \note           The function wait_some will rethrow any exceptions
    ///                 captured by the futures while becoming ready. If this
    ///                 behavior is undesirable, use \a wait_some_nothrow
    ///                 instead.
    ///
    template <typename R, std::size_t N>
    void wait_some(std::size_t n, std::array<future<R>, N>&& futures);

    /// The function \a wait_some is an operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the same list of futures
    /// after n of them finished executing.
    ///
    /// \param n        [in] The number of futures out of the arguments which
    ///                 have to become ready in order for the returned future
    ///                 to get ready.
    /// \param futures  [in] An arbitrary number of \a future or \a shared_future
    ///                 objects, possibly holding different types for which
    ///                 \a wait_some should wait.
    /// \param ec       [in,out] this represents the error status on exit, if
    ///                 this is pre-initialized to \a hpx#throws the function
    ///                 will throw on error instead.
    ///
    /// \note The function \a wait_all returns after \a n futures have become
    ///       ready. All input futures are still valid after \a wait_some
    ///       returns.
    ///
    /// \note           The function wait_some will rethrow any exceptions
    ///                 captured by the futures while becoming ready. If this
    ///                 behavior is undesirable, use \a wait_some_nothrow
    ///                 instead.
    ///
    template <typename... T>
    void wait_some(std::size_t n, T&&... futures);

    /// The function \a wait_some_n is an operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the same list of futures
    /// after n of them finished executing.
    ///
    /// \param n        [in] The number of futures out of the arguments which
    ///                 have to become ready in order for the returned future
    ///                 to get ready.
    /// \param first    [in] The iterator pointing to the first element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a when_all should wait.
    /// \param count    [in] The number of elements in the sequence starting at
    ///                 \a first.
    ///
    /// \note The function \a wait_some_n returns after \a n futures have become
    ///       ready. All input futures are still valid after \a wait_some_n
    ///       returns.
    ///
    /// \note           The function wait_some_n will rethrow any exceptions
    ///                 captured by the futures while becoming ready. If this
    ///                 behavior is undesirable, use \a wait_some_n_nothrow
    ///                 instead.
    ///
    template <typename InputIter>
    void wait_some_n(std::size_t n, InputIter first, std::size_t count);
}    // namespace hpx
#else

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_combinators/detail/throw_if_exceptional.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/futures/traits/acquire_shared_state.hpp>
#include <hpx/futures/traits/detail/future_traits.hpp>
#include <hpx/futures/traits/future_access.hpp>
#include <hpx/futures/traits/is_future.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/preprocessor/strip_parens.hpp>
#include <hpx/type_support/pack.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx {

    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        template <typename Sequence>
        struct wait_some;

        template <typename Sequence>
        struct set_wait_some_callback_impl
        {
            explicit set_wait_some_callback_impl(
                wait_some<Sequence>& wait) noexcept
              : wait_(wait)
            {
            }

            template <typename SharedState>
            void operator()(SharedState const& shared_state) const
            {
                if constexpr (!traits::is_shared_state_v<SharedState>)
                {
                    apply(shared_state);
                }
                else
                {
                    std::size_t counter =
                        wait_.count_.load(std::memory_order_acquire);

                    if (counter < wait_.needed_count_ && shared_state)
                    {
                        if (!shared_state->is_ready(std::memory_order_relaxed))
                        {
                            // handle future only if not enough futures are
                            // ready yet also, do not touch any futures which
                            // are already ready
                            shared_state->execute_deferred();

                            // execute_deferred might have made the future ready
                            if (!shared_state->is_ready(
                                    std::memory_order_relaxed))
                            {
                                shared_state->set_on_completed(
                                    util::deferred_call(
                                        &wait_some<Sequence>::on_future_ready,
                                        wait_.shared_from_this(),
                                        hpx::execution_base::this_thread::
                                            agent()));
                                return;
                            }
                        }

                        // check whether the current future is exceptional
                        if (!wait_.has_exceptional_results_ &&
                            shared_state->has_exception())
                        {
                            wait_.has_exceptional_results_ = true;
                        }
                    }

                    if (wait_.count_.fetch_add(1) + 1 == wait_.needed_count_)
                    {
                        wait_.goal_reached_on_calling_thread_ = true;
                    }
                }
            }

            template <typename Tuple, std::size_t... Is>
            HPX_FORCEINLINE void apply(
                Tuple const& tuple, hpx::util::index_pack<Is...>) const
            {
                ((*this)(hpx::get<Is>(tuple)), ...);
            }

            template <typename... Ts>
            HPX_FORCEINLINE void apply(hpx::tuple<Ts...> const& sequence) const
            {
                apply(sequence, hpx::util::make_index_pack_t<sizeof...(Ts)>());
            }

            template <typename Sequence_>
            HPX_FORCEINLINE void apply(Sequence_ const& sequence) const
            {
                std::for_each(sequence.begin(), sequence.end(), *this);
            }

            wait_some<Sequence>& wait_;
        };

        template <typename Sequence>
        void set_on_completed_callback(wait_some<Sequence>& wait)
        {
            set_wait_some_callback_impl<Sequence> callback(wait);
            callback.apply(wait.values_);
        }

        template <typename Sequence>
        struct wait_some
          : std::enable_shared_from_this<wait_some<Sequence>>    //-V690
        {
        public:
            void on_future_ready(hpx::execution_base::agent_ref ctx)
            {
                if (count_.fetch_add(1) + 1 == needed_count_)
                {
                    // reactivate waiting thread only if it's not us
                    if (ctx != hpx::execution_base::this_thread::agent())
                    {
                        ctx.resume();
                    }
                    else
                    {
                        goal_reached_on_calling_thread_ = true;
                    }
                }
            }

        private:
            wait_some(wait_some const&) = delete;
            wait_some(wait_some&&) = delete;

            wait_some& operator=(wait_some const&) = delete;
            wait_some& operator=(wait_some&&) = delete;

        public:
            using argument_type = Sequence;

            wait_some(argument_type const& values, std::size_t n) noexcept
              : values_(values)
              , count_(0)
              , needed_count_(n)
            {
            }

            bool operator()()
            {
                // set callback functions to executed wait future is ready
                set_on_completed_callback(*this);

                // if all of the requested futures are already set, our
                // callback above has already been called often enough, otherwise
                // we suspend ourselves
                if (!goal_reached_on_calling_thread_)
                {
                    // wait for any of the futures to return to become ready
                    hpx::execution_base::this_thread::suspend(
                        "hpx::detail::wait_some::operator()");
                }

                // at least N futures should be ready
                HPX_ASSERT(
                    count_.load(std::memory_order_acquire) >= needed_count_);

                return has_exceptional_results_;
            }

            argument_type const& values_;
            std::atomic<std::size_t> count_;
            std::size_t const needed_count_;
            bool goal_reached_on_calling_thread_ = false;
            bool has_exceptional_results_ = false;
        };

        template <typename T>
        auto get_wait_some_frame(T const& values, std::size_t n)
        {
            return std::make_shared<hpx::detail::wait_some<T>>(values, n);
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    inline constexpr struct wait_some_nothrow_t final
      : hpx::functional::tag<wait_some_nothrow_t>
    {
    private:
        template <typename Future>
        static bool wait_some_nothrow_impl(
            std::size_t n, std::vector<Future> const& values)
        {
            static_assert(hpx::traits::is_future_v<Future>,
                "invalid use of hpx::wait_some");

            if (n == 0)
            {
                return false;
            }

            if (n > values.size())
            {
                HPX_THROW_EXCEPTION(hpx::error::bad_parameter, "hpx::wait_some",
                    "number of results to wait for is out of bounds");
                return false;
            }

            auto lazy_values = traits::acquire_shared_state_disp()(values);
            auto f = detail::get_wait_some_frame(lazy_values, n);

            return (*f)();
        }

        template <typename Future>
        friend bool tag_invoke(wait_some_nothrow_t, std::size_t n,
            std::vector<Future> const& values)
        {
            return wait_some_nothrow_t::wait_some_nothrow_impl(n, values);
        }

        template <typename Future>
        friend HPX_FORCEINLINE bool tag_invoke(
            wait_some_nothrow_t, std::size_t n, std::vector<Future>& values)
        {
            return wait_some_nothrow_t::wait_some_nothrow_impl(
                n, const_cast<std::vector<Future> const&>(values));
        }

        template <typename Future>
        friend HPX_FORCEINLINE bool tag_invoke(
            wait_some_nothrow_t, std::size_t n, std::vector<Future>&& values)
        {
            return wait_some_nothrow_t::wait_some_nothrow_impl(
                n, const_cast<std::vector<Future> const&>(values));
        }

        template <typename Future, std::size_t N>
        static bool wait_some_nothrow_impl(
            std::size_t n, std::array<Future, N> const& values)
        {
            static_assert(
                hpx::traits::is_future_v<Future>, "invalid use of wait_some");

            if (n == 0)
            {
                return false;
            }

            if (n > values.size())
            {
                HPX_THROW_EXCEPTION(hpx::error::bad_parameter, "hpx::wait_some",
                    "number of results to wait for is out of bounds");
                return false;
            }

            auto lazy_values = traits::acquire_shared_state_disp()(values);
            auto f = detail::get_wait_some_frame(lazy_values, n);

            return (*f)();
        }

        template <typename Future, std::size_t N>
        friend bool tag_invoke(wait_some_nothrow_t, std::size_t n,
            std::array<Future, N> const& values)
        {
            return wait_some_nothrow_t::wait_some_nothrow_impl(n, values);
        }

        template <typename Future, std::size_t N>
        friend HPX_FORCEINLINE bool tag_invoke(wait_some_nothrow_t,
            std::size_t n, std::array<Future, N>& lazy_values)
        {
            return wait_some_nothrow_t::wait_some_nothrow_impl(
                n, const_cast<std::array<Future, N> const&>(lazy_values));
        }

        template <typename Future, std::size_t N>
        friend HPX_FORCEINLINE bool tag_invoke(wait_some_nothrow_t,
            std::size_t n, std::array<Future, N>&& lazy_values)
        {
            return wait_some_nothrow_t::wait_some_nothrow_impl(
                n, const_cast<std::array<Future, N> const&>(lazy_values));
        }

        template <typename Iterator,
            typename Enable =
                std::enable_if_t<hpx::traits::is_iterator_v<Iterator>>>
        friend bool tag_invoke(
            wait_some_nothrow_t, std::size_t n, Iterator begin, Iterator end)
        {
            auto values = traits::acquire_shared_state<Iterator>()(begin, end);
            auto f = detail::get_wait_some_frame(values, n);

            return (*f)();
        }

        friend bool tag_invoke(wait_some_nothrow_t, std::size_t n)
        {
            if (n != 0)
            {
                HPX_THROW_EXCEPTION(hpx::error::bad_parameter,
                    "hpx::wait_some_nothrow",
                    "number of results to wait for is out of bounds");
            }
            return false;
        }

        template <typename T>
        friend bool tag_invoke(
            wait_some_nothrow_t, std::size_t n, hpx::future<T>&& f)
        {
            if (n != 1)
            {
                HPX_THROW_EXCEPTION(hpx::error::bad_parameter, "hpx::wait_some",
                    "number of results to wait for is out of bounds");
                return false;
            }

            f.wait();
            return f.has_exception();
        }

        template <typename T>
        friend bool tag_invoke(
            wait_some_nothrow_t, std::size_t n, hpx::shared_future<T>&& f)
        {
            if (n != 1)
            {
                HPX_THROW_EXCEPTION(hpx::error::bad_parameter, "hpx::wait_some",
                    "number of results to wait for is out of bounds");
                return false;
            }

            f.wait();
            return f.has_exception();
        }

        template <typename... Ts>
        friend bool tag_invoke(wait_some_nothrow_t, std::size_t n, Ts&&... ts)
        {
            if (n == 0)
            {
                return false;
            }

            if (n > sizeof...(Ts))
            {
                HPX_THROW_EXCEPTION(hpx::error::bad_parameter,
                    "hpx::lcos::wait_some",
                    "number of results to wait for is out of bounds");
                return false;
            }

            using result_type =
                hpx::tuple<traits::detail::shared_state_ptr_for_t<Ts>...>;

            result_type values(traits::detail::get_shared_state(ts)...);
            auto f = detail::get_wait_some_frame(values, n);

            return (*f)();
        }
    } wait_some_nothrow{};

    ///////////////////////////////////////////////////////////////////////////
    inline constexpr struct wait_some_t final
      : hpx::functional::tag<wait_some_t>
    {
    private:
        template <typename Future>
        friend void tag_invoke(
            wait_some_t, std::size_t n, std::vector<Future> const& values)
        {
            if (hpx::wait_some_nothrow(n, values))
            {
                hpx::detail::throw_if_exceptional(values);
            }
        }

        template <typename Future>
        friend void tag_invoke(
            wait_some_t, std::size_t n, std::vector<Future>& values)
        {
            if (hpx::wait_some_nothrow(
                    n, const_cast<std::vector<Future> const&>(values)))
            {
                hpx::detail::throw_if_exceptional(values);
            }
        }

        template <typename Future>
        friend void tag_invoke(
            wait_some_t, std::size_t n, std::vector<Future>&& values)
        {
            if (hpx::wait_some_nothrow(
                    n, const_cast<std::vector<Future> const&>(values)))
            {
                hpx::detail::throw_if_exceptional(values);
            }
        }

        template <typename Future, std::size_t N>
        friend void tag_invoke(wait_some_t, std::size_t n,
            std::array<Future, N> const& lazy_values)
        {
            if (hpx::wait_some_nothrow(n, lazy_values))
            {
                hpx::detail::throw_if_exceptional(lazy_values);
            }
        }

        template <typename Future, std::size_t N>
        friend void tag_invoke(
            wait_some_t, std::size_t n, std::array<Future, N>& lazy_values)
        {
            if (hpx::wait_some_nothrow(
                    n, const_cast<std::array<Future, N> const&>(lazy_values)))
            {
                hpx::detail::throw_if_exceptional(lazy_values);
            }
        }

        template <typename Future, std::size_t N>
        friend void tag_invoke(
            wait_some_t, std::size_t n, std::array<Future, N>&& lazy_values)
        {
            if (hpx::wait_some_nothrow(
                    n, const_cast<std::array<Future, N> const&>(lazy_values)))
            {
                hpx::detail::throw_if_exceptional(lazy_values);
            }
        }

        template <typename Iterator,
            typename Enable =
                std::enable_if_t<hpx::traits::is_iterator_v<Iterator>>>
        friend void tag_invoke(
            wait_some_t, std::size_t n, Iterator begin, Iterator end)
        {
            auto values = traits::acquire_shared_state<Iterator>()(begin, end);
            auto f = detail::get_wait_some_frame(values, n);

            if ((*f)())
            {
                hpx::detail::throw_if_exceptional(values);
            }
        }

        friend void tag_invoke(wait_some_t, std::size_t n)
        {
            if (n != 0)
            {
                HPX_THROW_EXCEPTION(hpx::error::bad_parameter, "hpx::wait_some",
                    "number of results to wait for is out of bounds");
            }
        }

        template <typename T>
        friend void tag_invoke(wait_some_t, std::size_t n, hpx::future<T>&& f)
        {
            if (hpx::wait_some_nothrow(n, HPX_MOVE(f)))
            {
                hpx::detail::throw_if_exceptional(f);
            }
        }

        template <typename T>
        friend void tag_invoke(
            wait_some_t, std::size_t n, hpx::shared_future<T>&& f)
        {
            if (hpx::wait_some_nothrow(n, HPX_MOVE(f)))
            {
                hpx::detail::throw_if_exceptional(f);
            }
        }

        template <typename... Ts>
        friend void tag_invoke(wait_some_t, std::size_t n, Ts&&... ts)
        {
            if (hpx::wait_some_nothrow(n, ts...))
            {
                hpx::detail::throw_if_exceptional(HPX_FORWARD(Ts, ts)...);
            }
        }
    } wait_some{};

    ///////////////////////////////////////////////////////////////////////////
    inline constexpr struct wait_some_n_nothrow_t final
      : hpx::functional::tag<wait_some_n_nothrow_t>
    {
    private:
        template <typename Iterator,
            typename Enable =
                std::enable_if_t<hpx::traits::is_iterator_v<Iterator>>>
        friend bool tag_invoke(wait_some_n_nothrow_t, std::size_t n,
            Iterator begin, std::size_t count)
        {
            auto values =
                traits::acquire_shared_state<Iterator>()(begin, count);
            auto f = detail::get_wait_some_frame(values, n);

            return (*f)();
        }
    } wait_some_n_nothrow{};

    ///////////////////////////////////////////////////////////////////////////
    inline constexpr struct wait_some_n_t final
      : hpx::functional::tag<wait_some_n_t>
    {
    private:
        template <typename Iterator,
            typename Enable =
                std::enable_if_t<hpx::traits::is_iterator_v<Iterator>>>
        friend void tag_invoke(
            wait_some_n_t, std::size_t n, Iterator begin, std::size_t count)
        {
            auto values =
                traits::acquire_shared_state<Iterator>()(begin, count);
            auto f = detail::get_wait_some_frame(values, n);

            if ((*f)())
            {
                hpx::detail::throw_if_exceptional(values);
            }
        }
    } wait_some_n{};
}    // namespace hpx

namespace hpx::lcos {

    template <typename Future>
    HPX_DEPRECATED_V(
        1, 8, "hpx::lcos::wait_some is deprecated. Use hpx::wait_some instead.")
    void wait_some(std::size_t n, std::vector<Future> const& lazy_values,
        error_code& = throws)
    {
        hpx::wait_some(n, lazy_values);
    }

    template <typename Future>
    HPX_DEPRECATED_V(
        1, 8, "hpx::lcos::wait_some is deprecated. Use hpx::wait_some instead.")
    void wait_some(
        std::size_t n, std::vector<Future>& lazy_values, error_code& = throws)
    {
        hpx::wait_some(n, const_cast<std::vector<Future> const&>(lazy_values));
    }

    template <typename Future>
    HPX_DEPRECATED_V(
        1, 8, "hpx::lcos::wait_some is deprecated. Use hpx::wait_some instead.")
    void wait_some(
        std::size_t n, std::vector<Future>&& lazy_values, error_code& = throws)
    {
        hpx::wait_some(n, const_cast<std::vector<Future> const&>(lazy_values));
    }

    template <typename Future, std::size_t N>
    HPX_DEPRECATED_V(
        1, 8, "hpx::lcos::wait_some is deprecated. Use hpx::wait_some instead.")
    void wait_some(std::size_t n, std::array<Future, N> const& lazy_values,
        error_code& = throws)
    {
        hpx::wait_some(n, lazy_values);
    }

    template <typename Future, std::size_t N>
    HPX_DEPRECATED_V(
        1, 8, "hpx::lcos::wait_some is deprecated. Use hpx::wait_some instead.")
    void wait_some(
        std::size_t n, std::array<Future, N>& lazy_values, error_code& = throws)
    {
        hpx::wait_some(
            n, const_cast<std::array<Future, N> const&>(lazy_values));
    }

    template <typename Future, std::size_t N>
    HPX_DEPRECATED_V(
        1, 8, "hpx::lcos::wait_some is deprecated. Use hpx::wait_some instead.")
    void wait_some(std::size_t n, std::array<Future, N>&& lazy_values,
        error_code& = throws)
    {
        hpx::wait_some(
            n, const_cast<std::array<Future, N> const&>(lazy_values));
    }

    template <typename Iterator,
        typename Enable =
            std::enable_if_t<hpx::traits::is_iterator_v<Iterator>>>
    HPX_DEPRECATED_V(
        1, 8, "hpx::lcos::wait_some is deprecated. Use hpx::wait_some instead.")
    void wait_some(
        std::size_t n, Iterator begin, Iterator end, error_code& = throws)
    {
        hpx::wait_some(n, begin, end);
    }

    template <typename Iterator,
        typename Enable =
            std::enable_if_t<hpx::traits::is_iterator_v<Iterator>>>
    HPX_DEPRECATED_V(
        1, 8, "hpx::lcos::wait_some is deprecated. Use hpx::wait_some instead.")
    Iterator wait_some_n(
        std::size_t n, Iterator begin, std::size_t count, error_code& = throws)
    {
        hpx::wait_some(n, begin, count);
    }

    HPX_DEPRECATED_V(
        1, 8, "hpx::lcos::wait_some is deprecated. Use hpx::wait_some instead.")
    inline void wait_some(std::size_t n, error_code& = throws)
    {
        hpx::wait_some(n);
    }

    template <typename T>
    HPX_DEPRECATED_V(
        1, 8, "hpx::lcos::wait_some is deprecated. Use hpx::wait_some instead.")
    void wait_some(std::size_t n, hpx::future<T>&& f, error_code& = throws)
    {
        hpx::wait_some(n, HPX_MOVE(f));
    }

    template <typename T>
    HPX_DEPRECATED_V(
        1, 8, "hpx::lcos::wait_some is deprecated. Use hpx::wait_some instead.")
    void wait_some(
        std::size_t n, hpx::shared_future<T>&& f, error_code& = throws)
    {
        hpx::wait_some(n, HPX_MOVE(f));
    }

    template <typename... Ts>
    HPX_DEPRECATED_V(
        1, 8, "hpx::lcos::wait_some is deprecated. Use hpx::wait_some instead.")
    void wait_some(std::size_t n, error_code&, Ts&&... ts)
    {
        hpx::wait_some(n, HPX_FORWARD(Ts, ts)...);
    }

    template <typename... Ts>
    HPX_DEPRECATED_V(
        1, 8, "hpx::lcos::wait_some is deprecated. Use hpx::wait_some instead.")
    void wait_some(std::size_t n, Ts&&... ts)
    {
        hpx::wait_some(n, HPX_FORWARD(Ts, ts)...);
    }
}    // namespace hpx::lcos

#endif    // DOXYGEN
