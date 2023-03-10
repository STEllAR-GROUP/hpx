//  Copyright (c) 2007-2022 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file when_some.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    /// Result type for \a when_some, contains a sequence of futures and
    /// indices pointing to ready futures.
    template <typename Sequence>
    struct when_some_result
    {
        /// List of indices of futures that have become ready
        std::vector<std::size_t> indices;

        /// The sequence of futures as passed to \a hpx::when_some
        Sequence futures;
    };

    /// The function \a when_some is an operator allowing to join on the result
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
    /// \param last     [in] The iterator pointing to the last element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a when_all should wait.
    ///
    /// \note The future returned by the function \a when_some becomes ready
    ///       when at least \a n argument futures have become ready.
    ///
    /// \return   Returns a when_some_result holding the same list of futures
    ///           as has been passed to when_some and indices pointing to
    ///           ready futures.
    ///           - future<when_some_result<Container<future<R>>>>: If the input
    ///             cardinality is unknown at compile time and the futures
    ///             are all of the same type. The order of the futures in the
    ///             output container will be the same as given by the input
    ///             iterator.
    ///
    /// \note Calling this version of \a when_some where first == last, returns
    ///       a future with an empty container that is immediately ready.
    ///       Each future and shared_future is waited upon and then copied into
    ///       the collection of the output (returned) future, maintaining the
    ///       order of the futures in the input collection.
    ///       The future returned by \a when_some will not throw an exception,
    ///       but the futures held in the output collection may.
    ///
    template <typename InputIter,
        typename Container = vector<
            future<typename std::iterator_traits<InputIter>::value_type>>>
    future<when_some_result<Container>> when_some(
        std::size_t n, Iterator first, Iterator last);

    /// The function \a when_some is an operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the same list of futures
    /// after n of them finished executing.
    ///
    /// \param n        [in] The number of futures out of the arguments which
    ///                 have to become ready in order for the returned future
    ///                 to get ready.
    /// \param futures  [in] A container holding an arbitrary amount of \a future
    ///                 or \a shared_future objects for which \a when_some
    ///                 should wait.
    ///
    /// \note The future returned by the function \a when_some becomes ready
    ///       when at least \a n argument futures have become ready.
    ///
    /// \return   Returns a when_some_result holding the same list of futures
    ///           as has been passed to when_some and indices pointing to
    ///           ready futures.
    ///           - future<when_some_result<Container<future<R>>>>: If the input
    ///             cardinality is unknown at compile time and the futures
    ///             are all of the same type. The order of the futures in the
    ///             output container will be the same as given by the input
    ///             iterator.
    ///
    /// \note Each future and shared_future is waited upon and then copied into
    ///       the collection of the output (returned) future, maintaining the
    ///       order of the futures in the input collection.
    ///       The future returned by \a when_some will not throw an exception,
    ///       but the futures held in the output collection may.
    ///
    template <typename Range>
    future<when_some_result<Range>> when_some(std::size_t n, Range&& futures);

    /// The function \a when_some is an operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the same list of futures
    /// after n of them finished executing.
    ///
    /// \param n        [in] The number of futures out of the arguments which
    ///                 have to become ready in order for the returned future
    ///                 to get ready.
    /// \param futures  [in] An arbitrary number of \a future or \a shared_future
    ///                 objects, possibly holding different types for which
    ///                 \a when_some should wait.
    ///
    /// \note The future returned by the function \a when_some becomes ready
    ///       when at least \a n argument futures have become ready.
    ///
    /// \return   Returns a when_some_result holding the same list of futures
    ///           as has been passed to when_some and an index pointing to a
    ///           ready future..
    ///           - future<when_some_result<tuple<future<T0>, future<T1>...>>>:
    ///             If inputs are fixed in number and are of heterogeneous
    ///             types. The inputs can be any arbitrary number of future
    ///             objects.
    ///           - future<when_some_result<tuple<>>> if \a when_some is
    ///             called with zero arguments.
    ///             The returned future will be initially ready.
    ///
    /// \note Each future and shared_future is waited upon and then copied into
    ///       the collection of the output (returned) future, maintaining the
    ///       order of the futures in the input collection.
    ///       The future returned by \a when_some will not throw an exception,
    ///       but the futures held in the output collection may.
    ///
    template <typename... Ts>
    future<when_some_result<tuple<future<T>...>>> when_some(
        std::size_t n, Ts&&... futures);

    /// The function \a when_some_n is an operator allowing to join on the result
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
    /// \note The future returned by the function \a when_some_n becomes ready
    ///       when at least \a n argument futures have become ready.
    ///
    /// \return   Returns a when_some_result holding the same list of futures
    ///           as has been passed to when_some and indices pointing to
    ///           ready futures.
    ///           - future<when_some_result<Container<future<R>>>>: If the input
    ///             cardinality is unknown at compile time and the futures
    ///             are all of the same type. The order of the futures in the
    ///             output container will be the same as given by the input
    ///             iterator.
    ///
    /// \note Calling this version of \a when_some_n where count == 0, returns
    ///       a future with the same elements as the arguments that is
    ///       immediately ready. Possibly none of the futures in that container
    ///       are ready.
    ///       Each future and shared_future is waited upon and then copied into
    ///       the collection of the output (returned) future, maintaining the
    ///       order of the futures in the input collection.
    ///       The future returned by \a when_some_n will not throw an exception,
    ///       but the futures held in the output collection may.
    ///
    template <typename InputIter,
        typename Container = vector<
            future<typename std::iterator_traits<InputIter>::value_type>>>
    future<when_some_result<Container>> when_some_n(
        std::size_t n, Iterator first, std::size_t count);
}    // namespace hpx

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/futures/futures_factory.hpp>
#include <hpx/futures/traits/acquire_future.hpp>
#include <hpx/futures/traits/acquire_shared_state.hpp>
#include <hpx/futures/traits/detail/future_traits.hpp>
#include <hpx/futures/traits/future_access.hpp>
#include <hpx/futures/traits/is_future.hpp>
#include <hpx/futures/traits/is_future_range.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/type_support/pack.hpp>
#include <hpx/util/detail/reserve.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <iterator>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx {

    template <typename Sequence>
    struct when_some_result
    {
        when_some_result() = default;

        explicit when_some_result(Sequence&& futures) noexcept
          : indices()
          , futures(HPX_MOVE(futures))
        {
        }

        std::vector<std::size_t> indices;
        Sequence futures;
    };
}    // namespace hpx

namespace hpx::lcos::detail {

    ///////////////////////////////////////////////////////////////////////
    template <typename Sequence>
    struct when_some;

    template <typename Sequence>
    struct set_when_some_callback_impl
    {
        explicit set_when_some_callback_impl(when_some<Sequence>& when) noexcept
          : when_(when)
          , idx_(0)
        {
        }

        template <typename Future>
        std::enable_if_t<traits::is_future_v<Future>> operator()(
            Future& future) const
        {
            std::size_t counter = when_.count_.load(std::memory_order_seq_cst);
            if (counter < when_.needed_count_)
            {
                // handle future only if not enough futures are ready
                // yet also, do not touch any futures which are already
                // ready

                auto shared_state = traits::detail::get_shared_state(future);

                if (shared_state &&
                    !shared_state->is_ready(std::memory_order_relaxed))
                {
                    shared_state->execute_deferred();

                    // execute_deferred might have made the future ready
                    if (!shared_state->is_ready(std::memory_order_relaxed))
                    {
                        shared_state->set_on_completed(util::deferred_call(
                            &detail::when_some<Sequence>::on_future_ready,
                            when_.shared_from_this(), idx_,
                            hpx::execution_base::this_thread::agent()));
                        ++idx_;

                        return;
                    }
                }

                {
                    using mutex_type =
                        typename detail::when_some<Sequence>::mutex_type;
                    std::lock_guard<mutex_type> l(when_.mtx_);
                    when_.values_.indices.push_back(idx_);
                }

                if (when_.count_.fetch_add(1) + 1 == when_.needed_count_)
                {
                    when_.goal_reached_on_calling_thread_.store(
                        true, std::memory_order_release);
                }
            }
            ++idx_;
        }

        template <typename Sequence_>
        HPX_FORCEINLINE std::enable_if_t<traits::is_future_range_v<Sequence_>>
        operator()(Sequence_& sequence) const
        {
            apply(sequence);
        }

        template <typename Tuple, std::size_t... Is>
        HPX_FORCEINLINE void apply(Tuple& tuple, util::index_pack<Is...>) const
        {
            ((*this)(hpx::get<Is>(tuple)), ...);
        }

        template <typename... Ts>
        HPX_FORCEINLINE void apply(hpx::tuple<Ts...>& sequence) const
        {
            apply(sequence, util::make_index_pack_t<sizeof...(Ts)>());
        }

        template <typename Sequence_>
        HPX_FORCEINLINE void apply(Sequence_& sequence) const
        {
            std::for_each(sequence.begin(), sequence.end(), *this);
        }

        detail::when_some<Sequence>& when_;
        mutable std::size_t idx_;
    };

    template <typename Sequence>
    HPX_FORCEINLINE void set_on_completed_callback(
        detail::when_some<Sequence>& when)
    {
        set_when_some_callback_impl<Sequence> callback(when);
        callback.apply(when.values_.futures);
    }

    template <typename Sequence>
    struct when_some
      : std::enable_shared_from_this<when_some<Sequence>>    //-V690
    {
        using mutex_type = hpx::spinlock;

    public:
        void on_future_ready(
            std::size_t idx, hpx::execution_base::agent_ref ctx)
        {
            std::size_t const new_count = count_.fetch_add(1) + 1;
            if (new_count <= needed_count_)
            {
                {
                    std::lock_guard<mutex_type> l(this->mtx_);
                    values_.indices.push_back(idx);
                }

                if (new_count == needed_count_)
                {
                    if (ctx != hpx::execution_base::this_thread::agent())
                    {
                        ctx.resume();
                    }
                    else
                    {
                        goal_reached_on_calling_thread_.store(
                            true, std::memory_order_release);
                    }
                }
            }
        }

    private:
        when_some(when_some const&) = delete;
        when_some(when_some&&) = delete;

        when_some& operator=(when_some const&) = delete;
        when_some& operator=(when_some&&) = delete;

    public:
        using argument_type = Sequence;

        when_some(argument_type&& values, std::size_t n) noexcept
          : values_(HPX_MOVE(values))
          , count_(0)
          , needed_count_(n)
          , goal_reached_on_calling_thread_(false)
        {
        }

        when_some_result<Sequence> operator()()
        {
            // set callback functions to executed when future is ready
            set_on_completed_callback(*this);

            // if all of the requested futures are already set, our
            // callback above has already been called often enough, otherwise
            // we suspend ourselves
            if (!goal_reached_on_calling_thread_.load(
                    std::memory_order_acquire))
            {
                // wait for any of the futures to return to become ready
                hpx::execution_base::this_thread::suspend(
                    "hpx::lcos::detail::when_some::operator()");
            }

            // at least N futures should be ready
            HPX_ASSERT(count_.load(std::memory_order_acquire) >= needed_count_);

            return HPX_MOVE(values_);
        }

        mutable mutex_type mtx_;
        when_some_result<Sequence> values_;
        std::atomic<std::size_t> count_;
        std::size_t needed_count_;
        std::atomic<bool> goal_reached_on_calling_thread_;
    };
}    // namespace hpx::lcos::detail

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    inline constexpr struct when_some_t final
      : hpx::functional::tag<when_some_t>
    {
    private:
        template <typename Range,
            typename Enable =
                std::enable_if_t<traits::is_future_range_v<Range>>>
        friend auto tag_invoke(when_some_t, std::size_t n, Range&& lazy_values)
        {
            using result_type = std::decay_t<Range>;

            if (n == 0)
            {
                return hpx::make_ready_future(when_some_result<result_type>());
            }

            result_type values =
                traits::acquire_future<result_type>()(lazy_values);

            if (n > values.size())
            {
                return hpx::make_exceptional_future<
                    when_some_result<result_type>>(HPX_GET_EXCEPTION(
                    hpx::error::bad_parameter, "hpx::when_some",
                    "number of results to wait for is out of bounds"));
            }

            auto f = std::make_shared<lcos::detail::when_some<result_type>>(
                HPX_MOVE(values), n);

            lcos::local::futures_factory<when_some_result<result_type>()> p(
                [f = HPX_MOVE(f)]() -> when_some_result<result_type> {
                    return (*f)();
                });

            auto result = p.get_future();
            p.post();

            return result;
        }

        template <typename Iterator,
            typename Enable =
                std::enable_if_t<hpx::traits::is_iterator_v<Iterator>>>
        friend decltype(auto) tag_invoke(
            when_some_t, std::size_t n, Iterator begin, Iterator end)
        {
            using value_type = lcos::detail::future_iterator_traits_t<Iterator>;

            std::vector<value_type> values;
            traits::detail::reserve_if_random_access_by_range(
                values, begin, end);

            std::transform(begin, end, std::back_inserter(values),
                traits::acquire_future_disp());

            return tag_invoke(when_some_t{}, n, HPX_MOVE(values));
        }

        friend decltype(auto) tag_invoke(when_some_t, std::size_t n)
        {
            using result_type = hpx::tuple<>;

            if (n == 0)
            {
                return hpx::make_ready_future(when_some_result<result_type>());
            }

            return hpx::make_exceptional_future<when_some_result<result_type>>(
                HPX_GET_EXCEPTION(hpx::error::bad_parameter, "hpx::when_some",
                    "number of results to wait for is out of bounds"));
        }

        ///////////////////////////////////////////////////////////////////////////
        template <typename T, typename... Ts,
            typename Enable = std::enable_if_t<!(
                traits::is_future_range_v<T> && sizeof...(Ts) == 0)>>
        friend auto tag_invoke(when_some_t, std::size_t n, T&& t, Ts&&... ts)
        {
            using result_type = hpx::tuple<traits::acquire_future_t<T>,
                traits::acquire_future_t<Ts>...>;

            if (n == 0)
            {
                return hpx::make_ready_future(when_some_result<result_type>());
            }

            if (n > 1 + sizeof...(Ts))
            {
                return hpx::make_exceptional_future<
                    when_some_result<result_type>>(HPX_GET_EXCEPTION(
                    hpx::error::bad_parameter, "hpx::when_some",
                    "number of results to wait for is out of bounds"));
            }

            traits::acquire_future_disp func;
            result_type values(
                func(HPX_FORWARD(T, t)), func(HPX_FORWARD(Ts, ts))...);

            auto f = std::make_shared<lcos::detail::when_some<result_type>>(
                HPX_MOVE(values), n);

            lcos::local::futures_factory<when_some_result<result_type>()> p(
                [f = HPX_MOVE(f)]() -> when_some_result<result_type> {
                    return (*f)();
                });

            auto result = p.get_future();
            p.post();

            return result;
        }
    } when_some{};

    ///////////////////////////////////////////////////////////////////////////
    inline constexpr struct when_some_n_t final
      : hpx::functional::tag<when_some_n_t>
    {
    private:
        template <typename Iterator,
            typename Enable =
                std::enable_if_t<hpx::traits::is_iterator_v<Iterator>>>
        friend decltype(auto) tag_invoke(
            when_some_n_t, std::size_t n, Iterator begin, std::size_t count)
        {
            using value_type = lcos::detail::future_iterator_traits_t<Iterator>;

            std::vector<value_type> values;
            values.reserve(count);

            traits::acquire_future_disp func;
            for (std::size_t i = 0; i != count; ++i)
            {
                values.push_back(func(*begin++));
            }

            return hpx::when_some(n, HPX_MOVE(values));
        }
    } when_some_n{};
}    // namespace hpx

namespace hpx::lcos {

    template <typename Range>
    HPX_DEPRECATED_V(
        1, 8, "hpx::lcos::when_some is deprecated. Use hpx::when_some instead.")
    std::enable_if_t<traits::is_future_range_v<Range>,
        hpx::future<
            when_some_result<std::decay_t<Range>>>> when_some(std::size_t n,
        Range&& values, error_code& = throws)
    {
        return hpx::when_some(n, HPX_FORWARD(Range, values));
    }

    template <typename Iterator,
        typename Container =
            std::vector<lcos::detail::future_iterator_traits_t<Iterator>>,
        typename Enable =
            std::enable_if_t<hpx::traits::is_iterator_v<Iterator>>>
    HPX_DEPRECATED_V(
        1, 8, "hpx::lcos::when_some is deprecated. Use hpx::when_some instead.")
    hpx::future<when_some_result<Container>> when_some(
        std::size_t n, Iterator begin, Iterator end, error_code& = throws)
    {
        return hpx::when_some(n, begin, end);
    }

    template <typename Iterator,
        typename Container =
            std::vector<lcos::detail::future_iterator_traits_t<Iterator>>,
        typename Enable =
            std::enable_if_t<hpx::traits::is_iterator_v<Iterator>>>
    HPX_DEPRECATED_V(1, 8,
        "hpx::lcos::when_some_n is deprecated. Use hpx::when_some_n instead.")
    hpx::future<when_some_result<Container>> when_some_n(
        std::size_t n, Iterator begin, std::size_t count, error_code& = throws)
    {
        return hpx::when_some(n, begin, count);
    }

    HPX_DEPRECATED_V(
        1, 8, "hpx::lcos::when_some is deprecated. Use hpx::when_some instead.")
    inline hpx::future<when_some_result<hpx::tuple<>>> when_some(
        std::size_t n, error_code& = throws)
    {
        return hpx::when_some(n);
    }

    template <typename T, typename... Ts>
    HPX_DEPRECATED_V(
        1, 8, "hpx::lcos::when_some is deprecated. Use hpx::when_some instead.")
    std::enable_if_t<!(traits::is_future_range_v<T> && sizeof...(Ts) == 0),
        hpx::future<when_some_result<hpx::tuple<traits::acquire_future_t<T>,
            traits::acquire_future_t<Ts>...>>>> when_some(std::size_t n, T&& t,
        Ts&&... ts)
    {
        return hpx::when_some(n, HPX_FORWARD(T, t), HPX_FORWARD(Ts, ts)...);
    }

    template <typename T, typename... Ts>
    HPX_DEPRECATED_V(
        1, 8, "hpx::lcos::when_some is deprecated. Use hpx::when_some instead.")
    std::enable_if_t<!(traits::is_future_range_v<T> && sizeof...(Ts) == 0),
        hpx::future<when_some_result<hpx::tuple<traits::acquire_future_t<T>,
            traits::acquire_future_t<Ts>...>>>> when_some(std::size_t n,
        error_code&, T&& t, Ts&&... ts)
    {
        return hpx::when_some(n, HPX_FORWARD(T, t), HPX_FORWARD(Ts, ts)...);
    }

    template <typename Container>
    using when_some_result HPX_DEPRECATED_V(1, 8,
        "hpx::lcos::when_some_result is deprecated. Use hpx::when_some_result "
        "instead.") = hpx::when_some_result<Container>;
}    // namespace hpx::lcos

#endif    // DOXYGEN
