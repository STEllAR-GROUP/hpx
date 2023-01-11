//  Copyright (c) 2007-2022 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file when_any.hpp

#pragma once

#if defined(DOXYGEN)
/// Top level HPX namespace
namespace hpx {
    ///////////////////////////////////////////////////////////////////////////
    /// Result type for \a when_any, contains a sequence of futures and an
    /// index pointing to a ready future.
    template <typename Sequence>
    struct when_any_result
    {
        std::size_t index;    ///< The index of a future which has become ready
        Sequence futures;     ///< The sequence of futures as passed to
                              ///< \a hpx::when_any
    };

    /// \brief function \a when_any creates a future object that becomes
    ///        when at least one element in a set of \a future and \a shared_future
    ///        objects becomes ready. It is a non-deterministic choice operator.
    ///        It OR-composes all given future objects and returns a new future
    ///        object representing the same list of futures after one future of
    ///        that list finishes execution.
    ///
    /// \param first    [in] The iterator pointing to the first element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a when_any should wait.
    /// \param last     [in] The iterator pointing to the last element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a when_any should wait.
    ///
    /// \return   Returns a when_any_result holding the same list of futures
    ///           as has been passed to when_any and an index pointing to a
    ///           ready future.
    ///           - future<when_any_result<Container<future<R>>>>: If the input
    ///             cardinality is unknown at compile time and the futures
    ///             are all of the same type. The order of the futures in the
    ///             output container will be the same as given by the input
    ///             iterator.
    template <typename InputIter,
        typename Container = vector<
            future<typename std::iterator_traits<InputIter>::value_type>>>
    future<when_any_result<Container>> when_any(
        InputIter first, InputIter last);

    /// \copybrief when_any(InputIter first, InputIter last)
    ///
    /// \param values   [in] A range holding an arbitrary amount of \a futures
    ///                 or \a shared_future objects for which \a when_any should
    ///                 wait.
    ///
    /// \return   Returns a when_any_result holding the same list of futures
    ///           as has been passed to when_any and an index pointing to a
    ///           ready future.
    ///           - future<when_any_result<Container<future<R>>>>: If the input
    ///             cardinality is unknown at compile time and the futures
    ///             are all of the same type. The order of the futures in the
    ///             output container will be the same as given by the input
    ///             iterator.
    template <typename Range>
    future<when_any_result<Range>> when_any(Range& values);

    /// \copybrief when_any(InputIter first, InputIter last)
    ///
    /// \param futures  [in] An arbitrary number of \a future or \a shared_future
    ///                 objects, possibly holding different types for which
    ///                 \a when_any should wait.
    ///
    /// \return   Returns a when_any_result holding the same list of futures
    ///           as has been passed to when_any and an index pointing to a
    ///           ready future..
    ///           - future<when_any_result<tuple<future<T0>, future<T1>...>>>:
    ///             If inputs are fixed in number and are of heterogeneous
    ///             types. The inputs can be any arbitrary number of future
    ///             objects.
    ///           - future<when_any_result<tuple<>>> if \a when_any is called
    ///             with zero arguments.
    ///             The returned future will be initially ready.
    template <typename... T>
    future<when_any_result<tuple<future<T>...>>> when_any(T&&... futures);

    /// \brief function \a when_any_n creates a future object that becomes
    ///        when at least one element in a set of \a future and \a shared_future
    ///        objects becomes ready. It is a non-deterministic choice operator.
    ///        It OR-composes all given future objects and returns a new future
    ///        object representing the same list of futures after one future of
    ///        that list finishes execution.
    ///
    /// \param first    [in] The iterator pointing to the first element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a when_any_n should wait.
    /// \param count    [in] The number of elements in the sequence starting at
    ///                 \a first.
    ///
    /// \return   Returns a when_any_result holding the same list of futures
    ///           as has been passed to when_any and an index pointing to a
    ///           ready future.
    ///           - future<when_any_result<Container<future<R>>>>: If the input
    ///             cardinality is unknown at compile time and the futures
    ///             are all of the same type. The order of the futures in the
    ///             output container will be the same as given by the input
    ///             iterator.
    ///
    /// \note     None of the futures in the input sequence are invalidated.
    template <typename InputIter,
        typename Container = vector<
            future<typename std::iterator_traits<InputIter>::value_type>>>
    future<when_any_result<Container>> when_any_n(
        InputIter first, std::size_t count);
}    // namespace hpx

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_combinators/when_any.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/execution_base/this_thread.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/futures/futures_factory.hpp>
#include <hpx/futures/traits/acquire_future.hpp>
#include <hpx/futures/traits/detail/future_traits.hpp>
#include <hpx/futures/traits/future_access.hpp>
#include <hpx/futures/traits/is_future.hpp>
#include <hpx/futures/traits/is_future_range.hpp>
#include <hpx/type_support/pack.hpp>
#include <hpx/util/detail/reserve.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx {

    template <typename Sequence>
    struct when_any_result
    {
        static constexpr std::size_t index_error() noexcept
        {
            return static_cast<std::size_t>(-1);
        }

        when_any_result() noexcept
          : index(index_error())
          , futures()
        {
        }

        explicit when_any_result(Sequence&& futures) noexcept
          : index(index_error())
          , futures(HPX_MOVE(futures))
        {
        }

        when_any_result(when_any_result const& rhs)
          : index(rhs.index)
          , futures(rhs.futures)
        {
        }

        when_any_result(when_any_result&& rhs) noexcept
          : index(rhs.index)
          , futures(HPX_MOVE(rhs.futures))
        {
            rhs.index = index_error();
        }

        when_any_result& operator=(when_any_result const& rhs)
        {
            if (this != &rhs)
            {
                index = rhs.index;
                futures = rhs.futures;
            }
            return *this;
        }

        when_any_result& operator=(when_any_result&& rhs) noexcept
        {
            if (this != &rhs)
            {
                index = rhs.index;
                rhs.index = index_error();
                futures = HPX_MOVE(rhs.futures);
            }
            return *this;
        }

        std::size_t index;
        Sequence futures;
    };
}    // namespace hpx

namespace hpx::lcos::detail {

    ///////////////////////////////////////////////////////////////////////
    template <typename Sequence>
    struct when_any;

    template <typename Sequence>
    struct set_when_any_callback_impl
    {
        explicit set_when_any_callback_impl(when_any<Sequence>& when) noexcept
          : when_(when)
          , idx_(0)
        {
        }

        template <typename Future>
        std::enable_if_t<hpx::traits::is_future_v<Future>> operator()(
            Future& future) const
        {
            std::size_t index = when_.index_.load(std::memory_order_seq_cst);

            if (index == when_any_result<Sequence>::index_error())
            {
                using shared_state_ptr =
                    hpx::traits::detail::shared_state_ptr_for_t<Future>;
                shared_state_ptr const& shared_state =
                    traits::detail::get_shared_state(future);

                if (shared_state &&
                    !shared_state->is_ready(std::memory_order_relaxed))
                {
                    // handle future only if not enough futures are ready
                    // yet also, do not touch any futures which are already
                    // ready
                    shared_state->execute_deferred();

                    // execute_deferred might have made the future ready
                    if (!shared_state->is_ready(std::memory_order_relaxed))
                    {
                        shared_state->set_on_completed(util::deferred_call(
                            &detail::when_any<Sequence>::on_future_ready,
                            when_.shared_from_this(), idx_,
                            hpx::execution_base::this_thread::agent()));
                        ++idx_;
                        return;
                    }
                }

                if (when_.index_.compare_exchange_strong(index, idx_))
                {
                    when_.goal_reached_on_calling_thread_ = true;
                }
            }
            ++idx_;
        }

        template <typename Sequence_>
        HPX_FORCEINLINE
            std::enable_if_t<hpx::traits::is_future_range_v<Sequence_>>
            operator()(Sequence_& sequence) const
        {
            apply(sequence);
        }

        template <typename Tuple, std::size_t... Is>
        HPX_FORCEINLINE void apply(
            Tuple& tuple, hpx::util::index_pack<Is...>) const
        {
            ((*this)(hpx::get<Is>(tuple)), ...);
        }

        template <typename... Ts>
        HPX_FORCEINLINE void apply(hpx::tuple<Ts...>& sequence) const
        {
            apply(sequence, hpx::util::make_index_pack_t<sizeof...(Ts)>());
        }

        template <typename Sequence_>
        HPX_FORCEINLINE void apply(Sequence_& sequence) const
        {
            std::for_each(sequence.begin(), sequence.end(), *this);
        }

        detail::when_any<Sequence>& when_;
        mutable std::size_t idx_;
    };

    template <typename Sequence>
    HPX_FORCEINLINE void set_on_completed_callback(
        detail::when_any<Sequence>& when)
    {
        set_when_any_callback_impl<Sequence> callback(when);
        callback.apply(when.lazy_values_.futures);
    }

    ///////////////////////////////////////////////////////////////////////
    template <typename Sequence>
    struct when_any
      : std::enable_shared_from_this<when_any<Sequence>>    //-V690
    {
    public:
        void on_future_ready(
            std::size_t idx, hpx::execution_base::agent_ref ctx)
        {
            std::size_t index_not_initialized =
                when_any_result<Sequence>::index_error();
            if (index_.compare_exchange_strong(index_not_initialized, idx))
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
        when_any(when_any const&) = delete;
        when_any(when_any&) = delete;

        when_any& operator=(when_any const&) = delete;
        when_any& operator=(when_any&&) = delete;

    public:
        using argument_type = Sequence;

        explicit when_any(argument_type&& lazy_values) noexcept
          : lazy_values_(HPX_MOVE(lazy_values))
          , index_(when_any_result<Sequence>::index_error())
          , goal_reached_on_calling_thread_(false)
        {
        }

        when_any_result<Sequence> operator()()
        {
            // set callback functions to executed when future is ready
            set_on_completed_callback(*this);

            // if one of the requested futures is already set, our
            // callback above has already been called often enough, otherwise
            // we suspend ourselves
            if (!goal_reached_on_calling_thread_)
            {
                // wait for any of the futures to return to become ready
                hpx::execution_base::this_thread::suspend(
                    "hpx::lcos::detail::when_any::operator()");
            }

            // that should not happen
            HPX_ASSERT(
                index_.load() != when_any_result<Sequence>::index_error());

            lazy_values_.index = index_.load();
            return HPX_MOVE(lazy_values_);
        }

        when_any_result<Sequence> lazy_values_;
        std::atomic<std::size_t> index_;
        bool goal_reached_on_calling_thread_;
    };
}    // namespace hpx::lcos::detail

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    inline constexpr struct when_any_t final : hpx::functional::tag<when_any_t>
    {
    private:
        template <typename Range,
            typename Enable =
                std::enable_if_t<hpx::traits::is_future_range_v<Range>>>
        friend hpx::future<hpx::when_any_result<std::decay_t<Range>>>
        tag_invoke(when_any_t, Range&& values)
        {
            using result_type = std::decay_t<Range>;

            auto f = std::make_shared<lcos::detail::when_any<result_type>>(
                hpx::traits::acquire_future<result_type>()(values));

            lcos::local::futures_factory<hpx::when_any_result<result_type>()> p(
                [f = HPX_MOVE(f)]() -> hpx::when_any_result<result_type> {
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
            when_any_t, Iterator begin, Iterator end)
        {
            using value_type =
                hpx::lcos::detail::future_iterator_traits_t<Iterator>;

            std::vector<value_type> values;
            traits::detail::reserve_if_random_access_by_range(
                values, begin, end);

            std::move(begin, end, std::back_inserter(values));
            return tag_invoke(when_any_t{}, HPX_MOVE(values));
        }

        friend auto tag_invoke(when_any_t)
        {
            return hpx::make_ready_future(hpx::when_any_result<hpx::tuple<>>());
        }

        ///////////////////////////////////////////////////////////////////////////
        template <typename T, typename... Ts,
            typename Enable = std::enable_if_t<!(
                hpx::traits::is_future_range_v<T> && sizeof...(Ts) == 0)>>
        friend auto tag_invoke(when_any_t, T&& t, Ts&&... ts)
        {
            using result_type = hpx::tuple<hpx::traits::acquire_future_t<T>,
                hpx::traits::acquire_future_t<Ts>...>;

            hpx::traits::acquire_future_disp func;
            result_type values(
                func(HPX_FORWARD(T, t)), func(HPX_FORWARD(Ts, ts))...);

            auto f = std::make_shared<lcos::detail::when_any<result_type>>(
                HPX_MOVE(values));

            lcos::local::futures_factory<hpx::when_any_result<result_type>()> p(
                [f = HPX_MOVE(f)]() -> hpx::when_any_result<result_type> {
                    return (*f)();
                });

            auto result = p.get_future();
            p.post();

            return result;
        }
    } when_any{};

    ///////////////////////////////////////////////////////////////////////////
    inline constexpr struct when_any_n_t final
      : hpx::functional::tag<when_any_n_t>
    {
    private:
        template <typename Iterator,
            typename Enable =
                std::enable_if_t<hpx::traits::is_iterator_v<Iterator>>>
        friend decltype(auto) tag_invoke(
            when_any_n_t, Iterator begin, std::size_t count)
        {
            using value_type =
                hpx::lcos::detail::future_iterator_traits_t<Iterator>;

            std::vector<value_type> values;
            values.reserve(count);

            while (count-- != 0)
            {
                // NOLINTNEXTLINE(bugprone-macro-repeated-side-effects)
                values.push_back(HPX_MOVE(*begin++));
            }
            return hpx::when_any(HPX_MOVE(values));
        }
    } when_any_n{};
}    // namespace hpx

namespace hpx::lcos {

    template <typename... Ts>
    HPX_DEPRECATED_V(
        1, 8, "hpx::lcos::when_any is deprecated. Use hpx::when_any instead.")
    auto when_any(Ts&&... ts)
    {
        return hpx::when_any(HPX_FORWARD(Ts, ts)...);
    }

    template <typename Iterator,
        typename Enable =
            std::enable_if_t<hpx::traits::is_iterator_v<Iterator>>>
    HPX_DEPRECATED_V(1, 8,
        "hpx::lcos::when_any_n is deprecated. Use hpx::when_any_n instead.")
    auto when_any_n(Iterator begin, std::size_t count)
    {
        return hpx::when_any_n(begin, count);
    }

    template <typename Container>
    using when_any_result HPX_DEPRECATED_V(1, 8,
        "hpx::lcos::when_all_result is deprecated. Use hpx::when_all_result "
        "instead.") = hpx::when_any_result<Container>;
}    // namespace hpx::lcos

#endif    // DOXYGEN
