//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/futures/futures_factory.hpp>
#include <hpx/async_combinators/wait_all.hpp>
#include <hpx/futures/traits/acquire_shared_state.hpp>
#include <hpx/futures/traits/future_access.hpp>
#include <hpx/futures/traits/future_traits.hpp>

#include <boost/dynamic_bitset.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>
#include <type_traits>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    namespace detail
    {
        template <typename Future>
        struct wait_acquire_future
        {
            template <typename R>
            HPX_FORCEINLINE hpx::future<R>
            operator()(hpx::future<R>& future) const
            {
                return std::move(future);
            }

            template <typename R>
            HPX_FORCEINLINE hpx::shared_future<R>
            operator()(hpx::shared_future<R>& future) const
            {
                return future;
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // This version has a callback to be invoked for each future when it
        // gets ready.
        template <typename Future, typename F>
        struct wait_each
        {
        protected:
            void on_future_ready_(hpx::execution_base::agent_ref ctx)
            {
                std::size_t oldcount = ready_count_.fetch_add(1);
                HPX_ASSERT(oldcount < lazy_values_.size());

                if (oldcount + 1 == lazy_values_.size())
                {
                    // reactivate waiting thread only if it's not us
                    if (ctx != hpx::execution_base::this_thread::agent())
                        ctx.resume();
                    else
                        goal_reached_on_calling_thread_ = true;
                }
            }

            template <typename Index>
            void on_future_ready(
                std::false_type, Index i, hpx::execution_base::agent_ref ctx)
            {
                if (lazy_values_[i].has_value()) {
                    if (success_counter_)
                        ++*success_counter_;
                    // invoke callback function
                    f_(i, lazy_values_[i].get());
                }

                // keep track of ready futures
                on_future_ready_(ctx);
            }

            template <typename Index>
            void on_future_ready(
                std::true_type, Index i, hpx::execution_base::agent_ref ctx)
            {
                if (lazy_values_[i].has_value()) {
                    if (success_counter_)
                        ++*success_counter_;
                    // invoke callback function
                    f_(i);
                }

                // keep track of ready futures
                on_future_ready_(ctx);
            }

        public:
            typedef std::vector<Future> argument_type;

            template <typename F_>
            wait_each(argument_type const& lazy_values, F_ && f,
                    std::atomic<std::size_t>* success_counter)
              : lazy_values_(lazy_values),
                ready_count_(0),
                f_(std::forward<F>(f)),
                success_counter_(success_counter),
                goal_reached_on_calling_thread_(false)
            {}

            template <typename F_>
            wait_each(argument_type && lazy_values, F_ && f,
                    std::atomic<std::size_t>* success_counter)
              : lazy_values_(std::move(lazy_values)),
                ready_count_(0),
                f_(std::forward<F>(f)),
                success_counter_(success_counter),
                goal_reached_on_calling_thread_(false)
            {}

            wait_each(wait_each && rhs)
              : lazy_values_(std::move(rhs.lazy_values_)),
                ready_count_(rhs.ready_count_.load()),
                f_(std::move(rhs.f_)),
                success_counter_(rhs.success_counter_),
                goal_reached_on_calling_thread_(
                    rhs.goal_reached_on_calling_thread_)
            {
                rhs.success_counter_ = nullptr;
                rhs.goal_reached_on_calling_thread_ = false;
            }

            wait_each& operator= (wait_each && rhs)
            {
                if (this != &rhs) {
                    lazy_values_ = std::move(rhs.lazy_values_);
                    ready_count_.store(rhs.ready_count_.load());
                    rhs.ready_count_ = 0;
                    f_ = std::move(rhs.f_);
                    success_counter_ = rhs.success_counter_;
                    rhs.success_counter_ = nullptr;
                    goal_reached_on_calling_thread_ =
                        rhs.goal_reached_on_calling_thread_;
                    rhs.goal_reached_on_calling_thread_ = false;
                }
                return *this;
            }

            std::vector<Future> operator()()
            {
                ready_count_.store(0);
                goal_reached_on_calling_thread_ = false;

                // set callback functions to executed when future is ready
                std::size_t size = lazy_values_.size();
                auto ctx = hpx::execution_base::this_thread::agent();
                for (std::size_t i = 0; i != size; ++i)
                {
                    typedef
                        typename traits::detail::shared_state_ptr_for<Future>::type
                        shared_state_ptr;
                    shared_state_ptr current =
                        traits::detail::get_shared_state(lazy_values_[i]);

                    current->execute_deferred();
                    current->set_on_completed([=]() -> void {
                        using is_void = std::is_void<
                            typename traits::future_traits<Future>::type>;
                        return on_future_ready(is_void{}, i, ctx);
                    });
                }

                // If all of the requested futures are already set then our
                // callback above has already been called, otherwise we suspend
                // ourselves.
                if (!goal_reached_on_calling_thread_)
                {
                    // wait for all of the futures to return to become ready
                    hpx::execution_base::this_thread::suspend(
                        "hpx::lcos::detail::wait_each::operator()");
                }

                // all futures should be ready
                HPX_ASSERT(ready_count_ == size);

                return std::move(lazy_values_);
            }

            std::vector<Future> lazy_values_;
            std::atomic<std::size_t> ready_count_;
            typename std::remove_reference<F>::type f_;
            std::atomic<std::size_t>* success_counter_;
            bool goal_reached_on_calling_thread_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    // Asynchronous versions.

    /// The one argument version is special in the sense that it returns the
    /// expected value directly (without wrapping it into a tuple).
    template <typename Future, typename F>
    inline typename std::enable_if<
        !std::is_void<typename traits::future_traits<Future>::type>::value
      , std::size_t
    >::type
    wait(Future && f1, F && f)
    {
        f(0, f1.get());
        return 1;
    }

    template <typename Future, typename F>
    inline typename std::enable_if< //-V659
        std::is_void<typename traits::future_traits<Future>::type>::value
      , std::size_t
    >::type
    wait(Future && f1, F && f)
    {
        f1.get();
        f(0);
        return 1;
    }

    //////////////////////////////////////////////////////////////////////////
    // This overload of wait() will make sure that the passed function will be
    // invoked as soon as a value becomes available, it will not wait for all
    // results to be there.
    template <typename Future, typename F>
    inline std::size_t wait(std::vector<Future>& lazy_values, F&& f,
        std::int32_t /* suspend_for */ = 10)
    {
        typedef std::vector<Future> return_type;

        if (lazy_values.empty())
            return 0;

        return_type lazy_values_;
        lazy_values_.reserve(lazy_values.size());
        std::transform(lazy_values.begin(), lazy_values.end(),
            std::back_inserter(lazy_values_),
            detail::wait_acquire_future<Future>());

        std::atomic<std::size_t> success_counter(0);
        lcos::local::futures_factory<return_type()> p(
            detail::wait_each<Future, F>(std::move(lazy_values_),
                std::forward<F>(f), &success_counter));

        p.apply();
        p.get_future().get();

        return success_counter.load();
    }

    template <typename Future, typename F>
    inline std::size_t
    wait(std::vector<Future> && lazy_values, F && f,
        std::int32_t suspend_for = 10)
    {
        return wait(lazy_values, std::forward<F>(f), suspend_for);
    }

    template <typename Future, typename F>
    inline std::size_t wait(std::vector<Future> const& lazy_values, F&& f,
        std::int32_t /* suspend_for */ = 10)
    {
        typedef std::vector<Future> return_type;

        if (lazy_values.empty())
            return 0;

        return_type lazy_values_;
        lazy_values_.reserve(lazy_values.size());
        std::transform(lazy_values.begin(), lazy_values.end(),
            std::back_inserter(lazy_values_),
            detail::wait_acquire_future<Future>());

        std::atomic<std::size_t> success_counter(0);
        lcos::local::futures_factory<return_type()> p(
            detail::wait_each<Future, F>(std::move(lazy_values_),
                std::forward<F>(f), &success_counter));

        p.apply();
        p.get_future().get();

        return success_counter.load();
    }
}}

