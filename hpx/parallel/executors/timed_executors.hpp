//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/execute_at_executor.hpp

#if !defined(HPX_PARALLEL_TIMED_EXECUTORS_JAN_06_2017_0802AM)
#define HPX_PARALLEL_TIMED_EXECUTORS_JAN_06_2017_0802AM

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/traits/detail/wrap_int.hpp>
#include <hpx/traits/executor_traits.hpp>
#include <hpx/traits/is_launch_policy.hpp>
#include <hpx/util/steady_clock.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/decay.hpp>

#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/executors/execution.hpp>
#include <hpx/parallel/executors/parallel_executor.hpp>
#include <hpx/parallel/executors/sequenced_executor.hpp>

#include <chrono>
#include <functional>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { namespace execution
{
    namespace detail
    {
        /// \cond NOINTERNAL

        ///////////////////////////////////////////////////////////////////////
        template <typename Tag>
        struct sync_execute_at_helper
        {
            template <typename Executor, typename F>
            auto operator()(hpx::future<void> && fut, Executor && exec, F && f) const
            ->  decltype(
                    execution::sync_execute(std::forward<Executor>(exec),
                        std::forward<F>(f))
                )
            {
                fut.get();        // rethrow exceptions
                return execution::sync_execute(std::forward<Executor>(exec),
                    std::forward<F>(f));
            }

            template <typename Executor, typename F, typename ... Ts>
            static auto
            call(hpx::traits::detail::wrap_int,
                    Executor && exec,
                    std::chrono::steady_clock::time_point const& abs_time,
                    F && f, Ts &&... ts)
            ->  decltype(
                    execution::async_execute(std::forward<Executor>(exec),
                        std::forward<F>(f), std::forward<Ts>(ts)...).get()
                )
            {
                auto predecessor = make_ready_future_at(abs_time);
                return execution::then_execute(sequenced_executor(),
                    hpx::util::bind(
                        hpx::util::one_shot(sync_execute_at_helper()),
                        hpx::util::placeholders::_1, std::ref(exec),
                        hpx::util::deferred_call(
                            std::forward<F>(f), std::forward<Ts>(ts)...)),
                    predecessor).get();
            }

            template <typename Executor, typename F, typename ... Ts>
            static auto
            call(int,
                    Executor && exec,
                    std::chrono::steady_clock::time_point const& abs_time,
                    F && f, Ts &&... ts)
            ->  decltype(
                    exec.sync_execute_at(abs_time, std::forward<F>(f),
                        std::forward<Ts>(ts)...)
                )
            {
                return exec.sync_execute_at(abs_time, std::forward<F>(f),
                    std::forward<Ts>(ts)...);
            }
        };

        template <>
        struct sync_execute_at_helper<sequenced_execution_tag>
        {
            template <typename Executor, typename F, typename ... Ts>
            static auto
            call(hpx::traits::detail::wrap_int,
                    Executor && exec,
                    std::chrono::steady_clock::time_point const& abs_time,
                    F && f, Ts &&... ts)
            ->  decltype(
                    execution::sync_execute(std::forward<Executor>(exec),
                        std::forward<F>(f), std::forward<Ts>(ts)...))
            {
                this_thread::sleep_until(abs_time);
                return execution::sync_execute(std::forward<Executor>(exec),
                    std::forward<F>(f), std::forward<Ts>(ts)...);
            }

            template <typename Executor, typename F, typename ... Ts>
            static auto
            call(int,
                    Executor && exec,
                    std::chrono::steady_clock::time_point const& abs_time,
                    F && f, Ts &&... ts)
            ->  decltype(
                    exec.sync_execute_at(abs_time, std::forward<F>(f),
                        std::forward<Ts>(ts)...))
            {
                return exec.sync_execute_at(abs_time, std::forward<F>(f),
                    std::forward<Ts>(ts)...);
            }
        };

        template <typename Executor, typename F, typename ... Ts>
        auto call_sync_execute_at(Executor && exec,
                std::chrono::steady_clock::time_point const& abs_time,
                F && f, Ts &&... ts)
        ->  decltype(
                sync_execute_at_helper<
                        typename hpx::traits::executor_execution_category<
                            typename hpx::util::decay<Executor>::type
                        >::type
                    >::call(0, std::forward<Executor>(exec), abs_time,
                        std::forward<F>(f), std::forward<Ts>(ts)...)
            )
        {
            typedef typename hpx::traits::executor_execution_category<
                    typename hpx::util::decay<Executor>::type
                >::type tag;

            return sync_execute_at_helper<tag>::call(0,
                std::forward<Executor>(exec), abs_time, std::forward<F>(f),
                std::forward<Ts>(ts)...);
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Tag>
        struct async_execute_at_helper
        {
            template <typename Executor, typename F>
            auto operator()(hpx::future<void> && fut, Executor && exec, F && f) const
            ->  decltype(
                    execution::async_execute(std::forward<Executor>(exec),
                        std::forward<F>(f))
                )
            {
                fut.get();        // rethrow exceptions
                return execution::async_execute(std::forward<Executor>(exec),
                    std::forward<F>(f));
            }

            template <typename Executor, typename F, typename ... Ts>
            static auto
            call(hpx::traits::detail::wrap_int,
                    Executor && exec,
                    std::chrono::steady_clock::time_point const& abs_time,
                    F && f, Ts &&... ts)
            ->  decltype(
                    execution::async_execute(std::forward<Executor>(exec),
                        std::forward<F>(f), std::forward<Ts>(ts)...)
                )
            {
                auto predecessor = make_ready_future_at(abs_time);
                return execution::then_execute(sequenced_executor(),
                    hpx::util::bind(
                        hpx::util::one_shot(async_execute_at_helper()),
                        hpx::util::placeholders::_1, std::forward<Executor>(exec),
                        hpx::util::deferred_call(
                            std::forward<F>(f), std::forward<Ts>(ts)...)),
                    predecessor);
            }

            template <typename Executor, typename F, typename ... Ts>
            static auto
            call(int,
                    Executor && exec,
                    std::chrono::steady_clock::time_point const& abs_time,
                    F && f, Ts &&... ts)
            ->  decltype(
                    exec.async_execute_at(abs_time, std::forward<F>(f),
                        std::forward<Ts>(ts)...)
                )
            {
                return exec.async_execute_at(abs_time, std::forward<F>(f),
                    std::forward<Ts>(ts)...);
            }
        };

        template <>
        struct async_execute_at_helper<sequential_execution_tag>
        {
            template <typename Executor, typename F, typename ... Ts>
            static auto
            call(hpx::traits::detail::wrap_int,
                    Executor && exec,
                    std::chrono::steady_clock::time_point const& abs_time,
                    F && f, Ts &&... ts)
            ->  decltype(
                    execution::async_execute(std::forward<Executor>(exec),
                        std::forward<F>(f), std::forward<Ts>(ts)...)
                )
            {
                this_thread::sleep_until(abs_time);
                return execution::async_execute(std::forward<Executor>(exec),
                    std::forward<F>(f), std::forward<Ts>(ts)...);
            }

            template <typename Executor, typename F, typename ... Ts>
            static auto call(int,
                    Executor && exec,
                    std::chrono::steady_clock::time_point const& abs_time,
                    F && f, Ts &&... ts)
            ->  decltype(
                    exec.async_execute_at(abs_time, std::forward<F>(f),
                        std::forward<Ts>(ts)...)
                )
            {
                return exec.async_execute_at(abs_time, std::forward<F>(f),
                    std::forward<Ts>(ts)...);
            }
        };

        template <typename Executor, typename F, typename ... Ts>
        auto call_async_execute_at(Executor && exec,
                std::chrono::steady_clock::time_point const& abs_time,
                F && f, Ts &&... ts)
        ->  decltype(
                async_execute_at_helper<
                        typename hpx::traits::executor_execution_category<
                            typename hpx::util::decay<Executor>::type
                        >::type
                    >::call(0, std::forward<Executor>(exec), abs_time,
                        std::forward<F>(f), std::forward<Ts>(ts)...)
            )
        {
            typedef typename hpx::traits::executor_execution_category<
                    typename hpx::util::decay<Executor>::type
                >::type tag;
            return async_execute_at_helper<tag>::call(0,
                std::forward<Executor>(exec), abs_time,
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Tag>
        struct apply_execute_at_helper
        {
            template <typename Executor, typename F>
            void operator()(hpx::future<void> && fut, Executor && exec, F && f) const
            {
                fut.get();        // rethrow exceptions
                execution::post(std::forward<Executor>(exec), std::forward<F>(f));
            }

            template <typename Executor, typename F, typename ... Ts>
            static void
            call(hpx::traits::detail::wrap_int,
                    Executor && exec,
                    std::chrono::steady_clock::time_point const& abs_time,
                    F && f, Ts &&... ts)
            {
                auto predecessor = make_ready_future_at(abs_time);
                execution::then_execute(sequenced_executor(),
                    hpx::util::bind(
                        hpx::util::one_shot(apply_execute_at_helper()),
                        hpx::util::placeholders::_1, std::forward<Executor>(exec),
                        hpx::util::deferred_call(std::forward<F>(f),
                            std::forward<Ts>(ts)...)),
                    predecessor);
            }

            template <typename Executor, typename F, typename ... Ts>
            static auto call(int,
                    Executor && exec,
                    std::chrono::steady_clock::time_point const& abs_time,
                    F && f, Ts &&... ts)
            ->  decltype(
                    exec.apply_execute_at(abs_time, std::forward<F>(f),
                        std::forward<Ts>(ts)...)
                )
            {
                return exec.apply_execute_at(abs_time, std::forward<F>(f),
                    std::forward<Ts>(ts)...);
            }
        };

        template <>
        struct apply_execute_at_helper<sequential_execution_tag>
        {
            template <typename Executor, typename F, typename ... Ts>
            static void
            call(hpx::traits::detail::wrap_int,
                    Executor && exec,
                    std::chrono::steady_clock::time_point const& abs_time,
                    F && f, Ts &&... ts)
           {
                this_thread::sleep_until(abs_time);
                execution::post(std::forward<Executor>(exec),
                    std::forward<F>(f), std::forward<Ts>(ts)...);
            }

            template <typename Executor, typename F, typename ... Ts>
            static auto
            call(int,
                    Executor && exec,
                    std::chrono::steady_clock::time_point const& abs_time,
                    F && f, Ts &&... ts)
            ->  decltype(
                    exec.apply_execute_at(abs_time, std::forward<F>(f),
                        std::forward<Ts>(ts)...)
                )
            {
                exec.apply_execute_at(abs_time, std::forward<F>(f),
                    std::forward<Ts>(ts)...);
            }
        };

        template <typename Executor, typename F, typename ... Ts>
        void call_apply_execute_at(Executor && exec,
            std::chrono::steady_clock::time_point const& abs_time, F && f,
            Ts &&... ts)
        {
            typedef typename hpx::traits::executor_execution_category<
                    typename hpx::util::decay<Executor>::type
                >::type tag;

            return apply_execute_at_helper<tag>::call(0,
                std::forward<Executor>(exec), abs_time, std::forward<F>(f),
                std::forward<Ts>(ts)...);
        }
        /// \endcond
    }

    ///////////////////////////////////////////////////////////////////////////
    // Executor allowing to run things at a given point in time
    template <typename BaseExecutor>
    struct timed_executor
    {
        typedef typename std::decay<BaseExecutor>::type base_executor_type;

        typedef typename hpx::traits::executor_execution_category<
                base_executor_type
            >::type execution_category;

        typedef typename hpx::traits::executor_parameters_type<
                base_executor_type
            >::type parameters_type;

        timed_executor(hpx::util::steady_time_point const& abs_time)
          : exec_(BaseExecutor()),
            execute_at_(abs_time.value())
        {}

        timed_executor(hpx::util::steady_duration const& rel_time)
          : exec_(BaseExecutor()),
            execute_at_(rel_time.from_now())
        {}

        template <typename Executor>
        timed_executor(Executor && exec,
                hpx::util::steady_time_point const& abs_time)
          : exec_(std::forward<Executor>(exec)),
            execute_at_(abs_time.value())
        {}

        template <typename Executor>
        timed_executor(Executor && exec,
                hpx::util::steady_duration const& rel_time)
          : exec_(std::forward<Executor>(exec)),
            execute_at_(rel_time.from_now())
        {}

        /// \cond NOINTERNAL
        bool operator==(timed_executor const& rhs) const noexcept
        {
            return exec_ == rhs.exec_ && execute_at_ == rhs.execute_at_;
        }

        bool operator!=(timed_executor const& rhs) const noexcept
        {
            return !(*this == rhs);
        }

        timed_executor const& context() const noexcept
        {
            return *this;
        }
        /// \endcond

        // OneWayExecutor interface
        template <typename F, typename ... Ts>
        typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type
        sync_execute(F && f, Ts &&... ts)
        {
            return detail::call_sync_execute_at(exec_, execute_at_,
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        // TwoWayExecutor interface
        template <typename F, typename ... Ts>
        hpx::future<
            typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type
        >
        async_execute(F && f, Ts &&... ts)
        {
            return detail::call_async_execute_at(exec_, execute_at_,
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        // NonBlockingOneWayExecutor (adapted) interface
        template <typename F, typename ... Ts>
        void apply_execute(F && f, Ts &&... ts)
        {
            detail::call_apply_execute_at(exec_, execute_at_,
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        BaseExecutor exec_;
        std::chrono::steady_clock::time_point execute_at_;
    };

    ///////////////////////////////////////////////////////////////////////////
    using sequenced_timed_executor =
        timed_executor<execution::sequenced_executor>;

    using parallel_timed_executor =
        timed_executor<execution::parallel_executor>;
}}}

namespace hpx { namespace traits
{
    /// \cond NOINTERNAL

    ///////////////////////////////////////////////////////////////////////////
    template <typename BaseExecutor>
    struct is_one_way_executor<
            parallel::execution::timed_executor<BaseExecutor> >
      : is_one_way_executor<typename std::decay<BaseExecutor>::type>
    {};

    template <typename BaseExecutor>
    struct is_two_way_executor<
            parallel::execution::timed_executor<BaseExecutor> >
      : is_two_way_executor<typename std::decay<BaseExecutor>::type>
    {};

    template <typename BaseExecutor>
    struct is_non_blocking_one_way_executor<
            parallel::execution::timed_executor<BaseExecutor> >
      : is_non_blocking_one_way_executor<typename std::decay<BaseExecutor>::type>
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <>
    struct is_one_way_executor<
            parallel::execution::sequenced_timed_executor>
      : std::true_type
    {};

    template <>
    struct is_two_way_executor<
            parallel::execution::parallel_timed_executor>
      : std::true_type
    {};
    /// \endcond
}}

#include <hpx/parallel/executors/timed_execution.hpp>
#include <hpx/parallel/executors/thread_timed_execution.hpp>

#endif
