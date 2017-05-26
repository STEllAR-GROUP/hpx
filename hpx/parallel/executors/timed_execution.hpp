//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_EXECUTORS_TIMED_EXECUTION_JAN_07_2017_0735AM)
#define HPX_PARALLEL_EXECUTORS_TIMED_EXECUTION_JAN_07_2017_0735AM

#include <hpx/config.hpp>
#include <hpx/parallel/executors/timed_execution_fwd.hpp>

#include <hpx/lcos/future.hpp>
#include <hpx/traits/detail/wrap_int.hpp>
#include <hpx/traits/executor_traits.hpp>
#include <hpx/traits/is_executor.hpp>
#include <hpx/util/steady_clock.hpp>

#include <hpx/parallel/executors/execution.hpp>
#include <hpx/parallel/executors/timed_executors.hpp>

#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace execution
{
    ///////////////////////////////////////////////////////////////////////////
    // Executor customization points
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename Executor, typename Enable = void>
        struct timed_post_fn_helper;

        template <typename Executor, typename Enable = void>
        struct timed_async_execute_fn_helper;

        template <typename Executor, typename Enable = void>
        struct timed_sync_execute_fn_helper;
        /// \endcond
    }

    // customization point for NonBlockingOneWayExecutor interface
    // post_at(), post_after()
    namespace detail
    {
        /// \cond NOINTERNAL

        ///////////////////////////////////////////////////////////////////////
        template <typename Executor>
        struct timed_post_fn_helper<Executor,
            typename std::enable_if<
                hpx::traits::is_one_way_executor<Executor>::value ||
                hpx::traits::is_two_way_executor<Executor>::value ||
                hpx::traits::is_non_blocking_one_way_executor<Executor>::value
            >::type>
        {
            template <typename NonBlockingOneWayExecutor, typename F,
                typename ... Ts>
            HPX_FORCEINLINE static auto
            call(NonBlockingOneWayExecutor && exec,
                hpx::util::steady_time_point const& abs_time,
                F && f, Ts &&... ts)
            {
                return execution::post(
                    timed_executor<NonBlockingOneWayExecutor&>(exec, abs_time),
                    std::forward<F>(f), std::forward<Ts>(ts)...);
            }

            template <typename NonBlockingOneWayExecutor, typename F,
                typename ... Ts>
            HPX_FORCEINLINE static auto
            call(NonBlockingOneWayExecutor && exec,
                hpx::util::steady_duration const& rel_time,
                F && f, Ts &&... ts)
            {
                return execution::post(
                    timed_executor<NonBlockingOneWayExecutor&>(exec, rel_time),
                    std::forward<F>(f), std::forward<Ts>(ts)...);
            }
        };

        // post_at dispatch point
        template <typename Executor, typename F, typename ... Ts>
        HPX_FORCEINLINE auto
        post_at(Executor && exec,
                hpx::util::steady_time_point const& abs_time,
                F && f, Ts &&... ts)
        ->  decltype(timed_post_fn_helper<
                    typename std::decay<Executor>::type
                >::call(std::forward<Executor>(exec), abs_time,
                    std::forward<F>(f), std::forward<Ts>(ts)...
            ))
        {
            return timed_post_fn_helper<
                    typename std::decay<Executor>::type
                >::call(std::forward<Executor>(exec), abs_time,
                    std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename Executor, typename F, typename ... Ts>
        HPX_FORCEINLINE
        auto customization_point<post_at_tag>::operator()(
            Executor && exec, hpx::util::steady_time_point const& abs_time,
            F && f, Ts &&... ts) const
        {
            return post_at(std::forward<Executor>(exec),
                abs_time, std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        // post_after dispatch point
        template <typename Executor, typename F, typename ... Ts>
        HPX_FORCEINLINE auto
        post_after(Executor && exec,
                hpx::util::steady_duration const& rel_time,
                F && f, Ts &&... ts)
        ->  decltype(timed_post_fn_helper<
                    typename std::decay<Executor>::type
                >::call(std::forward<Executor>(exec), rel_time,
                    std::forward<F>(f), std::forward<Ts>(ts)...
            ))
        {
            return timed_post_fn_helper<
                    typename std::decay<Executor>::type
                >::call(std::forward<Executor>(exec), rel_time,
                    std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename Executor, typename F, typename ... Ts>
        HPX_FORCEINLINE
        auto customization_point<post_after_tag>::operator()(
            Executor && exec, hpx::util::steady_duration const& rel_time,
            F && f, Ts &&... ts) const
        {
            return post_after(std::forward<Executor>(exec),
                rel_time, std::forward<F>(f), std::forward<Ts>(ts)...);
        }
        /// \endcond
    }

    ///////////////////////////////////////////////////////////////////////////
    // customization points for TwoWayExecutor interface
    // async_execute_at(), async_execute_after()
    namespace detail
    {
        /// \cond NOINTERNAL

        ///////////////////////////////////////////////////////////////////////
        template <typename Executor>
        struct timed_async_execute_fn_helper<Executor,
            typename std::enable_if<
                hpx::traits::is_two_way_executor<Executor>::value
            >::type>
        {
            template <typename TwoWayExecutor, typename F, typename ... Ts>
            HPX_FORCEINLINE static auto
            call(TwoWayExecutor && exec,
                hpx::util::steady_time_point const& abs_time,
                F && f, Ts &&... ts)
            {
                return execution::async_execute(
                    timed_executor<TwoWayExecutor&>(exec, abs_time),
                    std::forward<F>(f), std::forward<Ts>(ts)...);
            }

            template <typename TwoWayExecutor, typename F, typename ... Ts>
            HPX_FORCEINLINE static auto
            call(TwoWayExecutor && exec,
                hpx::util::steady_duration const& rel_time,
                F && f, Ts &&... ts)
            {
                return execution::async_execute(
                    timed_executor<TwoWayExecutor&>(exec, rel_time),
                    std::forward<F>(f), std::forward<Ts>(ts)...);
            }
        };

        // async_execute_at dispatch point
        template <typename Executor, typename F, typename ... Ts>
        HPX_FORCEINLINE auto
        async_execute_at(Executor && exec,
                hpx::util::steady_time_point const& abs_time,
                F && f, Ts &&... ts)
        ->  decltype(timed_async_execute_fn_helper<
                    typename std::decay<Executor>::type
                >::call(std::forward<Executor>(exec), abs_time,
                    std::forward<F>(f), std::forward<Ts>(ts)...
            ))
        {
            return timed_async_execute_fn_helper<
                    typename std::decay<Executor>::type
                >::call(std::forward<Executor>(exec), abs_time,
                    std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename Executor, typename F, typename ... Ts>
        HPX_FORCEINLINE
        auto customization_point<async_execute_at_tag>::operator()(
            Executor && exec, hpx::util::steady_time_point const& abs_time,
            F && f, Ts &&... ts) const
        {
            return async_execute_at(std::forward<Executor>(exec),
                abs_time, std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        // async_execute_after dispatch point
        template <typename Executor, typename F, typename ... Ts>
        HPX_FORCEINLINE auto
        async_execute_after(Executor && exec,
                hpx::util::steady_duration const& rel_time,
                F && f, Ts &&... ts)
        ->  decltype(timed_async_execute_fn_helper<
                    typename std::decay<Executor>::type
                >::call(std::forward<Executor>(exec), rel_time,
                    std::forward<F>(f), std::forward<Ts>(ts)...
            ))
        {
            return timed_async_execute_fn_helper<
                    typename std::decay<Executor>::type
                >::call(std::forward<Executor>(exec), rel_time,
                    std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename Executor, typename F, typename ... Ts>
        HPX_FORCEINLINE
        auto customization_point<async_execute_after_tag>::operator()(
            Executor && exec, hpx::util::steady_duration const& rel_time,
            F && f, Ts &&... ts) const
        {
            return async_execute_after(std::forward<Executor>(exec),
                rel_time, std::forward<F>(f), std::forward<Ts>(ts)...);
        }
        /// \endcond
    }

    ///////////////////////////////////////////////////////////////////////////
    // customization points for OneWayExecutor interface
    // sync_execute_at(), sync_execute_after()
    namespace detail
    {
        /// \cond NOINTERNAL

        ///////////////////////////////////////////////////////////////////////
        template <typename Executor>
        struct timed_sync_execute_fn_helper<Executor,
            typename std::enable_if<
                hpx::traits::is_one_way_executor<Executor>::value ||
                hpx::traits::is_two_way_executor<Executor>::value
            >::type>
        {
            template <typename OneWayExecutor, typename F, typename ... Ts>
            HPX_FORCEINLINE static auto
            call(OneWayExecutor && exec,
                hpx::util::steady_time_point const& abs_time,
                F && f, Ts &&... ts)
            {
                return execution::sync_execute(
                    timed_executor<OneWayExecutor&>(exec, abs_time),
                    std::forward<F>(f), std::forward<Ts>(ts)...);
            }

            template <typename OneWayExecutor, typename F, typename ... Ts>
            HPX_FORCEINLINE static auto
            call(OneWayExecutor && exec,
                hpx::util::steady_duration const& rel_time,
                F && f, Ts &&... ts)
            {
                return execution::sync_execute(
                    timed_executor<OneWayExecutor&>(exec, rel_time),
                    std::forward<F>(f), std::forward<Ts>(ts)...);
            }
        };

        // sync_execute_at dispatch point
        template <typename Executor, typename F, typename ... Ts>
        HPX_FORCEINLINE auto
        sync_execute_at(Executor && exec,
                hpx::util::steady_time_point const& abs_time,
                F && f, Ts &&... ts)
        ->  decltype(timed_sync_execute_fn_helper<
                    typename std::decay<Executor>::type
                >::call(std::forward<Executor>(exec), abs_time,
                    std::forward<F>(f), std::forward<Ts>(ts)...
            ))
        {
            return timed_sync_execute_fn_helper<
                    typename std::decay<Executor>::type
                >::call(std::forward<Executor>(exec), abs_time,
                    std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename Executor, typename F, typename ... Ts>
        HPX_FORCEINLINE
        auto customization_point<sync_execute_at_tag>::operator()(
            Executor && exec, hpx::util::steady_time_point const& abs_time,
            F && f, Ts &&... ts) const
        {
            return sync_execute_at(std::forward<Executor>(exec),
                abs_time, std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        // sync_execute_after dispatch point
        template <typename Executor, typename F, typename ... Ts>
        HPX_FORCEINLINE auto
        sync_execute_after(Executor && exec,
                hpx::util::steady_duration const& rel_time,
                F && f, Ts &&... ts)
        ->  decltype(timed_sync_execute_fn_helper<
                    typename std::decay<Executor>::type
                >::call(std::forward<Executor>(exec), rel_time,
                    std::forward<F>(f), std::forward<Ts>(ts)...
            ))
        {
            return timed_sync_execute_fn_helper<
                    typename std::decay<Executor>::type
                >::call(std::forward<Executor>(exec), rel_time,
                    std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename Executor, typename F, typename ... Ts>
        HPX_FORCEINLINE
        auto customization_point<sync_execute_after_tag>::operator()(
            Executor && exec, hpx::util::steady_duration const& rel_time,
            F && f, Ts &&... ts) const
        {
            return sync_execute_after(std::forward<Executor>(exec),
                rel_time, std::forward<F>(f), std::forward<Ts>(ts)...);
        }
        /// \endcond
    }
}}}

#endif

