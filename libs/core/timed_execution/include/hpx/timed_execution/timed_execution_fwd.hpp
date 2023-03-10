//  Copyright (c) 2017-2023 Hartmut Kaiser
//  Copyright (c) 2017 Google
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/execution/traits/executor_traits.hpp>
#include <hpx/execution_base/execution.hpp>
#include <hpx/execution_base/traits/is_executor.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>
#include <hpx/modules/timing.hpp>

#include <type_traits>
#include <utility>

namespace hpx::parallel::execution {

    ///////////////////////////////////////////////////////////////////////////
    // Executor customization points
    namespace detail {

        /// \cond NOINTERNAL
        template <typename Executor, typename Enable = void>
        struct timed_post_fn_helper;

        template <typename Executor, typename Enable = void>
        struct timed_async_execute_fn_helper;

        template <typename Executor, typename Enable = void>
        struct timed_sync_execute_fn_helper;
        /// \endcond
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////
    // extensions

    // forward declare timed_executor wrapper
    template <typename BaseExecutor>
    struct timed_executor;

    // define customization points

    // NonBlockingOneWayExecutor customization points: execution::post_at
    // and execution::post_after

    /// Customization point of asynchronous fire & forget execution agent
    /// creation supporting timed execution.
    ///
    /// This asynchronously (fire & forget) creates a single function
    /// invocation f() using the associated executor at the given point in
    /// time.
    ///
    /// \param exec [in] The executor object to use for scheduling of the
    ///             function \a f.
    /// \param abs_time [in] The point in time the given function should be
    ///             scheduled at to run.
    /// \param f    [in] The function which will be scheduled using the
    ///             given executor.
    /// \param ts... [in] Additional arguments to use to invoke \a f.
    ///
    /// \note This calls exec.post_at(abs_time, f, ts...), if
    ///       available, otherwise it emulates timed scheduling by delaying
    ///       calling execution::post() on the underlying non-time-scheduled
    ///       execution agent.
    ///
    inline constexpr struct post_at_t final
      : hpx::functional::detail::tag_fallback<post_at_t>
    {
    private:
        // clang-format off
        template <typename Executor, typename F, typename... Ts,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_any_v<Executor>
            )>
        // clang-format on
        friend HPX_FORCEINLINE decltype(auto) tag_fallback_invoke(post_at_t,
            Executor&& exec, hpx::chrono::steady_time_point const& abs_time,
            F&& f, Ts&&... ts)
        {
            return detail::timed_post_fn_helper<std::decay_t<Executor>>::call(
                HPX_FORWARD(Executor, exec), abs_time, HPX_FORWARD(F, f),
                HPX_FORWARD(Ts, ts)...);
        }
    } post_at{};

    /// Customization point of asynchronous fire & forget execution agent
    /// creation supporting timed execution.
    ///
    /// This asynchronously (fire & forget) creates a single function
    /// invocation f() using the associated executor at the given point in
    /// time.
    ///
    /// \param exec [in] The executor object to use for scheduling of the
    ///             function \a f.
    /// \param rel_time [in] The duration of time after which the given
    ///             function should be scheduled to run.
    /// \param f    [in] The function which will be scheduled using the
    ///             given executor.
    /// \param ts... [in] Additional arguments to use to invoke \a f.
    ///
    /// \note This calls exec.post_after(rel_time, f, ts...), if
    ///       available, otherwise it emulates timed scheduling by delaying
    ///       calling execution::post() on the underlying non-time-scheduled
    ///       execution agent.
    ///
    inline constexpr struct post_after_t final
      : hpx::functional::detail::tag_fallback<post_after_t>
    {
    private:
        // clang-format off
        template <typename Executor, typename F, typename... Ts,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_any_v<Executor>
            )>
        // clang-format on
        friend HPX_FORCEINLINE decltype(auto) tag_fallback_invoke(post_after_t,
            Executor&& exec, hpx::chrono::steady_duration const& rel_time,
            F&& f, Ts&&... ts)
        {
            return detail::timed_post_fn_helper<std::decay_t<Executor>>::call(
                HPX_FORWARD(Executor, exec), rel_time, HPX_FORWARD(F, f),
                HPX_FORWARD(Ts, ts)...);
        }
    } post_after{};

    ///////////////////////////////////////////////////////////////////////
    // TwoWayExecutor customization points: execution::async_execute_at,
    // execution::async_execute_after, execution::sync_execute_at, and
    // execution::sync_execute_after

    /// Customization point of asynchronous execution agent creation
    /// supporting timed execution.
    ///
    /// This asynchronously creates a single function invocation f() using
    /// the associated executor at the given point in time.
    ///
    /// \param exec [in] The executor object to use for scheduling of the
    ///             function \a f.
    /// \param abs_time [in] The point in time the given function should be
    ///             scheduled at to run.
    /// \param f    [in] The function which will be scheduled using the
    ///             given executor.
    /// \param ts... [in] Additional arguments to use to invoke \a f.
    ///
    /// \returns f(ts...)'s result through a future
    ///
    /// \note This calls exec.async_execute_at(abs_time, f, ts...), if
    ///       available, otherwise it emulates timed scheduling by delaying
    ///       calling execution::async_execute() on the underlying
    ///       non-time-scheduled execution agent.
    ///
    inline constexpr struct async_execute_at_t final
      : hpx::functional::detail::tag_fallback<async_execute_at_t>
    {
    private:
        // clang-format off
        template <typename Executor, typename F, typename... Ts,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_any_v<Executor>
            )>
        // clang-format on
        friend HPX_FORCEINLINE decltype(auto) tag_fallback_invoke(
            async_execute_at_t, Executor&& exec,
            hpx::chrono::steady_time_point const& abs_time, F&& f, Ts&&... ts)
        {
            return detail::timed_async_execute_fn_helper<
                std::decay_t<Executor>>::call(HPX_FORWARD(Executor, exec),
                abs_time, HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
        }
    } async_execute_at{};

    /// Customization point of asynchronous execution agent creation
    /// supporting timed execution.
    ///
    /// This asynchronously creates a single function invocation f() using
    /// the associated executor at the given point in time.
    ///
    /// \param exec [in] The executor object to use for scheduling of the
    ///             function \a f.
    /// \param rel_time [in] The duration of time after which the given
    ///             function should be scheduled to run.
    /// \param f    [in] The function which will be scheduled using the
    ///             given executor.
    /// \param ts... [in] Additional arguments to use to invoke \a f.
    ///
    /// \returns f(ts...)'s result through a future
    ///
    /// \note This calls exec.async_execute_after(rel_time, f, ts...), if
    ///       available, otherwise it emulates timed scheduling by delaying
    ///       calling execution::async_execute() on the underlying
    ///       non-time-scheduled execution agent.
    ///
    inline constexpr struct async_execute_after_t final
      : hpx::functional::detail::tag_fallback<async_execute_after_t>
    {
    private:
        // clang-format off
        template <typename Executor, typename F, typename... Ts,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_any_v<Executor>
            )>
        // clang-format on
        friend HPX_FORCEINLINE decltype(auto) tag_fallback_invoke(
            async_execute_after_t, Executor&& exec,
            hpx::chrono::steady_duration const& rel_time, F&& f, Ts&&... ts)
        {
            return detail::timed_async_execute_fn_helper<
                std::decay_t<Executor>>::call(HPX_FORWARD(Executor, exec),
                rel_time, HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
        }
    } async_execute_after{};

    /// Customization point of synchronous execution agent creation
    /// supporting timed execution.
    ///
    /// This synchronously creates a single function invocation f() using
    /// the associated executor at the given point in time.
    ///
    /// \param exec [in] The executor object to use for scheduling of the
    ///             function \a f.
    /// \param abs_time [in] The point in time the given function should be
    ///             scheduled at to run.
    /// \param f    [in] The function which will be scheduled using the
    ///             given executor.
    /// \param ts... [in] Additional arguments to use to invoke \a f.
    ///
    /// \returns f(ts...)'s result
    ///
    /// \note This calls exec.sync_execute_at(abs_time, f, ts...), if
    ///       available, otherwise it emulates timed scheduling by delaying
    ///       calling execution::sync_execute() on the underlying
    ///       non-time-scheduled execution agent.
    ///
    inline constexpr struct sync_execute_at_t final
      : hpx::functional::detail::tag_fallback<sync_execute_at_t>
    {
    private:
        // clang-format off
        template <typename Executor, typename F, typename... Ts,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_any_v<Executor>
            )>
        // clang-format on
        friend HPX_FORCEINLINE decltype(auto) tag_fallback_invoke(
            sync_execute_at_t, Executor&& exec,
            hpx::chrono::steady_time_point const& abs_time, F&& f, Ts&&... ts)
        {
            return detail::timed_sync_execute_fn_helper<
                std::decay_t<Executor>>::call(HPX_FORWARD(Executor, exec),
                abs_time, HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
        }
    } sync_execute_at{};

    /// Customization point of synchronous execution agent creation
    /// supporting timed execution.
    ///
    /// This synchronously creates a single function invocation f() using
    /// the associated executor at the given point in time.
    ///
    /// \param exec [in] The executor object to use for scheduling of the
    ///             function \a f.
    /// \param rel_time [in] The duration of time after which the given
    ///             function should be scheduled to run.
    /// \param f    [in] The function which will be scheduled using the
    ///             given executor.
    /// \param ts... [in] Additional arguments to use to invoke \a f.
    ///
    /// \returns f(ts...)'s result
    ///
    /// \note This calls exec.sync_execute_after(rel_time, f, ts...), if
    ///       available, otherwise it emulates timed scheduling by delaying
    ///       calling execution::sync_execute() on the underlying
    ///       non-time-scheduled execution agent.
    ///
    inline constexpr struct sync_execute_after_t final
      : hpx::functional::detail::tag_fallback<sync_execute_after_t>
    {
    private:
        // clang-format off
        template <typename Executor, typename F, typename... Ts,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_any_v<Executor>
            )>
        // clang-format on
        friend HPX_FORCEINLINE decltype(auto) tag_fallback_invoke(
            sync_execute_after_t, Executor&& exec,
            hpx::chrono::steady_duration const& rel_time, F&& f, Ts&&... ts)
        {
            return detail::timed_sync_execute_fn_helper<
                std::decay_t<Executor>>::call(HPX_FORWARD(Executor, exec),
                rel_time, HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
        }
    } sync_execute_after{};
}    // namespace hpx::parallel::execution
