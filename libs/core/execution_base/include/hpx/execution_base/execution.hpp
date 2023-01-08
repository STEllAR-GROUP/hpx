//  Copyright (c) 2017-2022 Hartmut Kaiser
//  Copyright (c) 2017 Google
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/execution_base/traits/is_executor.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>
#include <hpx/iterator_support/counting_shape.hpp>

#include <type_traits>
#include <utility>

namespace hpx::execution {

    ///////////////////////////////////////////////////////////////////////////
    /// Function invocations executed by a group of sequential execution agents
    /// execute in sequential order.
    struct sequenced_execution_tag
    {
    };

    /// Function invocations executed by a group of parallel execution agents
    /// execute in unordered fashion. Any such invocations executing in the
    /// same thread are indeterminately sequenced with respect to each other.
    ///
    /// \note \a parallel_execution_tag is weaker than
    ///       \a sequenced_execution_tag.
    struct parallel_execution_tag
    {
    };

    /// Function invocations executed by a group of vector execution agents are
    /// permitted to execute in unordered fashion when executed in different
    /// threads, and un-sequenced with respect to one another when executed in
    /// the same thread.
    ///
    /// \note \a unsequenced_execution_tag is weaker than
    ///       \a parallel_execution_tag.
    struct unsequenced_execution_tag
    {
    };
}    // namespace hpx::execution

namespace hpx::parallel::execution {

    ///////////////////////////////////////////////////////////////////////////
    // Executor customization points
    namespace detail {

        /// \cond NOINTERNAL
        template <typename Executor, typename Enable = void>
        struct async_execute_fn_helper;

        template <typename Executor, typename Enable = void>
        struct sync_execute_fn_helper;

        template <typename Executor, typename Enable = void>
        struct then_execute_fn_helper;

        template <typename Executor, typename Enable = void>
        struct post_fn_helper;

        template <typename Executor, typename Enable = void>
        struct bulk_async_execute_fn_helper;

        template <typename Executor, typename Enable = void>
        struct bulk_sync_execute_fn_helper;

        template <typename Executor, typename Enable = void>
        struct bulk_then_execute_fn_helper;
        /// \endcond
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    // define customization points

    // OneWayExecutor customization point: execution::sync_execute

    /// Customization point for synchronous execution agent creation.
    ///
    /// This synchronously creates a single function invocation f() using
    /// the associated executor. The execution of the supplied function
    /// synchronizes with the caller
    ///
    /// \param exec [in] The executor object to use for scheduling of the
    ///             function \a f.
    /// \param f    [in] The function which will be scheduled using the
    ///             given executor.
    /// \param ts   [in] Additional arguments to use to invoke \a f.
    ///
    /// \returns f(ts...)'s result
    ///
    /// \note It will call tag_invoke(sync_execute_t, exec, f, ts...) if it
    ///       exists. For two-way executors it will invoke asynch_execute_t
    ///       and wait for the task's completion before returning.
    ///
    inline constexpr struct sync_execute_t final
      : hpx::functional::detail::tag_fallback<sync_execute_t>
    {
    private:
        // clang-format off
        template <typename Executor, typename F, typename... Ts,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_any_v<Executor>
            )>
        // clang-format on
        friend HPX_FORCEINLINE decltype(auto) tag_fallback_invoke(
            sync_execute_t, Executor&& exec, F&& f, Ts&&... ts)
        {
            return detail::sync_execute_fn_helper<std::decay_t<Executor>>::call(
                HPX_FORWARD(Executor, exec), HPX_FORWARD(F, f),
                HPX_FORWARD(Ts, ts)...);
        }
    } sync_execute{};

    ///////////////////////////////////////////////////////////////////////////
    // TwoWayExecutor customization points: execution::async_execute,
    // execution::sync_execute, and execution::then_execute

    /// Customization point for asynchronous execution agent creation.
    ///
    /// This asynchronously creates a single function invocation f() using
    /// the associated executor.
    ///
    /// \param exec [in] The executor object to use for scheduling of the
    ///             function \a f.
    /// \param f    [in] The function which will be scheduled using the
    ///             given executor.
    /// \param ts   [in] Additional arguments to use to invoke \a f.
    ///
    /// \note Executors have to implement only `async_execute()`. All other
    ///       functions will be emulated by this or other customization
    ///       points in terms of this single basic primitive. However, some
    ///       executors will naturally specialize all operations for
    ///       maximum efficiency.
    ///
    /// \note This is valid for one way executors (calls
    ///       make_ready_future(exec.sync_execute(f, ts...) if it exists)
    ///       and for two way executors (calls exec.async_execute(f, ts...)
    ///       if it exists).
    ///
    /// \returns f(ts...)'s result through a future
    ///
    inline constexpr struct async_execute_t final
      : hpx::functional::detail::tag_fallback<async_execute_t>
    {
    private:
        // clang-format off
        template <typename Executor, typename F, typename... Ts,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_any_v<Executor>
            )>
        // clang-format on
        friend HPX_FORCEINLINE decltype(auto) tag_fallback_invoke(
            async_execute_t, Executor&& exec, F&& f, Ts&&... ts)
        {
            return detail::async_execute_fn_helper<
                std::decay_t<Executor>>::call(HPX_FORWARD(Executor, exec),
                HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
        }
    } async_execute{};

    /// Customization point for execution agent creation depending on a
    /// given future.
    ///
    /// This creates a single function invocation f() using the associated
    /// executor after the given future object has become ready.
    ///
    /// \param exec [in] The executor object to use for scheduling of the
    ///             function \a f.
    /// \param f    [in] The function which will be scheduled using the
    ///             given executor.
    /// \param predecessor [in] The future object the execution of the
    ///             given function depends on.
    /// \param ts   [in] Additional arguments to use to invoke \a f.
    ///
    /// \returns f(ts...)'s result through a future
    ///
    /// \note This is valid for two way executors (calls
    ///       exec.then_execute(f, predecessor, ts...) if it exists) and
    ///       for one way executors (calls predecessor.then(bind(f, ts...))).
    ///
    inline constexpr struct then_execute_t final
      : hpx::functional::detail::tag_fallback<then_execute_t>
    {
    private:
        // clang-format off
        template <typename Executor, typename F, typename Future, typename... Ts,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_any_v<Executor>
            )>
        // clang-format on
        friend HPX_FORCEINLINE decltype(auto) tag_fallback_invoke(
            then_execute_t, Executor&& exec, F&& f, Future&& predecessor,
            Ts&&... ts)
        {
            return detail::then_execute_fn_helper<std::decay_t<Executor>>::call(
                HPX_FORWARD(Executor, exec), HPX_FORWARD(F, f),
                HPX_FORWARD(Future, predecessor), HPX_FORWARD(Ts, ts)...);
        }
    } then_execute{};

    ///////////////////////////////////////////////////////////////////////
    // NonBlockingOneWayExecutor customization point: execution::post

    /// Customization point for asynchronous fire & forget execution
    /// agent creation.
    ///
    /// This asynchronously (fire & forget) creates a single function
    /// invocation f() using the associated executor.
    ///
    /// \param exec [in] The executor object to use for scheduling of the
    ///             function \a f.
    /// \param f    [in] The function which will be scheduled using the
    ///             given executor.
    /// \param ts   [in] Additional arguments to use to invoke \a f.
    ///
    /// \note This is valid for two way executors (calls
    ///       exec.post(f, ts...), if available, otherwise
    ///       it calls exec.async_execute(f, ts...) while discarding the
    ///       returned future), and for non-blocking two way executors
    ///       (calls exec.post(f, ts...) if it exists).
    ///
    inline constexpr struct post_t final
      : hpx::functional::detail::tag_fallback<post_t>
    {
    private:
        // clang-format off
        template <typename Executor, typename F, typename... Ts,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_any_v<Executor>
            )>
        // clang-format on
        friend HPX_FORCEINLINE decltype(auto) tag_fallback_invoke(
            post_t, Executor&& exec, F&& f, Ts&&... ts)
        {
            return detail::post_fn_helper<std::decay_t<Executor>>::call(
                HPX_FORWARD(Executor, exec), HPX_FORWARD(F, f),
                HPX_FORWARD(Ts, ts)...);
        }
    } post{};

    ///////////////////////////////////////////////////////////////////////
    // BulkTwoWayExecutor customization points:
    // execution::bulk_async_execute, execution::bulk_sync_execute,
    // execution::bulk_then_execute

    /// Bulk form of synchronous execution agent creation.
    ///
    /// \note This is deliberately different from the bulk_sync_execute
    ///       customization points specified in P0443.The bulk_sync_execute
    ///       customization point defined here is more generic and is used
    ///       as the workhorse for implementing the specified APIs.
    ///
    /// This synchronously creates a group of function invocations f(i)
    /// whose ordering is given by the execution_category associated with
    /// the executor. The function synchronizes the execution of all
    /// scheduled functions with the caller.
    ///
    /// Here \a i takes on all values in the index space implied by shape.
    /// All exceptions thrown by invocations of f(i) are reported in a
    /// manner consistent with parallel algorithm execution through the
    /// returned future.
    ///
    /// \param exec  [in] The executor object to use for scheduling of the
    ///              function \a f.
    /// \param f     [in] The function which will be scheduled using the
    ///              given executor.
    /// \param shape [in] The shape objects which defines the iteration
    ///              boundaries for the arguments to be passed to \a f.
    /// \param ts    [in] Additional arguments to use to invoke \a f.
    ///
    /// \returns The return type of \a executor_type::bulk_sync_execute
    ///          if defined by \a executor_type. Otherwise a vector holding
    ///          the returned values of each invocation of \a f except when
    ///          \a f returns void, which case void is returned.
    ///
    /// \note This calls exec.bulk_sync_execute(f, shape, ts...) if it
    ///       exists; otherwise it executes sync_execute(f, shape, ts...)
    ///       as often as needed.
    ///
    inline constexpr struct bulk_sync_execute_t final
      : hpx::functional::detail::tag_fallback<bulk_sync_execute_t>
    {
    private:
        // clang-format off
        template <typename Executor, typename F, typename Shape, typename... Ts,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_any_v<Executor> &&
               !std::is_integral_v<Shape>
            )>
        // clang-format on
        friend HPX_FORCEINLINE decltype(auto) tag_fallback_invoke(
            bulk_sync_execute_t, Executor&& exec, F&& f, Shape const& shape,
            Ts&&... ts)
        {
            return detail::bulk_sync_execute_fn_helper<
                std::decay_t<Executor>>::call(HPX_FORWARD(Executor, exec),
                HPX_FORWARD(F, f), shape, HPX_FORWARD(Ts, ts)...);
        }

        // clang-format off
        template <typename Executor, typename F, typename Shape, typename... Ts,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_any_v<Executor> &&
                std::is_integral_v<Shape>
            )>
        // clang-format on
        friend HPX_FORCEINLINE decltype(auto) tag_fallback_invoke(
            bulk_sync_execute_t tag, Executor&& exec, F&& f, Shape const& shape,
            Ts&&... ts)
        {
            return tag(HPX_FORWARD(Executor, exec), HPX_FORWARD(F, f),
                hpx::util::counting_shape(shape), HPX_FORWARD(Ts, ts)...);
        }
    } bulk_sync_execute{};

    /// Bulk form of asynchronous execution agent creation.
    ///
    /// \note This is deliberately different from the bulk_async_execute
    ///       customization points specified in P0443.The bulk_async_execute
    ///       customization point defined here is more generic and is used
    ///       as the workhorse for implementing the specified APIs.
    ///
    /// This asynchronously creates a group of function invocations f(i)
    /// whose ordering is given by the execution_category associated with
    /// the executor.
    ///
    /// Here \a i takes on all values in the index space implied by shape.
    /// All exceptions thrown by invocations of f(i) are reported in a
    /// manner consistent with parallel algorithm execution through the
    /// returned future.
    ///
    /// \param exec  [in] The executor object to use for scheduling of the
    ///              function \a f.
    /// \param f     [in] The function which will be scheduled using the
    ///              given executor.
    /// \param shape [in] The shape objects which defines the iteration
    ///              boundaries for the arguments to be passed to \a f.
    /// \param ts    [in] Additional arguments to use to invoke \a f.
    ///
    /// \returns The return type of \a executor_type::bulk_async_execute if
    ///          defined by \a executor_type. Otherwise a vector
    ///          of futures holding the returned values of each invocation
    ///          of \a f.
    ///
    /// \note This calls exec.bulk_async_execute(f, shape, ts...) if it
    ///       exists; otherwise it executes async_execute(f, shape, ts...)
    ///       as often as needed.
    ///
    inline constexpr struct bulk_async_execute_t final
      : hpx::functional::detail::tag_fallback<bulk_async_execute_t>
    {
    private:
        // clang-format off
        template <typename Executor, typename F, typename Shape, typename... Ts,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_any_v<Executor> &&
               !std::is_integral_v<Shape>
            )>
        // clang-format on
        friend HPX_FORCEINLINE decltype(auto) tag_fallback_invoke(
            bulk_async_execute_t, Executor&& exec, F&& f, Shape const& shape,
            Ts&&... ts)
        {
            return detail::bulk_async_execute_fn_helper<
                std::decay_t<Executor>>::call(HPX_FORWARD(Executor, exec),
                HPX_FORWARD(F, f), shape, HPX_FORWARD(Ts, ts)...);
        }

        // clang-format off
        template <typename Executor, typename F, typename Shape, typename... Ts,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_any_v<Executor> &&
                std::is_integral_v<Shape>
            )>
        // clang-format on
        friend HPX_FORCEINLINE decltype(auto) tag_fallback_invoke(
            bulk_async_execute_t tag, Executor&& exec, F&& f,
            Shape const& shape, Ts&&... ts)
        {
            return tag(HPX_FORWARD(Executor, exec), HPX_FORWARD(F, f),
                hpx::util::counting_shape(shape), HPX_FORWARD(Ts, ts)...);
        }
    } bulk_async_execute{};

    /// Bulk form of execution agent creation depending on a given future.
    ///
    /// \note This is deliberately different from the then_sync_execute
    ///       customization points specified in P0443.The bulk_then_execute
    ///       customization point defined here is more generic and is used
    ///       as the workhorse for implementing the specified APIs.
    ///
    /// This creates a group of function invocations f(i)
    /// whose ordering is given by the execution_category associated with
    /// the executor.
    ///
    /// Here \a i takes on all values in the index space implied by shape.
    /// All exceptions thrown by invocations of f(i) are reported in a
    /// manner consistent with parallel algorithm execution through the
    /// returned future.
    ///
    /// \param exec  [in] The executor object to use for scheduling of the
    ///              function \a f.
    /// \param f     [in] The function which will be scheduled using the
    ///              given executor.
    /// \param shape [in] The shape objects which defines the iteration
    ///              boundaries for the arguments to be passed to \a f.
    /// \param predecessor [in] The future object the execution of the
    ///             given function depends on.
    /// \param ts    [in] Additional arguments to use to invoke \a f.
    ///
    /// \returns The return type of \a executor_type::bulk_then_execute
    ///          if defined by \a executor_type. Otherwise a vector holding
    ///          the returned values of each invocation of \a f.
    ///
    /// \note This calls exec.bulk_then_execute(f, shape, pred, ts...) if it
    ///       exists; otherwise it executes
    ///       sync_execute(f, shape, pred.share(), ts...) (if this executor
    ///       is also an OneWayExecutor), or
    ///       async_execute(f, shape, pred.share(), ts...) (if this executor
    ///       is also a TwoWayExecutor) - as often as needed.
    ///
    inline constexpr struct bulk_then_execute_t final
      : hpx::functional::detail::tag_fallback<bulk_then_execute_t>
    {
    private:
        // clang-format off
        template <typename Executor, typename F, typename Shape,
            typename Future, typename... Ts,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_any_v<Executor> &&
               !std::is_integral_v<Shape>
            )>
        // clang-format on
        friend HPX_FORCEINLINE decltype(auto) tag_fallback_invoke(
            bulk_then_execute_t, Executor&& exec, F&& f, Shape const& shape,
            Future&& predecessor, Ts&&... ts)
        {
            return detail::bulk_then_execute_fn_helper<
                std::decay_t<Executor>>::call(HPX_FORWARD(Executor, exec),
                HPX_FORWARD(F, f), shape, HPX_FORWARD(Future, predecessor),
                HPX_FORWARD(Ts, ts)...);
        }

        // clang-format off
        template <typename Executor, typename F, typename Shape,
            typename Future, typename... Ts,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_any_v<Executor> &&
                std::is_integral_v<Shape>
            )>
        // clang-format on
        friend HPX_FORCEINLINE decltype(auto) tag_fallback_invoke(
            bulk_then_execute_t tag, Executor&& exec, F&& f, Shape const& shape,
            Future&& predecessor, Ts&&... ts)
        {
            return tag(HPX_FORWARD(Executor, exec), HPX_FORWARD(F, f),
                hpx::util::counting_shape(shape),
                HPX_FORWARD(Future, predecessor), HPX_FORWARD(Ts, ts)...);
        }
    } bulk_then_execute{};
}    // namespace hpx::parallel::execution
