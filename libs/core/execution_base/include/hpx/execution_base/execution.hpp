//  Copyright (c) 2017-2021 Hartmut Kaiser
//  Copyright (c) 2017 Google
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/execution_base/traits/is_executor.hpp>
#include <hpx/functional/tag_fallback_dispatch.hpp>
#include <hpx/iterator_support/counting_shape.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace execution {
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

    /// \cond NOINTERNAL
    struct task_policy_tag
    {
        constexpr task_policy_tag() = default;
    };

    /// \cond NOINTERNAL
    struct non_task_policy_tag
    {
        constexpr non_task_policy_tag() = default;
    };
}}    // namespace hpx::execution

namespace hpx { namespace parallel { namespace execution {

    using parallel_execution_tag HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::execution::parallel_execution_tag is deprecated. Use "
        "hpx::execution::parallel_execution_tag instead.") =
        hpx::execution::parallel_execution_tag;
    using sequenced_execution_tag HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::execution::sequenced_execution_tag is deprecated. Use "
        "hpx::execution::sequenced_execution_tag instead.") =
        hpx::execution::sequenced_execution_tag;
    using task_policy_tag HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::execution::task_policy_tag is deprecated. Use "
        "hpx::execution::task_policy_tag instead.") =
        hpx::execution::task_policy_tag;
    using unsequenced_execution_tag HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::execution::unsequenced_execution_tag is deprecated. "
        "Use hpx::execution::unsequenced_execution_tag instead.") =
        hpx::execution::unsequenced_execution_tag;
}}}    // namespace hpx::parallel::execution
/// \endcond

namespace hpx { namespace parallel { namespace execution {

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
    /// \note This is valid for one way executors only, it will call
    ///       exec.sync_execute(f, ts...) if it exists.
    ///
    HPX_INLINE_CONSTEXPR_VARIABLE struct sync_execute_t final
      : hpx::functional::tag_fallback<sync_execute_t>
    {
    private:
        // clang-format off
        template <typename Executor, typename F, typename... Ts,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_any<Executor>::value
            )>
        // clang-format on
        friend HPX_FORCEINLINE decltype(auto) tag_fallback_dispatch(
            sync_execute_t, Executor&& exec, F&& f, Ts&&... ts)
        {
            return detail::sync_execute_fn_helper<std::decay_t<Executor>>::call(
                std::forward<Executor>(exec), std::forward<F>(f),
                std::forward<Ts>(ts)...);
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
    HPX_INLINE_CONSTEXPR_VARIABLE struct async_execute_t final
      : hpx::functional::tag_fallback<async_execute_t>
    {
    private:
        // clang-format off
        template <typename Executor, typename F, typename... Ts,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_any<Executor>::value
            )>
        // clang-format on
        friend HPX_FORCEINLINE decltype(auto) tag_fallback_dispatch(
            async_execute_t, Executor&& exec, F&& f, Ts&&... ts)
        {
            return detail::async_execute_fn_helper<
                std::decay_t<Executor>>::call(std::forward<Executor>(exec),
                std::forward<F>(f), std::forward<Ts>(ts)...);
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
    HPX_INLINE_CONSTEXPR_VARIABLE struct then_execute_t final
      : hpx::functional::tag_fallback<then_execute_t>
    {
    private:
        // clang-format off
        template <typename Executor, typename F, typename Future, typename... Ts,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_any<Executor>::value
            )>
        // clang-format on
        friend HPX_FORCEINLINE decltype(auto) tag_fallback_dispatch(
            then_execute_t, Executor&& exec, F&& f, Future&& predecessor,
            Ts&&... ts)
        {
            return detail::then_execute_fn_helper<std::decay_t<Executor>>::call(
                std::forward<Executor>(exec), std::forward<F>(f),
                std::forward<Future>(predecessor), std::forward<Ts>(ts)...);
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
    HPX_INLINE_CONSTEXPR_VARIABLE struct post_t final
      : hpx::functional::tag_fallback<post_t>
    {
    private:
        // clang-format off
        template <typename Executor, typename F, typename... Ts,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_any<Executor>::value
            )>
        // clang-format on
        friend HPX_FORCEINLINE decltype(auto) tag_fallback_dispatch(
            post_t, Executor&& exec, F&& f, Ts&&... ts)
        {
            return detail::post_fn_helper<std::decay_t<Executor>>::call(
                std::forward<Executor>(exec), std::forward<F>(f),
                std::forward<Ts>(ts)...);
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
    HPX_INLINE_CONSTEXPR_VARIABLE struct bulk_sync_execute_t final
      : hpx::functional::tag_fallback<bulk_sync_execute_t>
    {
    private:
        // clang-format off
        template <typename Executor, typename F, typename Shape, typename... Ts,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_any<Executor>::value&&
                !std::is_integral<Shape>::value
            )>
        // clang-format on
        friend HPX_FORCEINLINE decltype(auto) tag_fallback_dispatch(
            bulk_sync_execute_t, Executor&& exec, F&& f, Shape const& shape,
            Ts&&... ts)
        {
            return detail::bulk_sync_execute_fn_helper<
                std::decay_t<Executor>>::call(std::forward<Executor>(exec),
                std::forward<F>(f), shape, std::forward<Ts>(ts)...);
        }

        // clang-format off
        template <typename Executor, typename F, typename Shape, typename... Ts,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_any<Executor>::value&&
                std::is_integral<Shape>::value
            )>
        // clang-format on
        friend HPX_FORCEINLINE decltype(auto) tag_fallback_dispatch(
            bulk_sync_execute_t, Executor&& exec, F&& f, Shape const& shape,
            Ts&&... ts)
        {
            return detail::bulk_sync_execute_fn_helper<
                std::decay_t<Executor>>::call(std::forward<Executor>(exec),
                std::forward<F>(f),
                hpx::util::detail::make_counting_shape(shape),
                std::forward<Ts>(ts)...);
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
    HPX_INLINE_CONSTEXPR_VARIABLE struct bulk_async_execute_t final
      : hpx::functional::tag_fallback<bulk_async_execute_t>
    {
    private:
        // clang-format off
        template <typename Executor, typename F, typename Shape, typename... Ts,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_any<Executor>::value &&
                !std::is_integral<Shape>::value
            )>
        // clang-format on
        friend HPX_FORCEINLINE decltype(auto) tag_fallback_dispatch(
            bulk_async_execute_t, Executor&& exec, F&& f, Shape const& shape,
            Ts&&... ts)
        {
            return detail::bulk_async_execute_fn_helper<
                std::decay_t<Executor>>::call(std::forward<Executor>(exec),
                std::forward<F>(f), shape, std::forward<Ts>(ts)...);
        }

        // clang-format off
        template <typename Executor, typename F, typename Shape, typename... Ts,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_any<Executor>::value &&
                std::is_integral<Shape>::value
            )>
        // clang-format on
        friend HPX_FORCEINLINE decltype(auto) tag_fallback_dispatch(
            bulk_async_execute_t, Executor&& exec, F&& f, Shape const& shape,
            Ts&&... ts)
        {
            return detail::bulk_async_execute_fn_helper<
                std::decay_t<Executor>>::call(std::forward<Executor>(exec),
                std::forward<F>(f),
                hpx::util::detail::make_counting_shape(shape),
                std::forward<Ts>(ts)...);
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
    HPX_INLINE_CONSTEXPR_VARIABLE struct bulk_then_execute_t final
      : hpx::functional::tag_fallback<bulk_then_execute_t>
    {
    private:
        // clang-format off
        template <typename Executor, typename F, typename Shape,
            typename Future, typename... Ts,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_any<Executor>::value&&
                !std::is_integral<Shape>::value
            )>
        // clang-format on
        friend HPX_FORCEINLINE decltype(auto) tag_fallback_dispatch(
            bulk_then_execute_t, Executor&& exec, F&& f, Shape const& shape,
            Future&& predecessor, Ts&&... ts)
        {
            return detail::bulk_then_execute_fn_helper<
                std::decay_t<Executor>>::call(std::forward<Executor>(exec),
                std::forward<F>(f), shape, std::forward<Future>(predecessor),
                std::forward<Ts>(ts)...);
        }

        // clang-format off
        template <typename Executor, typename F, typename Shape,
            typename Future, typename... Ts,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_any<Executor>::value&&
                std::is_integral<Shape>::value
            )>
        // clang-format on
        friend HPX_FORCEINLINE decltype(auto) tag_fallback_dispatch(
            bulk_then_execute_t, Executor&& exec, F&& f, Shape const& shape,
            Future&& predecessor, Ts&&... ts)
        {
            return detail::bulk_then_execute_fn_helper<
                std::decay_t<Executor>>::call(std::forward<Executor>(exec),
                std::forward<F>(f),
                hpx::util::detail::make_counting_shape(shape),
                std::forward<Future>(predecessor), std::forward<Ts>(ts)...);
        }
    } bulk_then_execute{};
}}}    // namespace hpx::parallel::execution
