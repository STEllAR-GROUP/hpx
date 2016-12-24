//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nounnamed

#if !defined(HPX_PARALLEL_EXECUTORS_EXECUTION_FWD_DEC_23_0712PM)
#define HPX_PARALLEL_EXECUTORS_EXECUTION_FWD_DEC_23_0712PM

#include <hpx/config.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(concurrency_v2) {
    namespace execution
{
    ///////////////////////////////////////////////////////////////////////////
    /// Function invocations executed by a group of sequential execution agents
    /// execute in sequential order.
    struct sequenced_execution_tag {};

    /// Function invocations executed by a group of parallel execution agents
    /// execute in unordered fashion. Any such invocations executing in the
    /// same thread are indeterminately sequenced with respect to each other.
    ///
    /// \note \a parallel_execution_tag is weaker than
    ///       \a sequenced_execution_tag.
    struct parallel_execution_tag {};

    /// Function invocations executed by a group of vector execution agents are
    /// permitted to execute in unordered fashion when executed in different
    /// threads, and un-sequenced with respect to one another when executed in
    /// the same thread.
    ///
    /// \note \a unsequenced_execution_tag is weaker than
    ///       \a parallel_execution_tag.
    struct unsequenced_execution_tag {};

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        // Helper facility to avoid ODR violations
        template <typename T>
        struct static_const
        {
            HPX_STATIC_CONSTEXPR T value{};
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    // Forward declaration of executor customization points
    namespace detail
    {
        template <typename Executor, typename Enable = void>
        struct execute_fn_helper;

        struct execute_fn
        {
            template <typename Executor, typename F, typename ... Ts>
            auto operator()(Executor const& exec, F && f, Ts &&... ts) const
            ->  decltype(execute_fn_helper<Executor>::call(
                    exec, std::forward<F>(f), std::forward<Ts>(ts)...
                ));
        };

        template <typename Executor, typename Enable = void>
        struct async_execute_fn_helper;

        struct async_execute_fn
        {
            template <typename Executor, typename F, typename ... Ts>
            auto operator()(Executor const& exec, F && f, Ts &&... ts) const
            ->  decltype(async_execute_fn_helper<Executor>::call(
                    exec, std::forward<F>(f), std::forward<Ts>(ts)...
                ));
        };

        template <typename Executor, typename Enable = void>
        struct sync_execute_fn_helper;

        struct sync_execute_fn
        {
            template <typename Executor, typename F, typename ... Ts>
            auto operator()(Executor const& exec, F && f, Ts &&... ts) const
            ->  decltype(sync_execute_fn_helper<Executor>::call(
                    exec, std::forward<F>(f), std::forward<Ts>(ts)...
                ));
        };

        template <typename Executor, typename Enable = void>
        struct then_execute_fn_helper;

        struct then_execute_fn
        {
            template <typename Executor, typename F, typename Future,
                typename ... Ts>
            auto operator()(Executor const& exec, F && f, Future& predecessor,
                    Ts &&... ts) const
            ->  decltype(then_execute_fn_helper<Executor>::call(
                    exec, std::forward<F>(f), predecessor, std::forward<Ts>(ts)...
                ));
        };

        template <typename Executor, typename Enable = void>
        struct post_fn_helper;

        struct post_fn
        {
            template <typename Executor, typename F, typename Future,
                typename ... Ts>
            auto operator()(Executor const& exec, F && f, Ts &&... ts) const
            ->  decltype(post_fn_helper<Executor>::call(
                    exec, std::forward<F>(f), std::forward<Ts>(ts)...
                ));
        };
    }

    namespace
    {
        // OneWayExecutor customization point: execution::execute
        constexpr detail::execute_fn const& execute =
            detail::static_const<detail::execute_fn>::value;

        // TwoWayExecutor customization points: execution::async_execute,
        // execution::sync_execute, and execution::then_execute
        constexpr detail::async_execute_fn const& async_execute =
            detail::static_const<detail::async_execute_fn>::value;

        constexpr detail::sync_execute_fn const& sync_execute =
            detail::static_const<detail::sync_execute_fn>::value;

        constexpr detail::then_execute_fn const& then_execute =
            detail::static_const<detail::then_execute_fn>::value;

        // NonBlockingOneWayExecutor customization point: execution::post
        constexpr detail::post_fn const& post =
            detail::static_const<detail::post_fn>::value;
    }
}}}}

#endif

