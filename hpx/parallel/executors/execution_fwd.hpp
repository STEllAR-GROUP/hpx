//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nounnamed

#if !defined(HPX_PARALLEL_EXECUTORS_EXECUTION_FWD_DEC_23_0712PM)
#define HPX_PARALLEL_EXECUTORS_EXECUTION_FWD_DEC_23_0712PM

#include <hpx/config.hpp>
#include <hpx/traits/executor_traits.hpp>

namespace hpx { namespace parallel { namespace execution
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

    /// \cond NOINTERNAL
    struct task_policy_tag
    {
        HPX_CONSTEXPR task_policy_tag() {}
    };

    ///////////////////////////////////////////////////////////////////////////
    // Define infrastructure for customization points
    namespace detail
    {
        struct post_tag {};
        struct sync_execute_tag {};
        struct async_execute_tag {};
        struct then_execute_tag {};
        struct bulk_sync_execute_tag {};
        struct bulk_async_execute_tag {};
        struct bulk_then_execute_tag {};

        template <typename Executor, typename ... Ts>
        struct undefined_customization_point;

        template <typename Tag>
        struct customization_point
        {
            template <typename Executor, typename ... Ts>
            auto operator()(Executor && exec, Ts &&... ts) const
            ->  undefined_customization_point<Executor, Ts...>
            {
                return undefined_customization_point<Executor, Ts...>{};
            }
        };

        // forward declare customization point implementations
        template <>
        struct customization_point<post_tag>
        {
            template <typename Executor, typename F, typename ... Ts>
            HPX_FORCEINLINE
            auto operator()(Executor && exec, F && f, Ts &&... ts) const;
        };

        template <>
        struct customization_point<sync_execute_tag>
        {
            template <typename Executor, typename F, typename ... Ts>
            HPX_FORCEINLINE
            auto operator()(Executor && exec, F && f, Ts &&... ts) const;
        };

        template <>
        struct customization_point<async_execute_tag>
        {
            template <typename Executor, typename F, typename ... Ts>
            HPX_FORCEINLINE
            auto operator()(Executor && exec, F && f, Ts &&... ts) const;
        };

        template <>
        struct customization_point<then_execute_tag>
        {
            template <typename Executor, typename F, typename Future,
                typename ... Ts>
            HPX_FORCEINLINE
            auto operator()(Executor && exec, F && f, Future& predecessor,
                Ts &&... ts) const;
        };

        template <>
        struct customization_point<bulk_sync_execute_tag>
        {
            template <typename Executor, typename F, typename Shape,
                typename ... Ts>
            HPX_FORCEINLINE
            auto operator()(Executor && exec, F && f, Shape const& shape,
                Ts &&... ts) const;
        };

        template <>
        struct customization_point<bulk_async_execute_tag>
        {
            template <typename Executor, typename F, typename Shape,
                typename ... Ts>
            HPX_FORCEINLINE
            auto operator()(Executor && exec, F && f, Shape const& shape,
                Ts &&... ts) const;
        };

        template <>
        struct customization_point<bulk_then_execute_tag>
        {
            template <typename Executor, typename F, typename Shape,
                typename Future, typename ... Ts>
            HPX_FORCEINLINE
            auto operator()(Executor && exec, F && f, Shape const& shape,
                Future& predecessor, Ts &&... ts) const;
        };

        // Helper facility to avoid ODR violations
        template <typename T>
        struct static_const
        {
            static T const value;
        };

        template <typename T>
        T const static_const<T>::value = T{};
    }
    /// \endcond

    // define customization points
    namespace
    {
        ///////////////////////////////////////////////////////////////////////
        // OneWayExecutor customization point: execution::execute

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
        constexpr detail::customization_point<detail::sync_execute_tag> const&
            sync_execute = detail::static_const<
                    detail::customization_point<detail::sync_execute_tag>
                >::value;

        ///////////////////////////////////////////////////////////////////////
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
        constexpr detail::customization_point<detail::async_execute_tag> const&
            async_execute = detail::static_const<
                    detail::customization_point<detail::async_execute_tag>
                >::value;

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
        constexpr detail::customization_point<detail::then_execute_tag> const&
            then_execute = detail::static_const<
                    detail::customization_point<detail::then_execute_tag>
                >::value;

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
        ///       exec.apply_execute(f, ts...), if available, otherwise
        ///       it calls exec.async_execute(f, ts...) while discarding the
        ///       returned future), and for non-blocking two way executors
        ///       (calls exec.post(f, ts...) if it exists).
        ///
        constexpr detail::customization_point<detail::post_tag> const&
            post = detail::static_const<
                    detail::customization_point<detail::post_tag>
                >::value;

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
        constexpr detail::customization_point<detail::bulk_sync_execute_tag> const&
            bulk_sync_execute = detail::static_const<
                    detail::customization_point<detail::bulk_sync_execute_tag>
                >::value;

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
        constexpr detail::customization_point<detail::bulk_async_execute_tag> const&
            bulk_async_execute = detail::static_const<
                    detail::customization_point<detail::bulk_async_execute_tag>
                >::value;

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
        constexpr detail::customization_point<detail::bulk_then_execute_tag> const&
            bulk_then_execute = detail::static_const<
                    detail::customization_point<detail::bulk_then_execute_tag>
                >::value;
    }
}}}

#endif

