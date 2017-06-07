//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nounnamed

#if !defined(HPX_PARALLEL_EXECUTORS_TIMED_EXECUTION_FWD_JAN_07_2017_0720AM)
#define HPX_PARALLEL_EXECUTORS_TIMED_EXECUTION_FWD_JAN_07_2017_0720AM

#include <hpx/config.hpp>
#include <hpx/parallel/executors/execution_fwd.hpp>
#include <hpx/traits/executor_traits.hpp>

namespace hpx { namespace util
{
    class steady_time_point;
    class steady_duration;
}}

namespace hpx { namespace parallel { namespace execution
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        /// \cond NOINTERNAL
        struct post_at_tag {};
        struct post_after_tag {};
        struct sync_execute_at_tag {};
        struct sync_execute_after_tag {};
        struct async_execute_at_tag {};
        struct async_execute_after_tag {};

        // forward declare customization point implementations
        template <>
        struct customization_point<post_at_tag>
        {
            template <typename Executor, typename F, typename ... Ts>
            HPX_FORCEINLINE
            auto operator()(Executor && exec,
                hpx::util::steady_time_point const& abs_time,
                F && f, Ts &&... ts) const;
        };

        template <>
        struct customization_point<post_after_tag>
        {
            template <typename Executor, typename F, typename ... Ts>
            HPX_FORCEINLINE
            auto operator()(Executor && exec,
                hpx::util::steady_duration const& rel_time,
                F && f, Ts &&... ts) const;
        };

        template <>
        struct customization_point<sync_execute_at_tag>
        {
            template <typename Executor, typename F, typename ... Ts>
            HPX_FORCEINLINE
            auto operator()(Executor && exec,
                hpx::util::steady_time_point const& abs_time,
                F && f, Ts &&... ts) const;
        };

        template <>
        struct customization_point<sync_execute_after_tag>
        {
            template <typename Executor, typename F, typename ... Ts>
            HPX_FORCEINLINE
            auto operator()(Executor && exec,
                hpx::util::steady_duration const& rel_time,
                F && f, Ts &&... ts) const;
        };

        template <>
        struct customization_point<async_execute_at_tag>
        {
            template <typename Executor, typename F, typename ... Ts>
            HPX_FORCEINLINE
            auto operator()(Executor && exec,
                hpx::util::steady_time_point const& abs_time,
                F && f, Ts &&... ts) const;
        };

        template <>
        struct customization_point<async_execute_after_tag>
        {
            template <typename Executor, typename F, typename ... Ts>
            HPX_FORCEINLINE
            auto operator()(Executor && exec,
                hpx::util::steady_duration const& rel_time,
                F && f, Ts &&... ts) const;
        };
        /// \endcond
    }

    // define customization points
    namespace
    {
        ///////////////////////////////////////////////////////////////////////
        // extensions

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
        /// \note This calls exec.apply_execute_at(abs_time, f, ts...), if
        ///       available, otherwise it emulates timed scheduling by delaying
        ///       calling execution::post() on the underlying non-time-scheduled
        ///       execution agent.
        ///
        constexpr detail::customization_point<detail::post_at_tag> const&
            post_at = detail::static_const<
                    detail::customization_point<detail::post_at_tag>
                >::value;

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
        /// \note This calls exec.apply_execute_after(rel_time, f, ts...), if
        ///       available, otherwise it emulates timed scheduling by delaying
        ///       calling execution::post() on the underlying non-time-scheduled
        ///       execution agent.
        ///
        constexpr detail::customization_point<detail::post_after_tag> const&
            post_after = detail::static_const<
                    detail::customization_point<detail::post_after_tag>
                >::value;

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
        constexpr detail::customization_point<detail::async_execute_at_tag> const&
            async_execute_at = detail::static_const<
                    detail::customization_point<detail::async_execute_at_tag>
                >::value;

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
        constexpr detail::customization_point<detail::async_execute_after_tag> const&
            async_execute_after = detail::static_const<
                    detail::customization_point<detail::async_execute_after_tag>
                >::value;

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
        constexpr detail::customization_point<detail::sync_execute_at_tag> const&
            sync_execute_at = detail::static_const<
                    detail::customization_point<detail::sync_execute_at_tag>
                >::value;

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
        constexpr detail::customization_point<detail::sync_execute_after_tag> const&
            sync_execute_after = detail::static_const<
                    detail::customization_point<detail::sync_execute_after_tag>
                >::value;
    }
}}}

#endif

