//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/thread_timed_executor_traits.hpp

#if !defined(HPX_PARALLEL_THREAD_TIMED_EXECUTOR_TRAITS_AUG_07_2015_0328PM)
#define HPX_PARALLEL_THREAD_TIMED_EXECUTOR_TRAITS_AUG_07_2015_0328PM

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/packaged_task.hpp>
#include <hpx/util/date_time_chrono.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/executors/thread_executor_traits.hpp>
#include <hpx/parallel/executors/timed_executor_traits.hpp>

#include <type_traits>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v3)
{
    ///////////////////////////////////////////////////////////////////////////
    /// The timed_executor_traits type is used to request execution agents from
    /// an executor. It is analogous to the interaction between containers and
    /// allocator_traits. The generated generated execution agents support
    /// timed scheduling functionality (in addition to what is supported
    /// execution agents generated using execution_traits type).
    ///
    /// This is the specialization for threads::executor types
    ///
    template <typename Executor>
    struct timed_executor_traits<Executor,
            typename std::enable_if<
                hpx::traits::is_threads_executor<Executor>::value
            >::type>
        : executor_traits<Executor>
    {
        /// The type of the executor associated with this instance of
        /// \a executor_traits
        typedef typename executor_traits<Executor>::executor_type
            executor_type;

        /// The category of agents created by the bulk-form execute() and
        /// async_execute()
        ///
        typedef typename executor_traits<Executor>::execution_category
            execution_category;

        /// The type of future returned by async_execute(). All
        /// threads::executors return hpx::future<T>.
        ///
        template <typename T>
        struct future
        {
            typedef hpx::future<T> type;
        };

        /// \brief Singleton form of asynchronous fire & forget execution agent
        ///        creation supporting timed execution.
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
        ///
        /// \note This calls exec.apply_execute_at(abs_time, f), if available,
        ///       otherwise it emulates timed scheduling by delaying calling
        ///       exec.apply_execute() on the underlying non-scheduled
        ///       execution agent while discarding the returned future.
        ///
        template <typename F>
        static void apply_execute_at(executor_type& sched,
            hpx::util::steady_time_point const& abs_time, F && f)
        {
            sched.add_at(abs_time, std::forward<F>(f), "apply_execute_at");
        }

        /// \brief Singleton form of asynchronous fire & forget execution agent
        ///        creation supporting timed execution.
        ///
        /// This asynchronously (fire & forget) creates a single function
        /// invocation f() using the associated executor after the given amount
        /// of time.
        ///
        /// \param exec [in] The executor object to use for scheduling of the
        ///             function \a f.
        /// \param rel_time [in] The duration of time after which the given
        ///             function should be scheduled to run.
        /// \param f    [in] The function which will be scheduled using the
        ///             given executor.
        ///
        template <typename F>
        static void apply_execute_after(executor_type& sched,
            hpx::util::steady_duration const& rel_time, F && f)
        {
            sched.add_after(rel_time, std::forward<F>(f), "apply_execute_at");
        }

        /// \brief Singleton form of asynchronous execution agent creation
        ///        supporting timed execution.
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
        ///
        /// \returns f()'s result through a future
        ///
        template <typename F>
        static hpx::future<typename hpx::util::result_of<F()>::type>
        async_execute_at(executor_type& sched,
                hpx::util::steady_time_point const& abs_time, F && f)
        {
            typedef typename hpx::util::result_of<F()>::type result_type;

            lcos::local::packaged_task<result_type()> task(std::forward<F>(f));
            hpx::future<result_type> result = task.get_future();
            sched.add_at(abs_time, std::move(task), "async_execute_at");
            return result;
        }

        /// \brief Singleton form of asynchronous execution agent creation
        ///        supporting timed execution.
        ///
        /// This asynchronously creates a single function invocation f() using
        /// the associated executor after the given amount of time.
        ///
        /// \param exec [in] The executor object to use for scheduling of the
        ///             function \a f.
        /// \param rel_time [in] The duration of time after which the given
        ///             function should be scheduled to run.
        /// \param f    [in] The function which will be scheduled using the
        ///             given executor.
        ///
        /// \returns f()'s result through a future
        ///
        template <typename F>
        static hpx::future<typename hpx::util::result_of<F()>::type>
        async_execute_after(executor_type& sched,
                hpx::util::steady_duration const& rel_time, F && f)
        {
            typedef typename hpx::util::result_of<F()>::type result_type;

            lcos::local::packaged_task<result_type()> task(std::forward<F>(f));
            hpx::future<result_type> result = task.get_future();
            sched.add_after(rel_time, std::move(task), "async_execute_after");
            return result;
        }

        /// \brief Singleton form of synchronous execution agent creation
        ///        supporting timed execution.
        ///
        /// This synchronously creates a single function invocation f() using
        /// the associated executor at the given point in time. The execution
        /// of the supplied function synchronizes with the caller.
        ///
        /// \param exec [in] The executor object to use for scheduling of the
        ///             function \a f.
        /// \param abs_time [in] The point in time the given function should be
        ///             scheduled at to run.
        /// \param f    [in] The function which will be scheduled using the
        ///             given executor.
        ///
        /// \returns f()'s result
        ///
        template <typename F>
        static typename hpx::util::result_of<F()>::type
        execute_at(executor_type& sched,
                hpx::util::steady_time_point const& abs_time, F && f)
        {
            return async_execute_at(sched, abs_time, std::forward<F>(f)).get();
        }

        /// \brief Singleton form of synchronous execution agent creation
        ///        supporting timed execution.
        ///
        /// This synchronously creates a single function invocation f() using
        /// the associated executor after the given amount of time. The
        /// execution of the supplied function synchronizes with the caller.
        ///
        /// \param exec [in] The executor object to use for scheduling of the
        ///             function \a f.
        /// \param rel_time [in] The duration of time after which the given
        ///             function should be scheduled to run.
        /// \param f    [in] The function which will be scheduled using the
        ///             given executor.
        ///
        /// \returns f()'s result
        ///
        template <typename F>
        static typename hpx::util::result_of<F()>::type
        execute_after(executor_type& sched,
                hpx::util::steady_duration const& rel_time, F && f)
        {
            return async_execute_after(sched, rel_time, std::forward<F>(f)).get();
        }
    };
}}}

#endif
