//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/executor_parameter_traits.hpp

#if !defined(HPX_PARALLEL_THREAD_EXECUTOR_PARAMETER_TRAITS_AUG_26_2015_1204PM)
#define HPX_PARALLEL_THREAD_EXECUTOR_PARAMETER_TRAITS_AUG_26_2015_1204PM

#include <hpx/config.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/executors/executor_parameter_traits.hpp>
#include <hpx/parallel/executors/thread_executor_traits.hpp>
#include <hpx/traits/is_executor_parameters.hpp>
#include <hpx/util/always_void.hpp>

#include <cstdarg>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v3)
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Parameters>
    struct executor_parameter_traits<Parameters,
        typename std::enable_if<
            hpx::traits::is_threads_executor<Parameters>::value
        >::type>
    {
        /// The type of the executor associated with this instance of
        /// \a executor_traits
        typedef Parameters executor_parameters_type;

        /// Returns whether the number of loop iterations to combine is
        /// different for each of the generated chunks.
        ///
        /// \param params   [in] The executor parameters object to use for
        ///                 determining whether the chunk size is variable.
        /// \param sched    [in] The executor object which will be used for
        ///                 scheduling of the tasks.
        ///
        /// \note This calls params.variable_chunk_size(exec), if available,
        ///       otherwise it returns false.
        ///
        template <typename Executor>
        static bool variable_chunk_size(executor_parameters_type& params,
            Executor& sched)
        {
            return detail::call_variable_chunk_size(params, sched);
        }

        /// Return the number of invocations of the given function \a f which
        /// should be combined into a single task
        ///
        /// \param params   [in] The executor parameters object to use for
        ///                 determining the chunk size for the given number of
        ///                 tasks \a num_tasks.
        /// \param sched    [in] The executor object which will be used used
        ///                 for scheduling of the the loop iterations.
        /// \param f        [in] The function which will be optionally scheduled
        ///                 using the given executor.
        /// \param num_tasks [in] The number of tasks the chunk size should be
        ///                 determined for
        ///
        /// \note  The parameter \a f is expected to be a nullary function
        ///        returning a `std::size_t` representing the number of
        ///        iteration the function has already executed (i.e. which
        ///        don't have to be scheduled anymore).
        ///
        template <typename Executor, typename F>
        static std::size_t get_chunk_size(executor_parameters_type& params,
            Executor& sched, F && f, std::size_t num_tasks)
        {
            return detail::call_get_chunk_size(params, sched,
                std::forward<F>(f), num_tasks);
        }

        /// Reset the internal round robin thread distribution scheme for the
        /// given executor.
        ///
        /// \param params   [in] The executor parameters object to use for
        ///                 resetting the thread distribution scheme.
        /// \param sched    [in] The executor object to use.
        ///
        template <typename Executor>
        static void reset_thread_distribution(executor_parameters_type&,
            Executor& sched)
        {
            sched.reset_thread_distribution();
        }

        /// Retrieve the number of (kernel-)threads used by the associated
        /// executor.
        ///
        /// \param exec  [in] The executor object to use for scheduling of the
        ///              function \a f.
        /// \param params [in] The executor parameters object to use as a
        ///              fallback if the executor does not expose
        ///
        /// \note This calls exec.processing_units_count() if it exists;
        ///       otherwise it forwards teh request to the executor parameters
        ///       object.
        ///
        static std::size_t processing_units_count(
            executor_parameters_type& params)
        {
            return detail::call_processing_units_parameter_count(params);
        }

        /// Mark the begin of a parallel algorithm execution
        ///
        /// \param params [in] The executor parameters object to use as a
        ///              fallback if the executor does not expose
        ///
        /// \note This calls params.mark_begin_execution(exec) if it exists;
        ///       otherwise it does nothing.
        ///
        static void mark_begin_execution(executor_parameters_type& params)
        {
            detail::call_mark_begin_execution(params);
        }

        /// Mark the end of a parallel algorithm execution
        ///
        /// \param params [in] The executor parameters object to use as a
        ///              fallback if the executor does not expose
        ///
        /// \note This calls params.mark_end_execution(exec) if it exists;
        ///       otherwise it does nothing.
        ///
        static void mark_end_execution(executor_parameters_type& params)
        {
            detail::call_mark_end_execution(params);
        }
    };
}}}

#endif
