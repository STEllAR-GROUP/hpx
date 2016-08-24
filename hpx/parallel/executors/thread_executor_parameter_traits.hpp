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

        /// The compile-time information about whether the number of loop
        /// iterations to combine is different for each of the generated chunks.
        ///
        /// \note This calls extracts parameters_type::has_variable_chunk_size,
        ///       if available, otherwise it returns std::false_type.
        ///
        typedef typename detail::extract_has_variable_chunk_size<
                executor_parameters_type
            >::type has_variable_chunk_size;

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
        /// \param cores    [in] The number of cores the number of chunks
        ///                 should be determined for.
        /// \param num_tasks [in] The number of tasks the chunk size should be
        ///                 determined for
        ///
        /// \note  The parameter \a f is expected to be a nullary function
        ///        returning a `std::size_t` representing the number of
        ///        iteration the function has already executed (i.e. which
        ///        don't have to be scheduled anymore).
        ///
        template <typename Parameters_, typename Executor, typename F>
        static std::size_t get_chunk_size(Parameters_ && params,
            Executor && sched, F && f, std::size_t cores, std::size_t num_tasks)
        {
            return detail::call_get_chunk_size(
                std::forward<Parameters_>(params), std::forward<Executor>(sched),
                std::forward<F>(f), cores, num_tasks);
        }

        /// Return the largest reasonable number of chunks to create for a
        /// single algorithm invocation.
        ///
        /// \param params   [in] The executor parameters object to use for
        ///                 determining the number of chunks for the given
        ///                 number of \a cores.
        /// \param exec     [in] The executor object which will be used used
        ///                 for scheduling of the the loop iterations.
        /// \param cores    [in] The number of cores the number of chunks
        ///                 should be determined for.
        /// \param num_tasks [in] The number of tasks the chunk size should be
        ///                 determined for
        ///
        template <typename Parameters_, typename Executor>
        static std::size_t maximal_number_of_chunks(
            Parameters_ && params, Executor && sched, std::size_t cores,
            std::size_t num_tasks)
        {
            return detail::call_maximal_number_of_chunks(
                std::forward<Parameters_>(params), std::forward<Executor>(sched),
                cores, num_tasks);
        }

        /// Reset the internal round robin thread distribution scheme for the
        /// given executor.
        ///
        /// \param params   [in] The executor parameters object to use for
        ///                 resetting the thread distribution scheme.
        /// \param sched    [in] The executor object to use.
        ///
        template <typename Parameters_, typename Executor>
        static void reset_thread_distribution(Parameters_ &&,
            Executor && sched)
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
        template <typename Parameters_>
        static std::size_t processing_units_count(Parameters_ && params)
        {
            return detail::call_processing_units_parameter_count(
                std::forward<Parameters_>(params));
        }

        /// Mark the begin of a parallel algorithm execution
        ///
        /// \param params [in] The executor parameters object to use as a
        ///              fallback if the executor does not expose
        ///
        /// \note This calls params.mark_begin_execution(exec) if it exists;
        ///       otherwise it does nothing.
        ///
        template <typename Parameters_>
        static void mark_begin_execution(Parameters_ && params)
        {
            detail::call_mark_begin_execution(
                std::forward<Parameters_>(params));
        }

        /// Mark the end of a parallel algorithm execution
        ///
        /// \param params [in] The executor parameters object to use as a
        ///              fallback if the executor does not expose
        ///
        /// \note This calls params.mark_end_execution(exec) if it exists;
        ///       otherwise it does nothing.
        ///
        template <typename Parameters_>
        static void mark_end_execution(Parameters_ && params)
        {
            detail::call_mark_end_execution(std::forward<Parameters_>(params));
        }
    };
}}}

#endif
