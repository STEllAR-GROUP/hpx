//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/thread_executor_information_traits.hpp

#if !defined(HPX_PARALLEL_THREAD_EXECUTOR_INFORMATION_TRAITS_AUG_26_2015_1141AM)
#define HPX_PARALLEL_THREAD_EXECUTOR_INFORMATION_TRAITS_AUG_26_2015_1141AM

#include <hpx/config.hpp>
#include <hpx/apply.hpp>
#include <hpx/async.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/executors/executor_information_traits.hpp>
#include <hpx/runtime/threads/policies/scheduler_mode.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/traits/is_launch_policy.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/result_of.hpp>
#include <hpx/util/unwrapped.hpp>

#include <cstddef>
#include <type_traits>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v3)
{
    ///////////////////////////////////////////////////////////////////////////
    /// Specialization for executor_traits for types which conform to
    /// traits::is_threads_executor<Executor>
    template <typename Executor>
    struct executor_information_traits<Executor,
        typename std::enable_if<
            hpx::traits::is_threads_executor<Executor>::value
        >::type>
    {
        /// The type of the executor associated with this instance of
        /// \a executor_traits
        typedef Executor executor_type;

        /// Retrieve the number of (kernel-)threads used by the associated
        /// executor. All threads::executors invoke
        /// hpx::get_os_thread_count(sched).
        ///
        /// \param sched  [in] The executor object to use for the number of
        ///               os-threads used to schedule tasks.
        ///
        /// \note This calls exec.os_thread_count() if it exists;
        ///       otherwise it executes hpx::get_os_thread_count().
        ///
        template <typename Parameters>
        static std::size_t processing_units_count(executor_type const& sched,
            Parameters&)
        {
            return hpx::get_os_thread_count(sched);
        }

        /// Retrieve whether this executor has operations pending or not.
        /// All threads::executors invoke sched.num_pending_closures().
        ///
        /// \param sched  [in] The executor object to use for querying the
        ///               number of pending tasks.
        ///
        static bool has_pending_closures(executor_type const& sched)
        {
            return sched.num_pending_closures();
        }

        /// Retrieve the bitmask describing the processing units the given
        /// thread is allowed to run on
        /// All threads::executors invoke sched.get_pu_mask().
        ///
        /// \param sched  [in] The executor object to use for querying the
        ///               number of pending tasks.
        ///
        static threads::mask_cref_type get_pu_mask(executor_type const& sched,
            threads::topology& topo, std::size_t thread_num)
        {
            return sched.get_pu_mask(topo, thread_num);
        }

        /// Set various modes of operation on the scheduler underneath the
        /// given executor.
        ///
        /// \param params   [in] The executor parameters object to use for
        ///                 resetting the thread distribution scheme.
        /// \param sched    [in] The executor object to use.
        ///
        template <typename Mode>
        static void set_scheduler_mode(executor_type& sched, Mode mode)
        {
            sched.set_scheduler_mode(
                static_cast<threads::policies::scheduler_mode>(mode)
            );
        }
    };
}}}

#endif
