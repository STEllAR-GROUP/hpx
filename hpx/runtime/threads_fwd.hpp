//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file threads_fwd.hpp

#ifndef HPX_RUNTIME_THREADS_FWD_HPP
#define HPX_RUNTIME_THREADS_FWD_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/threads/thread_data_fwd.hpp>
#include <hpx/runtime/threads/thread_enums.hpp>
#include <hpx/util/unique_function.hpp>

#include <boost/thread/mutex.hpp>

namespace hpx
{
    /// \namespace threads
    ///
    /// The namespace \a thread-manager contains all the definitions required
    /// for the scheduling, execution and general management of \a
    /// hpx#threadmanager#thread's.
    namespace threads
    {
        namespace policies
        {
            struct scheduler_base;

            struct lockfree_fifo;
            struct lockfree_lifo;

            // multi priority scheduler with work-stealing
            template <typename Mutex = boost::mutex
                    , typename PendingQueuing = lockfree_fifo
                    , typename StagedQueuing = lockfree_fifo
                    , typename TerminatedQueuing = lockfree_lifo
                     >
            class HPX_EXPORT local_priority_queue_scheduler;

            // single priority scheduler with work-stealing
            template <typename Mutex = boost::mutex
                    , typename PendingQueuing = lockfree_fifo
                    , typename StagedQueuing = lockfree_fifo
                    , typename TerminatedQueuing = lockfree_lifo
                     >
            class HPX_EXPORT local_queue_scheduler;

#if defined(HPX_HAVE_PERIODIC_PRIORITY_SCHEDULER)
            template <typename Mutex = boost::mutex
                    , typename PendingQueuing = lockfree_fifo
                    , typename StagedQueuing = lockfree_fifo
                    , typename TerminatedQueuing = lockfree_lifo
                     >
            class HPX_EXPORT periodic_priority_queue_scheduler;
#endif

#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
            // multi priority scheduler with no work-stealing
            template <typename Mutex = boost::mutex
                    , typename PendingQueuing = lockfree_fifo
                    , typename StagedQueuing = lockfree_fifo
                    , typename TerminatedQueuing = lockfree_lifo
                     >
            class HPX_EXPORT static_priority_queue_scheduler;
#endif

#if defined(HPX_HAVE_STATIC_SCHEDULER)
            // single priority scheduler with no work-stealing
            template <typename Mutex = boost::mutex
                    , typename PendingQueuing = lockfree_fifo
                    , typename StagedQueuing = lockfree_fifo
                    , typename TerminatedQueuing = lockfree_lifo
                     >
            class HPX_EXPORT static_queue_scheduler;
#endif

#if defined(HPX_HAVE_THROTTLE_SCHEDULER) && defined(HPX_HAVE_APEX)
            // single priority scheduler with work-stealing and throttling
            template <typename Mutex = boost::mutex
                    , typename PendingQueuing = lockfree_fifo
                    , typename StagedQueuing = lockfree_fifo
                    , typename TerminatedQueuing = lockfree_lifo
                     >
            class HPX_EXPORT throttle_queue_scheduler;
#endif

#if defined(HPX_HAVE_HIERARCHY_SCHEDULER)
            template <typename Mutex = boost::mutex
                    , typename PendingQueuing = lockfree_fifo
                    , typename StagedQueuing = lockfree_fifo
                    , typename TerminatedQueuing = lockfree_lifo
                     >
            class HPX_EXPORT hierarchy_scheduler;
#endif

            typedef local_priority_queue_scheduler<
                boost::mutex,
                lockfree_fifo, // FIFO pending queuing
                lockfree_fifo, // FIFO staged queuing
                lockfree_lifo  // LIFO terminated queuing
            > fifo_priority_queue_scheduler;

#if defined(HPX_HAVE_ABP_SCHEDULER)
            struct lockfree_abp_fifo;
            struct lockfree_abp_lifo;

            typedef local_priority_queue_scheduler<
                boost::mutex,
                lockfree_abp_fifo, // FIFO + ABP pending queuing
                lockfree_abp_fifo, // FIFO + ABP staged queuing
                lockfree_lifo  // LIFO terminated queuing
            > abp_fifo_priority_queue_scheduler;
#endif

            // define the default scheduler to use
            typedef fifo_priority_queue_scheduler queue_scheduler;

            class HPX_EXPORT callback_notifier;
        }
    }
}

#endif
