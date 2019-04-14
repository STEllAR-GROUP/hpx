//  Copyright (c)      2013 Thomas Heller
//  Copyright (c) 2007-2019 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADMANAGER_SCHEDULING_STATIC_PRIOTITY_QUEUE_HPP)
#define HPX_THREADMANAGER_SCHEDULING_STATIC_PRIOTITY_QUEUE_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
#include <hpx/compat/mutex.hpp>
#include <hpx/runtime/threads/policies/lockfree_queue_backends.hpp>
#include <hpx/runtime/threads/policies/local_priority_queue_scheduler.hpp>
#include <hpx/runtime/threads_fwd.hpp>
#include <hpx/util/assert.hpp>

#include <cstddef>
#include <cstdint>
#include <string>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace policies
{
    ///////////////////////////////////////////////////////////////////////////
    /// The static_priority_queue_scheduler maintains exactly one queue of work
    /// items (threads) per OS thread, where this OS thread pulls its next work
    /// from. Additionally it maintains separate queues: several for high
    /// priority threads and one for low priority threads.
    /// High priority threads are executed by the first N OS threads before any
    /// other work is executed. Low priority threads are executed by the last
    /// OS thread whenever no other work is available.
    /// This scheduler does not do any work stealing.
    template <typename Mutex = compat::mutex,
        typename PendingQueuing = lockfree_fifo,
        typename StagedQueuing = lockfree_fifo,
        typename TerminatedQueuing = lockfree_lifo>
    class HPX_EXPORT static_priority_queue_scheduler
        : public local_priority_queue_scheduler<
            Mutex, PendingQueuing, StagedQueuing, TerminatedQueuing>
    {
    public:
        using base_type = local_priority_queue_scheduler<Mutex, PendingQueuing,
            StagedQueuing, TerminatedQueuing>;

        using init_parameter_type = typename base_type::init_parameter_type;

        static_priority_queue_scheduler(init_parameter_type const& init,
                bool deferred_initialization = true)
          : base_type(init, deferred_initialization)
        {
            // disable thread stealing to begin with
            this->remove_scheduler_mode(policies::enable_stealing);
        }

        scheduler_mode get_scheduler_mode(std::size_t num_thread) const override
        {
            return scheduler_mode(
                this->base_type::get_scheduler_mode(num_thread) &
                ~policies::enable_stealing);
        }

        static std::string get_scheduler_name()
        {
            return "static_priority_queue_scheduler";
        }
    };
}}}

#include <hpx/config/warnings_suffix.hpp>

#endif
#endif

