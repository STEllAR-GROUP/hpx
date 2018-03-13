//  Copyright (c)      2013 Thomas Heller
//  Copyright (c) 2007-2017 Hartmut Kaiser
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
            Mutex, PendingQueuing, StagedQueuing, TerminatedQueuing
          >
    {
    public:
        typedef local_priority_queue_scheduler<
            Mutex, PendingQueuing, StagedQueuing, TerminatedQueuing
        > base_type;

        typedef typename base_type::init_parameter_type
            init_parameter_type;

        static_priority_queue_scheduler(init_parameter_type const& init,
                bool deferred_initialization = true)
          : base_type(init, deferred_initialization)
        {}

        virtual bool has_thread_stealing() const override { return false; }

        static std::string get_scheduler_name()
        {
            return "static_priority_queue_scheduler";
        }

        /// Return the next thread to be executed, return false if non is
        /// available
        bool get_next_thread(std::size_t num_thread, bool running,
            std::int64_t& idle_loop_count, threads::thread_data*& thrd) override
        {
            std::size_t queues_size = this->queues_.size();

            typedef typename base_type::thread_queue_type thread_queue_type;

            if (num_thread < this->high_priority_queues_.size())
            {
                thread_queue_type* q = this->high_priority_queues_[num_thread];

                q->increment_num_pending_accesses();
                if (q->get_next_thread(thrd))
                    return true;
                q->increment_num_pending_misses();
            }

            {
                HPX_ASSERT(num_thread < queues_size);
                thread_queue_type* q = this->queues_[num_thread];

                q->increment_num_pending_accesses();
                if (q->get_next_thread(thrd))
                    return true;
                q->increment_num_pending_misses();
            }

            // Limit access to the low priority queue to one worker thread
            if ((queues_size - 1) == num_thread)
                return this->low_priority_queue_.get_next_thread(thrd);

            return false;
        }
    };
}}}

#include <hpx/config/warnings_suffix.hpp>

#endif
#endif

