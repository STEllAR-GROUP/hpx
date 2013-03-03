//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADMANAGER_SCHEDULING_GLOBAL_QUEUE_JUN_18_2009_1116AM)
#define HPX_THREADMANAGER_SCHEDULING_GLOBAL_QUEUE_JUN_18_2009_1116AM

#include <vector>
#include <memory>

#include <hpx/config.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/block_profiler.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/policies/global_thread_queue.hpp>

#include <boost/mpl/bool.hpp>

#include <hpx/config/warnings_prefix.hpp>

// TODO: add branch prediction and function heat

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace policies
{
    ///////////////////////////////////////////////////////////////////////////
    /// The global_queue_scheduler maintains exactly one global queue of work
    /// items (threads), where all OS threads pull their next work item from.
    class global_queue_scheduler
    {
    private:
        // The maximum number of active threads this thread manager should
        // create. This number will be a constraint only as long as the work
        // items queue is not empty. Otherwise the number of active threads
        // will be incremented in steps equal to the \a min_add_new_count
        // specified above.
        enum { max_thread_count = 1000 };

    public:
        typedef boost::mpl::false_ has_periodic_maintenance;
        // the scheduler type takes one initialization parameter: the maxcount
        typedef std::size_t init_parameter_type;

        global_queue_scheduler(init_parameter_type max_count = 0)
          : queue_((0 == max_count)
                  ? static_cast<init_parameter_type>(max_thread_count)
                  : max_count)
        {}

        bool numa_sensitive() const { return false; }

        std::size_t get_pu_mask(topology const& topology, std::size_t num_thread) const
        {
            return topology.get_thread_affinity_mask(num_thread, false);
        }

        std::size_t get_pu_num(std::size_t num_thread) const
        {
            return num_thread;
        }

        std::size_t get_num_stolen_threads(bool reset) const
        {
            return 0;
        }

        ///////////////////////////////////////////////////////////////////////
        // This returns the current length of the queues (work items and new items)
        boost::int64_t get_queue_length(std::size_t num_thread = std::size_t(-1)) const
        {
            return queue_.get_queue_length();
        }

        ///////////////////////////////////////////////////////////////////////
        boost::int64_t get_thread_count(thread_state_enum state = unknown,
            thread_priority priority = thread_priority_default,
            std::size_t num_thread = std::size_t(-1), bool reset = false) const
        {
            return queue_.get_thread_count(state);
        }

        ///////////////////////////////////////////////////////////////////////
        void abort_all_suspended_threads()
        {
            return queue_.abort_all_suspended_threads(0);
        }

#if HPX_THREAD_MAINTAIN_QUEUE_WAITTIME
        ///////////////////////////////////////////////////////////////////////
        boost::int64_t get_average_thread_wait_time(
            std::size_t num_thread = std::size_t(-1)) const
        {
            return queue_.get_average_thread_wait_time();
        }

        boost::int64_t get_average_task_wait_time(
            std::size_t num_thread = std::size_t(-1)) const
        {
            return queue_.get_average_task_wait_time();
        }
#endif

        ///////////////////////////////////////////////////////////////////////
        bool cleanup_terminated(bool delete_all = false)
        {
            return queue_.cleanup_terminated(delete_all);
        }

        ///////////////////////////////////////////////////////////////////////
        // create a new thread and schedule it if the initial state is equal to
        // pending
        thread_id_type create_thread(thread_init_data& data,
            thread_state_enum initial_state, bool run_now, error_code& ec,
            std::size_t num_thread)
        {
            return queue_.create_thread(data, initial_state, run_now, num_thread, ec);
        }

        /// Return the next thread to be executed, return false if non is
        /// available
        bool get_next_thread(std::size_t num_thread, bool running,
            boost::int64_t& idle_loop_count, threads::thread_data*& thrd)
        {
            return queue_.get_next_thread(thrd, num_thread);
        }

        /// Schedule the passed thread
        void schedule_thread(threads::thread_data* thrd, std::size_t num_thread,
            thread_priority /*priority*/ = thread_priority_normal)
        {
            queue_.schedule_thread(thrd, num_thread);
        }

        void schedule_thread_last(threads::thread_data* thrd, std::size_t num_thread,
            thread_priority priority = thread_priority_normal)
        {
            schedule_thread(thrd, num_thread, priority);
        }

        /// Destroy the passed thread as it has been terminated
        bool destroy_thread(threads::thread_data* thrd, boost::int64_t& busy_count)
        {
            return queue_.destroy_thread(thrd, busy_count);
        }

        /// This is a function which gets called periodically by the thread
        /// manager to allow for maintenance tasks to be executed in the
        /// scheduler. Returns true if the OS thread calling this function
        /// has to be terminated (i.e. no more work has to be done).
        bool wait_or_add_new(std::size_t num_thread, bool running,
            boost::int64_t& idle_loop_count)
        {
            std::size_t added = 0;
            return queue_.wait_or_add_new(num_thread, running, idle_loop_count, added);
        }

        /// This function gets called by the threadmanager whenever new work
        /// has been added, allowing the scheduler to reactivate one or more of
        /// possibly idling OS threads
        void do_some_work(std::size_t num_thread = std::size_t(-1))
        {
            queue_.do_some_work();
        }

        ///////////////////////////////////////////////////////////////////////
        void on_start_thread(std::size_t num_thread)
        {
            queue_.on_start_thread(num_thread);
        }
        void on_stop_thread(std::size_t num_thread)
        {
            queue_.on_stop_thread(num_thread);
        }
        void on_error(std::size_t num_thread, boost::exception_ptr const& e)
        {
            queue_.on_error(num_thread, e);
        }

    private:
        global_thread_queue queue_;                ///< this manages all the threads
    };
}}}

#include <hpx/config/warnings_suffix.hpp>

#endif
