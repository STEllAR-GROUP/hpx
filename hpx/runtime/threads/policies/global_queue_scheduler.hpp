//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADMANAGER_SCHEDULING_GLOBAL_QUEUE_JUN_18_2009_1116AM)
#define HPX_THREADMANAGER_SCHEDULING_GLOBAL_QUEUE_JUN_18_2009_1116AM

#include <map>
#include <memory>

#include <hpx/config.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/block_profiler.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/threads/policies/thread_queue.hpp>

#include <boost/thread.hpp>
#include <boost/thread/condition.hpp>
#include <boost/bind.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/lockfree/fifo.hpp>
#include <boost/ptr_container/ptr_map.hpp>

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
        // the scheduler type takes one initialization parameter: the maxcount
        typedef std::size_t init_parameter_type;

        global_queue_scheduler(init_parameter_type max_count = 0)
          : queue_((0 == max_count) ? max_thread_count : max_count)
        {}

        ///////////////////////////////////////////////////////////////////////
        // This returns the current length of the queues (work items and new items)
        boost::int64_t get_queue_lengths(std::size_t num_thread = std::size_t(-1)) const
        {
            return queue_.get_queue_lengths();
        }

        ///////////////////////////////////////////////////////////////////////
        // create a new thread and schedule it if the initial state is equal to 
        // pending
        thread_id_type create_thread(thread_init_data& data, 
            thread_state initial_state, bool run_now, error_code& ec,
            std::size_t num_thread = std::size_t(-1))
        {
            return queue_.create_thread(data, initial_state, run_now, ec);
        }

        /// Return the next thread to be executed, return false if non is 
        /// available
        bool get_next_thread(std::size_t num_thread, threads::thread** thrd)
        {
            return queue_.get_next_thread(thrd);
        }

        /// Schedule the passed thread
        void schedule_thread(threads::thread* thrd, 
            std::size_t num_thread = std::size_t(-1))
        {
            queue_.schedule_thread(thrd);
        }

        /// Destroy the passed thread as it has been terminated
        void destroy_thread(threads::thread* thrd)
        {
            queue_.destroy_thread(thrd);
        }

        /// Return the number of existing threads, regardless of their state
        std::size_t get_thread_count(std::size_t num_thread) const
        {
            return queue_.get_thread_count();
        }

        /// This is a function which gets called periodically by the thread 
        /// manager to allow for maintenance tasks to be executed in the 
        /// scheduler. Returns true if the OS thread calling this function
        /// has to be terminated (i.e. no more work has to be done).
        bool wait_or_add_new(std::size_t num_thread, bool running)
        {
            return queue_.wait_or_add_new(num_thread, running);
        }

        /// This function gets called by the threadmanager whenever new work
        /// has been added, allowing the scheduler to reactivate one or more of
        /// possibly idling OS threads
        void do_some_work(std::size_t num_thread)
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
        thread_queue queue_;                ///< this manages all the threads
    };

}}}

#endif
