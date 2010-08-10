//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADMANAGER_SCHEDULING_LOCAL_QUEUE_AUG_25_2009_0137PM)
#define HPX_THREADMANAGER_SCHEDULING_LOCAL_QUEUE_AUG_25_2009_0137PM

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
#include <boost/noncopyable.hpp>
#include <boost/atomic.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace policies
{
    ///////////////////////////////////////////////////////////////////////////
    /// The local_queue_scheduler maintains exactly one queue of work items 
    /// (threads) per os thread, where this OS thread pulls its next work from.
    class local_queue_scheduler : boost::noncopyable
    {
    private:
        // The maximum number of active threads this thread manager should
        // create. This number will be a constraint only as long as the work
        // items queue is not empty. Otherwise the number of active threads 
        // will be incremented in steps equal to the \a min_add_new_count
        // specified above.
        enum { max_thread_count = 1000 };

    public:
        // the scheduler type takes two initialization parameters: 
        //    the number of queues
        //    the maxcount per queue
        struct init_parameter
        {
            init_parameter()
              : num_queues_(1), 
                max_queue_thread_count_(max_thread_count),
                numa_sensitive_(false)
            {}

            init_parameter(std::size_t num_queues, 
                    std::size_t max_queue_thread_count = max_thread_count, 
                    bool numa_sensitive = false) 
              : num_queues_(num_queues), 
                max_queue_thread_count_(max_queue_thread_count),
                numa_sensitive_(numa_sensitive)
            {}

            init_parameter(std::pair<std::size_t, std::size_t> const& init)
              : num_queues_(init.first), 
                max_queue_thread_count_(init.second),
                numa_sensitive_(false)
            {}

            std::size_t num_queues_;
            std::size_t max_queue_thread_count_;
            bool numa_sensitive_;
        };
        typedef init_parameter init_parameter_type;

        local_queue_scheduler(init_parameter_type const& init)
          : queues_(init.num_queues_), 
            curr_queue_(0),
            numa_sensitive_(init.numa_sensitive_)
        {
            BOOST_ASSERT(init.num_queues_ != 0);
            for (std::size_t i = 0; i < init.num_queues_; ++i) 
                queues_[i] = new thread_queue<false>(init.max_queue_thread_count_);
        }

        ~local_queue_scheduler()
        {
            for (std::size_t i = 0; i < queues_.size(); ++i) 
                delete queues_[i];
        }

        bool numa_sensitive() const { return numa_sensitive_; }

        ///////////////////////////////////////////////////////////////////////
        // This returns the current length of the queues (work items and new items)
        boost::int64_t get_queue_lengths(std::size_t num_thread = std::size_t(-1)) const
        {
            // either return queue length of one specific queue
            if (std::size_t(-1) != num_thread) {
                BOOST_ASSERT(num_thread < queues_.size());
                return queues_[num_thread]->get_queue_lengths();
            }

            // or cumulative queue lengths of all queues
            boost::int64_t result = 0;
            for (std::size_t i = 0; i < queues_.size(); ++i)
                result += queues_[i]->get_queue_lengths();
            return result;
        }

        ///////////////////////////////////////////////////////////////////////
        // create a new thread and schedule it if the initial state is equal to 
        // pending
        thread_id_type create_thread(thread_init_data& data, 
            thread_state_enum initial_state, bool run_now, error_code& ec,
            std::size_t num_thread = std::size_t(-1))
        {
            if (std::size_t(-1) != num_thread) {
                BOOST_ASSERT(num_thread < queues_.size());
                return queues_[num_thread]->create_thread(data, initial_state, 
                    run_now, ec);
            }

            return queues_[++curr_queue_ % queues_.size()]->create_thread(
                data, initial_state, run_now, ec);
        }

        /// Return the next thread to be executed, return false if non is 
        /// available
        bool get_next_thread(std::size_t num_thread, threads::thread** thrd)
        {
            BOOST_ASSERT(num_thread < queues_.size());
            if (queues_[num_thread]->get_next_thread(thrd))
                return true;

            // steal thread
            for (std::size_t i = 1; i < queues_.size(); ++i) {
                std::size_t idx = (i + num_thread) % queues_.size();
                if (queues_[idx]->get_next_thread(thrd))
                    return true;
            }
            return false;
        }

        /// Schedule the passed thread
        void schedule_thread(threads::thread* thrd, 
            std::size_t num_thread = std::size_t(-1))
        {
            if (std::size_t(-1) != num_thread) {
                BOOST_ASSERT(num_thread < queues_.size());
                queues_[num_thread]->schedule_thread(thrd);
            }
            else {
                queues_[++curr_queue_ % queues_.size()]->schedule_thread(thrd);
            }
        }

        /// Destroy the passed thread as it has been terminated
        void destroy_thread(threads::thread* thrd)
        {
            for (std::size_t i = 0; i < queues_.size(); ++i)
                queues_[i]->destroy_thread(thrd);
        }

        /// Return the number of existing threads, regardless of their state
        std::size_t get_thread_count(std::size_t num_thread) const
        {
            BOOST_ASSERT(num_thread < queues_.size());
            return queues_[num_thread]->get_thread_count();
        }

        /// This is a function which gets called periodically by the thread 
        /// manager to allow for maintenance tasks to be executed in the 
        /// scheduler. Returns true if the OS thread calling this function
        /// has to be terminated (i.e. no more work has to be done).
        bool wait_or_add_new(std::size_t num_thread, bool running,
            std::size_t& idle_loop_count)
        {
            BOOST_ASSERT(num_thread < queues_.size());

            std::size_t added = 0;
            bool result = queues_[num_thread]->wait_or_add_new(
                num_thread, running, idle_loop_count, added);

            // steal work items
            for (std::size_t i = 1; 0 == added && i < queues_.size(); ++i) {
                std::size_t idx = (i + num_thread) % queues_.size();
                result = queues_[num_thread]->wait_or_add_new(
                    idx, running, idle_loop_count, added, queues_[idx]);
            }

            return result;
        }

        /// This function gets called by the threadmanager whenever new work
        /// has been added, allowing the scheduler to reactivate one or more of
        /// possibly idling OS threads
        void do_some_work(std::size_t num_thread)
        {
            if (std::size_t(-1) != num_thread) {
                BOOST_ASSERT(num_thread < queues_.size());
                queues_[num_thread]->do_some_work();
            }
            else {
                for (std::size_t i = 0; i < queues_.size(); ++i)
                    queues_[i]->do_some_work();
            }
        }

        ///////////////////////////////////////////////////////////////////////
        void on_start_thread(std::size_t num_thread) 
        {
            queues_[num_thread]->on_start_thread(num_thread);
        } 
        void on_stop_thread(std::size_t num_thread)
        {
            queues_[num_thread]->on_stop_thread(num_thread);
        }
        void on_error(std::size_t num_thread, boost::exception_ptr const& e) 
        {
            queues_[num_thread]->on_error(num_thread, e);
        }

    private:
        std::vector<thread_queue<false>*> queues_;   ///< this manages all the PX threads
        boost::atomic<std::size_t> curr_queue_;
        bool numa_sensitive_;
    };

}}}

#endif
