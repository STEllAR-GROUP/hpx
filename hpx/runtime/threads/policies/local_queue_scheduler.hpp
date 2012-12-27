//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADMANAGER_SCHEDULING_LOCAL_QUEUE_AUG_25_2009_0137PM)
#define HPX_THREADMANAGER_SCHEDULING_LOCAL_QUEUE_AUG_25_2009_0137PM

#include <vector>
#include <memory>

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/block_profiler.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/runtime/threads/policies/thread_queue.hpp>

#include <boost/noncopyable.hpp>
#include <boost/atomic.hpp>
#include <boost/mpl/bool.hpp>

#include <hpx/config/warnings_prefix.hpp>

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
        typedef boost::mpl::false_ has_periodic_maintenance;
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

            init_parameter(std::pair<std::size_t, std::size_t> const& init,
                    bool numa_sensitive = false)
              : num_queues_(init.first),
                max_queue_thread_count_(init.second),
                numa_sensitive_(numa_sensitive)
            {}

            std::size_t num_queues_;
            std::size_t max_queue_thread_count_;
            bool numa_sensitive_;
        };
        typedef init_parameter init_parameter_type;

        local_queue_scheduler(init_parameter_type const& init)
          : queues_(init.num_queues_),
            curr_queue_(0),
            numa_sensitive_(init.numa_sensitive_),
            topology_(get_topology())
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

        std::size_t get_pu_num(std::size_t num_thread) const
        {
            return num_thread;
        }

        ///////////////////////////////////////////////////////////////////////
        // Queries the current length of the queues (work items and new items).
        boost::int64_t get_queue_length(std::size_t num_thread = std::size_t(-1)) const
        {
            // Return queue length of one specific queue.
            if (std::size_t(-1) != num_thread)
            {
                BOOST_ASSERT(num_thread < queues_.size());
                return queues_[num_thread]->get_queue_length();
            }

            // Cumulative queue lengths of all queues.
            boost::int64_t result = 0;
            for (std::size_t i = 0; i < queues_.size(); ++i)
                result += queues_[i]->get_queue_length();
            return result;
        }

        ///////////////////////////////////////////////////////////////////////
        // Queries the current thread count of the queues.
        boost::int64_t get_thread_count(thread_state_enum state = unknown,
            std::size_t num_thread = std::size_t(-1)) const
        {
            // Return thread count of one specific queue.
            if (std::size_t(-1) != num_thread)
            {
                BOOST_ASSERT(num_thread < queues_.size());
                return queues_[num_thread]->get_thread_count(state);
            }

            // Return the cumulative count for all queues.
            boost::int64_t result = 0;
            for (std::size_t i = 0; i < queues_.size(); ++i)
                result += queues_[i]->get_thread_count(state);
            return result;
        }

        ///////////////////////////////////////////////////////////////////////
        void abort_all_suspended_threads()
        {
            for (std::size_t i = 0; i < queues_.size(); ++i)
                queues_[i]->abort_all_suspended_threads(i);
        }

        ///////////////////////////////////////////////////////////////////////
        bool cleanup_terminated(bool delete_all = false)
        {
            bool empty = true;
            for (std::size_t i = 0; i < queues_.size(); ++i)
                empty = queues_[i]->cleanup_terminated(delete_all) && empty;
            return empty;
        }

        ///////////////////////////////////////////////////////////////////////
        // create a new thread and schedule it if the initial state is equal to
        // pending
        thread_id_type create_thread(thread_init_data& data,
            thread_state_enum initial_state, bool run_now, error_code& ec,
            std::size_t num_thread)
        {
            if (std::size_t(-1) == num_thread)
                num_thread = ++curr_queue_ % queues_.size();

            BOOST_ASSERT(num_thread < queues_.size());
            return queues_[num_thread]->create_thread(data, initial_state,
                run_now, num_thread, ec);
        }

        /// Return the next thread to be executed, return false if non is
        /// available
        bool get_next_thread(std::size_t num_thread, bool running,
            boost::int64_t& idle_loop_count, threads::thread_data*& thrd)
        {
            // first try to get the next thread from our own queue
            BOOST_ASSERT(num_thread < queues_.size());
            if (queues_[num_thread]->get_next_thread(thrd, num_thread))
                return true;

//             // no work available, try to fill our own queue
//             // this favors filling a queue over stealing other work
//             std::size_t added = 0;
//             bool result = queues_[num_thread]->wait_or_add_new(
//                 num_thread, running, idle_loop_count, added);
//
//             if (result) return false;   // terminated
//
//             // retry this queue if work has been added
//             if (added && queues_[num_thread]->get_next_thread(thrd))
//                 return true;     // more work available now

            // steal thread from other queue
            for (std::size_t i = 1; i < queues_.size(); ++i) {
                std::size_t idx = (i + num_thread) % queues_.size();
                if (queues_[idx]->get_next_thread(thrd, num_thread))
                    return true;
            }
            return false;
        }

        /// Schedule the passed thread
        void schedule_thread(threads::thread_data* thrd, std::size_t num_thread,
            thread_priority /*priority*/ = thread_priority_normal)
        {
            if (std::size_t(-1) != num_thread) {
                BOOST_ASSERT(num_thread < queues_.size());
                queues_[num_thread]->schedule_thread(thrd, num_thread);
            }
            else {
                queues_[++curr_queue_ % queues_.size()]->schedule_thread(thrd, num_thread);
            }
        }

        void schedule_thread_last(threads::thread_data* thrd, std::size_t num_thread,
            thread_priority priority = thread_priority_normal)
        {
            schedule_thread(thrd, num_thread, priority);
        }

        /// Destroy the passed thread as it has been terminated
        bool destroy_thread(threads::thread_data* thrd, boost::int64_t& busy_count)
        {
            for (std::size_t i = 0; i < queues_.size(); ++i) {
                if (queues_[i]->destroy_thread(thrd, busy_count))
                    return true;
            }
            return false;
        }

        /// This is a function which gets called periodically by the thread
        /// manager to allow for maintenance tasks to be executed in the
        /// scheduler. Returns true if the OS thread calling this function
        /// has to be terminated (i.e. no more work has to be done).
        bool wait_or_add_new(std::size_t num_thread, bool running,
            boost::int64_t& idle_loop_count)
        {
            BOOST_ASSERT(num_thread < queues_.size());

            std::size_t added = 0;
            bool result = queues_[num_thread]->wait_or_add_new(
                num_thread, running, idle_loop_count, added);

            if (0 == added) {
                // steal work items: first try to steal from other cores in the
                // same numa node
                boost::uint64_t core_mask
                    = topology_.get_thread_affinity_mask(num_thread, numa_sensitive_);
                boost::uint64_t node_mask
                    = topology_.get_numa_node_affinity_mask(num_thread, numa_sensitive_);

                if (core_mask && node_mask) {
                    boost::uint64_t m = 0x01LL;
                    for (std::size_t i = 0; (0 == added) && i < queues_.size();
                         m <<= 1, ++i)
                    {
                        if (m == core_mask || !(m & node_mask))
                            continue;         // don't steal from ourselves

                        std::size_t idx = least_significant_bit_set(m);
                        BOOST_ASSERT(idx < queues_.size());

                        result = queues_[num_thread]->wait_or_add_new(idx,
                            running, idle_loop_count, added, queues_[idx]) && result;
                    }
                }

                // if nothing found ask everybody else
                for (std::size_t i = 1; 0 == added && i < queues_.size(); ++i) {
                    std::size_t idx = (i + num_thread) % queues_.size();
                    result = queues_[num_thread]->wait_or_add_new(idx, running,
                        idle_loop_count, added, queues_[idx]) && result;
                }

#if HPX_THREAD_MINIMAL_DEADLOCK_DETECTION
                // no new work is available, are we deadlocked?
                if (HPX_UNLIKELY(0 == added /*&& 0 == num_thread*/ && LHPX_ENABLED(error))) {
                    bool suspended_only = true;

                    for (std::size_t i = 0; suspended_only && i < queues_.size(); ++i) {
                        suspended_only = queues_[i]->dump_suspended_threads(
                            i, idle_loop_count, running);
                    }

                    if (HPX_UNLIKELY(suspended_only)) {
                        if (running) {
                            LTM_(error)
                                << "queue(" << num_thread << "): "
                                << "no new work available, are we deadlocked?";
                        }
                        else {
                            LHPX_CONSOLE_(boost::logging::level::error) << "  [TM] "
                                  << "queue(" << num_thread << "): "
                                  << "no new work available, are we deadlocked?\n";
                        }
                    }
                }
#endif
            }
            return result && 0 == added;
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
        topology const& topology_;
    };
}}}

#include <hpx/config/warnings_suffix.hpp>

#endif
