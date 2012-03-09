//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//  Copyright (c) 2011      Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADMANAGER_SCHEDULING_PERIODIC_PRIORITY_QUEUE_HPP)
#define HPX_THREADMANAGER_SCHEDULING_PERIODIC_PRIORITY_QUEUE_HPP

#include <vector>
#include <memory>

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/block_profiler.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/runtime/threads/policies/thread_queue.hpp>

#include <boost/noncopyable.hpp>
#include <boost/atomic.hpp>
#include <boost/mpl/bool.hpp>

#include <hpx/config/warnings_prefix.hpp>

// TODO: add branch prediction and function heat

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace policies
{
    ///////////////////////////////////////////////////////////////////////////
    /// The local_periodic_priority_scheduler maintains exactly one queue of work
    /// items (threads) per OS thread, where this OS thread pulls its next work
    /// from. Additionally it maintains separate queues: several for high
    /// priority threads and one for low priority threads.
    /// High priority threads are executed by the first N OS threads before any
    /// other work is executed. Low priority threads are executed by the last
    /// OS thread whenever no other work is available.
    class local_periodic_priority_scheduler : boost::noncopyable
    {
    private:
        // The maximum number of active threads this thread manager should
        // create. This number will be a constraint only as long as the work
        // items queue is not empty. Otherwise the number of active threads
        // will be incremented in steps equal to the \a min_add_new_count
        // specified above.
        enum {
            max_thread_count = 1000,
            min_thread_count = 10
        };

    public:

        typedef boost::mpl::true_ has_periodic_maintenance;

        // the scheduler type takes two initialization parameters:
        //    the number of queues
        //    the number of high priority queues
        //    the maxcount per queue
        struct init_parameter
        {
            init_parameter()
              : num_queues_(1),
                max_queue_thread_count_(max_thread_count),
                numa_sensitive_(false)
            {}

            init_parameter(std::size_t num_queues,
                    std::size_t num_high_priority_queues,
                    std::size_t max_queue_thread_count = max_thread_count,
                    bool numa_sensitive = false)
              : num_queues_(num_queues),
                num_high_priority_queues_(num_high_priority_queues),
                max_queue_thread_count_(max_queue_thread_count),
                numa_sensitive_(numa_sensitive)
            {}

            init_parameter(std::pair<std::size_t, std::size_t> const& init,
                    bool numa_sensitive = false)
              : num_queues_(init.first),
                num_high_priority_queues_(init.first),
                max_queue_thread_count_(init.second),
                numa_sensitive_(numa_sensitive)
            {}

            std::size_t num_queues_;
            std::size_t num_high_priority_queues_;
            std::size_t max_queue_thread_count_;
            bool numa_sensitive_;
        };
        typedef init_parameter init_parameter_type;

        typedef hpx::util::spinlock mutex_type;


        local_periodic_priority_scheduler(init_parameter_type const& init)
          : queues_(init.num_queues_),
            high_priority_queues_(init.num_high_priority_queues_),
            low_priority_queue_(init.max_queue_thread_count_),
            curr_queue_(0),
            numa_sensitive_(init.numa_sensitive_),
            topology_(get_topology())
        {
            BOOST_ASSERT(init.num_queues_ != 0);
            for (std::size_t i = 0; i < init.num_queues_; ++i)
            {
                queues_[i] = new thread_queue<false>(init.max_queue_thread_count_);
            }

            BOOST_ASSERT(init.num_high_priority_queues_ != 0);
            BOOST_ASSERT(init.num_high_priority_queues_ <= init.num_queues_);
            for (std::size_t i = 0; i < init.num_high_priority_queues_; ++i) {
                high_priority_queues_[i] =
                    new thread_queue<false>(init.max_queue_thread_count_);
            }
        }

        ~local_periodic_priority_scheduler()
        {
            for (std::size_t i = 0; i < queues_.size(); ++i)
            {
                delete queues_[i];
            }
            for (std::size_t i = 0; i < high_priority_queues_.size(); ++i)
                delete high_priority_queues_[i];
        }

        bool numa_sensitive() const { return numa_sensitive_; }

        std::size_t get_pu_num(std::size_t num_thread) const
        {
            return num_thread;
        }

        ///////////////////////////////////////////////////////////////////////
        void abort_all_suspended_threads()
        {
            for (std::size_t i = 0; i < queues_.size(); ++i)
                queues_[i]->abort_all_suspended_threads(i);

            for (std::size_t i = 0; i < high_priority_queues_.size(); ++i) {
                std::size_t queue_num = queues_.size() + i;
                high_priority_queues_[i]->abort_all_suspended_threads(queue_num);
            }

            low_priority_queue_.abort_all_suspended_threads(
                queues_.size()+high_priority_queues_.size());
        }

        ///////////////////////////////////////////////////////////////////////
        bool cleanup_terminated()
        {
            bool empty = true;
            for (std::size_t i = 0; i < queues_.size(); ++i)
                empty = queues_[i]->cleanup_terminated() && empty;

            for (std::size_t i = 0; i < high_priority_queues_.size(); ++i)
                empty = high_priority_queues_[i]->cleanup_terminated() && empty;

            empty = low_priority_queue_.cleanup_terminated() && empty;
            return empty;
        }

        ///////////////////////////////////////////////////////////////////////
        // create a new thread and schedule it if the initial state is equal to
        // pending
        thread_id_type create_thread(thread_init_data& data,
            thread_state_enum initial_state, bool run_now, error_code& ec,
            std::size_t num_thread)
        {
            // try to figure out the NUMA node where the data lives
            if (numa_sensitive_ && std::size_t(-1) == num_thread) {
                boost::uint64_t mask
                    = topology_.get_thread_affinity_mask_from_lva(data.lva);
                if (mask) {
                    std::size_t m = 0x01LL;
                    for (std::size_t i = 0; i < queues_.size(); m <<= 1, ++i)
                    {
                        if (!(m & mask))
                            continue;
                        num_thread = i;
                        break;
                    }
                }
            }
            if (std::size_t(-1) == num_thread)
                num_thread = ++curr_queue_ % queues_.size();


            // now create the thread
            if (data.priority == thread_priority_critical) {
                BOOST_ASSERT(run_now == true);
                std::size_t num = num_thread % high_priority_queues_.size();
                return high_priority_queues_[num]->create_thread(data,
                    initial_state, run_now, queues_.size(), ec);
            }
            else if (data.priority == thread_priority_low) {
                return low_priority_queue_.create_thread(data, initial_state,
                    run_now, queues_.size()+high_priority_queues_.size(), ec);
            }

            BOOST_ASSERT(num_thread < queues_.size());
            return queues_[num_thread]->create_thread(data, initial_state,
                run_now, num_thread, ec);
        }

        /// Return the next thread to be executed, return false if non is
        /// available
        bool get_next_thread(std::size_t num_thread, bool running,
            std::size_t& idle_loop_count, threads::thread*& thrd)
        {
            // master thread only: first try to get a priority thread
            if (num_thread < high_priority_queues_.size())
            {
                bool result = high_priority_queues_[num_thread]->
                    get_next_thread(thrd, queues_.size()+num_thread);
                if (result) return true;
            }

            // try to get the next thread from our own queue
            BOOST_ASSERT(num_thread < queues_.size());
            if (queues_[num_thread]->get_next_thread(thrd, num_thread))
                return true;

            // try to execute low priority work if no other work is available
            if (queues_.size()-1 == num_thread &&
                low_priority_queue_.get_next_thread(
                    thrd, queues_.size()+high_priority_queues_.size()))
            {
                return true;
            }

            // steal thread from other queue, first try high priority queues,
            // then normal ones

            for (std::size_t i = 1; i < high_priority_queues_.size(); ++i) {
                std::size_t idx = (i + num_thread) % high_priority_queues_.size();
                if (high_priority_queues_[idx]->
                        get_next_thread(thrd, queues_.size()+num_thread))
                {
                    return true;
                }
            }

            for (std::size_t i = 1; i < queues_.size(); ++i) {
                std::size_t idx = (i + num_thread) % queues_.size();
                if (queues_[idx]->get_next_thread(thrd, num_thread))
                    return true;
            }
            return false;
        }

        /// Schedule the passed thread
        void schedule_thread(threads::thread* thrd, std::size_t num_thread,
            thread_priority priority = thread_priority_normal)
        {
            if (std::size_t(-1) == num_thread)
                num_thread = ++curr_queue_ % queues_.size();

            if (priority == thread_priority_critical) {
                std::size_t num = num_thread % high_priority_queues_.size();
                high_priority_queues_[num]->schedule_thread(
                    thrd, queues_.size()+num);
            }
            else if (priority == thread_priority_low) {
                low_priority_queue_.schedule_thread(thrd,
                    queues_.size()+high_priority_queues_.size());
            }
            else {
                BOOST_ASSERT(num_thread < queues_.size());
                queues_[num_thread]->schedule_thread(thrd, num_thread);
            }
        }

        void schedule_thread_last(threads::thread* thrd, std::size_t num_thread,
            thread_priority priority = thread_priority_normal)
        {
            schedule_thread(thrd, num_thread, priority);
        }

        /// Destroy the passed thread as it has been terminated
        bool destroy_thread(threads::thread* thrd)
        {
            for (std::size_t i = 0; i < high_priority_queues_.size(); ++i)
            {
                if (high_priority_queues_[i]->destroy_thread(thrd))
                    return true;
            }

            if (low_priority_queue_.destroy_thread(thrd))
                return true;

            for (std::size_t i = 0; i < queues_.size(); ++i)
            {
                if (queues_[i]->destroy_thread(thrd))
                    return true;
            }
            return false;
        }

        ///////////////////////////////////////////////////////////////////////
        // This returns the current length of the queues (work items and new items)
        boost::int64_t get_queue_length(std::size_t num_thread = std::size_t(-1)) const
        {
            // Return queue length of one specific queue.
            if (std::size_t(-1) != num_thread) {
                BOOST_ASSERT(num_thread < queues_.size());

                boost::int64_t count = 0;

                if (num_thread < high_priority_queues_.size())
                {
                    count =  high_priority_queues_[num_thread]->get_thread_count();
                }
                if (num_thread == queues_.size()-1)
                {
                    count += low_priority_queue_.get_thread_count();
                }
                return count + queues_[num_thread]->get_queue_length();
            }

            // Cumulative queue lengths of all queues.
            boost::int64_t result = 0;
            for (std::size_t i = 0; i < high_priority_queues_.size(); ++i)
                result += high_priority_queues_[i]->get_queue_length();

            result += low_priority_queue_.get_queue_length();

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

                boost::int64_t result = 0;
                if (num_thread < high_priority_queues_.size())
                {
                    result = high_priority_queues_[num_thread]->get_thread_count(state);
                }
                if (queues_.size()-1 == num_thread)
                {
                    result += low_priority_queue_.get_thread_count(state);
                }
                return result + queues_[num_thread]->get_thread_count(state);
            }

            // Return the cumulative count for all queues.
            boost::int64_t result = 0;
            for (std::size_t i = 0; i < high_priority_queues_.size(); ++i)
                result += high_priority_queues_[i]->get_thread_count(state);

            result += low_priority_queue_.get_thread_count(state);

            for (std::size_t i = 0; i < queues_.size(); ++i)
                result += queues_[i]->get_thread_count(state);

            return result;
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

            std::size_t queues_size = queues_.size();
            if ((queues_size-1 == num_thread) && (0 == added)) {
                // Convert low priority tasks to threads before attempting to
                // steal from other OS thread.
                result = low_priority_queue_.wait_or_add_new(
                    num_thread, running, idle_loop_count, added) && result;
            }

            if (0 == added) {
                // steal work items: first try to steal from other cores in
                // the same NUMA node
                boost::uint64_t core_mask
                    = topology_.get_thread_affinity_mask(num_thread, numa_sensitive_);
                boost::uint64_t node_mask
                    = topology_.get_numa_node_affinity_mask(num_thread, numa_sensitive_);

                if (core_mask && node_mask) {
                    boost::uint64_t m = 0x01LL;
                    for (std::size_t i = 0; (0 == added) && i < queues_size;
                         m <<= 1, ++i)
                    {
                        if (m == core_mask || !(m & node_mask))
                            continue;         // don't steal from ourselves

                        result = queues_[num_thread]->wait_or_add_new(i,
                            running, idle_loop_count, added, queues_[i]) && result;
                    }
                }

                // if nothing found ask everybody else
                for (std::size_t i = 1; 0 == added && i < queues_size; ++i) {
                    std::size_t idx = (i + num_thread) % queues_size;
                    result = queues_[num_thread]->wait_or_add_new(idx, running,
                        idle_loop_count, added, queues_[idx]) && result;
                }

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
            }
            return result && 0 == added;
        }

        bool periodic_maintenance(bool running)
        {
            // periodic maintenance redistributes work and is responsible that
            // every OS-Thread has enough work

            {
                // Calculate the average ...
                boost::int64_t average_task_count = 0;
                boost::int64_t average_work_count = 0;
                for(std::size_t i = 0; i < high_priority_queues_.size(); ++i)
                {
                    average_task_count += high_priority_queues_[i]->get_task_length();
                    average_work_count += high_priority_queues_[i]->get_work_length();
                }
                average_task_count = average_task_count / high_priority_queues_.size();
                average_work_count = average_work_count / high_priority_queues_.size();

                // Remove items from queues that have more than the average
                thread_queue<false> tmp_queue;
                for(std::size_t i = 0; i < high_priority_queues_.size(); ++i)
                {
                    boost::int64_t task_items = high_priority_queues_[i]->get_task_length();
                    boost::int64_t work_items = high_priority_queues_[i]->get_work_length();
                    if(task_items > average_task_count)
                    {
                        boost::int64_t count = task_items - average_task_count;
                        tmp_queue.move_task_items_from(high_priority_queues_[i], count);
                    }
                    if(work_items > average_work_count)
                    {
                        boost::int64_t count = work_items - average_work_count;
                        tmp_queue.move_work_items_from(high_priority_queues_[i], count, i + queues_.size());
                    }
                }

                // And readd them to the queues which didn't have enough work ...
                for(std::size_t i = 0; i < high_priority_queues_.size(); ++i)
                {
                    boost::int64_t task_items = high_priority_queues_[i]->get_task_length();
                    boost::int64_t work_items = high_priority_queues_[i]->get_work_length();
                    if(task_items < average_task_count)
                    {
                        boost::int64_t count = average_task_count - task_items;
                        high_priority_queues_[i]->move_task_items_from(&tmp_queue, count);
                    }
                    if(work_items < average_work_count)
                    {
                        boost::int64_t count = average_work_count - work_items;
                        high_priority_queues_[i]->move_work_items_from(&tmp_queue, count, i + queues_.size());
                    }
                }

                // Some items might remain in the tmp_queue ... readd them round robin
                {
                    std::size_t i = 0;
                    while(tmp_queue.get_task_length())
                    {
                        high_priority_queues_[i]->move_task_items_from(&tmp_queue, 1);
                        i = (i + 1) % high_priority_queues_.size();
                    }
                }
                {
                    std::size_t i = 0;
                    while(tmp_queue.get_work_length())
                    {
                        high_priority_queues_[i]->move_work_items_from(&tmp_queue, 1, i + queues_.size());
                        i = (i + 1) % high_priority_queues_.size();
                    }
                }
            }

            {
                // Calculate the average ...
                boost::int64_t average_task_count = 0;
                boost::int64_t average_work_count = 0;
                for(std::size_t i = 0; i < queues_.size(); ++i)
                {
                    average_task_count += queues_[i]->get_task_length();
                    average_work_count += queues_[i]->get_work_length();
                }
                average_task_count = average_task_count / queues_.size();
                average_work_count = average_work_count / queues_.size();

                // Remove items from queues that have more than the average
                thread_queue<false> tmp_queue;
                for(std::size_t i = 0; i < queues_.size(); ++i)
                {
                    boost::int64_t task_items = queues_[i]->get_task_length();
                    boost::int64_t work_items = queues_[i]->get_work_length();
                    if(task_items > average_task_count)
                    {
                        boost::int64_t count = task_items - average_task_count;
                        tmp_queue.move_task_items_from(queues_[i], count);
                    }
                    if(work_items > average_work_count)
                    {
                        boost::int64_t count = work_items - average_work_count;
                        tmp_queue.move_work_items_from(queues_[i], count, i + queues_.size());
                    }
                }

                // And readd them to the queues which didn't have enough work ...
                for(std::size_t i = 0; i < queues_.size(); ++i)
                {
                    boost::int64_t task_items = queues_[i]->get_task_length();
                    boost::int64_t work_items = queues_[i]->get_work_length();
                    if(task_items < average_task_count)
                    {
                        boost::int64_t count = average_task_count - task_items;
                        queues_[i]->move_task_items_from(&tmp_queue, count);
                    }
                    if(work_items < average_work_count)
                    {
                        boost::int64_t count = average_work_count - work_items;
                        queues_[i]->move_work_items_from(&tmp_queue, count, i + queues_.size());
                    }
                }
                // Some items might remain in the tmp_queue ... readd them round robin
                {
                    std::size_t i = 0;
                    while(tmp_queue.get_task_length())
                    {
                        queues_[i]->move_task_items_from(&tmp_queue, 1);
                        i = (i + 1) % queues_.size();
                    }
                }
                {
                    std::size_t i = 0;
                    while(tmp_queue.get_work_length())
                    {
                        queues_[i]->move_work_items_from(&tmp_queue, 1, i + queues_.size());
                        i = (i + 1) % queues_.size();
                    }
                }
            }

            return true;
        }

        /// This function gets called by the thread-manager whenever new work
        /// has been added, allowing the scheduler to reactivate one or more of
        /// possibly idling OS threads
        void do_some_work(std::size_t num_thread)
        {
            if (std::size_t(-1) != num_thread) {
                BOOST_ASSERT(num_thread < queues_.size());

                if (num_thread < high_priority_queues_.size())
                    high_priority_queues_[num_thread]->do_some_work();
                if (num_thread == queues_.size()-1)
                    low_priority_queue_.do_some_work();

                queues_[num_thread]->do_some_work();
            }
            else {
                for (std::size_t i = 0; i < high_priority_queues_.size(); ++i)
                    high_priority_queues_[i]->do_some_work();
                for (std::size_t i = 0; i < queues_.size(); ++i)
                    queues_[i]->do_some_work();
                low_priority_queue_.do_some_work();
            }
        }

        ///////////////////////////////////////////////////////////////////////
        void on_start_thread(std::size_t num_thread)
        {
            if (num_thread < high_priority_queues_.size())
                high_priority_queues_[num_thread]->on_start_thread(num_thread);
            if (num_thread == queues_.size()-1)
                low_priority_queue_.on_start_thread(num_thread);

            queues_[num_thread]->on_start_thread(num_thread);
        }
        void on_stop_thread(std::size_t num_thread)
        {
            if (num_thread < high_priority_queues_.size())
                high_priority_queues_[num_thread]->on_stop_thread(num_thread);
            if (num_thread == queues_.size()-1)
                low_priority_queue_.on_stop_thread(num_thread);

            queues_[num_thread]->on_stop_thread(num_thread);
        }
        void on_error(std::size_t num_thread, boost::exception_ptr const& e)
        {
            if (num_thread < high_priority_queues_.size())
                high_priority_queues_[num_thread]->on_error(num_thread, e);
            if (num_thread == queues_.size()-1)
                low_priority_queue_.on_error(num_thread, e);

            queues_[num_thread]->on_error(num_thread, e);
        }

    private:
        std::vector<thread_queue<false>*> queues_;   ///< this manages all the PX threads
        std::vector<thread_queue<false>*> high_priority_queues_;
        thread_queue<false> low_priority_queue_;
        boost::atomic<std::size_t> curr_queue_;
        bool numa_sensitive_;
        topology const& topology_;
    };
}}}

#include <hpx/config/warnings_suffix.hpp>

#endif
