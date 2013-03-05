//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADMANAGER_SCHEDULING_ABP_PRIOTITY_QUEUE_DEC_22_2011_0807PM)
#define HPX_THREADMANAGER_SCHEDULING_ABP_PRIOTITY_QUEUE_DEC_22_2011_0807PM

#include <vector>
#include <memory>

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/get_and_reset_value.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/runtime/threads/policies/thread_deque.hpp>

#include <boost/noncopyable.hpp>
#include <boost/atomic.hpp>
#include <boost/mpl/bool.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace policies
{
    class abp_priority_queue_scheduler : boost::noncopyable
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

        abp_priority_queue_scheduler(init_parameter_type const& init)
          : queues_(init.num_queues_),
            high_priority_queues_(init.num_high_priority_queues_),
            low_priority_queue_(init.max_queue_thread_count_),
            curr_queue_(0),
            numa_sensitive_(init.numa_sensitive_),
            topology_(get_topology()),
            stolen_threads_(0)
        {
            BOOST_ASSERT(init.num_queues_ != 0);
            for (std::size_t i = 0; i < init.num_queues_; ++i)
                queues_[i] = new thread_deque(init.max_queue_thread_count_);

            BOOST_ASSERT(init.num_high_priority_queues_ != 0);
            BOOST_ASSERT(init.num_high_priority_queues_ <= init.num_queues_);
            for (std::size_t i = 0; i < init.num_high_priority_queues_; ++i) {
                high_priority_queues_[i] =
                    new thread_deque(init.max_queue_thread_count_);
            }
        }

        ~abp_priority_queue_scheduler()
        {
            for (std::size_t i = 0; i < queues_.size(); ++i)
                delete queues_[i];
            for (std::size_t i = 0; i < high_priority_queues_.size(); ++i)
                delete high_priority_queues_[i];
        }

        bool numa_sensitive() const { return numa_sensitive_; }

        std::size_t get_pu_mask(topology const& topology, std::size_t num_thread) const
        {
            return topology.get_thread_affinity_mask(num_thread, numa_sensitive_);
        }

        std::size_t get_pu_num(std::size_t num_thread) const
        {
            return num_thread;
        }

        std::size_t get_num_stolen_threads(std::size_t num_thread, bool reset)
        {
            if (num_thread != std::size_t(-1))
                return 0;
            return util::get_and_reset_value(stolen_threads_, reset);
        }

        ///////////////////////////////////////////////////////////////////////////
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

        ///////////////////////////////////////////////////////////////////////////
        bool cleanup_terminated(bool delete_all = false)
        {
            bool empty = true;
            for (std::size_t i = 0; i < queues_.size(); ++i)
                empty = queues_[i]->cleanup_terminated(delete_all) && empty;

            for (std::size_t i = 0; i < high_priority_queues_.size(); ++i)
                empty = high_priority_queues_[i]->cleanup_terminated(delete_all) && empty;

            empty = low_priority_queue_.cleanup_terminated(delete_all) && empty;
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
                boost::uint64_t mask = 0;
#if HPX_THREAD_MAINTAIN_TARGET_ADDRESS
                mask = topology_.get_thread_affinity_mask_from_lva(data.lva);
#endif
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
                BOOST_ASSERT(run_now );
                std::size_t num = num_thread % high_priority_queues_.size();
                return high_priority_queues_[num]->create_thread(data,
                    initial_state, run_now, ec);
            }
            else if (data.priority == thread_priority_low) {
                return low_priority_queue_.create_thread(data, initial_state,
                    run_now, ec);
            }

            BOOST_ASSERT(num_thread < queues_.size());
            return queues_[num_thread]->create_thread(data, initial_state,
                run_now, ec);
        }

        ///////////////////////////////////////////////////////////////////////////
        // Return the next thread to be executed, return false if none is available
        // TODO: shouldn't we try to fill the queue before stealing?
        bool get_next_thread(std::size_t num_thread, bool running,
            boost::int64_t& idle_loop_count, threads::thread_data*& thrd)
        {
            // master thread only: first try to get a priority thread
            std::size_t high_priority_queue_size = high_priority_queues_.size();
            std::size_t queue_size = queues_.size();
            if (num_thread < high_priority_queue_size)
            {
                bool result = high_priority_queues_[num_thread]->
                    get_next_thread(thrd);
                if (result) return true;
            }

            // try to get the next thread from our own queue
            BOOST_ASSERT(num_thread < queue_size);
            if (queues_[num_thread]->get_next_thread(thrd))
                return true;

            // try to execute low priority work if no other work is available
            if (queue_size-1 == num_thread)
            {
                if (low_priority_queue_.steal_next_thread(thrd))
                    return true;
            }

            // steal thread from other queue, first try high priority queues,
            // then normal ones
            for (std::size_t i = 1; i < high_priority_queue_size; ++i) {
                std::size_t idx = (i + num_thread) % high_priority_queue_size;
                if (high_priority_queues_[idx]->steal_next_thread(thrd))
                {
                    ++stolen_threads_;
                    return true;
                }
            }

            // steal thread from other queue
            for (std::size_t i = 1; i < queue_size; ++i) {
                std::size_t idx = (i + num_thread) % queue_size;
                if (queues_[idx]->steal_next_thread(thrd))
                {
                    ++stolen_threads_;
                    return true;
                }
            }
            return false;
        }

        /// Schedule the passed thread
        void schedule_thread(threads::thread_data* thrd, std::size_t num_thread,
            thread_priority priority = thread_priority_normal)
        {
            if (std::size_t(-1) == num_thread)
                num_thread = ++curr_queue_ % queues_.size();

            if (priority == thread_priority_critical) {
                std::size_t num = num_thread % high_priority_queues_.size();
                high_priority_queues_[num]->schedule_thread(thrd);
            }
            else if (priority == thread_priority_low) {
                low_priority_queue_.schedule_thread(thrd);
            }
            else {
                BOOST_ASSERT(num_thread < queues_.size());
                queues_[num_thread]->schedule_thread(thrd);
            }
        }

        void schedule_thread_last(threads::thread_data* thrd, std::size_t num_thread,
            thread_priority priority = thread_priority_normal)
        {
            if (std::size_t(-1) == num_thread)
                num_thread = ++curr_queue_ % queues_.size();

            if (priority == thread_priority_critical) {
                std::size_t num = num_thread % high_priority_queues_.size();
                high_priority_queues_[num]->schedule_thread_last(thrd);
            }
            else if (priority == thread_priority_low) {
                low_priority_queue_.schedule_thread_last(thrd);
            }
            else {
                BOOST_ASSERT(num_thread < queues_.size());
                queues_[num_thread]->schedule_thread_last(thrd);
            }
        }

        /// Destroy the passed thread as it has been terminated
        bool destroy_thread(threads::thread_data* thrd, boost::int64_t& busy_count)
        {
            for (std::size_t i = 0; i < high_priority_queues_.size(); ++i)
            {
                if (high_priority_queues_[i]->destroy_thread(thrd, busy_count))
                    return true;
            }

            if (low_priority_queue_.destroy_thread(thrd, busy_count))
                return true;

            for (std::size_t i = 0; i < queues_.size(); ++i)
            {
                if (queues_[i]->destroy_thread(thrd, busy_count))
                    return true;
            }
            return false;
        }

        ///////////////////////////////////////////////////////////////////////
        // Queries the current length of the queues (work items and new items).
        boost::int64_t get_queue_length(std::size_t num_thread = std::size_t(-1)) const
        {
            // Return queue length of one specific queue.
            boost::int64_t count = 0;
            if (std::size_t(-1) != num_thread)
            {
                BOOST_ASSERT(num_thread < queues_.size());

                if (num_thread < high_priority_queues_.size())
                    count = high_priority_queues_[num_thread]->get_queue_length();

                if (num_thread == queues_.size()-1)
                    count += low_priority_queue_.get_queue_length();

                return count + queues_[num_thread]->get_queue_length();
            }

            // Cumulative queue lengths of all queues.
            for (std::size_t i = 0; i < high_priority_queues_.size(); ++i)
                count += high_priority_queues_[i]->get_queue_length();

            count += low_priority_queue_.get_queue_length();

            for (std::size_t i = 0; i < queues_.size(); ++i)
                count += queues_[i]->get_queue_length();

            return count;
        }

        ///////////////////////////////////////////////////////////////////////
        // Queries the current thread count of the queues.
        boost::int64_t get_thread_count(thread_state_enum state = unknown,
            thread_priority priority = thread_priority_default,
            std::size_t num_thread = std::size_t(-1), bool reset = false) const
        {
            // Return thread count of one specific queue.
            boost::int64_t count = 0;
            if (std::size_t(-1) != num_thread)
            {
                BOOST_ASSERT(num_thread < queues_.size());

                switch (priority) {
                case thread_priority_default:
                    {
                        if (num_thread < high_priority_queues_.size())
                            count = high_priority_queues_[num_thread]->get_thread_count(state);

                        if (queues_.size()-1 == num_thread)
                            count += low_priority_queue_.get_thread_count(state);

                        return count + queues_[num_thread]->get_thread_count(state);
                    }

                case thread_priority_low:
                    {
                        if (queues_.size()-1 == num_thread)
                            return low_priority_queue_.get_thread_count(state);
                        break;
                    }

                case thread_priority_normal:
                    return queues_[num_thread]->get_thread_count(state);

                case thread_priority_critical:
                    {
                        if (num_thread < high_priority_queues_.size())
                            return high_priority_queues_[num_thread]->get_thread_count(state);
                        break;
                    }

                default:
                case thread_priority_unknown:
                    {
                        HPX_THROW_EXCEPTION(bad_parameter,
                            "abp_priority_queue_scheduler::get_thread_count",
                            "unknown thread priority value (thread_priority_unknown)");
                        return 0;
                    }
                }
                return 0;
            }

            switch (priority) {
            case thread_priority_default:
                {
                    // Return the cumulative count for all queues.
                    for (std::size_t i = 0; i < high_priority_queues_.size(); ++i)
                        count += high_priority_queues_[i]->get_thread_count(state);

                    count += low_priority_queue_.get_thread_count(state);

                    for (std::size_t i = 0; i < queues_.size(); ++i)
                        count += queues_[i]->get_thread_count(state);
                }

            case thread_priority_low:
                return low_priority_queue_.get_thread_count(state);

            case thread_priority_normal:
                {
                    for (std::size_t i = 0; i < queues_.size(); ++i)
                        count += queues_[i]->get_thread_count(state);
                    break;
                }

            case thread_priority_critical:
                {
                    for (std::size_t i = 0; i < high_priority_queues_.size(); ++i)
                        count += high_priority_queues_[i]->get_thread_count(state);
                    break;
                }

            default:
            case thread_priority_unknown:
                {
                    HPX_THROW_EXCEPTION(bad_parameter,
                        "abp_priority_queue_scheduler::get_thread_count",
                        "unknown thread priority value (thread_priority_unknown)");
                    return 0;
                }
            }
            return count;
        }

#if HPX_THREAD_MAINTAIN_QUEUE_WAITTIME
        ///////////////////////////////////////////////////////////////////////
        boost::int64_t get_average_thread_wait_time(
            std::size_t num_thread = std::size_t(-1)) const
        {
            // Return average thread wait time of one specific queue.
            boost::uint64_t wait_time = 0;
            boost::uint64_t count = 0;
            if (std::size_t(-1) != num_thread)
            {
                BOOST_ASSERT(num_thread < queues_.size());

                if (num_thread < high_priority_queues_.size())
                {
                    wait_time = high_priority_queues_[num_thread]->
                        get_average_thread_wait_time();
                    ++count;
                }

                if (queues_.size()-1 == num_thread)
                {
                    wait_time += low_priority_queue_.
                        get_average_thread_wait_time();
                    ++count;
                }

                wait_time += queues_[num_thread]->get_average_thread_wait_time();
                return wait_time / (count + 1);
            }

            // Return the cumulative average thread wait time for all queues.
            for (std::size_t i = 0; i < high_priority_queues_.size(); ++i)
            {
                wait_time += high_priority_queues_[i]->get_average_thread_wait_time();
                ++count;
            }

            wait_time += low_priority_queue_.get_average_thread_wait_time();

            for (std::size_t i = 0; i < queues_.size(); ++i)
            {
                wait_time += queues_[i]->get_average_thread_wait_time();
                ++count;
            }

            return wait_time / (count + 1);
        }

        boost::int64_t get_average_task_wait_time(
            std::size_t num_thread = std::size_t(-1)) const
        {
            // Return average task wait time of one specific queue.
            boost::uint64_t wait_time = 0;
            boost::uint64_t count = 0;
            if (std::size_t(-1) != num_thread)
            {
                BOOST_ASSERT(num_thread < queues_.size());

                if (num_thread < high_priority_queues_.size())
                {
                    wait_time = high_priority_queues_[num_thread]->
                        get_average_task_wait_time();
                    ++count;
                }

                if (queues_.size()-1 == num_thread)
                {
                    wait_time += low_priority_queue_.
                        get_average_task_wait_time();
                    ++count;
                }

                wait_time += queues_[num_thread]->get_average_task_wait_time();
                return wait_time / (count + 1);
            }

            // Return the cumulative average task wait time for all queues.
            for (std::size_t i = 0; i < high_priority_queues_.size(); ++i)
            {
                wait_time += high_priority_queues_[i]->
                    get_average_task_wait_time();
                ++count;
            }

            wait_time += low_priority_queue_.get_average_task_wait_time();

            for (std::size_t i = 0; i < queues_.size(); ++i)
            {
                wait_time += queues_[i]->get_average_task_wait_time();
                ++count;
            }

            return wait_time / (count + 1);
        }
#endif

        // Note that this is more like terminate_or_add_new
        bool wait_or_add_new(std::size_t num_thread, bool running,
                             boost::int64_t& idle_loop_count)
        {
            BOOST_ASSERT(num_thread < queues_.size());

            std::size_t added = 0;
            bool result = queues_[num_thread]->add_new_or_terminate(
                num_thread, running, added);
            if (0 != added) return result;

            std::size_t queues_size = queues_.size();
            if (queues_size-1 == num_thread) {
                // Convert low priority tasks to threads before attempting to
                // steal from other OS thread.
                result = low_priority_queue_.add_new_or_terminate(
                    num_thread, running, added) && result;
                if (0 != added) return result;
            }

            // steal work items: first try to steal from other cores in the
            // same NUMA node
            boost::uint64_t core_mask
                = topology_.get_thread_affinity_mask(num_thread, numa_sensitive_);
            boost::uint64_t node_mask
                = topology_.get_numa_node_affinity_mask(num_thread, numa_sensitive_);

            if (core_mask && node_mask) {
                boost::uint64_t m = 0x01LL;
                for (std::size_t i = 1; i < queues_size; m <<= 1, ++i)
                {
                    if (i == num_thread || !(m & node_mask))
                        continue;         // don't steal from ourselves

                    result = queues_[num_thread]->steal_new_or_terminate(
                        i, running, added, queues_[i]) && result;
                    if (0 != added)
                    {
                        stolen_threads_ += added;
                        return result;
                    }
                }
            }

            // if nothing found ask everybody else
            for (std::size_t i = 1; i < queues_size; ++i) {
                std::size_t idx = (i + num_thread) % queues_size;
                result = queues_[num_thread]->steal_new_or_terminate(
                    idx, running, added, queues_[idx]) && result;
                if (0 != added)
                {
                    stolen_threads_ += added;
                    return result;
                }
            }

#if HPX_THREAD_MINIMAL_DEADLOCK_DETECTION
            // no new work is available, are we deadlocked?
            if (/*0 == num_thread &&*/ LHPX_ENABLED(error)) {
                bool suspended_only = true;

                for (std::size_t i = 0; suspended_only && i < queues_size; ++i) {
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
                        LHPX_CONSOLE_(hpx::util::logging::level::error) << "  [TM] "
                              << "queue(" << num_thread << "): "
                              << "no new work available, are we deadlocked?\n";
                    }
                }
            }
#endif
            return result;
        }

        // no-op for local scheduling
        void do_some_work(std::size_t num_thread) {}

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
        std::vector<thread_deque*> queues_; ///< this manages all the PX threads
        std::vector<thread_deque*> high_priority_queues_;
        thread_deque low_priority_queue_;
        boost::atomic<std::size_t> curr_queue_;
        bool numa_sensitive_;
        topology const& topology_;
        boost::atomic<std::size_t> stolen_threads_;
    };
}}}

#include <hpx/config/warnings_suffix.hpp>

#endif

