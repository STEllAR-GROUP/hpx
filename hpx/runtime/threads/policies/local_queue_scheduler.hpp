//  Copyright (c) 2007-2013 Hartmut Kaiser
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
#include <hpx/util/get_and_reset_value.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/runtime/threads/policies/thread_queue.hpp>
#include <hpx/runtime/threads/policies/affinity_data.hpp>
#include <hpx/runtime/threads/policies/scheduler_base.hpp>

#include <boost/noncopyable.hpp>
#include <boost/atomic.hpp>
#include <boost/mpl/bool.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace policies
{
    ///////////////////////////////////////////////////////////////////////////
    /// The local_queue_scheduler maintains exactly one queue of work items
    /// (threads) per OS thread, where this OS thread pulls its next work from.
    template <typename Mutex>
    class local_queue_scheduler : public scheduler_base
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
                pu_offset_(0),
                pu_step_(1),
                numa_sensitive_(false),
                affinity_domain_("pu"),
                affinity_desc_()
            {}

            init_parameter(std::size_t num_queues,
                    std::size_t max_queue_thread_count = max_thread_count,
                    bool numa_sensitive = false,
                    std::size_t pu_offset = 0,
                    std::size_t pu_step = 1,
                    std::string const& affinity = "pu",
                    std::string const& affinity_desc = "")
              : num_queues_(num_queues),
                max_queue_thread_count_(max_queue_thread_count),
                pu_offset_(pu_offset), pu_step_(pu_step),
                numa_sensitive_(numa_sensitive),
                affinity_domain_(affinity),
                affinity_desc_(affinity_desc)
            {}

            std::size_t num_queues_;
            std::size_t max_queue_thread_count_;
            std::size_t pu_offset_;
            std::size_t pu_step_;
            bool numa_sensitive_;
            std::string affinity_domain_;
            std::string affinity_desc_;
        };
        typedef init_parameter init_parameter_type;

        local_queue_scheduler(init_parameter_type const& init)
          : queues_(init.num_queues_),
            curr_queue_(0),
            affinity_data_(init.num_queues_, init.pu_offset_, init.pu_step_,
                init.affinity_domain_, init.affinity_desc_),
            numa_sensitive_(init.numa_sensitive_),
            topology_(get_topology())
        {
            BOOST_ASSERT(init.num_queues_ != 0);
            for (std::size_t i = 0; i < init.num_queues_; ++i)
                queues_[i] = new thread_queue<Mutex>(init.max_queue_thread_count_);
        }

        ~local_queue_scheduler()
        {
            for (std::size_t i = 0; i < queues_.size(); ++i)
                delete queues_[i];
        }

        bool numa_sensitive() const { return numa_sensitive_; }

        threads::mask_cref_type get_pu_mask(topology const& topology, std::size_t num_thread) const
        {
            return affinity_data_.get_pu_mask(topology, num_thread, numa_sensitive_);
        }

        std::size_t get_pu_num(std::size_t num_thread) const
        {
            return affinity_data_.get_pu_num(num_thread);
        }

        std::size_t get_num_stolen_threads(std::size_t num_thread, bool reset)
        {
            std::size_t num_stolen_threads = 0;
            if (num_thread == std::size_t(-1)) 
            {
                for (std::size_t i = 0; i < queues_.size(); ++i)
                    num_stolen_threads += queues_[i]->get_num_stolen_threads(reset);
                return num_stolen_threads;
            }

            return queues_[num_thread]->get_num_stolen_threads(reset);
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
            thread_priority priority = thread_priority_default,
            std::size_t num_thread = std::size_t(-1), bool reset = false) const
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

#if HPX_THREAD_MAINTAIN_QUEUE_WAITTIME
        ///////////////////////////////////////////////////////////////////////
        boost::int64_t get_average_thread_wait_time(
            std::size_t num_thread = std::size_t(-1)) const
        {
            // Return average thread wait time of one specific queue.
            if (std::size_t(-1) != num_thread)
            {
                BOOST_ASSERT(num_thread < queues_.size());
                return queues_[num_thread]->get_average_thread_wait_time();
            }

            // Return the cumulative average thread wait time for all queues.
            boost::int64_t wait_time = 0;
            for (std::size_t i = 0; i < queues_.size(); ++i)
                wait_time += queues_[i]->get_average_thread_wait_time();

            return wait_time / queues_.size();
        }

        boost::int64_t get_average_task_wait_time(
            std::size_t num_thread = std::size_t(-1)) const
        {
            // Return average task wait time of one specific queue.
            if (std::size_t(-1) != num_thread)
            {
                BOOST_ASSERT(num_thread < queues_.size());
                return queues_[num_thread]->get_average_task_wait_time();
            }

            // Return the cumulative average task wait time for all queues.
            boost::int64_t wait_time = 0;
            for (std::size_t i = 0; i < queues_.size(); ++i)
                wait_time += queues_[i]->get_average_task_wait_time();

            return wait_time / queues_.size();
        }
#endif

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
            std::size_t queue_size = queues_.size();
            if (std::size_t(-1) == num_thread)
                num_thread = ++curr_queue_ % queue_size;
            if (num_thread >= queue_size)
                num_thread %= queue_size;

            BOOST_ASSERT(num_thread < queue_size);
            return queues_[num_thread]->create_thread(data, initial_state,
                run_now, num_thread, ec);
        }

        /// Return the next thread to be executed, return false if none is
        /// available
        bool get_next_thread(std::size_t num_thread, bool running,
            boost::int64_t& idle_loop_count, threads::thread_data_base*& thrd)
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
                {
                    queues_[idx]->increment_num_stolen_threads();
                    return true;
                }
            }
            return false;
        }

        /// Schedule the passed thread
        void schedule_thread(threads::thread_data_base* thrd, std::size_t num_thread,
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

        void schedule_thread_last(threads::thread_data_base* thrd, std::size_t num_thread,
            thread_priority priority = thread_priority_normal)
        {
            local_queue_scheduler::schedule_thread(thrd, num_thread, priority);
        }

        /// Destroy the passed thread as it has been terminated
        bool destroy_thread(threads::thread_data_base* thrd, boost::int64_t& busy_count)
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
                // same NUMA node
                mask_cref_type node_mask = topology_.get_numa_node_affinity_mask(
                    num_thread, numa_sensitive_);

#if !defined(HPX_NATIVE_MIC)        // we know that the MIC has one NUMA domain only
                mask_cref_type core_mask = topology_.get_thread_affinity_mask(
                    num_thread, numa_sensitive_);
                if (any(core_mask) && any(node_mask))
#endif
                {
                    for (std::size_t i = 0; (0 == added) && i < queues_.size();
                         ++i)
                    {
                        if (i == num_thread || !test(node_mask, i))
                            continue;         // don't steal from ourselves

                        result = queues_[num_thread]->wait_or_add_new(i,
                            running, idle_loop_count, added, queues_[i]) && result;
                        if (0 != added)
                        {
                            queues_[num_thread]->increment_num_stolen_threads(added);
                            return result;
                        }
                    }
                }

#if !defined(HPX_NATIVE_MIC)        // we know that the MIC has one NUMA domain only
                // if nothing found ask everybody else
                for (std::size_t i = 1; 0 == added && i < queues_.size(); ++i) {
                    std::size_t idx = (i + num_thread) % queues_.size();
                    result = queues_[num_thread]->wait_or_add_new(idx, running,
                        idle_loop_count, added, queues_[idx]) && result;
                    if (0 != added)
                    {
                        queues_[num_thread]->increment_num_stolen_threads(added);
                        return result;
                    }
                }
#endif

#if HPX_THREAD_MINIMAL_DEADLOCK_DETECTION
                // no new work is available, are we deadlocked?
                if (HPX_UNLIKELY(0 == added && minimal_deadlock_detection && LHPX_ENABLED(error))) {
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
                            LHPX_CONSOLE_(hpx::util::logging::level::error) << "  [TM] "
                                  << "queue(" << num_thread << "): "
                                  << "no new work available, are we deadlocked?\n";
                        }
                    }
                }
#endif
            }
            return result && 0 == added;
        }

        /// This function gets called by the thread-manager whenever new work
        /// has been added, allowing the scheduler to reactivate one or more of
        /// possibly idling OS threads
        void do_some_work(std::size_t num_thread = std::size_t(-1))
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
        void add_punit(std::size_t virt_core, std::size_t thread_num)
        {
            affinity_data_.add_punit(virt_core, thread_num);
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
        std::vector<thread_queue<Mutex>*> queues_;   ///< this manages all the HPX threads
        boost::atomic<std::size_t> curr_queue_;
        detail::affinity_data affinity_data_;
        bool numa_sensitive_;
        topology const& topology_;
    };
}}}

#include <hpx/config/warnings_suffix.hpp>

#endif
