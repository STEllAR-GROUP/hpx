//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_shared_priority_scheduler)
#define HPX_shared_priority_scheduler

#include <hpx/config.hpp>
#include <hpx/compat/mutex.hpp>
#include <hpx/runtime/threads/policies/affinity_data.hpp>
#include <hpx/runtime/threads/policies/lockfree_queue_backends.hpp>
#include <hpx/runtime/threads/policies/scheduler_base.hpp>
#include <hpx/runtime/threads/policies/thread_queue.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/runtime/threads_fwd.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util_fwd.hpp>

#include <boost/atomic.hpp>
#include <boost/exception_ptr.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

// TODO: add branch prediction and function heat

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace policies
{
    ///////////////////////////////////////////////////////////////////////////
    /// The shared_priority_scheduler maintains exactly one queue of work
    /// items (threads) per OS thread, where this OS thread pulls its next
    /// work
    /// from. Additionally it maintains separate queues: several for high
    /// priority threads and one for low priority threads.
    /// High priority threads are executed by the first N OS threads before any
    /// other work is executed. Low priority threads are executed by the last
    /// OS thread whenever no other work is available.
    template <typename Mutex = compat::mutex,
        typename PendingQueuing = lockfree_fifo,
        typename StagedQueuing = lockfree_fifo,
        typename TerminatedQueuing = lockfree_lifo>
    class HPX_EXPORT shared_priority_scheduler : public scheduler_base
    {
    protected:
        // The maximum number of active threads this thread manager should
        // create. This number will be a constraint only as long as the work
        // items queue is not empty. Otherwise the number of active threads
        // will be incremented in steps equal to the \a min_add_new_count
        // specified above.
        // FIXME: this is specified both here, and in thread_queue.
        enum { max_thread_count = 1000 };

    public:
        typedef std::false_type has_periodic_maintenance;

        typedef thread_queue<
            Mutex, PendingQueuing, StagedQueuing, TerminatedQueuing
        > thread_queue_type;

        // the scheduler type takes two initialization parameters:
        //    the number of queues
        //    the number of high priority queues
        //    the maxcount per queue
        struct init_parameter
        {
            init_parameter()
              : num_queues_(1),
                max_queue_thread_count_(max_thread_count),
                numa_sensitive_(0),
                description_("local_priority_queue_scheduler")
            {}

            init_parameter(std::size_t num_queues,
                    std::size_t num_high_priority_queues = std::size_t(-1),
                    std::size_t max_queue_thread_count = max_thread_count,
                    std::size_t numa_sensitive = 0,
                    char const* description = "local_priority_queue_scheduler")
              : num_queues_(num_queues),
                max_queue_thread_count_(max_queue_thread_count),
                numa_sensitive_(numa_sensitive),
                description_(description)
            {}

            init_parameter(std::size_t num_queues, char const* description)
              : num_queues_(num_queues),
                max_queue_thread_count_(max_thread_count),
                numa_sensitive_(false),
                description_(description)
            {}

            std::size_t num_queues_;
            std::size_t max_queue_thread_count_;
            std::size_t numa_sensitive_;
            char const* description_;
        };
        typedef init_parameter init_parameter_type;

        shared_priority_scheduler(init_parameter_type const& init,
                bool deferred_initialization = true)
          : scheduler_base(init.num_queues_, init.description_),
            max_queue_thread_count_(init.max_queue_thread_count_),
            queues_(init.num_queues_),
            high_priority_queue_(init.max_queue_thread_count_),
            low_priority_queue_(init.max_queue_thread_count_),
            curr_queue_(0),
            numa_sensitive_(init.numa_sensitive_)
        {
            victim_threads_.clear();
            victim_threads_.resize(init.num_queues_);

            if (!deferred_initialization)
            {
                BOOST_ASSERT(init.num_queues_ != 0);
                for (std::size_t i = 0; i < init.num_queues_; ++i)
                    queues_[i] = new thread_queue_type(init.max_queue_thread_count_);
            }
        }

        virtual ~shared_priority_scheduler()
        {
            for (std::size_t i = 0; i != queues_.size(); ++i)
                delete queues_[i];
        }

        bool numa_sensitive() const { return numa_sensitive_ != 0; }

        static std::string get_scheduler_name()
        {
            return "shared_priority_scheduler";
        }

#ifdef HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES
        std::uint64_t get_creation_time(bool reset)
        {
            std::uint64_t time = 0;

            time += high_priority_queue_.get_creation_time(reset);

            time += low_priority_queue_.get_creation_time(reset);

            for (std::size_t i = 0; i != queues_.size(); ++i)
                time += queues_[i]->get_creation_time(reset);

            return time;
        }

        std::uint64_t get_cleanup_time(bool reset)
        {
            std::uint64_t time = 0;

            time += high_priority_queue_.get_cleanup_time(reset);

            time += low_priority_queue_.get_cleanup_time(reset);

            for (std::size_t i = 0; i != queues_.size(); ++i)
                time += queues_[i]->get_cleanup_time(reset);

            return time;
        }
#endif

#ifdef HPX_HAVE_THREAD_STEALING_COUNTS
        std::int64_t get_num_pending_misses(std::size_t num_thread, bool reset)
        {
            std::int64_t num_pending_misses = 0;
            if (num_thread == std::size_t(-1))
            {
                num_pending_misses += high_priority_queue_.
                    get_num_pending_misses(reset);

                for (std::size_t i = 0; i != queues_.size(); ++i)
                    num_pending_misses += queues_[i]->
                        get_num_pending_misses(reset);

                num_pending_misses += low_priority_queue_.
                    get_num_pending_misses(reset);

                return num_pending_misses;
            }

            num_pending_misses += queues_[num_thread]->
                get_num_pending_misses(reset);

            num_pending_misses += high_priority_queue_.
                get_num_pending_misses(reset);

            if (num_thread == 0)
            {
                num_pending_misses += low_priority_queue_.
                    get_num_pending_misses(reset);
            }
            return num_pending_misses;
        }

        std::int64_t get_num_pending_accesses(std::size_t num_thread, bool reset)
        {
            std::int64_t num_pending_accesses = 0;
            if (num_thread == std::size_t(-1))
            {
                num_pending_accesses += high_priority_queue_.
                    get_num_pending_accesses(reset);

                for (std::size_t i = 0; i != queues_.size(); ++i)
                    num_pending_accesses += queues_[i]->
                        get_num_pending_accesses(reset);

                num_pending_accesses += low_priority_queue_.
                    get_num_pending_accesses(reset);

                return num_pending_accesses;
            }

            num_pending_accesses += queues_[num_thread]->
                get_num_pending_accesses(reset);

            num_pending_accesses += high_priority_queue_.
                get_num_pending_accesses(reset);

            if (num_thread == 0)
            {
                num_pending_accesses += low_priority_queue_.
                    get_num_pending_accesses(reset);
            }
            return num_pending_accesses;
        }

        std::int64_t get_num_stolen_from_pending(std::size_t num_thread, bool reset)
        {
            std::int64_t num_stolen_threads = 0;
            if (num_thread == std::size_t(-1))
            {
                num_stolen_threads += high_priority_queue_.
                    get_num_stolen_from_pending(reset);

                for (std::size_t i = 0; i != queues_.size(); ++i)
                    num_stolen_threads += queues_[i]->
                        get_num_stolen_from_pending(reset);

                num_stolen_threads += low_priority_queue_.
                    get_num_stolen_from_pending(reset);

                return num_stolen_threads;
            }

            num_stolen_threads += queues_[num_thread]->
                get_num_stolen_from_pending(reset);

            num_stolen_threads += high_priority_queue_.
                get_num_stolen_from_pending(reset);

            if (num_thread == 0)
            {
                num_stolen_threads += low_priority_queue_.
                    get_num_stolen_from_pending(reset);
            }
            return num_stolen_threads;
        }

        std::int64_t get_num_stolen_to_pending(std::size_t num_thread, bool reset)
        {
            std::int64_t num_stolen_threads = 0;
            if (num_thread == std::size_t(-1))
            {
                num_stolen_threads += high_priority_queue_.
                    get_num_stolen_to_pending(reset);

                for (std::size_t i = 0; i != queues_.size(); ++i)
                    num_stolen_threads += queues_[i]->
                        get_num_stolen_to_pending(reset);

                num_stolen_threads += low_priority_queue_.
                    get_num_stolen_to_pending(reset);

                return num_stolen_threads;
            }

            num_stolen_threads += queues_[num_thread]->
                get_num_stolen_to_pending(reset);

            num_stolen_threads += high_priority_queue_.
                get_num_stolen_to_pending(reset);

            if (num_thread == 0)
            {
                num_stolen_threads += low_priority_queue_.
                    get_num_stolen_to_pending(reset);
            }
            return num_stolen_threads;
        }

        std::int64_t get_num_stolen_from_staged(std::size_t num_thread, bool reset)
        {
            std::int64_t num_stolen_threads = 0;
            if (num_thread == std::size_t(-1))
            {
                num_stolen_threads += high_priority_queue_.
                    get_num_stolen_from_staged(reset);

                for (std::size_t i = 0; i != queues_.size(); ++i)
                    num_stolen_threads += queues_[i]->
                        get_num_stolen_from_staged(reset);

                num_stolen_threads += low_priority_queue_.
                    get_num_stolen_from_staged(reset);

                return num_stolen_threads;
            }

            num_stolen_threads += queues_[num_thread]->
                get_num_stolen_from_staged(reset);

            num_stolen_threads += high_priority_queue_.
                get_num_stolen_from_staged(reset);

            if (num_thread == 0)
            {
                num_stolen_threads += low_priority_queue_.
                    get_num_stolen_from_staged(reset);
            }
            return num_stolen_threads;
        }

        std::int64_t get_num_stolen_to_staged(std::size_t num_thread, bool reset)
        {
            std::int64_t num_stolen_threads = 0;
            if (num_thread == std::size_t(-1))
            {
                num_stolen_threads += high_priority_queue_.
                    get_num_stolen_to_staged(reset);

                for (std::size_t i = 0; i != queues_.size(); ++i)
                    num_stolen_threads += queues_[i]->
                        get_num_stolen_to_staged(reset);

                num_stolen_threads += low_priority_queue_.
                    get_num_stolen_to_staged(reset);

                return num_stolen_threads;
            }

            num_stolen_threads += queues_[num_thread]->
                get_num_stolen_to_staged(reset);

            num_stolen_threads += high_priority_queue_.
                get_num_stolen_to_staged(reset);

                if (num_thread == 0)
            {
                num_stolen_threads += low_priority_queue_.
                    get_num_stolen_to_staged(reset);
            }
            return num_stolen_threads;
        }
#endif

        ///////////////////////////////////////////////////////////////////////
        void abort_all_suspended_threads()
        {
            for (std::size_t i = 0; i != queues_.size(); ++i)
                queues_[i]->abort_all_suspended_threads();

            high_priority_queue_.abort_all_suspended_threads();

            low_priority_queue_.abort_all_suspended_threads();
        }

        ///////////////////////////////////////////////////////////////////////
        bool cleanup_terminated(bool delete_all = false)
        {
            bool empty = true;
            for (std::size_t i = 0; i != queues_.size(); ++i)
                empty = queues_[i]->cleanup_terminated(delete_all) && empty;
            if (!delete_all)
                return empty;

            empty = high_priority_queue_.
                cleanup_terminated(delete_all) && empty;

            empty = low_priority_queue_.cleanup_terminated(delete_all) && empty;
            return empty;
        }

        ///////////////////////////////////////////////////////////////////////
        // create a new thread and schedule it if the initial state is equal to
        // pending
        void create_thread(thread_init_data& data, thread_id_type* id,
            thread_state_enum initial_state, bool run_now, error_code& ec,
            std::size_t num_thread)
        {
#ifdef HPX_HAVE_THREAD_TARGET_ADDRESS
//             // try to figure out the NUMA node where the data lives
//             if (numa_sensitive_ && std::size_t(-1) == num_thread) {
//                 mask_cref_type mask =
//                     topology_.get_thread_affinity_mask_from_lva(data.lva);
//                 if (any(mask)) {
//                     num_thread = find_first(mask);
//                 }
//             }
#endif
            std::size_t queue_size = queues_.size();

            if (std::size_t(-1) == num_thread)
                num_thread = curr_queue_++ % queue_size;

            if (num_thread >= queue_size)
                num_thread %= queue_size;

            // now create the thread
            if (data.priority == thread_priority_critical) {
                high_priority_queue_.create_thread(data, id,
                    initial_state, run_now, ec);
                return;
            }

            if (data.priority == thread_priority_boost) {
                data.priority = thread_priority_normal;
                std::size_t num = num_thread;
                high_priority_queue_.create_thread(data, id,
                    initial_state, run_now, ec);
                return;
            }

            if (data.priority == thread_priority_low) {
                low_priority_queue_.create_thread(data, id, initial_state,
                    run_now, ec);
                return;
            }

            HPX_ASSERT(num_thread < queue_size);
            queues_[num_thread]->create_thread(data, id, initial_state,
                run_now, ec);
        }

        /// Return the next thread to be executed, return false if none is
        /// available
        virtual bool get_next_thread(std::size_t num_thread, bool running,
            std::int64_t& idle_loop_count, threads::thread_data*& thrd)
        {
            std::size_t queues_size = queues_.size();

            HPX_ASSERT(num_thread < queues_size);
            thread_queue_type* this_queue = queues_[num_thread];

            bool result =
                high_priority_queue_.get_next_thread(thrd);

            high_priority_queue_.increment_num_pending_accesses();
            if (result)
                return true;
            high_priority_queue_.increment_num_pending_misses();

            {
                bool result = this_queue->get_next_thread(thrd);

                this_queue->increment_num_pending_accesses();
                if (result)
                    return true;
                this_queue->increment_num_pending_misses();

                bool have_staged = this_queue->
                    get_staged_queue_length(boost::memory_order_relaxed) != 0;

                // Give up, we should have work to convert.
                if (have_staged)
                    return false;
            }

            // JB - no need to steal from another HP queue as HP queue is shared anyway
            for (std::size_t idx: victim_threads_[num_thread])
            {
                HPX_ASSERT(idx != num_thread);

                if (queues_[idx]->get_next_thread(thrd, running))
                {
                    queues_[idx]->increment_num_stolen_from_pending();
                    this_queue->increment_num_stolen_to_pending();
                    return true;
                }
            }

            return low_priority_queue_.get_next_thread(thrd);
        }

        /// Schedule the passed thread
        void schedule_thread(threads::thread_data* thrd,
            std::size_t num_thread,
            thread_priority priority = thread_priority_normal)
        {
            if (std::size_t(-1) == num_thread)
                num_thread = curr_queue_++ % queues_.size();

            if (priority == thread_priority_critical ||
                priority == thread_priority_boost)
            {
                high_priority_queue_.schedule_thread(thrd);
            }
            else if (priority == thread_priority_low) {
                low_priority_queue_.schedule_thread(thrd);
            }
            else {
                HPX_ASSERT(num_thread < queues_.size());
                queues_[num_thread]->schedule_thread(thrd);
            }
        }

        void schedule_thread_last(threads::thread_data* thrd,
            std::size_t num_thread,
            thread_priority priority = thread_priority_normal)
        {
            if (std::size_t(-1) == num_thread)
                num_thread = curr_queue_++ % queues_.size();

            if (priority == thread_priority_critical ||
                priority == thread_priority_boost)
            {
                high_priority_queue_.schedule_thread(thrd, true);
            }
            else if (priority == thread_priority_low) {
                low_priority_queue_.schedule_thread(thrd, true);
            }
            else {
                HPX_ASSERT(num_thread < queues_.size());
                queues_[num_thread]->schedule_thread(thrd, true);
            }
        }

        /// Destroy the passed thread as it has been terminated
        bool destroy_thread(threads::thread_data* thrd, std::int64_t& busy_count)
        {
            if (high_priority_queue_.destroy_thread(thrd, busy_count))
                return true;

            for (std::size_t i = 0; i != queues_.size(); ++i)
            {
                if (queues_[i]->destroy_thread(thrd, busy_count))
                    return true;
            }

            if (low_priority_queue_.destroy_thread(thrd, busy_count))
                return true;

            // the thread has to belong to one of the queues, always
            HPX_ASSERT(false);

            return false;
        }

        ///////////////////////////////////////////////////////////////////////
        // This returns the current length of the queues (work items and new items)
        std::int64_t get_queue_length(std::size_t num_thread = std::size_t(-1)) const
        {
            // Return queue length of one specific queue.
            std::int64_t count = 0;
            if (std::size_t(-1) != num_thread) {
                HPX_ASSERT(num_thread < queues_.size());

                count = high_priority_queue_.get_queue_length();

                if (num_thread == queues_.size()-1)
                    count += low_priority_queue_.get_queue_length();

                return count + queues_[num_thread]->get_queue_length();
            }

            // Cumulative queue lengths of all queues.
            count += high_priority_queue_.get_queue_length();

            count += low_priority_queue_.get_queue_length();

            for (std::size_t i = 0; i != queues_.size(); ++i)
                count += queues_[i]->get_queue_length();

            return count;
        }

        ///////////////////////////////////////////////////////////////////////
        // Queries the current thread count of the queues.
        std::int64_t get_thread_count(thread_state_enum state = unknown,
            thread_priority priority = thread_priority_default,
            std::size_t num_thread = std::size_t(-1), bool reset = false) const
        {
            // Return thread count of one specific queue.
            std::int64_t count = 0;
            if (std::size_t(-1) != num_thread)
            {
                HPX_ASSERT(num_thread < queues_.size());

                switch (priority) {
                case thread_priority_default:
                    {
                        count = high_priority_queue_.
                            get_thread_count(state);

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

                case thread_priority_boost:
                case thread_priority_critical:
                    {
                        return high_priority_queue_.get_thread_count(state);
                        break;
                    }

                default:
                case thread_priority_unknown:
                    {
                        HPX_THROW_EXCEPTION(bad_parameter,
                            "shared_priority_scheduler::get_thread_count",
                            "unknown thread priority value "
                            "(thread_priority_unknown)");
                        return 0;
                    }
                }
                return 0;
            }

            // Return the cumulative count for all queues.
            switch (priority) {
            case thread_priority_default:
                {
                    count += high_priority_queue_.get_thread_count(state);

                    count += low_priority_queue_.get_thread_count(state);

                    for (std::size_t i = 0; i != queues_.size(); ++i)
                        count += queues_[i]->get_thread_count(state);

                    break;
                }

            case thread_priority_low:
                return low_priority_queue_.get_thread_count(state);

            case thread_priority_normal:
                {
                    for (std::size_t i = 0; i != queues_.size(); ++i)
                        count += queues_[i]->get_thread_count(state);
                    break;
                }

            case thread_priority_boost:
            case thread_priority_critical:
                {
                    count += high_priority_queue_.get_thread_count(state);
                    break;
                }

            default:
            case thread_priority_unknown:
                {
                    HPX_THROW_EXCEPTION(bad_parameter,
                        "shared_priority_scheduler::get_thread_count",
                        "unknown thread priority value "
                        "(thread_priority_unknown)");
                    return 0;
                }
            }
            return count;
        }

        ///////////////////////////////////////////////////////////////////////
        // Enumerate matching threads from all queues
        bool enumerate_threads(
            util::function_nonser<bool(thread_id_type)> const& f,
            thread_state_enum state = unknown) const
        {
            bool result = true;
            result = result &&
                high_priority_queue_.enumerate_threads(f, state);

            result = result && low_priority_queue_.enumerate_threads(f, state);

            for (std::size_t i = 0; i != queues_.size(); ++i)
            {
                result = result && queues_[i]->enumerate_threads(f, state);
            }
            return result;
        }

#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
        ///////////////////////////////////////////////////////////////////////
        // Queries the current average thread wait time of the queues.
        std::int64_t get_average_thread_wait_time(
            std::size_t num_thread = std::size_t(-1)) const
        {
            // Return average thread wait time of one specific queue.
            std::uint64_t wait_time = 0;
            std::uint64_t count = 0;
            if (std::size_t(-1) != num_thread)
            {
                HPX_ASSERT(num_thread < queues_.size());

                wait_time = high_priority_queue_->get_average_thread_wait_time();
                ++count;

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
            wait_time += high_priority_queue_.get_average_thread_wait_time();
            ++count;

            wait_time += low_priority_queue_.get_average_thread_wait_time();

            for (std::size_t i = 0; i != queues_.size(); ++i)
            {
                wait_time += queues_[i]->get_average_thread_wait_time();
                ++count;
            }

            return wait_time / (count + 1);
        }

        ///////////////////////////////////////////////////////////////////////
        // Queries the current average task wait time of the queues.
        std::int64_t get_average_task_wait_time(
            std::size_t num_thread = std::size_t(-1)) const
        {
            // Return average task wait time of one specific queue.
            std::uint64_t wait_time = 0;
            std::uint64_t count = 0;
            if (std::size_t(-1) != num_thread)
            {
                HPX_ASSERT(num_thread < queues_.size());

                wait_time = high_priority_queue_->get_average_task_wait_time();
                ++count;

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
            wait_time += high_priority_queue_.get_average_task_wait_time();
            ++count;

            wait_time += low_priority_queue_.get_average_task_wait_time();

            for (std::size_t i = 0; i != queues_.size(); ++i)
            {
                wait_time += queues_[i]->get_average_task_wait_time();
                ++count;
            }

            return wait_time / (count + 1);
        }
#endif

        /// This is a function which gets called periodically by the thread
        /// manager to allow for maintenance tasks to be executed in the
        /// scheduler. Returns true if the OS thread calling this function
        /// has to be terminated (i.e. no more work has to be done).
        virtual bool wait_or_add_new(std::size_t num_thread, bool running,
            std::int64_t& idle_loop_count)
        {
            std::size_t added = 0;
            bool result = true;

            thread_queue_type* this_queue = queues_[num_thread];

            result = high_priority_queue_.wait_or_add_new(running,
                idle_loop_count, added) && result;
            if (0 != added) return result;

            result = this_queue->wait_or_add_new(
                running, idle_loop_count, added) && result;
            if (0 != added) return result;

            for (std::size_t idx: victim_threads_[num_thread])
            {
                HPX_ASSERT(idx != num_thread);

                result = this_queue->wait_or_add_new(running,
                    idle_loop_count, added, queues_[idx]) && result;
                if (0 != added)
                {
                    queues_[idx]->increment_num_stolen_from_staged(added);
                    this_queue->increment_num_stolen_to_staged(added);
                    return result;
                }
            }

            result = low_priority_queue_.wait_or_add_new(running,
                idle_loop_count, added) && result;
            if (0 != added) return result;

            return result;
        }

        ///////////////////////////////////////////////////////////////////////
        void on_start_thread(std::size_t num_thread)
        {
//@TODO: JB Check this as we might create the same HP queue N times
            if (nullptr == queues_[num_thread])
            {
                queues_[num_thread] =
                    new thread_queue_type(max_queue_thread_count_);
            }

            // forward this call to all queues etc.
            if (num_thread == queues_.size()-1)
                high_priority_queue_.on_start_thread(num_thread);
            if (num_thread == queues_.size()-1)
                low_priority_queue_.on_start_thread(num_thread);

            queues_[num_thread]->on_start_thread(num_thread);

            std::size_t num_threads = queues_.size();
            // get numa domain masks of all queues...
            std::vector<mask_type> numa_masks(num_threads);
            std::vector<mask_type> core_masks(num_threads);
            for (std::size_t i = 0; i != num_threads; ++i)
            {
                std::size_t num_pu = get_resource_partitioner().get_affinity_data()->get_pu_num(i);
                numa_masks[i] =
                    get_resource_partitioner().get_topology().get_numa_node_affinity_mask(num_pu, numa_sensitive_ != 0);
                core_masks[i] =
                    get_resource_partitioner().get_topology().get_core_affinity_mask(num_pu, numa_sensitive_ != 0);
            }

            // iterate over the number of threads again to determine where to
            // steal from
            std::ptrdiff_t radius =
                static_cast<std::ptrdiff_t>((num_threads / 2.0) + 0.5);
            victim_threads_[num_thread].reserve(num_threads);
            std::size_t num_pu = get_resource_partitioner().get_affinity_data()->get_pu_num(num_thread);
            mask_cref_type pu_mask =
                get_resource_partitioner().get_topology().get_thread_affinity_mask(num_pu, numa_sensitive_ != 0);
            mask_cref_type numa_mask = numa_masks[num_thread];
            mask_cref_type core_mask = core_masks[num_thread];

            // we allow the thread on the boundary of the NUMA domain to steal
            mask_type first_mask = mask_type();
            resize(first_mask, mask_size(pu_mask));

            std::size_t first = find_first(numa_mask);
            if (first != std::size_t(-1))
                set(first_mask, first);
            else
                first_mask = pu_mask;

            auto iterate = [&](hpx::util::function_nonser<bool(std::size_t)> f)
            {
                // check our neighbors in a radial fashion (left and right
                // alternating, increasing distance each iteration)
                int i = 1;
                for (/**/; i < radius; ++i)
                {
                    std::ptrdiff_t left =
                        (static_cast<std::ptrdiff_t>(num_thread) - i) %
                            static_cast<std::ptrdiff_t>(num_threads);
                    if (left < 0)
                        left = num_threads + left;

                    if (f(std::size_t(left)))
                    {
                        victim_threads_[num_thread].push_back(
                            static_cast<std::size_t>(left));
                    }

                    std::size_t right = (num_thread + i) % num_threads;
                    if (f(right))
                    {
                        victim_threads_[num_thread].push_back(right);
                    }
                }
                if ((num_threads % 2) == 0)
                {
                    std::size_t right = (num_thread + i) % num_threads;
                    if (f(right))
                    {
                        victim_threads_[num_thread].push_back(right);
                    }
                }
            };

            // check for threads which share the same core...
            iterate(
                [&](std::size_t other_num_thread)
                {
                    return any(core_mask & core_masks[other_num_thread]);
                }
            );

            // check for threads which share the same numa domain...
            iterate(
                [&](std::size_t other_num_thread)
                {
                    return
                        !any(core_mask & core_masks[other_num_thread])
                        && any(numa_mask & numa_masks[other_num_thread]);
                }
            );

            // check for the rest and if we are numa aware
            if (numa_sensitive_ != 2 && any(first_mask & pu_mask))
            {
                iterate(
                    [&](std::size_t other_num_thread)
                    {
                        return !any(numa_mask & numa_masks[other_num_thread]);
                    }
                );
            }
        }

        void on_stop_thread(std::size_t num_thread)
        {
            high_priority_queue_.on_stop_thread(num_thread);
            if (num_thread == queues_.size()-1)
                low_priority_queue_.on_stop_thread(num_thread);

            queues_[num_thread]->on_stop_thread(num_thread);
        }

        void on_error(std::size_t num_thread, std::exception_ptr const& e)
        {
            high_priority_queue_.on_error(num_thread, e);
            if (num_thread == queues_.size()-1)
                low_priority_queue_.on_error(num_thread, e);

            queues_[num_thread]->on_error(num_thread, e);
        }

        void reset_thread_distribution()
        {
            curr_queue_.store(0);
        }

    protected:
        std::size_t max_queue_thread_count_;
        std::vector<thread_queue_type*> queues_;
        thread_queue_type high_priority_queue_;
        thread_queue_type low_priority_queue_;
        boost::atomic<std::size_t> curr_queue_;
        std::size_t numa_sensitive_;

        std::vector<std::vector<std::size_t> > victim_threads_;
    };
}}}

#include <hpx/config/warnings_suffix.hpp>

#endif

