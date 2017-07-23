//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_shared_priority_scheduler)
#define HPX_shared_priority_scheduler

#include <hpx/config.hpp>
#include <hpx/compat/mutex.hpp>
#include <hpx/runtime/threads/policies/lockfree_queue_backends.hpp>
#include <hpx/runtime/threads/policies/scheduler_base.hpp>
#include <hpx/runtime/threads/policies/thread_queue.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/runtime/threads_fwd.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util_fwd.hpp>

#if !defined(HPX_MSVC)
#include <plugins/parcelport/parcelport_logging.hpp>
#endif

#include <boost/atomic.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#if !defined(HPX_MSVC)
static std::chrono::system_clock::time_point log_t_start =
    std::chrono::high_resolution_clock::now();

#define LOG_CUSTOM_WORKER(x)                                                   \
    dummy << "<CUSTOM> " << THREAD_ID << " time " << decimal(16) << nowt       \
          << ' ';                                                              \
    if (parent_pool)                                                           \
        dummy << "pool " << std::setfill(' ') << std::setw(8)                  \
              << parent_pool->get_pool_name() << " " << x << std::endl;        \
    else                                                                       \
        dummy << "pool (unset) " << x << std::endl;                            \
    std::cout << dummy.str().c_str();

#define LOG_CUSTOM_MSG(x)                                                      \
    std::stringstream dummy;                                                   \
    auto now = std::chrono::high_resolution_clock::now();                      \
    auto nowt = std::chrono::duration_cast<std::chrono::microseconds>(         \
        now - log_t_start)                                                     \
                    .count();                                                  \
    LOG_CUSTOM_WORKER(x);

#define LOG_CUSTOM_MSG2(x)                                                     \
    dummy.str(std::string());                                                  \
    LOG_CUSTOM_WORKER(x);

#define THREAD_DESC(thrd)                                                      \
    "\"" << thrd->get_description().get_description() << "\" "                 \
         << hexpointer(thrd)

#define THREAD_DESC2(data, thrd)                                               \
    "\"" << data.description.get_description() << "\" "                        \
         << hexpointer(thrd ? thrd->get() : 0)

#undef LOG_CUSTOM_MSG
#undef LOG_CUSTOM_MSG2
#define LOG_CUSTOM_MSG(x)
#define LOG_CUSTOM_MSG2(x)
#else
#define LOG_CUSTOM_MSG(x)
#define LOG_CUSTOM_MSG2(x)
#endif

#include <hpx/config/warnings_prefix.hpp>

// TODO: add branch prediction and function heat

///////////////////////////////////////////////////////////////////////////////
namespace hpx {
namespace threads {
    namespace policies {
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
            typename PendingQueuing = lockfree_abp_fifo,
            typename StagedQueuing = lockfree_fifo,
            typename TerminatedQueuing = lockfree_lifo>
        class shared_priority_scheduler : public scheduler_base
        {
        protected:
            // The maximum number of active threads this thread manager should
            // create. This number will be a constraint only as long as the work
            // items queue is not empty. Otherwise the number of active threads
            // will be incremented in steps equal to the \a min_add_new_count
            // specified above.
            // FIXME: this is specified both here, and in thread_queue.
            enum
            {
                max_thread_count = 1000
            };

        public:
            typedef std::false_type has_periodic_maintenance;

            typedef thread_queue<Mutex, PendingQueuing, StagedQueuing,
                TerminatedQueuing>
                thread_queue_type;

            shared_priority_scheduler(std::size_t num_worker_queues,
                std::size_t num_high_priority_queues_,
                char const* description,
                int max_tasks = max_thread_count)
              : scheduler_base(num_worker_queues, description)
              , max_queue_thread_count_(max_tasks)
              , queues_(num_worker_queues)
              , high_priority_queues_(num_high_priority_queues_)
              , low_priority_queue_(max_tasks)
              , curr_queue_(0)
              , curr_hp_queue_(0)
              , numa_sensitive_(false)
              , initialized(false)
            {
                LOG_CUSTOM_MSG("Constructing scheduler with num queues "
                    << decnumber(num_worker_queues));
                victim_threads_.clear();
                victim_threads_.resize(num_worker_queues);
                //
                BOOST_ASSERT(num_worker_queues != 0);
                for (std::size_t i = 0; i < num_worker_queues; ++i)
                    queues_[i] = new thread_queue_type(max_tasks);

                for (std::size_t i = 0; i < num_high_priority_queues_; ++i)
                    high_priority_queues_[i] = new thread_queue_type(max_tasks);
            }

            virtual ~shared_priority_scheduler()
            {
                LOG_CUSTOM_MSG(this->get_description()
                    << " - Deleting shared_priority_scheduler ");
                for (std::size_t i = 0; i != queues_.size(); ++i)
                {
                    LOG_CUSTOM_MSG2("Deleting queue " << i);
                    delete queues_[i];
                }
            }

            bool numa_sensitive() const
            {
                return numa_sensitive_ != 0;
            }

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
            std::int64_t get_num_pending_misses(
                std::size_t pool_queue_num, bool reset)
            {
                std::int64_t num_pending_misses = 0;
                if (pool_queue_num == std::size_t(-1))
                {
                    for (std::size_t i = 0; i != high_priority_queues_.size();
                         ++i)
                        num_pending_misses +=
                            high_priority_queues_[i]->get_num_pending_misses(
                                reset);

                    for (std::size_t i = 0; i != queues_.size(); ++i)
                        num_pending_misses +=
                            queues_[i]->get_num_pending_misses(reset);

                    num_pending_misses +=
                        low_priority_queue_.get_num_pending_misses(reset);

                    return num_pending_misses;
                }

                num_pending_misses +=
                    queues_[pool_queue_num]->get_num_pending_misses(reset);

                //            num_pending_misses += high_priority_queue_.
                //                get_num_pending_misses(reset);

                if (pool_queue_num == 0)
                {
                    num_pending_misses +=
                        low_priority_queue_.get_num_pending_misses(reset);
                }
                return num_pending_misses;
            }

            std::int64_t get_num_pending_accesses(
                std::size_t pool_queue_num, bool reset)
            {
                std::int64_t num_pending_accesses = 0;
                if (pool_queue_num == std::size_t(-1))
                {
                    //                num_pending_accesses += high_priority_queue_.
                    //                    get_num_pending_accesses(reset);

                    for (std::size_t i = 0; i != queues_.size(); ++i)
                        num_pending_accesses +=
                            queues_[i]->get_num_pending_accesses(reset);

                    num_pending_accesses +=
                        low_priority_queue_.get_num_pending_accesses(reset);

                    return num_pending_accesses;
                }

                num_pending_accesses +=
                    queues_[pool_queue_num]->get_num_pending_accesses(reset);

                //            num_pending_accesses += high_priority_queue_.
                //                get_num_pending_accesses(reset);

                if (pool_queue_num == 0)
                {
                    num_pending_accesses +=
                        low_priority_queue_.get_num_pending_accesses(reset);
                }
                return num_pending_accesses;
            }

            std::int64_t get_num_stolen_from_pending(
                std::size_t pool_queue_num, bool reset)
            {
                std::int64_t num_stolen_threads = 0;
                if (pool_queue_num == std::size_t(-1))
                {
                    //                num_stolen_threads += high_priority_queue_.
                    //                    get_num_stolen_from_pending(reset);

                    for (std::size_t i = 0; i != queues_.size(); ++i)
                        num_stolen_threads +=
                            queues_[i]->get_num_stolen_from_pending(reset);

                    num_stolen_threads +=
                        low_priority_queue_.get_num_stolen_from_pending(reset);

                    return num_stolen_threads;
                }

                num_stolen_threads +=
                    queues_[pool_queue_num]->get_num_stolen_from_pending(reset);

                //            num_stolen_threads += high_priority_queue_.
                //                get_num_stolen_from_pending(reset);

                if (pool_queue_num == 0)
                {
                    num_stolen_threads +=
                        low_priority_queue_.get_num_stolen_from_pending(reset);
                }
                return num_stolen_threads;
            }

            std::int64_t get_num_stolen_to_pending(
                std::size_t pool_queue_num, bool reset)
            {
                std::int64_t num_stolen_threads = 0;
                if (pool_queue_num == std::size_t(-1))
                {
                    //                num_stolen_threads += high_priority_queue_.
                    //                    get_num_stolen_to_pending(reset);

                    for (std::size_t i = 0; i != queues_.size(); ++i)
                        num_stolen_threads +=
                            queues_[i]->get_num_stolen_to_pending(reset);

                    num_stolen_threads +=
                        low_priority_queue_.get_num_stolen_to_pending(reset);

                    return num_stolen_threads;
                }

                num_stolen_threads +=
                    queues_[pool_queue_num]->get_num_stolen_to_pending(reset);

                //            num_stolen_threads += high_priority_queue_.
                //                get_num_stolen_to_pending(reset);

                if (pool_queue_num == 0)
                {
                    num_stolen_threads +=
                        low_priority_queue_.get_num_stolen_to_pending(reset);
                }
                return num_stolen_threads;
            }

            std::int64_t get_num_stolen_from_staged(
                std::size_t pool_queue_num, bool reset)
            {
                std::int64_t num_stolen_threads = 0;
                if (pool_queue_num == std::size_t(-1))
                {
                    //                num_stolen_threads += high_priority_queue_.
                    //                    get_num_stolen_from_staged(reset);

                    for (std::size_t i = 0; i != queues_.size(); ++i)
                        num_stolen_threads +=
                            queues_[i]->get_num_stolen_from_staged(reset);

                    num_stolen_threads +=
                        low_priority_queue_.get_num_stolen_from_staged(reset);

                    return num_stolen_threads;
                }

                num_stolen_threads +=
                    queues_[pool_queue_num]->get_num_stolen_from_staged(reset);

                //            num_stolen_threads += high_priority_queue_.
                //                get_num_stolen_from_staged(reset);

                if (pool_queue_num == 0)
                {
                    num_stolen_threads +=
                        low_priority_queue_.get_num_stolen_from_staged(reset);
                }
                return num_stolen_threads;
            }

            std::int64_t get_num_stolen_to_staged(
                std::size_t pool_queue_num, bool reset)
            {
                std::int64_t num_stolen_threads = 0;
                if (pool_queue_num == std::size_t(-1))
                {
                    //                num_stolen_threads += high_priority_queue_.
                    //                    get_num_stolen_to_staged(reset);

                    for (std::size_t i = 0; i != queues_.size(); ++i)
                        num_stolen_threads +=
                            queues_[i]->get_num_stolen_to_staged(reset);

                    num_stolen_threads +=
                        low_priority_queue_.get_num_stolen_to_staged(reset);

                    return num_stolen_threads;
                }

                num_stolen_threads +=
                    queues_[pool_queue_num]->get_num_stolen_to_staged(reset);

                //            num_stolen_threads += high_priority_queue_.
                //                get_num_stolen_to_staged(reset);

                if (pool_queue_num == 0)
                {
                    num_stolen_threads +=
                        low_priority_queue_.get_num_stolen_to_staged(reset);
                }
                return num_stolen_threads;
            }
#endif

            ///////////////////////////////////////////////////////////////////////
            void abort_all_suspended_threads()
            {
                LOG_CUSTOM_MSG("abort_all_suspended_threads");
                low_priority_queue_.abort_all_suspended_threads();

                for (std::size_t i = 0; i != queues_.size(); ++i)
                    queues_[i]->abort_all_suspended_threads();

                for (std::size_t i = 0; i != high_priority_queues_.size(); ++i)
                    high_priority_queues_[i]->abort_all_suspended_threads();
            }

            ///////////////////////////////////////////////////////////////////////
            bool cleanup_terminated(bool delete_all = false)
            {
                bool empty = true;
                for (std::size_t i = 0; i != queues_.size(); ++i)
                {
                    //LOG_CUSTOM_MSG(hexpointer(this) << "cleanup_terminated " << delete_all << " " << i);
                    empty = queues_[i]->cleanup_terminated(delete_all) && empty;
                }
                if (!delete_all)
                    return empty;

                //LOG_CUSTOM_MSG(hexpointer(this) << "high_priority_queue_ cleanup_terminated " << empty);
                for (std::size_t i = 0; i != high_priority_queues_.size(); ++i)
                {
                    empty = high_priority_queues_[i]->cleanup_terminated(
                                delete_all) &&
                        empty;
                }

                //LOG_CUSTOM_MSG2(hexpointer(this) << "low_priority_queue_ cleanup_terminated " << empty);
                empty =
                    low_priority_queue_.cleanup_terminated(delete_all) && empty;
                return empty;
            }

            inline std::size_t global_to_local_thread_num(std::size_t n)
            {
                return n - this->parent_pool->get_thread_offset();
            }

            ///////////////////////////////////////////////////////////////////////
            // create a new thread and schedule it if the initial state is equal
            // to pending
            void create_thread(thread_init_data& data, thread_id_type* thrd,
                thread_state_enum initial_state, bool run_now, error_code& ec,
                std::size_t pool_queue_num)
            {
                HPX_ASSERT(data.scheduler_base == this);

                if (pool_queue_num == std::size_t(-1))
                {
                    std::size_t t = threads::detail::thread_num_tss_
                                        .get_worker_thread_num();
                    pool_queue_num = global_to_local_thread_num(t);
                }

                // now create the thread
                if (data.priority == thread_priority_high ||
                    data.priority == thread_priority_high_recursive ||
                    data.priority == thread_priority_boost)
                {
                    // boosted threads return to normal after being queued
                    if (data.priority == thread_priority_boost)
                    {
                        data.priority = thread_priority_normal;
                    }
                    // if a non worker threads started this ...
                    if (pool_queue_num >= queues_.size())
                    {
                        pool_queue_num = curr_hp_queue_++ % queues_.size();
                    }
                    //
                    std::size_t hp_queue_num = hp_queue_lookup_[pool_queue_num];
                    high_priority_queues_[hp_queue_num]->create_thread(
                        data, thrd, initial_state, run_now, ec);
                    LOG_CUSTOM_MSG("create_thread thread_priority_high "
                        << THREAD_DESC2(data, thrd) << "hp_queue_num "
                        << hexnumber(pool_queue_num) << "scheduler "
                        << hexpointer(data.scheduler_base));
                    return;
                }

                if (data.priority == thread_priority_low)
                {
                    low_priority_queue_.create_thread(
                        data, thrd, initial_state, run_now, ec);
                    LOG_CUSTOM_MSG("create_thread thread_priority_low "
                        << THREAD_DESC2(data, thrd) << "pool_queue_num "
                        << hexnumber(pool_queue_num) << "scheduler "
                        << hexpointer(data.scheduler_base));
                    return;
                }

                // if a non worker threads started this ...
                if (pool_queue_num >= queues_.size())
                {
                    pool_queue_num = curr_queue_++ % queues_.size();
                }
                //
                queues_[pool_queue_num]->create_thread(
                    data, thrd, initial_state, run_now, ec);
                LOG_CUSTOM_MSG("create_thread thread_priority_normal "
                    << THREAD_DESC2(data, thrd) << "pool_queue_num "
                    << hexnumber(pool_queue_num) << "scheduler "
                    << hexpointer(data.scheduler_base));
            }

            /// Return the next thread to be executed, return false if none available
            virtual bool get_next_thread(std::size_t pool_queue_num,
                bool running, std::int64_t& idle_loop_count,
                threads::thread_data*& thrd)
            {
                // is there a high priority task we can take from the queue assigned to us
                std::size_t tq = hp_queue_lookup_[pool_queue_num];
                auto high_priority_queue = high_priority_queues_[tq];
                bool result = high_priority_queue->get_next_thread(thrd);
                // if, not how about an HP task from somewhere else?

                if (!result)
                    for (std::size_t i = 0; i < high_priority_queues_.size();
                         ++i)
                    {
                        if (i != tq)
                        {
                            high_priority_queue = high_priority_queues_[i];
                            if (high_priority_queue
                                    ->get_pending_queue_length() > 1)
                            {
                                result = high_priority_queue->get_next_thread(
                                    thrd, false, true);
                                if (result)
                                    break;
                            }
                        }
                    }

                // counter for access queries
                high_priority_queue->increment_num_pending_accesses();
                if (result)
                {
                    HPX_ASSERT(thrd->get_scheduler_base() == this);
                    LOG_CUSTOM_MSG("get_next_thread high priority "
                        << THREAD_DESC(thrd) << decnumber(idle_loop_count)
                        << "pool_queue_num " << decnumber(pool_queue_num));
                    return true;
                }
                // counter for high priority misses
                high_priority_queue->increment_num_pending_misses();

                // try to get a task from the queue associated with the requested thread
                std::size_t queues_size = queues_.size();
                HPX_ASSERT(pool_queue_num < queues_size);
                thread_queue_type* this_queue = queues_[pool_queue_num];
                {
                    bool result = this_queue->get_next_thread(thrd);

                    // counter for access queries
                    this_queue->increment_num_pending_accesses();
                    if (result)
                    {
                        HPX_ASSERT(thrd->get_scheduler_base() == this);
                        LOG_CUSTOM_MSG("get_next_thread normal "
                            << THREAD_DESC(thrd) << decnumber(idle_loop_count)
                            << "pool_queue_num " << decnumber(pool_queue_num));
                        return true;
                    }
                    // counter for misses
                    this_queue->increment_num_pending_misses();

                    bool have_staged = this_queue->get_staged_queue_length(
                                           boost::memory_order_relaxed) != 0;

                    // Give up, we should have work to convert.
                    if (have_staged)
                    {
                        LOG_CUSTOM_MSG("get_next_thread have_staged"
                            << "\" (unset) "
                            << "\" " << hexpointer(thrd)
                            << decnumber(idle_loop_count) << "pool_queue_num "
                            << decnumber(pool_queue_num));
                        return false;
                    }
                }

                // if we didn't get a task from the requested thread queue
                // try stealing from another
                for (std::size_t idx : victim_threads_[pool_queue_num])
                {
                    HPX_ASSERT(idx != pool_queue_num);

                    if (queues_[idx]->get_next_thread(thrd, false, true))
                    {
                        LOG_CUSTOM_MSG("get_next_thread stealing "
                            << THREAD_DESC(thrd) << "pool_queue_num "
                            << decnumber(pool_queue_num) << "victim thread "
                            << decnumber(idx));
                        queues_[idx]->increment_num_stolen_from_pending();
                        this_queue->increment_num_stolen_to_pending();
                        return true;
                    }
                }

                // if all else failed, then try the low priority queue
                auto ret = low_priority_queue_.get_next_thread(thrd);
                if (ret)
                {
                    LOG_CUSTOM_MSG("get_next_thread low priority "
                        << THREAD_DESC(thrd) << "pool_queue_num "
                        << decnumber(pool_queue_num));
                    return true;
                }
                // counter for low priority misses
                low_priority_queue_.increment_num_pending_misses();

                return ret;
            }

            /// Schedule the passed thread
            void schedule_thread(threads::thread_data* thrd,
                std::size_t pool_queue_num,
                thread_priority priority = thread_priority_normal)
            {
                HPX_ASSERT(thrd->get_scheduler_base() == this);

                if (pool_queue_num == std::size_t(-1))
                {
                    std::size_t t = threads::detail::thread_num_tss_
                                        .get_worker_thread_num();
                    pool_queue_num = global_to_local_thread_num(t);
                }

                if (priority == thread_priority_high ||
                    priority == thread_priority_high_recursive ||
                    priority == thread_priority_boost)
                {
                    std::size_t hp_queue_num =
                        curr_hp_queue_++ % high_priority_queues_.size();
                    high_priority_queues_[hp_queue_num]->schedule_thread(thrd);
                    LOG_CUSTOM_MSG("schedule_thread high priority "
                        << THREAD_DESC(thrd) << "pool_queue_num "
                        << decnumber(pool_queue_num));
                }
                else if (priority == thread_priority_low)
                {
                    low_priority_queue_.schedule_thread(thrd);
                    LOG_CUSTOM_MSG("schedule_thread low priority "
                        << THREAD_DESC(thrd) << "pool_queue_num "
                        << decnumber(pool_queue_num));
                }
                else
                {
                    pool_queue_num = curr_queue_++ % queues_.size();
                    queues_[pool_queue_num]->schedule_thread(thrd);
                }
            }

            /// Put task on the back of the queue
            void schedule_thread_last(threads::thread_data* thrd,
                std::size_t num_queue,
                thread_priority priority = thread_priority_normal)
            {
                HPX_ASSERT(thrd->get_scheduler_base() == this);

                if (priority == thread_priority_high ||
                    priority == thread_priority_high_recursive ||
                    priority == thread_priority_boost)
                {
                    LOG_CUSTOM_MSG("schedule_thread last thread_priority_high "
                        << THREAD_DESC(thrd));
                    std::size_t hp_queue_num =
                        curr_hp_queue_++ % high_priority_queues_.size();
                    high_priority_queues_[hp_queue_num]->schedule_thread(
                        thrd, true);
                }
                else if (priority == thread_priority_low)
                {
                    LOG_CUSTOM_MSG("schedule_thread last thread_priority_low "
                        << THREAD_DESC(thrd));
                    low_priority_queue_.schedule_thread(thrd, true);
                }
                else
                {
                    num_queue = curr_queue_++ % queues_.size();
                    LOG_CUSTOM_MSG(
                        "schedule_thread last thread_priority_normal "
                        << THREAD_DESC(thrd) << "num_queue "
                        << decnumber(num_queue));
                    queues_[num_queue]->schedule_thread(thrd, true);
                }
            }

            /// Destroy the passed thread - as it has been terminated
            bool destroy_thread(
                threads::thread_data* thrd, std::int64_t& busy_count)
            {
                LOG_CUSTOM_MSG("destroy thread " << THREAD_DESC(thrd)
                                                 << " busy_count "
                                                 << decnumber(busy_count));
                HPX_ASSERT(thrd->get_scheduler_base() == this);

                for (std::size_t i = 0; i != high_priority_queues_.size(); ++i)
                {
                    if (high_priority_queues_[i]->destroy_thread(
                            thrd, busy_count))
                        return true;
                }

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
            // This returns the current length of the queues (work items and new
            // items)
            std::int64_t get_queue_length(
                std::size_t pool_queue_num = std::size_t(-1)) const
            {
                LOG_CUSTOM_MSG("get_queue_length"
                    << "pool_queue_num " << decnumber(pool_queue_num));
                // Return queue length of one specific queue.
                std::int64_t count = 0;
                if (std::size_t(-1) != pool_queue_num)
                {
                    HPX_ASSERT(pool_queue_num < queues_.size());

                    auto high_priority_queue =
                        high_priority_queues_[hp_queue_lookup_[pool_queue_num]];
                    count = high_priority_queue->get_queue_length();

                    if (pool_queue_num == queues_.size() - 1)
                        count += low_priority_queue_.get_queue_length();

                    return count + queues_[pool_queue_num]->get_queue_length();
                }

                // Cumulative queue lengths of all queues.
                for (std::size_t i = 0; i != high_priority_queues_.size(); ++i)
                {
                    count += high_priority_queues_[i]->get_queue_length();
                }

                count += low_priority_queue_.get_queue_length();

                for (std::size_t i = 0; i != queues_.size(); ++i)
                    count += queues_[i]->get_queue_length();

                return count;
            }

            ///////////////////////////////////////////////////////////////////////
            // Queries the current thread count of the queues.
            std::int64_t get_thread_count(thread_state_enum state = unknown,
                thread_priority priority = thread_priority_default,
                std::size_t pool_queue_num = std::size_t(-1),
                bool reset = false) const
            {
                LOG_CUSTOM_MSG("get_thread_count pool_queue_num "
                    << hexnumber(pool_queue_num));
                // Return thread count of one specific queue.
                std::int64_t count = 0;
                if (std::size_t(-1) != pool_queue_num)
                {
                    HPX_ASSERT(pool_queue_num < queues_.size());

                    switch (priority)
                    {
                    case thread_priority_default:
                    {
                        auto high_priority_queue = high_priority_queues_
                            [hp_queue_lookup_[pool_queue_num]];
                        count = high_priority_queue->get_thread_count(state);

                        if (queues_.size() - 1 == pool_queue_num)
                            count +=
                                low_priority_queue_.get_thread_count(state);

                        return count +
                            queues_[pool_queue_num]->get_thread_count(state);
                    }

                    case thread_priority_low:
                    {
                        if (queues_.size() - 1 == pool_queue_num)
                            return low_priority_queue_.get_thread_count(state);
                        break;
                    }

                    case thread_priority_normal:
                        return queues_[pool_queue_num]->get_thread_count(state);

                    case thread_priority_boost:
                    case thread_priority_high:
                    case thread_priority_high_recursive:
                    {
                        return high_priority_queues_
                            [hp_queue_lookup_[pool_queue_num]]
                                ->get_thread_count(state);
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
                switch (priority)
                {
                case thread_priority_default:
                {
                    for (std::size_t i = 0; i != high_priority_queues_.size();
                         ++i)
                        count +=
                            high_priority_queues_[i]->get_thread_count(state);

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
                case thread_priority_high:
                case thread_priority_high_recursive:
                {
                    for (std::size_t i = 0; i != high_priority_queues_.size();
                         ++i)
                        count +=
                            high_priority_queues_[i]->get_thread_count(state);
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
                for (std::size_t i = 0; i != high_priority_queues_.size(); ++i)
                {
                    result = result &&
                        high_priority_queues_[i]->enumerate_threads(f, state);
                }

                result =
                    result && low_priority_queue_.enumerate_threads(f, state);

                for (std::size_t i = 0; i != queues_.size(); ++i)
                {
                    result = result && queues_[i]->enumerate_threads(f, state);
                }
                return result;
            }

            /// This is a function which gets called periodically by the thread
            /// manager to allow for maintenance tasks to be executed in the
            /// scheduler. Returns true if the OS thread calling this function
            /// has to be terminated (i.e. no more work has to be done).
            virtual bool wait_or_add_new(std::size_t pool_queue_num,
                bool running, std::int64_t& idle_loop_count)
            {
                std::size_t added = 0;
                bool result = true;

                auto high_priority_queue =
                    high_priority_queues_[hp_queue_lookup_[pool_queue_num]];
                result = high_priority_queue->wait_or_add_new(
                             running, idle_loop_count, added) &&
                    result;
                if (0 != added)
                {
                    //LOG_CUSTOM_MSG(hexpointer(this) << "wait_or_add_new high_priority_queue_ " << "pool_queue_num " << decnumber(pool_queue_num) );
                    return result;
                }

                thread_queue_type* this_queue = queues_[pool_queue_num];
                result = this_queue->wait_or_add_new(
                             running, idle_loop_count, added) &&
                    result;
                if (0 != added)
                {
                    //LOG_CUSTOM_MSG(hexpointer(this) << "wait_or_add_new this_queue_ " << "pool_queue_num " << decnumber(pool_queue_num) );
                    return result;
                }

                // try all the threads we might steal from
                for (std::size_t idx : victim_threads_[pool_queue_num])
                {
                    HPX_ASSERT(idx != pool_queue_num);

                    result = this_queue->wait_or_add_new(running,
                                 idle_loop_count, added, queues_[idx]) &&
                        result;
                    if (0 != added)
                    {
                        queues_[idx]->increment_num_stolen_from_staged(added);
                        this_queue->increment_num_stolen_to_staged(added);
                        //LOG_CUSTOM_MSG(hexpointer(this) << "wait_or_add_new pool_queue_num " << decnumber(pool_queue_num) << "victim " << decnumber(idx));
                        return result;
                    }
                }

                result = low_priority_queue_.wait_or_add_new(
                             running, idle_loop_count, added) &&
                    result;
                if (0 != added)
                {
                    //LOG_CUSTOM_MSG(hexpointer(this) << "wait_or_add_new low_priority_queue_ " << "pool_queue_num " << decnumber(pool_queue_num) );
                }

                return result;
            }

            ///////////////////////////////////////////////////////////////////////
            void on_start_thread(std::size_t pool_queue_num)
            {
                LOG_CUSTOM_MSG("start thread with local thread num "
                    << decnumber(pool_queue_num));
                /*
            // forward this call to all queues etc.
            if (pool_queue_num == queues_.size() - 1)
                high_priority_queue_.on_start_thread(pool_queue_num);

            if (pool_queue_num == queues_.size() - 1)
                low_priority_queue_.on_start_thread(pool_queue_num);

            queues_[pool_queue_num]->on_start_thread(pool_queue_num);
*/
                // int pool_offset = get_resource_partitioner().

                //
                // @TODO. move this to init somewhere
                //
                std::lock_guard<hpx::lcos::local::spinlock> lock(init_mutex);
                if (!initialized)
                {
                    initialized = true;
                    //
                    std::size_t num_queues = queues_.size();
                    for (std::size_t i = 0; i < num_queues; ++i)
                    {
                        victim_threads_[i].clear();
                        victim_threads_[i].reserve(num_queues - 1);
                        for (std::size_t j = 0; j < num_queues; ++j)
                        {
                            std::size_t v = (j + i) % num_queues;
                            if (i != v)
                            {
                                victim_threads_[i].push_back(v);
                            }
                        }
                    }

                    hp_queue_lookup_.resize(num_queues, 0);
                    double s = double(high_priority_queues_.size());
                    for (std::size_t i = 0; i < num_queues; ++i)
                    {
                        hp_queue_lookup_[i] =
                            static_cast<int>(i * s / num_queues);
                    }
                }
            }

            void on_stop_thread(std::size_t pool_queue_num)
            {
                //            high_priority_queue_.on_stop_thread(pool_queue_num);
                if (pool_queue_num == queues_.size() - 1)
                    low_priority_queue_.on_stop_thread(pool_queue_num);

                queues_[pool_queue_num]->on_stop_thread(pool_queue_num);
            }

            void on_error(
                std::size_t pool_queue_num, std::exception_ptr const& e)
            {
                //high_priority_queue_.on_error(pool_queue_num, e);
                if (pool_queue_num == queues_.size() - 1)
                    low_priority_queue_.on_error(pool_queue_num, e);

                queues_[pool_queue_num]->on_error(pool_queue_num, e);
            }

            void reset_thread_distribution()
            {
                curr_queue_.store(0);
            }

        protected:
            std::size_t max_queue_thread_count_;
            std::vector<thread_queue_type*> queues_;
            std::vector<thread_queue_type*> high_priority_queues_;
            thread_queue_type low_priority_queue_;
            boost::atomic<std::size_t> curr_queue_;
            boost::atomic<std::size_t> curr_hp_queue_;
            std::size_t numa_sensitive_;
            std::size_t threads_per_hp_queue_;

            std::vector<std::vector<std::size_t>> victim_threads_;
            std::vector<std::size_t> hp_queue_lookup_;
            hpx::lcos::local::spinlock init_mutex;
            bool initialized;
        };
    }
}
}

#include <hpx/config/warnings_suffix.hpp>

#endif
