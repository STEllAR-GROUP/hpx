//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SHARED_PRIORITY_SCHEDULER)
#define HPX_SHARED_PRIORITY_SCHEDULER

#include <hpx/config.hpp>
#include <hpx/compat/mutex.hpp>
#include <hpx/runtime/threads/policies/lockfree_queue_backends.hpp>
#include <hpx/runtime/threads/policies/scheduler_base.hpp>
#include <hpx/runtime/threads/policies/thread_queue.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/runtime/threads_fwd.hpp>
#include <hpx/runtime/threads/detail/thread_num_tss.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util_fwd.hpp>

#if !defined(HPX_MSVC)
#include <plugins/parcelport/parcelport_logging.hpp>
#endif

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#include <numeric>

#define HPX_MAX_NUMA_PER_NODE 4
//#define SHARED_PRIORITY_SCHEDULER_DEBUG 1
//#define SHARED_PRIORITY_SCHEDULER_MINIMAL_DEBUG 1

// for debug reasons, we might want to pretend a single socket machine has 2 numa domains
//#define DEBUG_FORCE_2_NUMA_DOMAINS

// @TODO :
// min_tasks_to_steal_pending
// min_tasks_to_steal_staged

#if !defined(HPX_MSVC) && defined(SHARED_PRIORITY_SCHEDULER_DEBUG)
static std::chrono::high_resolution_clock::time_point log_t_start =
    std::chrono::high_resolution_clock::now();

#define LOG_CUSTOM_WORKER(x)                                                   \
    dummy << "<CUSTOM> " << THREAD_ID << " time " << decimal(16) << nowt       \
          << ' ';                                                              \
    if (parent_pool_)                                                          \
        dummy << "pool " << std::setfill(' ') << std::setw(7)                  \
              << parent_pool_->get_pool_name() << " " << x << std::endl;       \
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

#if defined(HPX_HAVE_THREAD_DESCRIPTION)
#define THREAD_DESC2(data, thrd)                                               \
    "\"" << data.description.get_description() << "\" "                        \
         << hexpointer(thrd ? thrd->get() : 0)
#else
#define THREAD_DESC2(data, thrd)                                               \
    hexpointer(thrd ? thrd : 0)
#endif

#else
#define LOG_CUSTOM_MSG(x)
#define LOG_CUSTOM_MSG2(x)
#endif

#if defined(HPX_MSVC)
#undef SHARED_PRIORITY_SCHEDULER_DEBUG
#undef LOG_CUSTOM_MSG
#undef LOG_CUSTOM_MSG2
#define LOG_CUSTOM_MSG(x)
#define LOG_CUSTOM_MSG2(x)
#endif

//
namespace hpx { namespace debug {
#ifdef SHARED_PRIORITY_SCHEDULER_MINIMAL_DEBUG
    template<typename T>
    void output(const std::string &name, const std::vector<T> &v) {
        std::cout << name.c_str() << "\t : {" << decnumber(v.size()) << "} : ";
        std::copy(std::begin(v), std::end(v), std::ostream_iterator<T>(std::cout, ", "));
        std::cout << "\n";
    }

    template<typename T, std::size_t N>
    void output(const std::string &name, const std::array<T, N> &v) {
        std::cout << name.c_str() << "\t : {" << decnumber(v.size()) << "} : ";
        std::copy(std::begin(v), std::end(v), std::ostream_iterator<T>(std::cout, ", "));
        std::cout << "\n";
    }

    template<typename Iter>
    void output(const std::string &name, Iter begin, Iter end) {
        std::cout << name.c_str() << "\t : {"
                  << decnumber(std::distance(begin,end)) << "} : ";
        std::copy(begin, end,
            std::ostream_iterator<typename std::iterator_traits<Iter>::
            value_type>(std::cout, ", "));
        std::cout << std::endl;
    }
#else
    template<typename T>
    void output(const std::string &name, const std::vector<T> &v) {
    }

    template<typename T, std::size_t N>
    void output(const std::string &name, const std::array<T, N> &v) {
    }

    template<typename Iter>
    void output(const std::string &name, Iter begin, Iter end) {
    }
#endif
}}

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
// if abp scheduler is disabled in hpx, then we cannot see certain structs
// ... so make them visible here
//
// FIFO + stealing at opposite end.
#if !defined(HPX_HAVE_ABP_SCHEDULER)
struct lockfree_abp_fifo;
struct lockfree_abp_lifo;

template <typename T>
struct lockfree_abp_fifo_backend
{
    typedef boost::lockfree::deque<T> container_type;
    typedef T value_type;
    typedef T& reference;
    typedef T const& const_reference;
    typedef std::uint64_t size_type;

    lockfree_abp_fifo_backend(
        size_type initial_size = 0
      , size_type num_thread = size_type(-1)
        )
      : queue_(std::size_t(initial_size))
    {}

    bool push(const_reference val, bool /*other_end*/ = false)
    {
        return queue_.push_left(val);
    }

    bool pop(reference val, bool steal = true)
    {
        if (steal)
            return queue_.pop_left(val);
        return queue_.pop_right(val);
    }

    bool empty()
    {
        return queue_.empty();
    }

  private:
    container_type queue_;
};

struct lockfree_abp_fifo
{
    template <typename T>
    struct apply
    {
        typedef lockfree_abp_fifo_backend<T> type;
    };
};

#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx {
namespace threads {
namespace policies {

    inline void spin_for_time(std::size_t microseconds, const char *task) {
#ifdef SHARED_PRIORITY_SCHEDULER_DEBUG
        hpx::util::annotate_function apex_profiler(task);
        std::this_thread::sleep_for(std::chrono::microseconds(microseconds));
#endif
    }

    typedef std::tuple<std::size_t, std::size_t, std::size_t> core_ratios;

    // ----------------------------------------------------------------
    // helper class to hold a set of queues
    // ----------------------------------------------------------------
    template <typename QueueType>
    struct queue_holder
    {
        void init(std::size_t cores,
                  std::size_t queues,
                  std::size_t max_tasks)
        {
            num_cores  = cores;
            num_queues = queues;
            scale      = num_cores==1 ? 0
                         : static_cast<double>(num_queues-1)/(num_cores-1);
            //
            queues_.resize(num_queues);
            for (std::size_t i = 0; i < num_queues; ++i) {
                queues_[i] = new QueueType(max_tasks);
            }
        }
        // ----------------------------------------------------------------
        ~queue_holder()
        {
            for(auto &q : queues_) delete q;
            queues_.clear();
        }

        // ----------------------------------------------------------------
        inline std::size_t get_queue_index(std::size_t id) const
        {
            return static_cast<std::size_t>(0.5 + id*scale);;
        }

        // ----------------------------------------------------------------
        inline QueueType * get_queue(std::size_t id) const
        {
            return queues_[get_queue_index(id)];
        }

        // ----------------------------------------------------------------
        inline bool get_next_thread(std::size_t id, threads::thread_data*& thrd)
        {
            // loop over all queues and take one task,
            // starting with the requested queue
            // then stealing from any other one in the container
            for (std::size_t i=0; i<num_queues; ++i) {
                std::size_t q = (id + i) % num_queues;
                if (queues_[q]->get_next_thread(thrd)) return true;
            }
            return false;
        }

        // ----------------------------------------------------------------
        inline std::size_t get_queue_length() const
        {
            std::size_t len = 0;
            for (auto &q : queues_) len += q->get_queue_length();
            return len;
        }

        // ----------------------------------------------------------------
        inline std::size_t get_thread_count(thread_state_enum state = unknown) const
        {
            std::size_t len = 0;
            for (auto &q : queues_) len += q->get_thread_count(state);
            return len;
        }

        // ----------------------------------------------------------------
        bool enumerate_threads(util::function_nonser<bool(thread_id_type)> const& f,
            thread_state_enum state = unknown) const
        {
            bool result = true;
            for (auto &q : queues_) result = result && q->enumerate_threads(f, state);
            return result;
        }

        // ----------------------------------------------------------------
        inline std::size_t size() const {
            return num_queues;
        }

        // ----------------------------------------------------------------
        std::size_t             num_cores;
        std::size_t             num_queues;
        double                  scale;
        std::vector<QueueType*> queues_;
    };

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

        shared_priority_scheduler(
            std::size_t num_worker_threads,
            core_ratios cores_per_queue,
            char const* description,
            int max_tasks = max_thread_count)
          : scheduler_base(num_worker_threads, description)
          , cores_per_queue_(cores_per_queue)
          , max_queue_thread_count_(max_tasks)
          , num_workers_(num_worker_threads)
          , num_domains_(1)
          , initialized_(false)
        {
            LOG_CUSTOM_MSG("Constructing scheduler with num threads "
                << decnumber(num_worker_threads));
            //
            HPX_ASSERT(num_worker_threads != 0);
        }

        virtual ~shared_priority_scheduler()
        {
            LOG_CUSTOM_MSG(this->get_description()
                << " - Deleting shared_priority_scheduler ");
        }

        static std::string get_scheduler_name()
        {
            return "shared_priority_scheduler";
        }

#ifdef HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES
        std::uint64_t get_creation_time(bool reset)
        {
            std::uint64_t time = 0;

            for (std::size_t i = 0; i != queues_.size(); ++i)
                time += high_priority_queues_[i]->get_creation_time(reset);

            time += low_priority_queue_.get_creation_time(reset);

            for (std::size_t i = 0; i != queues_.size(); ++i)
                time += queues_[i]->get_creation_time(reset);

            return time;
        }

        std::uint64_t get_cleanup_time(bool reset)
        {
            std::uint64_t time = 0;

            for (std::size_t i = 0; i != queues_.size(); ++i)
                time += high_priority_queues_[i]->get_cleanup_time(reset);

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
            return 0;
/*
            std::int64_t num_pending_misses = 0;
            if (pool_queue_num == std::size_t(-1))
            {
                for (std::size_t i = 0; i != high_priority_queues_.size();
                     ++i)
                {
                    num_pending_misses +=
                        high_priority_queues_[i]->get_num_pending_misses(
                            reset);
                }

                for (std::size_t i = 0; i != queues_.size(); ++i)
                {
                    num_pending_misses +=
                        queues_[i]->get_num_pending_misses(reset);
                }

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
            */
        }

        std::int64_t get_num_pending_accesses(
            std::size_t pool_queue_num, bool reset)
        {
            return 0;
            /*
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
            */
        }

        std::int64_t get_num_stolen_from_pending(
            std::size_t pool_queue_num, bool reset)
        {
            return 0;
/*
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
            */
        }

        std::int64_t get_num_stolen_to_pending(
            std::size_t pool_queue_num, bool reset)
        {
            return 0;
            /*
            std::int64_t num_stolen_threads = 0;
            if (pool_queue_num == std::size_t(-1))
            {
                for (std::size_t i = 0; i != queues_.size(); ++i)
                    num_stolen_threads +=
                        queues_[i]->get_num_stolen_to_pending(reset);

                num_stolen_threads +=
                    low_priority_queue_.get_num_stolen_to_pending(reset);

                return num_stolen_threads;
            }

            num_stolen_threads +=
                queues_[pool_queue_num]->get_num_stolen_to_pending(reset);

            if (pool_queue_num == 0)
            {
                num_stolen_threads +=
                    low_priority_queue_.get_num_stolen_to_pending(reset);
            }
            return num_stolen_threads;
            */

        }
#endif

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
        for (std::size_t i = 0; i != high_priority_queues_.size(); ++i)
        {
            wait_time += high_priority_queues_[i]->get_average_thread_wait_time();
            ++count;
        }

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
        for (std::size_t i = 0; i != high_priority_queues_.size(); ++i)
        {
            wait_time += high_priority_queues_[i]->
                get_average_task_wait_time();
            ++count;
        }

        wait_time += low_priority_queue_.get_average_task_wait_time();

        for (std::size_t i = 0; i != queues_.size(); ++i)
        {
            wait_time += queues_[i]->get_average_task_wait_time();
            ++count;
        }

        return wait_time / (count + 1);
    }
#endif

        // ------------------------------------------------------------
        void abort_all_suspended_threads()
        {
            LOG_CUSTOM_MSG("abort_all_suspended_threads");
            for (std::size_t d=0; d<num_domains_; ++d) {
                for (auto &queue : lp_queues_[d].queues_) {
                     queue->abort_all_suspended_threads();
                }

                for (auto &queue : np_queues_[d].queues_) {
                     queue->abort_all_suspended_threads();
                }

                for (auto &queue : hp_queues_[d].queues_) {
                     queue->abort_all_suspended_threads();
                }
            }
        }

        // ------------------------------------------------------------
        bool cleanup_terminated(bool delete_all)
        {
            LOG_CUSTOM_MSG("cleanup_terminated with delete_all "
                << delete_all);
            bool empty = true;
            //
            for (std::size_t d=0; d<num_domains_; ++d) {
                for (auto &queue : lp_queues_[d].queues_) {
                     empty = queue->cleanup_terminated(delete_all) && empty;
                }

                for (auto &queue : np_queues_[d].queues_) {
                     empty = queue->cleanup_terminated(delete_all) && empty;
                }

                for (auto &queue : hp_queues_[d].queues_) {
                     empty = queue->cleanup_terminated(delete_all) && empty;
                }
            }

            return empty;
        }

bool cleanup_terminated(std::size_t num_thread, bool delete_all) {
return cleanup_terminated(delete_all);
}

        ///////////////////////////////////////////////////////////////////////
        // create a new thread and schedule it if the initial state
        // is equal to pending
        void create_thread(thread_init_data& data, thread_id_type* thrd,
            thread_state_enum initial_state, error_code& ec) override
        {
            // safety check that task was created by this thread/scheduler
            HPX_ASSERT(data.scheduler_base == this);
            //
            std::size_t pool_queue_num = std::size_t(data.schedulehint);
            std::size_t domain_num = 0;
            std::size_t q_index = std::size_t(-1);
            //
            const char* const msgs[] = {"HINT_NONE","HINT     ","ERROR", "NORMAL   "};
            const char *msg = nullptr;
            //
            if (data.schedulehint == thread_schedule_hint_none || pool_queue_num == 32767)
            {
                msg = msgs[0];
                std::size_t thread_num =
                    threads::detail::thread_num_tss_.get_worker_thread_num();
                pool_queue_num = this->global_to_local_thread_index(thread_num);
                if (pool_queue_num<num_workers_) {
                    domain_num     = d_lookup_[pool_queue_num];
                    q_index        = q_lookup_[pool_queue_num];
                }
                else {
                    // this is a task being injected from a thread on another pool
                    // so leave threadnum and domain unset
                }
            }
            else if (pool_queue_num>=32768) {
                msg = msgs[1];
                domain_num = pool_queue_num - 32768;
                // if the thread creating the new task is on the domain
                // assigned to the new task - try to reuse the core as well
                std::size_t thread_num =
                    threads::detail::thread_num_tss_.get_worker_thread_num();
                pool_queue_num = this->global_to_local_thread_index(thread_num);
                if (d_lookup_[pool_queue_num]==domain_num) {
                    q_index = q_lookup_[pool_queue_num];
                }
                else {
                    q_index = counters_[domain_num];
                }
            }
            else if (pool_queue_num>num_workers_) {
                throw std::runtime_error("Bad thread number in create_thread");
            }
            else { // everything is ok
                msg = msgs[3];
                domain_num = d_lookup_[pool_queue_num];
                q_index    = q_lookup_[pool_queue_num];
            }

            LOG_CUSTOM_MSG("create_thread " << msg << " "
                           << "queue " << decnumber(pool_queue_num)
                           << "domain " << decnumber(domain_num)
                           << "qindex " << decnumber(q_index));

            // create the thread using priority to select queue
            if (data.priority == thread_priority_high ||
                data.priority == thread_priority_high_recursive ||
                data.priority == thread_priority_boost)
            {
                // boosted threads return to normal after being queued
                if (data.priority == thread_priority_boost)
                {
                    data.priority = thread_priority_normal;
                }

                hp_queues_[domain_num].
                    get_queue(q_index  % hp_queues_[domain_num].num_cores)->
                        create_thread(data, thrd, initial_state, ec);

                LOG_CUSTOM_MSG("create_thread thread_priority_high "
                    << THREAD_DESC2(data, thrd) << "queue_num "
                    << hexnumber(q_index) << "scheduler "
                    << hexpointer(data.scheduler_base));
                return;
            }

            if (data.priority == thread_priority_low)
            {
                lp_queues_[domain_num].
                    get_queue(q_index % lp_queues_[domain_num].num_cores)->
                        create_thread(data, thrd, initial_state, ec);

                LOG_CUSTOM_MSG("create_thread thread_priority_low "
                    << THREAD_DESC2(data, thrd) << "pool_queue_num "
                    << hexnumber(q_index) << "scheduler "
                    << hexpointer(data.scheduler_base));
                return;
            }

            // normal priority
            np_queues_[domain_num].
                get_queue(q_index % np_queues_[domain_num].num_cores)->
                    create_thread(data, thrd, initial_state, ec);

            LOG_CUSTOM_MSG2("create_thread thread_priority_normal "
                << THREAD_DESC2(data, thrd) << "pool_queue_num "
                << hexnumber(q_index) << "scheduler "
                << hexpointer(data.scheduler_base));
        }

        /// Return the next thread to be executed, return false if none available
        virtual bool get_next_thread(std::size_t pool_queue_num,
            bool running, std::int64_t& idle_loop_count,
            threads::thread_data*& thrd)
        {
//                LOG_CUSTOM_MSG("get_next_thread " << " queue "
//                                                  << decnumber(pool_queue_num));
            bool result = false;
            if (pool_queue_num == std::size_t(-1)) {
                throw std::runtime_error("");
            }

            // find the numa domain from the local thread index
            std::size_t domain_num = d_lookup_[pool_queue_num];

            // is there a high priority task, take first from our numa domain
            // and then try to steal from others
            for (std::size_t d=0; d<num_domains_; ++d) {
                std::size_t dom = (domain_num+d) % num_domains_;
                // set the preferred queue for this domain, if applicable
                std::size_t q_index = q_lookup_[pool_queue_num];
                // get next task, steal if from another domain
                result = hp_queues_[dom].get_next_thread(q_index, thrd);
                if (result) {
                    spin_for_time(1000, "HP task");
                    break;
                }
            }

            // try a normal priority task
            if (!result) {
                for (std::size_t d=0; d<num_domains_; ++d) {
                    std::size_t dom = (domain_num+d) % num_domains_;
                    // set the preferred queue for this domain, if applicable
                    std::size_t q_index = q_lookup_[pool_queue_num];
                    // get next task, steal if from another domain
                    result = np_queues_[dom].get_next_thread(q_index, thrd);
                    if (result) {
                        spin_for_time(1000, "NP task");
                        break;
                    }
                }
            }

            // low priority task
            if (!result) {
#ifdef JB_LP_STEALING
                for (std::size_t d=domain_num; d<domain_num+num_domains_; ++d) {
                    std::size_t dom = d % num_domains_;
                    // set the preferred queue for this domain, if applicable
                    std::size_t q_index = (dom==domain_num) ?
                                q_lookup_[pool_queue_num] : counters_[dom];

                    result = lp_queues_[dom].get_next_thread(q_index, thrd);
                    if (result)
                        break;
                }
#else
                // no cross domain stealing for LP queues
                result = lp_queues_[domain_num].get_next_thread(0, thrd);
                if (result) {
                    spin_for_time(1000, "LP task");
                }
#endif
            }
            if (result)
            {
                HPX_ASSERT(thrd->get_scheduler_base() == this);
                LOG_CUSTOM_MSG("got next thread "
                               << "queue " << decnumber(pool_queue_num)
                               << "domain " << decnumber(domain_num));
            }
/*
            // counter for access queries
            stealing_queue->increment_num_pending_accesses();
            if (result)
            {
                HPX_ASSERT(thrd->get_scheduler_base() == this);
                LOG_CUSTOM_MSG("get_next_thread high priority "
                    << THREAD_DESC(thrd) << decnumber(idle_loop_count)
                    << "pool_queue_num " << decnumber(pool_queue_num));
                return true;
            }
            // counter for high priority misses
            stealing_queue->increment_num_pending_misses();
*/

            return result;
        }

        /// Schedule the passed thread
        void schedule_thread(threads::thread_data* thrd,
            std::size_t pool_queue_num,
            std::size_t /*fallback*/,
            thread_priority priority = thread_priority_normal)
        {
            HPX_ASSERT(thrd->get_scheduler_base() == this);
            std::size_t domain_num;
            //
            if (pool_queue_num == std::size_t(-1) || pool_queue_num == 32767)
            {
                std::size_t t = threads::detail::thread_num_tss_
                                    .get_worker_thread_num();
                pool_queue_num = this->global_to_local_thread_index(t);
                domain_num     = d_lookup_[pool_queue_num];
                LOG_CUSTOM_MSG("schedule_thread HINT_NONE"
                               << " queue " << decnumber(pool_queue_num));
            }
            else if (pool_queue_num>=32768) {
                domain_num     = pool_queue_num - 32768;
                std::size_t t = threads::detail::thread_num_tss_
                                    .get_worker_thread_num();
                pool_queue_num = this->global_to_local_thread_index(t);
                LOG_CUSTOM_MSG("schedule_thread HINT domain " << domain_num
                               << " queue " << decnumber(pool_queue_num));
            }
            else if (pool_queue_num>num_workers_) {
                throw std::runtime_error("Bad thread number in schedule thread");
            }
            else {
                domain_num = d_lookup_[pool_queue_num];
                LOG_CUSTOM_MSG("schedule_thread fallback domain " << domain_num
                               << " queue " << decnumber(pool_queue_num));
            }
            std::size_t q_index = q_lookup_[pool_queue_num];

            LOG_CUSTOM_MSG("thread scheduled "
                          << "queue " << decnumber(pool_queue_num)
                          << "domain " << decnumber(domain_num)
                          << "qindex " << decnumber(q_index));

            if (priority == thread_priority_high ||
                priority == thread_priority_high_recursive ||
                priority == thread_priority_boost)
            {
                hp_queues_[domain_num].get_queue(q_index)->
                    schedule_thread(thrd, false);
            }
            else if (priority == thread_priority_low)
            {
                lp_queues_[domain_num].get_queue(q_index)->
                    schedule_thread(thrd, false);
            }
            else
            {
                np_queues_[domain_num].get_queue(q_index)->
                    schedule_thread(thrd, false);
            }
        }

        /// Put task on the back of the queue
        void schedule_thread_last(threads::thread_data* thrd,
            std::size_t pool_queue_num,
            thread_priority priority = thread_priority_normal)
        {

            LOG_CUSTOM_MSG("schedule_thread last @@@ " << " queue "
                                              << decnumber(pool_queue_num));

            HPX_ASSERT(thrd->get_scheduler_base() == this);
            std::size_t domain_num;
            //
            if (pool_queue_num == std::size_t(-1) || pool_queue_num == 32767)
            {
//                    std::cout << "Scheduler schedule_thread_last received pool_queue_num -1 \n";
                std::size_t t = threads::detail::thread_num_tss_
                                    .get_worker_thread_num();
                pool_queue_num = this->global_to_local_thread_index(t);
                domain_num     = d_lookup_[pool_queue_num];
            }
            else if (pool_queue_num>=32768) {
                domain_num     = pool_queue_num - 32768;
                pool_queue_num = std::size_t(-1);
//                    std::cout << "Scheduler schedule_thread_last numa domain "
//                              << std::dec << domain_num << std::endl;
            }
            else if (pool_queue_num>num_workers_) {
                throw std::runtime_error("Bad thread number in schedule thread");
            }
            else {
                domain_num = d_lookup_[pool_queue_num];
            }
            std::size_t q_index = q_lookup_[pool_queue_num];

            LOG_CUSTOM_MSG2("schedule_thread last (done) " << " queue "
                                              << decnumber(pool_queue_num));

            if (priority == thread_priority_high ||
                priority == thread_priority_high_recursive ||
                priority == thread_priority_boost)
            {
                hp_queues_[domain_num].get_queue(q_index)->
                    schedule_thread(thrd, true);
            }
            else if (priority == thread_priority_low)
            {
                lp_queues_[domain_num].get_queue(q_index)->
                    schedule_thread(thrd, true);
            }
            else
            {
                np_queues_[domain_num].get_queue(q_index)->
                    schedule_thread(thrd, true);
            }
        }

        //---------------------------------------------------------------------
        // Destroy the passed thread - as it has been terminated
        //---------------------------------------------------------------------
        void destroy_thread(threads::thread_data* thrd, std::int64_t& busy_count)
        {
            HPX_ASSERT(thrd->get_scheduler_base() == this);
            thrd->get_queue<thread_queue_type>().destroy_thread(thrd, busy_count);
        }

#ifdef _OLD_STYLE_FUNCTION
        bool destroy_thread(threads::thread_data* thrd,
                     std::int64_t& busy_count)
        {
            LOG_CUSTOM_MSG("destroy thread " << THREAD_DESC(thrd)
                                             << " busy_count "
                                             << decnumber(busy_count));

            HPX_ASSERT(thrd->get_scheduler_base() == this);

            for (std::size_t d=0; d<num_domains_; ++d) {
                if (hp_queues_[d].destroy_thread(thrd,busy_count)) return true;
                if (np_queues_[d].destroy_thread(thrd,busy_count)) return true;
                if (lp_queues_[d].destroy_thread(thrd,busy_count)) return true;
            }

            // the thread has to belong to one of the queues, always
            HPX_ASSERT(false);
            return false;
        }
#endif

        ///////////////////////////////////////////////////////////////////////
        // This returns the current length of the queues (work items and new
        // items)
        //---------------------------------------------------------------------
        std::int64_t get_queue_length(
            std::size_t pool_queue_num = std::size_t(-1)) const
        {
            LOG_CUSTOM_MSG("get_queue_length"
                << "pool_queue_num " << decnumber(pool_queue_num));

            std::int64_t count = 0;
            for (std::size_t d=0; d<num_domains_; ++d) {
                count += hp_queues_[d].get_queue_length();
                count += np_queues_[d].get_queue_length();
                count += lp_queues_[d].get_queue_length();
            }

            if (pool_queue_num != std::size_t(-1)) {
                // find the numa domain from the local thread index
                std::size_t domain = d_lookup_[pool_queue_num];
                std::size_t q_index = q_lookup_[pool_queue_num];
                // get next task, steal if from another domain
                std::int64_t result =
                    hp_queues_[domain].get_queue(q_index)->get_queue_length();
                if (result>0) return result;
                result =
                    np_queues_[domain].get_queue(q_index)->get_queue_length();
                if (result>0) return result;
                return
                    lp_queues_[domain].get_queue(q_index)->get_queue_length();
            }
            return count;
        }

        //---------------------------------------------------------------------
        // Queries the current thread count of the queues.
        //---------------------------------------------------------------------
        std::int64_t get_thread_count(thread_state_enum state = unknown,
            thread_priority priority = thread_priority_default,
            std::size_t pool_queue_num = std::size_t(-1),
            bool reset = false) const
        {
            LOG_CUSTOM_MSG("get_thread_count pool_queue_num "
                << hexnumber(pool_queue_num));

            std::int64_t count = 0;

            // if a specific worker id was requested
            if (pool_queue_num != std::size_t(-1)) {
                std::size_t domain_num = d_lookup_[pool_queue_num];
                std::size_t q_index = q_lookup_[pool_queue_num];
                //
                switch (priority) {
                case thread_priority_default: {
                    count += hp_queues_[domain_num].get_queue(q_index)->
                        get_thread_count(state);
                    count += np_queues_[domain_num].get_queue(q_index)->
                        get_thread_count(state);
                    count += lp_queues_[domain_num].get_queue(q_index)->
                        get_thread_count(state);
                    LOG_CUSTOM_MSG("default get_thread_count pool_queue_num "
                        << hexnumber(pool_queue_num) << decnumber(count));
                    return count;
                }
                case thread_priority_low: {
                    count = lp_queues_[domain_num].get_queue(q_index)->
                        get_thread_count(state);
                    LOG_CUSTOM_MSG("low get_thread_count pool_queue_num "
                        << hexnumber(pool_queue_num) << decnumber(count));
                    return count;
                }
                case thread_priority_normal: {
                    count = np_queues_[domain_num].get_queue(q_index)->
                        get_thread_count(state);
                    LOG_CUSTOM_MSG("normal get_thread_count pool_queue_num "
                        << hexnumber(pool_queue_num) << decnumber(count));
                    return count;
                }
                case thread_priority_boost:
                case thread_priority_high:
                case thread_priority_high_recursive: {
                    count = hp_queues_[domain_num].get_queue(q_index)->
                        get_thread_count(state);
                    LOG_CUSTOM_MSG("high get_thread_count pool_queue_num "
                        << hexnumber(pool_queue_num) << decnumber(count));
                    return count;
                }
                default:
                case thread_priority_unknown:
                    HPX_THROW_EXCEPTION(bad_parameter,
                        "shared_priority_queue_scheduler::get_thread_count",
                        "unknown thread priority (thread_priority_unknown)");
                    return 0;
                }
            }

            switch (priority) {
            case thread_priority_default: {
                for (std::size_t d=0; d<num_domains_; ++d) {
                    count += hp_queues_[d].get_thread_count(state);
                    count += np_queues_[d].get_thread_count(state);
                    count += lp_queues_[d].get_thread_count(state);
                }
//                    LOG_CUSTOM_MSG("default get_thread_count pool_queue_num "
//                        << decnumber(pool_queue_num) << decnumber(count));
                return count;
            }
            case thread_priority_low: {
                for (std::size_t d=0; d<num_domains_; ++d) {
                    count += lp_queues_[d].get_thread_count(state);
                }
                LOG_CUSTOM_MSG("low get_thread_count pool_queue_num "
                    << decnumber(pool_queue_num) << decnumber(count));
                return count;
            }
            case thread_priority_normal: {
                for (std::size_t d=0; d<num_domains_; ++d) {
                    count += np_queues_[d].get_thread_count(state);
                }
                LOG_CUSTOM_MSG("normal get_thread_count pool_queue_num "
                    << decnumber(pool_queue_num) << decnumber(count));
                return count;
            }
            case thread_priority_boost:
            case thread_priority_high:
            case thread_priority_high_recursive: {
                for (std::size_t d=0; d<num_domains_; ++d) {
                    count += hp_queues_[d].get_thread_count(state);
                }
                LOG_CUSTOM_MSG("high get_thread_count pool_queue_num "
                    << decnumber(pool_queue_num) << decnumber(count));
                return count;
            }
            default:
            case thread_priority_unknown:
                HPX_THROW_EXCEPTION(bad_parameter,
                    "shared_priority_queue_scheduler::get_thread_count",
                    "unknown thread priority (thread_priority_unknown)");
                return 0;
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

            LOG_CUSTOM_MSG("enumerate_threads");

            for (std::size_t d=0; d<num_domains_; ++d) {
                result = result &&
                    hp_queues_[d].enumerate_threads(f, state);
                result = result &&
                    np_queues_[d].enumerate_threads(f, state);
                result = result &&
                    lp_queues_[d].enumerate_threads(f, state);
            }
            return result;
        }

        ///////////////////////////////////////////////////////////////////////
        /* @TODO make sure on_start_thread is forwarded
         * currently it does nothing in the thread queue - but might one day
         * queues_[pool_queue_num]->on_start_thread(pool_queue_num);
        */
        void on_start_thread(std::size_t pool_queue_num)
        {
            LOG_CUSTOM_MSG("start thread with local thread num "
                << decnumber(pool_queue_num));

            std::lock_guard<hpx::lcos::local::spinlock> lock(init_mutex);
            if (!initialized_)
            {
                initialized_ = true;
                //
                auto &rp = resource::get_partitioner();
                auto const& topo = rp.get_topology();

                // For each worker thread, count which each numa domain they
                // belong to and build lists of useful indexes/refs
                num_domains_ = 1;
                std::array<std::size_t, HPX_MAX_NUMA_PER_NODE> q_counts_;
                std::fill(d_lookup_.begin(), d_lookup_.end(), 0);
                std::fill(q_lookup_.begin(), q_lookup_.end(), 0);
                std::fill(q_counts_.begin(), q_counts_.end(), 0);
                std::fill(counters_.begin(), counters_.end(), 0);
                //
                //std::srand(std::time(0));
                for (std::size_t local_id=0; local_id!=num_workers_; ++local_id)
                {
                    std::size_t global_id = local_to_global_thread_index(local_id);
                    std::size_t pu_num = rp.get_pu_num(global_id);
                    std::size_t domain = topo.get_numa_node_number(pu_num);
#ifdef DEBUG_FORCE_2_NUMA_DOMAINS
                    domain = std::rand() % 2;
#endif
                    d_lookup_[local_id] = domain;
                    num_domains_ = std::max(num_domains_, domain+1);
                }
                //
                for (std::size_t local_id=0; local_id!=num_workers_; ++local_id)
                {
                    q_lookup_[local_id] = q_counts_[d_lookup_[local_id]]++;
                }

                debug::output("d_lookup_ ", &d_lookup_[0], &d_lookup_[num_workers_]);
                debug::output("q_lookup_ ", &q_lookup_[0], &q_lookup_[num_workers_]);
                debug::output("counters_ ", &counters_[0], &counters_[num_domains_]);
                debug::output("q_counts_ ", &q_counts_[0], &q_counts_[num_domains_]);

                // create queue sets for each numa domain
                for (std::size_t i=0; i<num_domains_; ++i) {
                    // high priority
                    int queues = std::max(q_counts_[i]/std::get<0>(cores_per_queue_),
                                          std::size_t(1));
                    hp_queues_[i].init(
                        q_counts_[i], queues, max_queue_thread_count_);
                    LOG_CUSTOM_MSG2("Created HP queue for numa " << i
                                    << " cores " << q_counts_[i]
                                    << " queues " << queues);
                    // normal priority
                    queues = std::max(q_counts_[i]/std::get<1>(cores_per_queue_),
                                      std::size_t(1));
                    np_queues_[i].init(
                        q_counts_[i], queues, max_queue_thread_count_);
                    LOG_CUSTOM_MSG2("Created NP queue for numa " << i
                                    << " cores " << q_counts_[i]
                                    << " queues " << queues);
                    // low oriority
                    queues = std::max(q_counts_[i]/std::get<2>(cores_per_queue_),
                                      std::size_t(1));
                    lp_queues_[i].init(
                        q_counts_[i], queues, max_queue_thread_count_);
                    LOG_CUSTOM_MSG2("Created LP queue for numa " << i
                                    << " cores " << q_counts_[i]
                                    << " queues " << queues);
                }
            }
        }

        void on_stop_thread(std::size_t pool_queue_num)
        {
            LOG_CUSTOM_MSG("on_stop_thread with local thread num "
                << decnumber(pool_queue_num));

            if (pool_queue_num>num_workers_) {
                throw std::runtime_error("Bad thread number in schedule thread");
            }
/*
            // find the numa domain from the local thread index
            std::size_t domain_num = d_lookup_[pool_queue_num];

            // set the preferred queue for this domain, if applicable
            std::size_t q_index = pool_queue_num - q_offset_[domain_num];

            bool result = hp_queues_[d].get_queue(q_index)->on_stop_thread(pool_queue_num);

            queues_[pool_queue_num]->on_stop_thread(pool_queue_num);
*/
        }

        void on_error(
            std::size_t pool_queue_num, std::exception_ptr const& e)
        {
            LOG_CUSTOM_MSG("on_error with local thread num "
                << decnumber(pool_queue_num));

            if (pool_queue_num>num_workers_) {
                throw std::runtime_error("Bad thread number in schedule thread");
        }
/*
            if (pool_queue_num == queues_.size() - 1)
                low_priority_queue_.on_error(pool_queue_num, e);

            queues_[pool_queue_num]->on_error(pool_queue_num, e);
*/
        }

        void reset_thread_distribution()
        {
            // curr_queue_.store(0);
        }

    protected:
        typedef queue_holder<thread_queue_type> numa_queues;
        //
        std::array<numa_queues, HPX_MAX_NUMA_PER_NODE> np_queues_;
        std::array<numa_queues, HPX_MAX_NUMA_PER_NODE> hp_queues_;
        std::array<numa_queues, HPX_MAX_NUMA_PER_NODE> lp_queues_;
        std::array<std::size_t, HPX_MAX_NUMA_PER_NODE> counters_;
        // lookup domain from local worker index
        std::array<std::size_t, HPX_HAVE_MAX_CPU_COUNT> d_lookup_;
        // lookup sub domain queue index from local worker index
        std::array<std::size_t, HPX_HAVE_MAX_CPU_COUNT> q_lookup_;
        // number of cores per queue for HP, NP, LP queues
        core_ratios cores_per_queue_;
        // max storage size of any queue
        std::size_t max_queue_thread_count_;
        // number of worker threads assigned to this pool
        std::size_t num_workers_;
        // number of numa domains that the threads are occupying
        std::size_t num_domains_;
        // used to make sure the scheduler is only initialized once on a thread
        bool        initialized_;
        hpx::lcos::local::spinlock init_mutex;
    };
}}}

#include <hpx/config/warnings_suffix.hpp>

#endif
