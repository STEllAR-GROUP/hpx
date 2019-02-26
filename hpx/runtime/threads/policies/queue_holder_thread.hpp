//  Copyright (c) 2017-2018 John Biddiscombe
//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADMANAGER_SCHEDULING_QUEUE_HOLDER_THREAD)
#define HPX_THREADMANAGER_SCHEDULING_QUEUE_HOLDER_THREAD

#include <hpx/config.hpp>
#include <hpx/logging.hpp>
#include <hpx/runtime/threads/policies/thread_queue_init_parameters.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/type_support/unused.hpp>
#include <hpx/runtime/threads/policies/thread_queue_mc.hpp>
#include <hpx/runtime/threads/policies/lockfree_queue_backends.hpp>
#include <hpx/runtime/threads/thread_data_stackful.hpp>
#include <hpx/runtime/threads/thread_data_stackless.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <vector>
#include <unordered_set>
#include <list>
#include <map>

#include <atomic>
#include <mutex>
#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#define LOG_CUSTOM_MSG(x)

// ----------------------------------------------------------------
namespace hpx { namespace threads { namespace policies
{
    // apply the modulo operator only when needed
    // (i.e. when the input is greater than the ceiling)
    // NB: the numbers must be positive
    HPX_FORCEINLINE int fast_mod(const unsigned int input, const unsigned int ceil) {
        return input >= ceil ? input % ceil : input;
    }

    enum : std::size_t { max_thread_count = 1000 };
    enum : std::size_t { round_robin_rollover = 2 };

    // ----------------------------------------------------------------
    // Helper class to hold a set of queues.
    // ----------------------------------------------------------------
    template <typename QueueType>
    struct alignas(64) queue_holder_thread
    {
        // Queues that will store actual work for this thread
        // They might be shared between cores, so we use a pointer to
        // reference them
        QueueType *hp_queue_;
        QueueType *np_queue_;
        QueueType *lp_queue_;

        const std::int16_t  owner_mask_;
        const std::int16_t queue_index_;

        // we must use OS mutexes here because we cannot suspend an HPX
        // thread whilst processing the Queues for that thread, this code
        // is running at the OS level in effect.
        typedef std::mutex mutex_type;

        // mutex protecting the members
        mutable mutex_type mtx_;

        // every thread maintains lists of free thread data objects
        // sorted by their stack sizes
        using thread_heap_type =
            std::list<thread_id_type, util::internal_allocator<thread_id_type>>;

        thread_heap_type thread_heap_small_;
        thread_heap_type thread_heap_medium_;
        thread_heap_type thread_heap_large_;
        thread_heap_type thread_heap_huge_;
        thread_heap_type thread_heap_nostack_;

        // number of terminated threads to discard
        const int max_delete_count_;

        // number of terminated threads to collect before cleaning them up
        const int max_terminated_threads_;

        // these ought to be atomic, but if we get a race and assign a thread
        // to queue N instead of N+1 it doesn't really matter
        mutable int next_queue_counter_;
        mutable int counter_rollover_;

        // ----------------------------------------------------------------
        // ----------------------------------------------------------------
        // ----------------------------------------------------------------

        static util::internal_allocator<threads::thread_data> thread_alloc_;

        typedef util::tuple<thread_init_data, thread_state_enum> task_description;

        // -------------------------------------
        // thread map stores every task in this queue set
        // this is the type of a map holding all threads (except depleted/terminated)
        using thread_map_type = std::unordered_set<thread_id_type,
            std::hash<thread_id_type>, std::equal_to<thread_id_type>,
            util::internal_allocator<thread_id_type>>;
        thread_map_type             thread_map_;
        std::atomic<std::int64_t>   thread_map_count_;

        // -------------------------------------
        // terminated tasks
        // completed tasks that can be reused (stack space etc)
        using terminated_items_type = lockfree_fifo::apply<thread_data*>::type;
        terminated_items_type       terminated_items_;
        std::atomic<std::int64_t>   terminated_items_count_;

        // ----------------------------------------------------------------
        // ----------------------------------------------------------------
        // ----------------------------------------------------------------
        queue_holder_thread(
                QueueType *hp_queue, QueueType *np_queue, QueueType *lp_queue,
                int owner, int id,
                const thread_queue_init_parameters &init)
            : hp_queue_(hp_queue)
            , np_queue_(np_queue)
            , lp_queue_(lp_queue)
            , owner_mask_(owner)
            , queue_index_(id)
            , max_delete_count_(init.max_delete_count_)
            , max_terminated_threads_(init.max_terminated_threads_)
            , next_queue_counter_(id)
            , counter_rollover_(round_robin_rollover)
            , thread_map_count_(0)
            , terminated_items_(max_thread_count)
            , terminated_items_count_(0)
        {
            if (hp_queue_) hp_queue_->set_holder(this);
            if (np_queue_) np_queue_->set_holder(this);
            if (lp_queue_) lp_queue_->set_holder(this);
        }

        // ----------------------------------------------------------------
        ~queue_holder_thread()
        {
            if (owns_hp_queue()) delete hp_queue_;
            if (owns_np_queue()) delete np_queue_;
            if (owns_lp_queue()) delete lp_queue_;
            //
            for(auto t: thread_heap_small_)
                deallocate(get_thread_id_data(t));

            for(auto t: thread_heap_medium_)
                deallocate(get_thread_id_data(t));

            for(auto t: thread_heap_large_)
                deallocate(get_thread_id_data(t));

            for(auto t: thread_heap_huge_)
                deallocate(get_thread_id_data(t));

            for(auto t: thread_heap_nostack_)
                deallocate(get_thread_id_data(t));
        }

        // ----------------------------------------------------------------
        inline bool owns_hp_queue() const {
            return hp_queue_ && ((owner_mask_ & 1) != 0);
        }

        // ----------------------------------------------------------------
        inline bool owns_np_queue() const {
            return ((owner_mask_ & 2) != 0);
        }

        // ----------------------------------------------------------------
        inline bool owns_lp_queue() const {
            return lp_queue_ && ((owner_mask_ & 4) != 0);
        }

        // ----------------------------------------------------------------
        void debug(const char *txt, int idx, int new_tasks, int work_items, threads::thread_data* thrd)
        {
            LOG_CUSTOM_MSG(txt
                           << " queue " << dec4(idx)
                           << " new " << dec4(new_tasks)
                           << " work " << dec4(work_items)
                           << " map " << dec4(thread_map_count_)
                           << " terminated " << dec4(terminated_items_count_)
                           << THREAD_DESC(thrd)
                           );
        }

        // ----------------------------------------------------------------
        void debug_timed(int delay, const char *txt, int idx, int new_tasks, int work_items, threads::thread_data* thrd)
        {
//            static int counter = 0;
//            if (counter++ % 100000 == 0 ) {
                LOG_CUSTOM_MSG(txt
                               << " queue " << dec4(idx)
                               << " new " << dec4(new_tasks)
                               << " work " << dec4(work_items)
                               << " map " << dec4(thread_map_count_)
                               << " terminated " << dec4(terminated_items_count_)
                               << THREAD_DESC(thrd)
                               );
//            }
        }

        // ------------------------------------------------------------
        // return the next round robin thread index across all workers
        // using a batching of 10 per worker before incrementing
        inline unsigned int worker_next(const unsigned int workers) const
        {
            if (--counter_rollover_ == 0) {
                counter_rollover_ = round_robin_rollover;
                next_queue_counter_ = fast_mod(next_queue_counter_+1, workers);
            }
            return next_queue_counter_;
        }

        // ------------------------------------------------------------
        // return the next round robin thread index in the same numa domain
        // using a batching of 10 per worker before incrementing
        inline unsigned int numa_next(const unsigned int numacount) const
        {
            if (--counter_rollover_ == 0) {
                counter_rollover_ = round_robin_rollover;
                next_queue_counter_ = fast_mod(next_queue_counter_+1, numacount);
                return next_queue_counter_;
            }
            // we have to take the modulus just in case the last thread used
            // in worker_next() was on another domain and has an index too high
            next_queue_counter_ = fast_mod(next_queue_counter_, numacount);
            return next_queue_counter_;
        }

        // ------------------------------------------------------------
        void schedule_thread(threads::thread_data* thrd,
                             thread_priority priority, bool other_end=false)
        {
            if (hp_queue_ &&
                     (priority == thread_priority_high ||
                      priority == thread_priority_high_recursive ||
                      priority == thread_priority_boost))
            {
                hp_queue_->schedule_thread(thrd, other_end);
            }
            else if (lp_queue_ &&
                     (priority == thread_priority_low))
            {
                lp_queue_->schedule_thread(thrd, other_end);
            }
            else
            {
                np_queue_->schedule_thread(thrd, other_end);
            }
        }

        // ----------------------------------------------------------------
        bool cleanup_terminated(bool delete_all = false)
        {
            if (terminated_items_count_ == 0) return true;

            if (delete_all) {
                // do not lock mutex while deleting all threads, do it piece-wise
                while (true)
                {
                    std::lock_guard<mutex_type> lk(mtx_);
                    if (cleanup_terminated_locked(false))
                    {
                        return true;
                    }
                }
                return false;
            }

            std::lock_guard<mutex_type> lk(mtx_);
            return cleanup_terminated_locked(false);
        }

        // ----------------------------------------------------------------
        bool cleanup_terminated_locked(bool delete_all = false)
        {
            if (terminated_items_count_ == 0)
                return true;

            if (delete_all) {
                // delete all threads
                thread_data* todelete;
                while (terminated_items_.pop(todelete))
                {
                    thread_id_type tid(todelete);
                    --terminated_items_count_;
                    remove_from_thread_map(tid, true);
                    debug("deallocate", queue_index_,
                          np_queue_->new_tasks_count_.data_,
                          np_queue_->work_items_count_.data_,
                          nullptr);
                }
            }
            else {
                // delete only this many threads
                std::int64_t delete_count =
                    (std::max)(
                        static_cast<std::int64_t>(terminated_items_count_ / 10),
                        static_cast<std::int64_t>(max_delete_count_));

                thread_data* todelete;
                while (delete_count && terminated_items_.pop(todelete))
                {
                    thread_id_type tid(todelete);
                    --terminated_items_count_;
                    remove_from_thread_map(tid, false);
                    recycle_thread(tid);
                    debug("recycle   ", queue_index_,
                          np_queue_->new_tasks_count_.data_,
                          np_queue_->work_items_count_.data_,
                          todelete);
                    --delete_count;

                }
            }
            return terminated_items_count_ == 0;
        }

        // ----------------------------------------------------------------
        void create_thread(thread_init_data& data, thread_id_type* tid,
                           thread_state_enum state, bool run_now, error_code& ec)
        {
            // create the thread using priority to select queue
            if (hp_queue_ &&
                (data.priority == thread_priority_high ||
                 data.priority == thread_priority_high_recursive ||
                 data.priority == thread_priority_boost))
            {
                // boosted threads return to normal after being queued
                if (data.priority == thread_priority_boost) {
                    data.priority = thread_priority_normal;
                }

                return hp_queue_->create_thread(data, tid, state, run_now, ec);
            }

            if (lp_queue_ &&
                (data.priority == thread_priority_low))
            {
                return lp_queue_->create_thread(data, tid, state, run_now, ec);
            }

            // normal priority + anything unassigned above (no hp queues etc)
            np_queue_->create_thread(data, tid, state, run_now, ec);
//            LOG_CUSTOM_MSG2("create_thread thread_priority_normal "
//                            << "queue " << decnumber(q_index)
//                            << "domain " << decnumber(domain_num)
//                            << THREAD_DESC2(data, thrd)
//                            << "scheduler " << hexpointer(data.scheduler_base));
        }

        // ----------------------------------------------------------------
        template <typename Lock>
        void create_thread_object(threads::thread_id_type& tid,
            threads::thread_init_data& data, thread_state_enum state, Lock& lk)
        {
            HPX_ASSERT(lk.owns_lock());
            HPX_ASSERT(data.stacksize != 0);

            std::ptrdiff_t stacksize = data.stacksize;

            thread_heap_type* heap = nullptr;
            if (stacksize == get_stack_size(thread_stacksize_small))
            {
                heap = &thread_heap_small_;
            }
            else if (stacksize == get_stack_size(thread_stacksize_medium))
            {
                heap = &thread_heap_medium_;
            }
            else if (stacksize == get_stack_size(thread_stacksize_large))
            {
                heap = &thread_heap_large_;
            }
            else if (stacksize == get_stack_size(thread_stacksize_huge))
            {
                heap = &thread_heap_huge_;
            }
            else if (stacksize == get_stack_size(thread_stacksize_nostack))
            {
                heap = &thread_heap_nostack_;
            }
            HPX_ASSERT(heap);

            if (state == pending_do_not_schedule || state == pending_boost)
            {
                state = pending;
            }

            // Check for an unused thread object.
            if (!heap->empty())
            {
                // Take ownership of the thread object and rebind it.
                tid = heap->front();
                heap->pop_front();
                get_thread_id_data(tid)->rebind(data, state);
            }
            else
            {
                hpx::util::unlock_guard<Lock> ull(lk);

                // Allocate a new thread object.
                threads::thread_data* p = nullptr;
                if (stacksize == get_stack_size(thread_stacksize_nostack))
                {
                    p = threads::thread_data_stackless::create(
                        data, this, state);
                }
                else
                {
                    p = threads::thread_data_stackful::create(
                        data, this, state);
                }
                tid = thread_id_type(p);
            }
        }

        // ----------------------------------------------------------------
        void recycle_thread(thread_id_type tid)
        {
            std::ptrdiff_t stacksize = get_thread_id_data(tid)->get_stack_size();

            if (stacksize == get_stack_size(thread_stacksize_small))
            {
                thread_heap_small_.push_front(tid);
            }
            else if (stacksize == get_stack_size(thread_stacksize_medium))
            {
                thread_heap_medium_.push_front(tid);
            }
            else if (stacksize == get_stack_size(thread_stacksize_large))
            {
                thread_heap_large_.push_front(tid);
            }
            else if (stacksize == get_stack_size(thread_stacksize_huge))
            {
                thread_heap_huge_.push_front(tid);
            }
            else if (stacksize == get_stack_size(thread_stacksize_nostack))
            {
                thread_heap_nostack_.push_front(tid);
            }
            else
            {
                HPX_ASSERT_MSG(
                    false, util::format("Invalid stack size {1}", stacksize));
            }
        }

        // ----------------------------------------------------------------
        static void deallocate(threads::thread_data* p)
        {
            using threads::thread_data;
            p->~thread_data();
            thread_alloc_.deallocate(p, 1);
        }

        // ----------------------------------------------------------------
        void add_to_thread_map(
                threads::thread_id_type tid,
                std::unique_lock<mutex_type> &lk)
        {
            HPX_ASSERT(lk.owns_lock());

            // add a new entry in the map for this thread
            std::pair<thread_map_type::iterator, bool> p =
                thread_map_.insert(tid);

            if (/*HPX_UNLIKELY*/(!p.second)) {
                lk.unlock();
                HPX_THROW_EXCEPTION(hpx::out_of_memory,
                    "queue_helper::add_to_thread_map",
                    "Couldn't add new thread to the thread map");
            }

            ++thread_map_count_;

            // this thread has to be in the map now
            HPX_ASSERT(thread_map_.find(tid)!=thread_map_.end());
//            HPX_ASSERT(&thrd->get_queue<queue_holder_thread>() == this);
        }

        // ----------------------------------------------------------------
        void remove_from_thread_map(
                threads::thread_id_type tid,
                bool dealloc)
        {
            // this thread has to be in this map
            HPX_ASSERT(thread_map_.find(tid)  !=  thread_map_.end());
            HPX_ASSERT(thread_map_count_ >= 0);

            bool deleted = thread_map_.erase(tid) != 0;
            HPX_ASSERT(deleted);
            if (dealloc) {
                deallocate(get_thread_id_data(tid));
            }
            --thread_map_count_;
        }

        // ----------------------------------------------------------------
        bool get_next_thread(threads::thread_data*& thrd,
            bool allow_stealing, bool other_end) HPX_HOT
        {
            if (owns_hp_queue() &&
                    hp_queue_->get_next_thread(thrd, allow_stealing, other_end))
                return true;

            if (np_queue_->get_next_thread(thrd, allow_stealing, other_end))
                return true;

            if (owns_lp_queue() &&
                    lp_queue_->get_next_thread(thrd, allow_stealing, other_end))
                return true;

            return false;
        }

        // ----------------------------------------------------------------
        bool wait_or_add_new(
            bool running, std::int64_t& idle_loop_count, std::size_t& added, bool steal)
        {
            bool result = true;
            if (owns_hp_queue()) {
                result = hp_queue_->wait_or_add_new(running, added) && result;
            }

            if (owns_np_queue()) {
                result = np_queue_->wait_or_add_new(running, added) && result;
            }

            if (owns_lp_queue()) {
                result = lp_queue_->wait_or_add_new(running, added) && result;
            }
            return result;
        }

        // ----------------------------------------------------------------
        inline std::size_t get_queue_length()
        {
            std::size_t count = 0;
            count += owns_hp_queue() ? hp_queue_->get_queue_length() : 0;
            count += owns_np_queue() ? np_queue_->get_queue_length() : 0;
            count += owns_lp_queue() ? lp_queue_->get_queue_length() : 0;
            return count;
        }

        // ----------------------------------------------------------------
        inline std::size_t get_thread_count_staged(thread_priority priority) const
        {
            // Return thread count of one specific queue.
            switch (priority) {
                case thread_priority_default: {
                    std::int64_t count = 0;
                    count += owns_hp_queue() ? hp_queue_->get_queue_length_staged() : 0;
                    count += owns_np_queue() ? np_queue_->get_queue_length_staged() : 0;
                    count += owns_lp_queue() ? lp_queue_->get_queue_length_staged() : 0;
                    return count;
                }
                case thread_priority_low: {
                    return owns_lp_queue() ? lp_queue_->get_queue_length_staged() : 0;
                }
                case thread_priority_normal: {
                    return owns_np_queue() ? np_queue_->get_queue_length_staged() : 0;
                }
                case thread_priority_boost:
                case thread_priority_high:
                case thread_priority_high_recursive: {
                    return owns_hp_queue() ? hp_queue_->get_queue_length_staged() : 0;
                }
                default:
                case thread_priority_unknown: {
                    HPX_THROW_EXCEPTION(bad_parameter,
                        "queue_holder_thread::get_thread_count_staged",
                        "unknown thread priority value (thread_priority_unknown)");
                }
            }
            return 0;
        }

        // ----------------------------------------------------------------
        inline std::size_t get_thread_count_pending(thread_priority priority) const
        {
            // Return thread count of one specific queue.
            switch (priority) {
                case thread_priority_default: {
                    std::int64_t count = 0;
                    count += owns_hp_queue() ? hp_queue_->get_queue_length_pending() : 0;
                    count += owns_np_queue() ? np_queue_->get_queue_length_pending() : 0;
                    count += owns_lp_queue() ? lp_queue_->get_queue_length_pending() : 0;
                    return count;
                }
                case thread_priority_low: {
                    return owns_lp_queue() ? lp_queue_->get_queue_length_pending() : 0;
                }
                case thread_priority_normal: {
                    return owns_np_queue() ? np_queue_->get_queue_length_pending() : 0;
                }
                case thread_priority_boost:
                case thread_priority_high:
                case thread_priority_high_recursive: {
                    return owns_hp_queue() ? hp_queue_->get_queue_length_pending() : 0;
                }
                default:
                case thread_priority_unknown: {
                    HPX_THROW_EXCEPTION(bad_parameter,
                        "queue_holder_thread::get_thread_count_pending",
                        "unknown thread priority value (thread_priority_unknown)");
                }
            }
            return 0;
        }

        // ----------------------------------------------------------------
        inline std::size_t get_thread_count(thread_state_enum state = unknown,
            thread_priority priority = thread_priority_default) const
        {
            if (terminated == state)
                return terminated_items_count_;

            if (staged == state)
                return get_thread_count_staged(priority);

            if (pending == state)
                return get_thread_count_pending(priority);

            if (unknown == state)
                return thread_map_count_ +
                        get_thread_count_staged(priority) - terminated_items_count_;

            // acquire lock only if absolutely necessary
            std::lock_guard<mutex_type> lk(mtx_);

            std::int64_t num_threads = 0;
            thread_map_type::const_iterator end = thread_map_.end();
            for (thread_map_type::const_iterator it = thread_map_.begin();
                 it != end; ++it)
            {
                if (get_thread_id_data(*it)->get_state().state() == state)
                    ++num_threads;
            }
            return num_threads;
        }

        // ------------------------------------------------------------
        /// Destroy the passed thread as it has been terminated
        void destroy_thread(threads::thread_data* thrd, std::int64_t& busy_count)
        {
            HPX_ASSERT(&thrd->get_queue<queue_holder_thread>() == this);
            terminated_items_.push(thrd);
            std::int64_t count = ++terminated_items_count_;
            if (count > max_terminated_threads_)
            {
                cleanup_terminated(false);   // clean up all terminated threads
            }
            debug("destroy   ", -1,
                  np_queue_->new_tasks_count_.data_,
                  np_queue_->work_items_count_.data_,
                  thrd);
        }

        // ------------------------------------------------------------
        void abort_all_suspended_threads()
        {
            throw std::runtime_error("This function needs to be reimplemented");
            std::lock_guard<mutex_type> lk(mtx_);
            thread_map_type::iterator end =  thread_map_.end();
            for (thread_map_type::iterator it = thread_map_.begin();
                 it != end; ++it)
            {
                if (get_thread_id_data(*it)->get_state().state() == suspended)
                {
                    get_thread_id_data(*it)->set_state(pending, wait_abort);
                    // np queue always exists so use that as priority doesn't matter
                    np_queue_->schedule_thread(get_thread_id_data(*it), true);
                }
            }
        }

        // ------------------------------------------------------------
        bool enumerate_threads(
            util::function_nonser<bool(thread_id_type)> const& f,
            thread_state_enum state = unknown) const
        {
            std::uint64_t count = thread_map_count_;
            if (state == terminated)
            {
                count = terminated_items_count_;
            }
            else if (state == staged)
            {
                HPX_THROW_EXCEPTION(bad_parameter,
                    "queue_holder_thread::iterate_threads",
                    "can't iterate over thread ids of staged threads");
                return false;
            }

            std::vector<thread_id_type> tids;
            tids.reserve(static_cast<std::size_t>(count));

            if (state == unknown)
            {
                std::lock_guard<mutex_type> lk(mtx_);
                thread_map_type::const_iterator end =  thread_map_.end();
                for (thread_map_type::const_iterator it = thread_map_.begin();
                     it != end; ++it)
                {
                    tids.push_back(*it);
                }
            }
            else
            {
                std::lock_guard<mutex_type> lk(mtx_);
                thread_map_type::const_iterator end =  thread_map_.end();
                for (thread_map_type::const_iterator it = thread_map_.begin();
                     it != end; ++it)
                {
                    if (get_thread_id_data(*it)->get_state().state() == state)
                        tids.push_back(*it);
                }
            }

            // now invoke callback function for all matching threads
            for (thread_id_type const& id : tids)
            {
                if (!f(id))
                    return false;       // stop iteration
            }

            return true;
        }
    };

    template <typename QueueType>
    util::internal_allocator<threads::thread_data>
        queue_holder_thread<QueueType>::thread_alloc_;

#ifdef HPX_HAVE_THREAD_MINIMAL_DEADLOCK_DETECTION
    // ------------------------------------------------------------////
    // We globally control whether to do minimal deadlock detection using this
    // global bool variable. It will be set once by the runtime configuration
    // startup code
    extern bool minimal_deadlock_detection;
#endif

// ------------------------------------------------------------////////

}}}

#endif

