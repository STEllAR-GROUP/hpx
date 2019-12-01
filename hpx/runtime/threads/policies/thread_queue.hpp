//  Copyright (c) 2007-2019 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADMANAGER_THREAD_QUEUE_AUG_25_2009_0132PM)
#define HPX_THREADMANAGER_THREAD_QUEUE_AUG_25_2009_0132PM

#include <hpx/config.hpp>
#include <hpx/allocator_support/internal_allocator.hpp>
#include <hpx/assertion.hpp>
#include <hpx/concurrency/cache_line_data.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/errors.hpp>
#include <hpx/format.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/runtime/threads/policies/lockfree_queue_backends.hpp>
#include <hpx/runtime/threads/policies/queue_helpers.hpp>
#include <hpx/runtime/threads/policies/thread_queue_init_parameters.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/thread_data_stackful.hpp>
#include <hpx/runtime/threads/thread_data_stackless.hpp>
#include <hpx/thread_support/unlock_guard.hpp>
#include <hpx/timing/high_resolution_clock.hpp>
#include <hpx/util/get_and_reset_value.hpp>

#ifdef HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES
#include <hpx/util/tick_counter.hpp>
#endif

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace policies {
#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
    ///////////////////////////////////////////////////////////////////////////
    // We control whether to collect queue wait times using this global bool.
    // It will be set by any of the related performance counters. Once set it
    // stays set, thus no race conditions will occur.
    extern HPX_EXPORT bool maintain_queue_wait_times;
#endif
#ifdef HPX_HAVE_THREAD_MINIMAL_DEADLOCK_DETECTION
    ///////////////////////////////////////////////////////////////////////////
    // We globally control whether to do minimal deadlock detection using this
    // global bool variable. It will be set once by the runtime configuration
    // startup code
    extern bool minimal_deadlock_detection;
#endif

    ///////////////////////////////////////////////////////////////////////////
    // // Queue back-end interface:
    //
    // template <typename T>
    // struct queue_backend
    // {
    //     typedef ... container_type;
    //     typedef ... value_type;
    //     typedef ... reference;
    //     typedef ... const_reference;
    //     typedef ... size_type;
    //
    //     queue_backend(
    //         size_type initial_size = ...
    //       , size_type num_thread = ...
    //         );
    //
    //     bool push(const_reference val);
    //
    //     bool pop(reference val, bool steal = true);
    //
    //     bool empty();
    // };
    //
    // struct queue_policy
    // {
    //     template <typename T>
    //     struct apply
    //     {
    //         typedef ... type;
    //     };
    // };
    template <typename Mutex, typename PendingQueuing, typename StagedQueuing,
        typename TerminatedQueuing>
    class thread_queue
    {
    private:
        // we use a simple mutex to protect the data members for now
        using mutex_type = Mutex;

        // this is the type of a map holding all threads (except depleted ones)
        using thread_map_type = std::unordered_set<thread_id_type,
            std::hash<thread_id_type>, std::equal_to<thread_id_type>,
            util::internal_allocator<thread_id_type>>;

        using thread_heap_type =
            std::list<thread_id_type, util::internal_allocator<thread_id_type>>;

#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
        using task_description =
            util::tuple<thread_init_data, thread_state_enum, std::uint64_t>;
#else
        using task_description =
            util::tuple<thread_init_data, thread_state_enum>;
#endif

#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
        using thread_description = util::tuple<thread_data*, std::uint64_t>;
#else
        using thread_description = thread_data;
#endif

        using work_items_type =
            typename PendingQueuing::template apply<thread_description*>::type;

        using task_items_type =
            typename StagedQueuing::template apply<task_description*>::type;

        using terminated_items_type =
            typename TerminatedQueuing::template apply<thread_data*>::type;

    protected:
        template <typename Lock>
        void create_thread_object(threads::thread_id_type& thrd,
            threads::thread_init_data& data, thread_state_enum state, Lock& lk)
        {
            HPX_ASSERT(lk.owns_lock());
            HPX_ASSERT(data.stacksize > 0);

            std::ptrdiff_t stacksize = data.stacksize;

            thread_heap_type* heap = nullptr;

            if (stacksize == parameters_.small_stacksize_)
            {
                heap = &thread_heap_small_;
            }
            else if (stacksize == parameters_.medium_stacksize_)
            {
                heap = &thread_heap_medium_;
            }
            else if (stacksize == parameters_.large_stacksize_)
            {
                heap = &thread_heap_large_;
            }
            else if (stacksize == parameters_.huge_stacksize_)
            {
                heap = &thread_heap_huge_;
            }
            else if (stacksize == parameters_.nostack_stacksize_)
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
                thrd = heap->front();
                heap->pop_front();
                get_thread_id_data(thrd)->rebind(data, state);
            }
            else
            {
                hpx::util::unlock_guard<Lock> ull(lk);

                // Allocate a new thread object.
                threads::thread_data* p = nullptr;
                if (stacksize == parameters_.nostack_stacksize_)
                {
                    p = threads::thread_data_stackless::create(
                        data, this, state);
                }
                else
                {
                    p = threads::thread_data_stackful::create(
                        data, this, state);
                }
                thrd = thread_id_type(p);
            }
        }

        static util::internal_allocator<task_description>
            task_description_alloc_;

        ///////////////////////////////////////////////////////////////////////
        // add new threads if there is some amount of work available
        std::size_t add_new(std::int64_t add_count, thread_queue* addfrom,
            std::unique_lock<mutex_type>& lk, bool steal = false)
        {
            HPX_ASSERT(lk.owns_lock());

            if (HPX_UNLIKELY(0 == add_count))
                return 0;

            std::size_t added = 0;
            task_description* task = nullptr;
            while (add_count-- && addfrom->new_tasks_.pop(task, steal))
            {
#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
                if (maintain_queue_wait_times)
                {
                    addfrom->new_tasks_wait_ +=
                        util::high_resolution_clock::now() -
                        util::get<2>(*task);
                    ++addfrom->new_tasks_wait_count_;
                }
#endif
                // create the new thread
                threads::thread_init_data& data = util::get<0>(*task);
                thread_state_enum state = util::get<1>(*task);
                threads::thread_id_type thrd;

                create_thread_object(thrd, data, state, lk);

                task->~task_description();
                task_description_alloc_.deallocate(task, 1);

                // add the new entry to the map of all threads
                std::pair<thread_map_type::iterator, bool> p =
                    thread_map_.insert(thrd);

                if (HPX_UNLIKELY(!p.second))
                {
                    --addfrom->new_tasks_count_.data_;
                    lk.unlock();
                    HPX_THROW_EXCEPTION(hpx::out_of_memory,
                        "thread_queue::add_new",
                        "Couldn't add new thread to the thread map");
                    return 0;
                }

                ++thread_map_count_;

                // Decrement only after thread_map_count_ has been incremented
                --addfrom->new_tasks_count_.data_;

                // only insert the thread into the work-items queue if it is in
                // pending state
                if (state == pending)
                {
                    // pushing the new thread into the pending queue of the
                    // specified thread_queue
                    ++added;
                    schedule_thread(get_thread_id_data(thrd));
                }

                // this thread has to be in the map now
                HPX_ASSERT(thread_map_.find(thrd) != thread_map_.end());
                HPX_ASSERT(
                    &get_thread_id_data(thrd)->get_queue<thread_queue>() ==
                    this);
            }

            if (added)
            {
                LTM_(debug) << "add_new: added " << added
                            << " tasks to queues";    //-V128
            }
            return added;
        }

        ///////////////////////////////////////////////////////////////////////
        bool add_new_always(std::size_t& added, thread_queue* addfrom,
            std::unique_lock<mutex_type>& lk, bool steal = false)
        {
            HPX_ASSERT(lk.owns_lock());

#ifdef HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES
            util::tick_counter tc(add_new_time_);
#endif

            // create new threads from pending tasks (if appropriate)
            std::int64_t add_count = -1;    // default is no constraint

            // if we are desperate (no work in the queues), add some even if the
            // map holds more than max_thread_count
            if (HPX_LIKELY(parameters_.max_thread_count_))
            {
                std::int64_t count =
                    static_cast<std::int64_t>(thread_map_.size());
                if (parameters_.max_thread_count_ >=
                    count + parameters_.min_add_new_count_)
                {    //-V104
                    HPX_ASSERT(parameters_.max_thread_count_ - count <
                        (std::numeric_limits<std::int64_t>::max)());
                    add_count = static_cast<std::int64_t>(
                        parameters_.max_thread_count_ - count);
                    if (add_count < parameters_.min_add_new_count_)
                        add_count = parameters_.min_add_new_count_;
                    if (add_count > parameters_.max_add_new_count_)
                        add_count = parameters_.max_add_new_count_;
                }
                else if (work_items_.empty())
                {
                    // add this number of threads
                    add_count = parameters_.min_add_new_count_;

                    // increase max_thread_count
                    parameters_.max_thread_count_ +=
                        parameters_.min_add_new_count_;    //-V101
                }
                else
                {
                    return false;
                }
            }

            std::size_t addednew = add_new(add_count, addfrom, lk, steal);
            added += addednew;
            return addednew != 0;
        }

        void recycle_thread(thread_id_type thrd)
        {
            std::ptrdiff_t stacksize =
                get_thread_id_data(thrd)->get_stack_size();

            if (stacksize == parameters_.small_stacksize_)
            {
                thread_heap_small_.push_front(thrd);
            }
            else if (stacksize == parameters_.medium_stacksize_)
            {
                thread_heap_medium_.push_front(thrd);
            }
            else if (stacksize == parameters_.large_stacksize_)
            {
                thread_heap_large_.push_front(thrd);
            }
            else if (stacksize == parameters_.huge_stacksize_)
            {
                thread_heap_huge_.push_front(thrd);
            }
            else if (stacksize == parameters_.nostack_stacksize_)
            {
                thread_heap_nostack_.push_front(thrd);
            }
            else
            {
                HPX_ASSERT_MSG(
                    false, util::format("Invalid stack size {1}", stacksize));
            }
        }

    public:
        /// This function makes sure all threads which are marked for deletion
        /// (state is terminated) are properly destroyed.
        ///
        /// This returns 'true' if there are no more terminated threads waiting
        /// to be deleted.
        bool cleanup_terminated_locked(bool delete_all = false)
        {
#ifdef HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES
            util::tick_counter tc(cleanup_terminated_time_);
#endif

            if (terminated_items_count_ == 0)
                return true;

            if (delete_all)
            {
                // delete all threads
                thread_data* todelete;
                while (terminated_items_.pop(todelete))
                {
                    thread_id_type tid(todelete);
                    --terminated_items_count_;

                    // this thread has to be in this map
                    HPX_ASSERT(thread_map_.find(tid) != thread_map_.end());

                    bool deleted = thread_map_.erase(tid) != 0;
                    HPX_ASSERT(deleted);
                    if (deleted)
                    {
                        deallocate(todelete);
                        --thread_map_count_;
                        HPX_ASSERT(thread_map_count_ >= 0);
                    }
                }
            }
            else
            {
                // delete only this many threads
                std::int64_t delete_count = (std::min)(
                    static_cast<std::int64_t>(terminated_items_count_ / 10),
                    static_cast<std::int64_t>(parameters_.max_delete_count_));

                // delete at least this many threads
                delete_count = (std::max)(delete_count,
                    static_cast<std::int64_t>(parameters_.min_delete_count_));

                thread_data* todelete;
                while (delete_count && terminated_items_.pop(todelete))
                {
                    thread_id_type tid(todelete);
                    --terminated_items_count_;

                    thread_map_type::iterator it = thread_map_.find(tid);

                    // this thread has to be in this map
                    HPX_ASSERT(it != thread_map_.end());

                    recycle_thread(*it);

                    thread_map_.erase(it);
                    --thread_map_count_;
                    HPX_ASSERT(thread_map_count_ >= 0);

                    --delete_count;
                }
            }
            return terminated_items_count_ == 0;
        }

    public:
        bool cleanup_terminated(bool delete_all = false)
        {
            if (terminated_items_count_ == 0)
                return true;

            if (delete_all)
            {
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

        thread_queue(std::size_t queue_num = std::size_t(-1),
            thread_queue_init_parameters parameters = {})
          : parameters_(parameters)
          , thread_map_count_(0)
          , work_items_(128, queue_num)
#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
          , work_items_wait_(0)
          , work_items_wait_count_(0)
#endif
          , terminated_items_(128)
          , terminated_items_count_(0)
          , new_tasks_(128)
#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
          , new_tasks_wait_(0)
          , new_tasks_wait_count_(0)
#endif
          , thread_heap_small_()
          , thread_heap_medium_()
          , thread_heap_large_()
          , thread_heap_huge_()
          , thread_heap_nostack_()
#ifdef HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES
          , add_new_time_(0)
          , cleanup_terminated_time_(0)
#endif
#ifdef HPX_HAVE_THREAD_STEALING_COUNTS
          , pending_misses_(0)
          , pending_accesses_(0)
          , stolen_from_pending_(0)
          , stolen_from_staged_(0)
          , stolen_to_pending_(0)
          , stolen_to_staged_(0)
#endif
        {
            new_tasks_count_.data_ = 0;
            work_items_count_.data_ = 0;
        }

        static void deallocate(threads::thread_data* p)
        {
            p->destroy();
        }

        ~thread_queue()
        {
            for (auto t : thread_heap_small_)
                deallocate(get_thread_id_data(t));

            for (auto t : thread_heap_medium_)
                deallocate(get_thread_id_data(t));

            for (auto t : thread_heap_large_)
                deallocate(get_thread_id_data(t));

            for (auto t : thread_heap_huge_)
                deallocate(get_thread_id_data(t));

            for (auto t : thread_heap_nostack_)
                deallocate(get_thread_id_data(t));
        }

#ifdef HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES
        std::uint64_t get_creation_time(bool reset)
        {
            return util::get_and_reset_value(add_new_time_, reset);
        }

        std::uint64_t get_cleanup_time(bool reset)
        {
            return util::get_and_reset_value(cleanup_terminated_time_, reset);
        }
#endif

        ///////////////////////////////////////////////////////////////////////
        // This returns the current length of the queues (work items and new items)
        std::int64_t get_queue_length(
            std::memory_order order = std::memory_order_acquire) const
        {
            return work_items_count_.data_.load(order) +
                new_tasks_count_.data_.load(order);
        }

        // This returns the current length of the pending queue
        std::int64_t get_pending_queue_length(
            std::memory_order order = std::memory_order_acquire) const
        {
            return work_items_count_.data_.load(order);
        }

        // This returns the current length of the staged queue
        std::int64_t get_staged_queue_length(
            std::memory_order order = std::memory_order_acquire) const
        {
            return new_tasks_count_.data_.load(order);
        }

#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
        std::uint64_t get_average_task_wait_time() const
        {
            std::uint64_t count = new_tasks_wait_count_;
            if (count == 0)
                return 0;
            return new_tasks_wait_ / count;
        }

        std::uint64_t get_average_thread_wait_time() const
        {
            std::uint64_t count = work_items_wait_count_;
            if (count == 0)
                return 0;
            return work_items_wait_ / count;
        }
#endif

#ifdef HPX_HAVE_THREAD_STEALING_COUNTS
        std::int64_t get_num_pending_misses(bool reset)
        {
            return util::get_and_reset_value(pending_misses_, reset);
        }

        void increment_num_pending_misses(std::size_t num = 1)
        {
            pending_misses_.fetch_add(num, std::memory_order_relaxed);
        }

        std::int64_t get_num_pending_accesses(bool reset)
        {
            return util::get_and_reset_value(pending_accesses_, reset);
        }

        void increment_num_pending_accesses(std::size_t num = 1)
        {
            pending_accesses_.fetch_add(num, std::memory_order_relaxed);
        }

        std::int64_t get_num_stolen_from_pending(bool reset)
        {
            return util::get_and_reset_value(stolen_from_pending_, reset);
        }

        void increment_num_stolen_from_pending(std::size_t num = 1)
        {
            stolen_from_pending_.fetch_add(num, std::memory_order_relaxed);
        }

        std::int64_t get_num_stolen_from_staged(bool reset)
        {
            return util::get_and_reset_value(stolen_from_staged_, reset);
        }

        void increment_num_stolen_from_staged(std::size_t num = 1)
        {
            stolen_from_staged_.fetch_add(num, std::memory_order_relaxed);
        }

        std::int64_t get_num_stolen_to_pending(bool reset)
        {
            return util::get_and_reset_value(stolen_to_pending_, reset);
        }

        void increment_num_stolen_to_pending(std::size_t num = 1)
        {
            stolen_to_pending_.fetch_add(num, std::memory_order_relaxed);
        }

        std::int64_t get_num_stolen_to_staged(bool reset)
        {
            return util::get_and_reset_value(stolen_to_staged_, reset);
        }

        void increment_num_stolen_to_staged(std::size_t num = 1)
        {
            stolen_to_staged_.fetch_add(num, std::memory_order_relaxed);
        }
#else
        HPX_CXX14_CONSTEXPR void increment_num_pending_misses(
            std::size_t num = 1)
        {
        }
        HPX_CXX14_CONSTEXPR void increment_num_pending_accesses(
            std::size_t num = 1)
        {
        }
        HPX_CXX14_CONSTEXPR void increment_num_stolen_from_pending(
            std::size_t num = 1)
        {
        }
        HPX_CXX14_CONSTEXPR void increment_num_stolen_from_staged(
            std::size_t num = 1)
        {
        }
        HPX_CXX14_CONSTEXPR void increment_num_stolen_to_pending(
            std::size_t num = 1)
        {
        }
        HPX_CXX14_CONSTEXPR void increment_num_stolen_to_staged(
            std::size_t num = 1)
        {
        }
#endif

        ///////////////////////////////////////////////////////////////////////
        // create a new thread and schedule it if the initial state is equal to
        // pending
        void create_thread(thread_init_data& data, thread_id_type* id,
            thread_state_enum initial_state, bool run_now, error_code& ec)
        {
            // thread has not been created yet
            if (id)
                *id = invalid_thread_id;

            if (run_now)
            {
                threads::thread_id_type thrd;

                // The mutex can not be locked while a new thread is getting
                // created, as it might have that the current HPX thread gets
                // suspended.
                {
                    std::unique_lock<mutex_type> lk(mtx_);

                    create_thread_object(thrd, data, initial_state, lk);

                    // add a new entry in the map for this thread
                    std::pair<thread_map_type::iterator, bool> p =
                        thread_map_.insert(thrd);

                    if (HPX_UNLIKELY(!p.second))
                    {
                        lk.unlock();
                        HPX_THROWS_IF(ec, hpx::out_of_memory,
                            "thread_queue::create_thread",
                            "Couldn't add new thread to the map of threads");
                        return;
                    }
                    ++thread_map_count_;

                    // this thread has to be in the map now
                    HPX_ASSERT(thread_map_.find(thrd) != thread_map_.end());
                    HPX_ASSERT(
                        &get_thread_id_data(thrd)->get_queue<thread_queue>() ==
                        this);

                    // push the new thread in the pending queue thread
                    if (initial_state == pending)
                        schedule_thread(get_thread_id_data(thrd));

                    // return the thread_id of the newly created thread
                    if (id)
                        *id = thrd;

                    if (&ec != &throws)
                        ec = make_success_code();
                    return;
                }
            }

            // do not execute the work, but register a task description for
            // later thread creation
            ++new_tasks_count_.data_;

            task_description* td = task_description_alloc_.allocate(1);
#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
            new (td) task_description(std::move(data), initial_state,
                util::high_resolution_clock::now());
#else
            new (td)
                task_description(std::move(data), initial_state);    //-V106
#endif
            new_tasks_.push(td);
            if (&ec != &throws)
                ec = make_success_code();
        }

        void move_work_items_from(thread_queue* src, std::int64_t count)
        {
            thread_description* trd;
            while (src->work_items_.pop(trd))
            {
                --src->work_items_count_.data_;

#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
                if (maintain_queue_wait_times)
                {
                    std::uint64_t now = util::high_resolution_clock::now();
                    src->work_items_wait_ += now - util::get<1>(*trd);
                    ++src->work_items_wait_count_;
                    util::get<1>(*trd) = now;
                }
#endif

                bool finished = count == ++work_items_count_.data_;
                work_items_.push(trd);
                if (finished)
                    break;
            }
        }

        void move_task_items_from(thread_queue* src, std::int64_t count)
        {
            task_description* task;
            while (src->new_tasks_.pop(task))
            {
#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
                if (maintain_queue_wait_times)
                {
                    std::int64_t now = util::high_resolution_clock::now();
                    src->new_tasks_wait_ += now - util::get<2>(*task);
                    ++src->new_tasks_wait_count_;
                    util::get<2>(*task) = now;
                }
#endif

                bool finish = count == ++new_tasks_count_.data_;

                // Decrement only after the local new_tasks_count_ has
                // been incremented
                --src->new_tasks_count_.data_;

                if (new_tasks_.push(task))
                {
                    if (finish)
                        break;
                }
                else
                {
                    --new_tasks_count_.data_;
                }
            }
        }

        /// Return the next thread to be executed, return false if none is
        /// available
        bool get_next_thread(threads::thread_data*& thrd,
            bool allow_stealing = false, bool steal = false) HPX_HOT
        {
            std::int64_t work_items_count =
                work_items_count_.data_.load(std::memory_order_relaxed);

            if (allow_stealing &&
                parameters_.min_tasks_to_steal_pending_ > work_items_count)
            {
                return false;
            }

#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
            thread_description* tdesc;
            if (0 != work_items_count && work_items_.pop(tdesc, steal))
            {
                --work_items_count_.data_;

                if (maintain_queue_wait_times)
                {
                    work_items_wait_ += util::high_resolution_clock::now() -
                        util::get<1>(*tdesc);
                    ++work_items_wait_count_;
                }

                thrd = util::get<0>(*tdesc);
                delete tdesc;

                return true;
            }
#else
            if (0 != work_items_count && work_items_.pop(thrd, steal))
            {
                --work_items_count_.data_;
                return true;
            }
#endif
            return false;
        }

        /// Schedule the passed thread
        void schedule_thread(threads::thread_data* thrd, bool other_end = false)
        {
            ++work_items_count_.data_;
#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
            work_items_.push(new thread_description(
                                 thrd, util::high_resolution_clock::now()),
                other_end);
#else
            work_items_.push(thrd, other_end);
#endif
        }

        /// Destroy the passed thread as it has been terminated
        void destroy_thread(
            threads::thread_data* thrd, std::int64_t& busy_count)
        {
            HPX_ASSERT(&thrd->get_queue<thread_queue>() == this);
            terminated_items_.push(thrd);

            std::int64_t count = ++terminated_items_count_;
            if (count > parameters_.max_terminated_threads_)
            {
                cleanup_terminated(true);    // clean up all terminated threads
            }
        }

        ///////////////////////////////////////////////////////////////////////
        /// Return the number of existing threads with the given state.
        std::int64_t get_thread_count(thread_state_enum state = unknown) const
        {
            if (terminated == state)
                return terminated_items_count_;

            if (staged == state)
                return new_tasks_count_.data_;

            if (unknown == state)
            {
                return thread_map_count_ + new_tasks_count_.data_ -
                    terminated_items_count_;
            }

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

        ///////////////////////////////////////////////////////////////////////
        void abort_all_suspended_threads()
        {
            std::lock_guard<mutex_type> lk(mtx_);
            thread_map_type::iterator end = thread_map_.end();
            for (thread_map_type::iterator it = thread_map_.begin(); it != end;
                 ++it)
            {
                auto thrd = get_thread_id_data(*it);
                if (thrd->get_state().state() == suspended)
                {
                    thrd->set_state(pending, wait_abort);
                    schedule_thread(thrd);
                }
            }
        }

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
                    "thread_queue::iterate_threads",
                    "can't iterate over thread ids of staged threads");
                return false;
            }

            std::vector<thread_id_type> ids;
            ids.reserve(static_cast<std::size_t>(count));

            if (state == unknown)
            {
                std::lock_guard<mutex_type> lk(mtx_);
                thread_map_type::const_iterator end = thread_map_.end();
                for (thread_map_type::const_iterator it = thread_map_.begin();
                     it != end; ++it)
                {
                    ids.push_back(*it);
                }
            }
            else
            {
                std::lock_guard<mutex_type> lk(mtx_);
                thread_map_type::const_iterator end = thread_map_.end();
                for (thread_map_type::const_iterator it = thread_map_.begin();
                     it != end; ++it)
                {
                    if (get_thread_id_data(*it)->get_state().state() == state)
                        ids.push_back(*it);
                }
            }

            // now invoke callback function for all matching threads
            for (thread_id_type const& id : ids)
            {
                if (!f(id))
                    return false;    // stop iteration
            }

            return true;
        }

        /// This is a function which gets called periodically by the thread
        /// manager to allow for maintenance tasks to be executed in the
        /// scheduler. Returns true if the OS thread calling this function
        /// has to be terminated (i.e. no more work has to be done).
        inline bool wait_or_add_new(
            bool, std::size_t& added, bool steal = false) HPX_HOT
        {
            if (0 == new_tasks_count_.data_.load(std::memory_order_relaxed))
            {
                return true;
            }

            // No obvious work has to be done, so a lock won't hurt too much.
            //
            // We prefer to exit this function (some kind of very short
            // busy waiting) to blocking on this lock. Locking fails either
            // when a thread is currently doing thread maintenance, which
            // means there might be new work, or the thread owning the lock
            // just falls through to the cleanup work below (no work is available)
            // in which case the current thread (which failed to acquire
            // the lock) will just retry to enter this loop.
            std::unique_lock<mutex_type> lk(mtx_, std::try_to_lock);
            if (!lk.owns_lock())
                return false;    // avoid long wait on lock

            // stop running after all HPX threads have been terminated
            return !add_new_always(added, this, lk, steal);
        }

        inline bool wait_or_add_new(bool running, std::size_t& added,
            thread_queue* addfrom, bool steal = false) HPX_HOT
        {
            // try to generate new threads from task lists, but only if our
            // own list of threads is empty
            if (0 == work_items_count_.data_.load(std::memory_order_relaxed))
            {
                // see if we can avoid grabbing the lock below

                // don't try to steal if there are only a few tasks left on
                // this queue
                if (running &&
                    parameters_.min_tasks_to_steal_staged_ >
                        addfrom->new_tasks_count_.data_.load(
                            std::memory_order_relaxed))
                {
                    return false;
                }

                // No obvious work has to be done, so a lock won't hurt too much.
                //
                // We prefer to exit this function (some kind of very short
                // busy waiting) to blocking on this lock. Locking fails either
                // when a thread is currently doing thread maintenance, which
                // means there might be new work, or the thread owning the lock
                // just falls through to the cleanup work below (no work is available)
                // in which case the current thread (which failed to acquire
                // the lock) will just retry to enter this loop.
                std::unique_lock<mutex_type> lk(mtx_, std::try_to_lock);
                if (!lk.owns_lock())
                    return false;    // avoid long wait on lock

                // stop running after all HPX threads have been terminated
                bool added_new = add_new_always(added, addfrom, lk, steal);
                if (!added_new)
                {
                    // Before exiting each of the OS threads deletes the
                    // remaining terminated HPX threads
                    // REVIEW: Should we be doing this if we are stealing?
                    bool canexit = cleanup_terminated_locked(true);
                    if (!running && canexit)
                    {
                        // we don't have any registered work items anymore
                        //do_some_work();       // notify possibly waiting threads
                        return true;    // terminate scheduling loop
                    }
                    return false;
                }
                else
                {
                    cleanup_terminated_locked();
                    return false;
                }
            }

            bool canexit = cleanup_terminated(true);
            if (!running && canexit)
            {
                // we don't have any registered work items anymore
                return true;    // terminate scheduling loop
            }

            return false;
        }

        ///////////////////////////////////////////////////////////////////////
        bool dump_suspended_threads(
            std::size_t num_thread, std::int64_t& idle_loop_count, bool running)
        {
#ifndef HPX_HAVE_THREAD_MINIMAL_DEADLOCK_DETECTION
            return false;
#else
            if (minimal_deadlock_detection)
            {
                std::lock_guard<mutex_type> lk(mtx_);
                return detail::dump_suspended_threads(
                    num_thread, thread_map_, idle_loop_count, running);
            }
            return false;
#endif
        }

        ///////////////////////////////////////////////////////////////////////
        void on_start_thread(std::size_t num_thread) {}
        void on_stop_thread(std::size_t num_thread) {}
        void on_error(std::size_t num_thread, std::exception_ptr const& e) {}

    private:
        thread_queue_init_parameters parameters_;

        mutable mutex_type mtx_;    // mutex protecting the members

        thread_map_type thread_map_;    // mapping of thread id's to HPX-threads

        // overall count of work items
        std::atomic<std::int64_t> thread_map_count_;

        work_items_type work_items_;    // list of active work items

#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
        // overall wait time of work items
        std::atomic<std::int64_t> work_items_wait_;
        // overall number of work items in queue
        std::atomic<std::int64_t> work_items_wait_count_;
#endif
        // list of terminated threads
        terminated_items_type terminated_items_;
        // count of terminated items
        std::atomic<std::int64_t> terminated_items_count_;

        task_items_type new_tasks_;    // list of new tasks to run

#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
        // overall wait time of new tasks
        std::atomic<std::int64_t> new_tasks_wait_;
        // overall number tasks waited
        std::atomic<std::int64_t> new_tasks_wait_count_;
#endif

        thread_heap_type thread_heap_small_;
        thread_heap_type thread_heap_medium_;
        thread_heap_type thread_heap_large_;
        thread_heap_type thread_heap_huge_;
        thread_heap_type thread_heap_nostack_;

#ifdef HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES
        std::uint64_t add_new_time_;
        std::uint64_t cleanup_terminated_time_;
#endif

#ifdef HPX_HAVE_THREAD_STEALING_COUNTS
        // # of times our associated worker-thread couldn't find work in work_items
        std::atomic<std::int64_t> pending_misses_;

        // # of times our associated worker-thread looked for work in work_items
        std::atomic<std::int64_t> pending_accesses_;

        // count of work_items stolen from this queue
        std::atomic<std::int64_t> stolen_from_pending_;
        // count of new_tasks stolen from this queue
        std::atomic<std::int64_t> stolen_from_staged_;
        // count of work_items stolen to this queue from other queues
        std::atomic<std::int64_t> stolen_to_pending_;
        // count of new_tasks stolen to this queue from other queues
        std::atomic<std::int64_t> stolen_to_staged_;
#endif
        // count of new tasks to run, separate to new cache line to avoid false
        // sharing
        util::cache_line_data<std::atomic<std::int64_t>> new_tasks_count_;

        // count of active work items
        util::cache_line_data<std::atomic<std::int64_t>> work_items_count_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Mutex, typename PendingQueuing, typename StagedQueuing,
        typename TerminatedQueuing>
    util::internal_allocator<typename thread_queue<Mutex, PendingQueuing,
        StagedQueuing, TerminatedQueuing>::task_description>
        thread_queue<Mutex, PendingQueuing, StagedQueuing,
            TerminatedQueuing>::task_description_alloc_;
}}}    // namespace hpx::threads::policies

#endif
