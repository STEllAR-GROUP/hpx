//  Copyright (c) 2007-2019 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/allocator_support/internal_allocator.hpp>
#include <hpx/assert.hpp>
#include <hpx/concurrency/cache_line_data.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/schedulers/deadlock_detection.hpp>
#include <hpx/schedulers/lockfree_queue_backends.hpp>
#include <hpx/schedulers/maintain_queue_wait_times.hpp>
#include <hpx/schedulers/queue_holder_thread.hpp>
#include <hpx/schedulers/thread_queue.hpp>
#include <hpx/thread_support/unlock_guard.hpp>
#include <hpx/threading_base/thread_data.hpp>
#include <hpx/threading_base/thread_queue_init_parameters.hpp>
#include <hpx/timing/high_resolution_clock.hpp>
#include <hpx/topology/topology.hpp>
#include <hpx/util/get_and_reset_value.hpp>

#ifdef HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES
#include <hpx/timing/tick_counter.hpp>
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

#if !defined(THREAD_QUEUE_MC_DEBUG)
#if defined(HPX_DEBUG)
#define THREAD_QUEUE_MC_DEBUG false
#else
#define THREAD_QUEUE_MC_DEBUG false
#endif
#endif

//#define DEBUG_QUEUE_EXTRA 1

namespace hpx {
    static hpx::debug::enable_print<THREAD_QUEUE_MC_DEBUG> tqmc_deb("_TQ_MC_");
}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace policies {

    template <typename Mutex, typename PendingQueuing, typename StagedQueuing,
        typename TerminatedQueuing>
    class thread_queue_mc
    {
    public:
        // we use a simple mutex to protect the data members for now
        typedef Mutex mutex_type;

        using thread_queue_type = thread_queue_mc<Mutex, PendingQueuing,
            StagedQueuing, TerminatedQueuing>;

        using thread_heap_type =
            std::list<thread_id_type, util::internal_allocator<thread_id_type>>;

        using task_description = thread_init_data;
        using thread_description = thread_data;

        typedef
            typename PendingQueuing::template apply<thread_description*>::type
                work_items_type;

        typedef concurrentqueue_fifo::apply<task_description>::type
            task_items_type;

    public:
        // ----------------------------------------------------------------
        // Take thread init data from the new work queue and convert it into
        // full thread_data items that are added to the pending queue.
        //
        // New work items are taken from the queue owned by 'addfrom' and
        // added to the pending queue of this thread holder
        //
        // This is not thread safe, only the thread owning the holder should
        // call this function
        std::size_t add_new(
            std::int64_t add_count, thread_queue_type* addfrom, bool stealing)
        {
            if (addfrom->new_tasks_count_.data_.load(
                    std::memory_order_relaxed) == 0)
            {
                return 0;
            }
            //

            std::size_t added = 0;
            task_description task;
            while (add_count-- && addfrom->new_task_items_.pop(task, stealing))
            {
                // create the new thread
                threads::thread_init_data& data = task;
                threads::thread_id_type tid;

                holder_->create_thread_object(tid, data);
                holder_->add_to_thread_map(tid);
                // Decrement only after thread_map_count_ has been incremented
                --addfrom->new_tasks_count_.data_;

                tqmc_deb.debug(debug::str<>("add_new"), "stealing", stealing,
                    debug::threadinfo<threads::thread_id_type*>(&tid));

                // insert the thread into work-items queue if in pending state
                if (data.initial_state == thread_schedule_state::pending)
                {
                    // pushing the new thread into the pending queue of the
                    // specified thread_queue
                    ++added;
                    schedule_work(get_thread_id_data(tid), stealing);
                }
            }

            return added;
        }

    public:
        explicit thread_queue_mc(const thread_queue_init_parameters& parameters,
            std::size_t queue_num = std::size_t(-1))
          : parameters_(parameters)
          , queue_index_(static_cast<int>(queue_num))
          , holder_(nullptr)
          , new_task_items_(1024)
          , work_items_(1024)
        {
            new_tasks_count_.data_ = 0;
            work_items_count_.data_ = 0;
        }

        // ----------------------------------------------------------------
        void set_holder(queue_holder_thread<thread_queue_type>* holder)
        {
            holder_ = holder;
            tqmc_deb.debug(debug::str<>("set_holder"), "D",
                debug::dec<2>(holder_->domain_index_), "Q",
                debug::dec<3>(queue_index_));
        }

        // ----------------------------------------------------------------
        ~thread_queue_mc() {}

        // ----------------------------------------------------------------
        // This returns the current length of the queues (work items and new items)
        std::int64_t get_queue_length() const
        {
            return work_items_count_.data_.load(std::memory_order_relaxed) +
                new_tasks_count_.data_.load(std::memory_order_relaxed);
        }

        // ----------------------------------------------------------------
        // This returns the current length of the pending queue
        std::int64_t get_queue_length_pending() const
        {
            return work_items_count_.data_.load(std::memory_order_relaxed);
        }

        // ----------------------------------------------------------------
        // This returns the current length of the staged queue
        std::int64_t get_queue_length_staged(
            std::memory_order order = std::memory_order_relaxed) const
        {
            return new_tasks_count_.data_.load(order);
        }

        // ----------------------------------------------------------------
        // Return the number of existing threads with the given state.
        std::int64_t get_thread_count() const
        {
            HPX_THROW_EXCEPTION(bad_parameter, "get_thread_count",
                "use get_queue_length_staged/get_queue_length_pending");
            return 0;
        }

        // create a new thread and schedule it if the initial state is equal to
        // pending
        void create_thread(
            thread_init_data& data, thread_id_type* id, error_code& ec)
        {
            // thread has not been created yet
            if (id)
                *id = invalid_thread_id;

            if (data.stacksize == threads::thread_stacksize::current)
            {
                data.stacksize = get_self_stacksize_enum();
            }

            HPX_ASSERT(data.stacksize != threads::thread_stacksize::current);

            if (data.run_now)
            {
                threads::thread_id_type tid;
                holder_->create_thread_object(tid, data);
                holder_->add_to_thread_map(tid);

                // push the new thread in the pending queue thread
                if (data.initial_state == thread_schedule_state::pending)
                    schedule_work(get_thread_id_data(tid), false);

                // return the thread_id of the newly created thread
                if (id)
                    *id = tid;

                if (&ec != &throws)
                    ec = make_success_code();
                return;
            }

            // do not execute the work, but register a task description for
            // later thread creation
            ++new_tasks_count_.data_;

            new_task_items_.push(task_description(std::move(data)));

            if (&ec != &throws)
                ec = make_success_code();
        }

        // ----------------------------------------------------------------
        /// Return the next thread to be executed, return false if none is
        /// available
        bool get_next_thread(threads::thread_data*& thrd, bool other_end,
            bool check_new = false) HPX_HOT
        {
            std::int64_t work_items_count_count =
                work_items_count_.data_.load(std::memory_order_relaxed);

            if (0 != work_items_count_count && work_items_.pop(thrd, other_end))
            {
                --work_items_count_.data_;
                tqmc_deb.debug(debug::str<>("get_next_thread"), "stealing",
                    other_end, "D", debug::dec<2>(holder_->domain_index_), "Q",
                    debug::dec<3>(queue_index_), "n",
                    debug::dec<4>(new_tasks_count_.data_), "w",
                    debug::dec<4>(work_items_count_.data_),
                    debug::threadinfo<threads::thread_data*>(thrd));
                return true;
            }
            if (check_new && add_new(32, this, false) > 0)
            {
                // use check_now false to prevent infinite recursion
                return get_next_thread(thrd, other_end, false);
            }
            return false;
        }

        // ----------------------------------------------------------------
        /// Schedule the passed thread (put it on the ready work queue)
        void schedule_work(threads::thread_data* thrd, bool other_end)
        {
            ++work_items_count_.data_;
            tqmc_deb.debug(debug::str<>("schedule_work"), "stealing", other_end,
                "D", debug::dec<2>(holder_->domain_index_), "Q",
                debug::dec<3>(queue_index_), "n",
                debug::dec<4>(new_tasks_count_.data_), "w",
                debug::dec<4>(work_items_count_.data_),
                debug::threadinfo<threads::thread_data*>(thrd));
            //
            work_items_.push(thrd, other_end);
#ifdef DEBUG_QUEUE_EXTRA
            debug_queue(work_items_);
#endif
        }

        ///////////////////////////////////////////////////////////////////////
        void on_start_thread(std::size_t /* num_thread */) {}
        void on_stop_thread(std::size_t /* num_thread */) {}
        void on_error(
            std::size_t /* num_thread */, std::exception_ptr const& /* e */)
        {
        }

        // pops all tasks off the queue, prints info and pushes them back on
        // just because we can't iterate over the queue/stack in general
#if defined(DEBUG_QUEUE_EXTRA)
        void debug_queue(work_items_type& q)
        {
            std::unique_lock<std::mutex> Lock(debug_mtx_);
            //
            work_items_type work_items_copy_;
            int x = 0;
            thread_description* thrd;
            tqmc_deb.debug(debug::str<>("debug_queue"), "Pop work items");
            while (q.pop(thrd))
            {
                tqmc_deb.debug(debug::str<>("debug_queue"), x++,
                    debug::threadinfo<threads::thread_data*>(thrd));
                work_items_copy_.push(thrd);
            }
            tqmc_deb.debug(debug::str<>("debug_queue"), "Push work items");
            while (work_items_copy_.pop(thrd))
            {
                q.push(thrd);
                tqmc_deb.debug(debug::str<>("debug_queue"), --x,
                    debug::threadinfo<threads::thread_data*>(thrd));
            }
            tqmc_deb.debug(debug::str<>("debug_queue"), "Finished");
        }
#endif

    public:
        /*const*/ thread_queue_init_parameters parameters_;

        int const queue_index_;

        queue_holder_thread<thread_queue_type>* holder_;

        // count of new tasks to run, separate to new cache line to avoid false
        // sharing

        task_items_type new_task_items_;
        work_items_type work_items_;

        util::cache_line_data<std::atomic<std::int32_t>> new_tasks_count_;
        util::cache_line_data<std::atomic<std::int32_t>> work_items_count_;

#ifdef DEBUG_QUEUE_EXTRA
        std::mutex debug_mtx_;
#endif
    };

}}}    // namespace hpx::threads::policies
