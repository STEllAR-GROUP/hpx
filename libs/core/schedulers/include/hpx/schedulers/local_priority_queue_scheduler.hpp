//  Copyright (c) 2007-2019 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/affinity/affinity_data.hpp>
#include <hpx/assert.hpp>
#include <hpx/concurrency/cache_line_data.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/schedulers/deadlock_detection.hpp>
#include <hpx/schedulers/lockfree_queue_backends.hpp>
#include <hpx/schedulers/thread_queue.hpp>
#include <hpx/threading_base/scheduler_base.hpp>
#include <hpx/threading_base/thread_data.hpp>
#include <hpx/threading_base/thread_num_tss.hpp>
#include <hpx/threading_base/thread_queue_init_parameters.hpp>
#include <hpx/topology/topology.hpp>

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

// TODO: add branch prediction and function heat

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace policies {
    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_CXX11_STD_ATOMIC_128BIT)
    using default_local_priority_queue_scheduler_terminated_queue =
        lockfree_lifo;
#else
    using default_local_priority_queue_scheduler_terminated_queue =
        lockfree_fifo;
#endif

    ///////////////////////////////////////////////////////////////////////////
    /// The local_priority_queue_scheduler maintains exactly one queue of work
    /// items (threads) per OS thread, where this OS thread pulls its next work
    /// from. Additionally it maintains separate queues: several for high
    /// priority threads and one for low priority threads.
    /// High priority threads are executed by the first N OS threads before any
    /// other work is executed. Low priority threads are executed by the last
    /// OS thread whenever no other work is available.
    template <typename Mutex = std::mutex,
        typename PendingQueuing = lockfree_fifo,
        typename StagedQueuing = lockfree_fifo,
        typename TerminatedQueuing =
            default_local_priority_queue_scheduler_terminated_queue>
    class HPX_CORE_EXPORT local_priority_queue_scheduler : public scheduler_base
    {
    public:
        typedef std::false_type has_periodic_maintenance;

        typedef thread_queue<Mutex, PendingQueuing, StagedQueuing,
            TerminatedQueuing>
            thread_queue_type;

        // the scheduler type takes two initialization parameters:
        //    the number of queues
        //    the number of high priority queues
        //    the maxcount per queue
        struct init_parameter
        {
            init_parameter(std::size_t num_queues,
                detail::affinity_data const& affinity_data,
                std::size_t num_high_priority_queues = std::size_t(-1),
                thread_queue_init_parameters thread_queue_init = {},
                char const* description = "local_priority_queue_scheduler")
              : num_queues_(num_queues)
              , num_high_priority_queues_(
                    num_high_priority_queues == std::size_t(-1) ?
                        num_queues :
                        num_high_priority_queues)
              , thread_queue_init_(thread_queue_init)
              , affinity_data_(affinity_data)
              , description_(description)
            {
            }

            init_parameter(std::size_t num_queues,
                detail::affinity_data const& affinity_data,
                char const* description)
              : num_queues_(num_queues)
              , num_high_priority_queues_(num_queues)
              , thread_queue_init_()
              , affinity_data_(affinity_data)
              , description_(description)
            {
            }

            std::size_t num_queues_;
            std::size_t num_high_priority_queues_;
            thread_queue_init_parameters thread_queue_init_;
            detail::affinity_data const& affinity_data_;
            char const* description_;
        };
        typedef init_parameter init_parameter_type;

        local_priority_queue_scheduler(init_parameter_type const& init,
            bool deferred_initialization = true)
          : scheduler_base(
                init.num_queues_, init.description_, init.thread_queue_init_)
          , curr_queue_(0)
          , affinity_data_(init.affinity_data_)
          , num_queues_(init.num_queues_)
          , num_high_priority_queues_(init.num_high_priority_queues_)
          , low_priority_queue_(0, thread_queue_init_)
          , queues_(num_queues_)
          , high_priority_queues_(num_queues_)
          , victim_threads_(num_queues_)
        {
            if (!deferred_initialization)
            {
                HPX_ASSERT(num_queues_ != 0);
                for (std::size_t i = 0; i != num_queues_; ++i)
                {
                    queues_[i].data_ =
                        new thread_queue_type(i, thread_queue_init_);
                }

                HPX_ASSERT(num_high_priority_queues_ != 0);
                HPX_ASSERT(num_high_priority_queues_ <= num_queues_);
                for (std::size_t i = 0; i != num_high_priority_queues_; ++i)
                {
                    high_priority_queues_[i].data_ =
                        new thread_queue_type(i, thread_queue_init_);
                }
                for (std::size_t i = num_high_priority_queues_;
                     i != num_queues_; ++i)
                {
                    high_priority_queues_[i].data_ = nullptr;
                }
            }
        }

        ~local_priority_queue_scheduler() override
        {
            for (std::size_t i = 0; i != num_queues_; ++i)
            {
                delete queues_[i].data_;
            }

            for (std::size_t i = 0; i != num_high_priority_queues_; ++i)
            {
                delete high_priority_queues_[i].data_;
            }
        }

        static std::string get_scheduler_name()
        {
            return "local_priority_queue_scheduler";
        }

#ifdef HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES
        std::uint64_t get_creation_time(bool reset) override
        {
            std::uint64_t time = 0;

            for (std::size_t i = 0; i != num_high_priority_queues_; ++i)
            {
                time +=
                    high_priority_queues_[i].data_->get_creation_time(reset);
            }

            time += low_priority_queue_.get_creation_time(reset);

            for (std::size_t i = 0; i != num_queues_; ++i)
            {
                time += queues_[i].data_->get_creation_time(reset);
            }
            return time;
        }

        std::uint64_t get_cleanup_time(bool reset) override
        {
            std::uint64_t time = 0;

            for (std::size_t i = 0; i != num_high_priority_queues_; ++i)
            {
                time += high_priority_queues_[i].data_->get_cleanup_time(reset);
            }

            time += low_priority_queue_.get_cleanup_time(reset);

            for (std::size_t i = 0; i != num_queues_; ++i)
            {
                time += queues_[i].data_->get_cleanup_time(reset);
            }
            return time;
        }
#endif

#ifdef HPX_HAVE_THREAD_STEALING_COUNTS
        std::int64_t get_num_pending_misses(
            std::size_t num_thread, bool reset) override
        {
            std::int64_t num_pending_misses = 0;
            if (num_thread == std::size_t(-1))
            {
                for (std::size_t i = 0; i != num_high_priority_queues_; ++i)
                {
                    num_pending_misses +=
                        high_priority_queues_[i].data_->get_num_pending_misses(
                            reset);
                }
                for (std::size_t i = 0; i != num_queues_; ++i)
                {
                    num_pending_misses +=
                        queues_[i].data_->get_num_pending_misses(reset);
                }
                num_pending_misses +=
                    low_priority_queue_.get_num_pending_misses(reset);

                return num_pending_misses;
            }

            num_pending_misses +=
                queues_[num_thread].data_->get_num_pending_misses(reset);

            if (num_thread < num_high_priority_queues_)
            {
                num_pending_misses += high_priority_queues_[num_thread]
                                          .data_->get_num_pending_misses(reset);
            }
            if (num_thread == num_queues_ - 1)
            {
                num_pending_misses +=
                    low_priority_queue_.get_num_pending_misses(reset);
            }
            return num_pending_misses;
        }

        std::int64_t get_num_pending_accesses(
            std::size_t num_thread, bool reset) override
        {
            std::int64_t num_pending_accesses = 0;
            if (num_thread == std::size_t(-1))
            {
                for (std::size_t i = 0; i != num_high_priority_queues_; ++i)
                {
                    num_pending_accesses +=
                        high_priority_queues_[i]
                            .data_->get_num_pending_accesses(reset);
                }
                for (std::size_t i = 0; i != num_queues_; ++i)
                {
                    num_pending_accesses +=
                        queues_[i].data_->get_num_pending_accesses(reset);
                }
                num_pending_accesses +=
                    low_priority_queue_.get_num_pending_accesses(reset);

                return num_pending_accesses;
            }

            num_pending_accesses +=
                queues_[num_thread].data_->get_num_pending_accesses(reset);

            if (num_thread < num_high_priority_queues_)
            {
                num_pending_accesses +=
                    high_priority_queues_[num_thread]
                        .data_->get_num_pending_accesses(reset);
            }
            if (num_thread == num_queues_ - 1)
            {
                num_pending_accesses +=
                    low_priority_queue_.get_num_pending_accesses(reset);
            }
            return num_pending_accesses;
        }

        std::int64_t get_num_stolen_from_pending(
            std::size_t num_thread, bool reset) override
        {
            std::int64_t num_stolen_threads = 0;
            if (num_thread == std::size_t(-1))
            {
                for (std::size_t i = 0; i != num_high_priority_queues_; ++i)
                {
                    num_stolen_threads +=
                        high_priority_queues_[i]
                            .data_->get_num_stolen_from_pending(reset);
                }
                for (std::size_t i = 0; i != num_queues_; ++i)
                {
                    num_stolen_threads +=
                        queues_[i].data_->get_num_stolen_from_pending(reset);
                }
                num_stolen_threads +=
                    low_priority_queue_.get_num_stolen_from_pending(reset);

                return num_stolen_threads;
            }

            num_stolen_threads +=
                queues_[num_thread].data_->get_num_stolen_from_pending(reset);

            if (num_thread < num_high_priority_queues_)
            {
                num_stolen_threads +=
                    high_priority_queues_[num_thread]
                        .data_->get_num_stolen_from_pending(reset);
            }
            if (num_thread == num_queues_ - 1)
            {
                num_stolen_threads +=
                    low_priority_queue_.get_num_stolen_from_pending(reset);
            }
            return num_stolen_threads;
        }

        std::int64_t get_num_stolen_to_pending(
            std::size_t num_thread, bool reset) override
        {
            std::int64_t num_stolen_threads = 0;
            if (num_thread == std::size_t(-1))
            {
                for (std::size_t i = 0; i != num_high_priority_queues_; ++i)
                {
                    num_stolen_threads +=
                        high_priority_queues_[i]
                            .data_->get_num_stolen_to_pending(reset);
                }
                for (std::size_t i = 0; i != num_queues_; ++i)
                {
                    num_stolen_threads +=
                        queues_[i].data_->get_num_stolen_to_pending(reset);
                }
                num_stolen_threads +=
                    low_priority_queue_.get_num_stolen_to_pending(reset);

                return num_stolen_threads;
            }

            num_stolen_threads +=
                queues_[num_thread].data_->get_num_stolen_to_pending(reset);

            if (num_thread < num_high_priority_queues_)
            {
                num_stolen_threads +=
                    high_priority_queues_[num_thread]
                        .data_->get_num_stolen_to_pending(reset);
            }
            if (num_thread == num_queues_ - 1)
            {
                num_stolen_threads +=
                    low_priority_queue_.get_num_stolen_to_pending(reset);
            }
            return num_stolen_threads;
        }

        std::int64_t get_num_stolen_from_staged(
            std::size_t num_thread, bool reset) override
        {
            std::int64_t num_stolen_threads = 0;
            if (num_thread == std::size_t(-1))
            {
                for (std::size_t i = 0; i != num_high_priority_queues_; ++i)
                {
                    num_stolen_threads +=
                        high_priority_queues_[i]
                            .data_->get_num_stolen_from_staged(reset);
                }

                for (std::size_t i = 0; i != num_queues_; ++i)
                {
                    num_stolen_threads +=
                        queues_[i].data_->get_num_stolen_from_staged(reset);
                }
                num_stolen_threads +=
                    low_priority_queue_.get_num_stolen_from_staged(reset);

                return num_stolen_threads;
            }

            num_stolen_threads +=
                queues_[num_thread].data_->get_num_stolen_from_staged(reset);

            if (num_thread < num_high_priority_queues_)
            {
                num_stolen_threads +=
                    high_priority_queues_[num_thread]
                        .data_->get_num_stolen_from_staged(reset);
            }
            if (num_thread == num_queues_ - 1)
            {
                num_stolen_threads +=
                    low_priority_queue_.get_num_stolen_from_staged(reset);
            }
            return num_stolen_threads;
        }

        std::int64_t get_num_stolen_to_staged(
            std::size_t num_thread, bool reset) override
        {
            std::int64_t num_stolen_threads = 0;
            if (num_thread == std::size_t(-1))
            {
                for (std::size_t i = 0; i != num_high_priority_queues_; ++i)
                {
                    num_stolen_threads +=
                        high_priority_queues_[i]
                            .data_->get_num_stolen_to_staged(reset);
                }
                for (std::size_t i = 0; i != num_queues_; ++i)
                {
                    num_stolen_threads +=
                        queues_[i].data_->get_num_stolen_to_staged(reset);
                }
                num_stolen_threads +=
                    low_priority_queue_.get_num_stolen_to_staged(reset);

                return num_stolen_threads;
            }

            num_stolen_threads +=
                queues_[num_thread].data_->get_num_stolen_to_staged(reset);

            if (num_thread < num_high_priority_queues_)
            {
                num_stolen_threads +=
                    high_priority_queues_[num_thread]
                        .data_->get_num_stolen_to_staged(reset);
            }
            if (num_thread == num_queues_ - 1)
            {
                num_stolen_threads +=
                    low_priority_queue_.get_num_stolen_to_staged(reset);
            }
            return num_stolen_threads;
        }
#endif

        ///////////////////////////////////////////////////////////////////////
        void abort_all_suspended_threads() override
        {
            for (std::size_t i = 0; i != num_queues_; ++i)
                queues_[i].data_->abort_all_suspended_threads();

            for (std::size_t i = 0; i != num_high_priority_queues_; ++i)
            {
                high_priority_queues_[i].data_->abort_all_suspended_threads();
            }
            low_priority_queue_.abort_all_suspended_threads();
        }

        ///////////////////////////////////////////////////////////////////////
        bool cleanup_terminated(bool delete_all) override
        {
            bool empty = true;

            for (std::size_t i = 0; i != num_queues_; ++i)
            {
                empty =
                    queues_[i].data_->cleanup_terminated(delete_all) && empty;
            }
            if (!delete_all)
                return empty;

            for (std::size_t i = 0; i != num_high_priority_queues_; ++i)
            {
                empty = high_priority_queues_[i].data_->cleanup_terminated(
                            delete_all) &&
                    empty;
            }
            empty = low_priority_queue_.cleanup_terminated(delete_all) && empty;

            return empty;
        }

        bool cleanup_terminated(
            std::size_t num_thread, bool delete_all) override
        {
            bool empty =
                queues_[num_thread].data_->cleanup_terminated(delete_all);
            if (!delete_all)
                return empty;

            if (num_thread < num_high_priority_queues_)
            {
                empty =
                    high_priority_queues_[num_thread].data_->cleanup_terminated(
                        delete_all) &&
                    empty;
            }
            return empty;
        }

        ///////////////////////////////////////////////////////////////////////
        // create a new thread and schedule it if the initial state is equal to
        // pending
        void create_thread(
            thread_init_data& data, thread_id_type* id, error_code& ec) override
        {
            // NOTE: This scheduler ignores NUMA hints.
            std::size_t num_thread =
                data.schedulehint.mode == thread_schedule_hint_mode::thread ?
                data.schedulehint.hint :
                std::size_t(-1);

            if (std::size_t(-1) == num_thread)
            {
                num_thread = curr_queue_++ % num_queues_;
            }
            else if (num_thread >= num_queues_)
            {
                num_thread %= num_queues_;
            }

            std::unique_lock<pu_mutex_type> l;
            num_thread = select_active_pu(l, num_thread);

            data.schedulehint.mode = thread_schedule_hint_mode::thread;
            data.schedulehint.hint = static_cast<std::int16_t>(num_thread);

            // now create the thread
            if (data.priority == thread_priority::high_recursive ||
                data.priority == thread_priority::high ||
                data.priority == thread_priority::boost)
            {
                if (data.priority == thread_priority::boost)
                {
                    data.priority = thread_priority::normal;
                }
                std::size_t num = num_thread % num_high_priority_queues_;

                high_priority_queues_[num].data_->create_thread(data, id, ec);
                return;
            }

            if (data.priority == thread_priority::low)
            {
                low_priority_queue_.create_thread(data, id, ec);
                return;
            }

            HPX_ASSERT(num_thread < num_queues_);
            queues_[num_thread].data_->create_thread(data, id, ec);
        }

        /// Return the next thread to be executed, return false if none is
        /// available
        bool get_next_thread(std::size_t num_thread, bool running,
            threads::thread_data*& thrd, bool enable_stealing) override
        {
            HPX_ASSERT(num_thread < num_queues_);
            thread_queue_type* this_high_priority_queue = nullptr;
            thread_queue_type* this_queue = queues_[num_thread].data_;

            if (num_thread < num_high_priority_queues_)
            {
                this_high_priority_queue =
                    high_priority_queues_[num_thread].data_;
                bool result = this_high_priority_queue->get_next_thread(thrd);

                this_high_priority_queue->increment_num_pending_accesses();
                if (result)
                    return true;
                this_high_priority_queue->increment_num_pending_misses();
            }

            {
                bool result = this_queue->get_next_thread(thrd);

                this_queue->increment_num_pending_accesses();
                if (result)
                    return true;
                this_queue->increment_num_pending_misses();

                bool have_staged = this_queue->get_staged_queue_length(
                                       std::memory_order_relaxed) != 0;

                // Give up, we should have work to convert.
                if (have_staged)
                    return false;
            }

            if (!running)
            {
                return false;
            }

            if (enable_stealing)
            {
                for (std::size_t idx : victim_threads_[num_thread].data_)
                {
                    HPX_ASSERT(idx != num_thread);

                    if (idx < num_high_priority_queues_ &&
                        num_thread < num_high_priority_queues_)
                    {
                        thread_queue_type* q = high_priority_queues_[idx].data_;
                        if (q->get_next_thread(thrd, running))
                        {
                            q->increment_num_stolen_from_pending();
                            this_high_priority_queue
                                ->increment_num_stolen_to_pending();
                            return true;
                        }
                    }

                    if (queues_[idx].data_->get_next_thread(thrd, running))
                    {
                        queues_[idx].data_->increment_num_stolen_from_pending();
                        this_queue->increment_num_stolen_to_pending();
                        return true;
                    }
                }
            }

            return low_priority_queue_.get_next_thread(thrd);
        }

        /// Schedule the passed thread
        void schedule_thread(threads::thread_data* thrd,
            threads::thread_schedule_hint schedulehint,
            bool allow_fallback = false,
            thread_priority priority = thread_priority::normal) override
        {
            // NOTE: This scheduler ignores NUMA hints.
            std::size_t num_thread = std::size_t(-1);
            if (schedulehint.mode == thread_schedule_hint_mode::thread)
            {
                num_thread = schedulehint.hint;
            }
            else
            {
                allow_fallback = false;
            }

            if (std::size_t(-1) == num_thread)
            {
                num_thread = curr_queue_++ % num_queues_;
            }
            else if (num_thread >= num_queues_)
            {
                num_thread %= num_queues_;
            }

            std::unique_lock<pu_mutex_type> l;
            num_thread = select_active_pu(l, num_thread, allow_fallback);

            if (priority == thread_priority::high_recursive ||
                priority == thread_priority::high ||
                priority == thread_priority::boost)
            {
                std::size_t num = num_thread % num_high_priority_queues_;
                high_priority_queues_[num].data_->schedule_thread(thrd);
            }
            else if (priority == thread_priority::low)
            {
                low_priority_queue_.schedule_thread(thrd);
            }
            else
            {
                HPX_ASSERT(num_thread < num_queues_);
                queues_[num_thread].data_->schedule_thread(thrd);
            }
        }

        void schedule_thread_last(threads::thread_data* thrd,
            threads::thread_schedule_hint schedulehint,
            bool allow_fallback = false,
            thread_priority priority = thread_priority::normal) override
        {
            // NOTE: This scheduler ignores NUMA hints.
            std::size_t num_thread = std::size_t(-1);
            if (schedulehint.mode == thread_schedule_hint_mode::thread)
            {
                num_thread = schedulehint.hint;
            }
            else
            {
                allow_fallback = false;
            }

            if (std::size_t(-1) == num_thread)
            {
                num_thread = curr_queue_++ % num_queues_;
            }
            else if (num_thread >= num_queues_)
            {
                num_thread %= num_queues_;
            }

            std::unique_lock<pu_mutex_type> l;
            num_thread = select_active_pu(l, num_thread, allow_fallback);

            if (priority == thread_priority::high_recursive ||
                priority == thread_priority::high ||
                priority == thread_priority::boost)
            {
                std::size_t num = num_thread % num_high_priority_queues_;
                high_priority_queues_[num].data_->schedule_thread(thrd, true);
            }
            else if (priority == thread_priority::low)
            {
                low_priority_queue_.schedule_thread(thrd, true);
            }
            else
            {
                HPX_ASSERT(num_thread < num_queues_);
                queues_[num_thread].data_->schedule_thread(thrd, true);
            }
        }

        /// Destroy the passed thread as it has been terminated
        void destroy_thread(threads::thread_data* thrd) override
        {
            HPX_ASSERT(thrd->get_scheduler_base() == this);
            thrd->get_queue<thread_queue_type>().destroy_thread(thrd);
        }

        ///////////////////////////////////////////////////////////////////////
        // This returns the current length of the queues (work items and new items)
        std::int64_t get_queue_length(
            std::size_t num_thread = std::size_t(-1)) const override
        {
            // Return queue length of one specific queue.
            std::int64_t count = 0;
            if (std::size_t(-1) != num_thread)
            {
                HPX_ASSERT(num_thread < num_queues_);

                if (num_thread < num_high_priority_queues_)
                {
                    count = high_priority_queues_[num_thread]
                                .data_->get_queue_length();
                }
                if (num_thread == num_queues_ - 1)
                    count += low_priority_queue_.get_queue_length();

                return count + queues_[num_thread].data_->get_queue_length();
            }

            // Cumulative queue lengths of all queues.
            for (std::size_t i = 0; i != num_high_priority_queues_; ++i)
            {
                count += high_priority_queues_[i].data_->get_queue_length();
            }
            count += low_priority_queue_.get_queue_length();

            for (std::size_t i = 0; i != num_queues_; ++i)
                count += queues_[i].data_->get_queue_length();

            return count;
        }

        ///////////////////////////////////////////////////////////////////////
        // Queries the current thread count of the queues.
        std::int64_t get_thread_count(
            thread_schedule_state state = thread_schedule_state::unknown,
            thread_priority priority = thread_priority::default_,
            std::size_t num_thread = std::size_t(-1),
            bool /* reset */ = false) const override
        {
            // Return thread count of one specific queue.
            std::int64_t count = 0;
            if (std::size_t(-1) != num_thread)
            {
                HPX_ASSERT(num_thread < num_queues_);

                switch (priority)
                {
                case thread_priority::default_:
                {
                    if (num_thread < num_high_priority_queues_)
                    {
                        count = high_priority_queues_[num_thread]
                                    .data_->get_thread_count(state);
                    }
                    if (num_queues_ - 1 == num_thread)
                        count += low_priority_queue_.get_thread_count(state);

                    return count +
                        queues_[num_thread].data_->get_thread_count(state);
                }

                case thread_priority::low:
                {
                    if (num_queues_ - 1 == num_thread)
                        return low_priority_queue_.get_thread_count(state);
                    break;
                }

                case thread_priority::normal:
                    return queues_[num_thread].data_->get_thread_count(state);

                case thread_priority::boost:
                case thread_priority::high:
                case thread_priority::high_recursive:
                {
                    if (num_thread < num_high_priority_queues_)
                    {
                        return high_priority_queues_[num_thread]
                            .data_->get_thread_count(state);
                    }
                    break;
                }

                default:
                case thread_priority::unknown:
                {
                    HPX_THROW_EXCEPTION(bad_parameter,
                        "local_priority_queue_scheduler::get_thread_count",
                        "unknown thread priority value "
                        "(thread_priority::unknown)");
                    return 0;
                }
                }
                return 0;
            }

            // Return the cumulative count for all queues.
            switch (priority)
            {
            case thread_priority::default_:
            {
                for (std::size_t i = 0; i != num_high_priority_queues_; ++i)
                {
                    count +=
                        high_priority_queues_[i].data_->get_thread_count(state);
                }
                count += low_priority_queue_.get_thread_count(state);

                for (std::size_t i = 0; i != num_queues_; ++i)
                {
                    count += queues_[i].data_->get_thread_count(state);
                }
                break;
            }

            case thread_priority::low:
                return low_priority_queue_.get_thread_count(state);

            case thread_priority::normal:
            {
                for (std::size_t i = 0; i != num_queues_; ++i)
                {
                    count += queues_[i].data_->get_thread_count(state);
                }
                break;
            }

            case thread_priority::boost:
            case thread_priority::high:
            case thread_priority::high_recursive:
            {
                for (std::size_t i = 0; i != num_high_priority_queues_; ++i)
                {
                    count +=
                        high_priority_queues_[i].data_->get_thread_count(state);
                }
                break;
            }

            default:
            case thread_priority::unknown:
            {
                HPX_THROW_EXCEPTION(bad_parameter,
                    "local_priority_queue_scheduler::get_thread_count",
                    "unknown thread priority value "
                    "(thread_priority::unknown)");
                return 0;
            }
            }
            return count;
        }

        // Queries whether a given core is idle
        bool is_core_idle(std::size_t num_thread) const override
        {
            if (num_thread < num_queues_ &&
                queues_[num_thread].data_->get_queue_length() != 0)
            {
                return false;
            }
            if (num_thread < num_high_priority_queues_ &&
                high_priority_queues_[num_thread].data_->get_queue_length() !=
                    0)
            {
                return false;
            }
            return true;
        }

        ///////////////////////////////////////////////////////////////////////
        // Enumerate matching threads from all queues
        bool enumerate_threads(
            util::function_nonser<bool(thread_id_type)> const& f,
            thread_schedule_state state =
                thread_schedule_state::unknown) const override
        {
            bool result = true;
            for (std::size_t i = 0; i != num_high_priority_queues_; ++i)
            {
                result = result &&
                    high_priority_queues_[i].data_->enumerate_threads(f, state);
            }

            result = result && low_priority_queue_.enumerate_threads(f, state);

            for (std::size_t i = 0; i != num_queues_; ++i)
            {
                result =
                    result && queues_[i].data_->enumerate_threads(f, state);
            }
            return result;
        }

#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
        ///////////////////////////////////////////////////////////////////////
        // Queries the current average thread wait time of the queues.
        std::int64_t get_average_thread_wait_time(
            std::size_t num_thread = std::size_t(-1)) const override
        {
            // Return average thread wait time of one specific queue.
            std::uint64_t wait_time = 0;
            std::uint64_t count = 0;
            if (std::size_t(-1) != num_thread)
            {
                HPX_ASSERT(num_thread < num_queues_);

                if (num_thread < num_high_priority_queues_)
                {
                    wait_time = high_priority_queues_[num_thread]
                                    .data_->get_average_thread_wait_time();
                    ++count;
                }

                if (num_queues_ - 1 == num_thread)
                {
                    wait_time +=
                        low_priority_queue_.get_average_thread_wait_time();
                    ++count;
                }

                wait_time +=
                    queues_[num_thread].data_->get_average_thread_wait_time();
                return wait_time / (count + 1);
            }

            // Return the cumulative average thread wait time for all queues.
            for (std::size_t i = 0; i != num_high_priority_queues_; ++i)
            {
                wait_time += high_priority_queues_[i]
                                 .data_->get_average_thread_wait_time();
                ++count;
            }

            wait_time += low_priority_queue_.get_average_thread_wait_time();

            for (std::size_t i = 0; i != num_queues_; ++i)
            {
                wait_time += queues_[i].data_->get_average_thread_wait_time();
                ++count;
            }

            return wait_time / (count + 1);
        }

        ///////////////////////////////////////////////////////////////////////
        // Queries the current average task wait time of the queues.
        std::int64_t get_average_task_wait_time(
            std::size_t num_thread = std::size_t(-1)) const override
        {
            // Return average task wait time of one specific queue.
            std::uint64_t wait_time = 0;
            std::uint64_t count = 0;
            if (std::size_t(-1) != num_thread)
            {
                HPX_ASSERT(num_thread < num_queues_);

                if (num_thread < num_high_priority_queues_)
                {
                    wait_time = high_priority_queues_[num_thread]
                                    .data_->get_average_task_wait_time();
                    ++count;
                }

                if (num_queues_ - 1 == num_thread)
                {
                    wait_time +=
                        low_priority_queue_.get_average_task_wait_time();
                    ++count;
                }

                wait_time +=
                    queues_[num_thread].data_->get_average_task_wait_time();
                return wait_time / (count + 1);
            }

            // Return the cumulative average task wait time for all queues.
            for (std::size_t i = 0; i != num_high_priority_queues_; ++i)
            {
                wait_time += high_priority_queues_[i]
                                 .data_->get_average_task_wait_time();
                ++count;
            }

            wait_time += low_priority_queue_.get_average_task_wait_time();

            for (std::size_t i = 0; i != num_queues_; ++i)
            {
                wait_time += queues_[i].data_->get_average_task_wait_time();
                ++count;
            }

            return wait_time / (count + 1);
        }
#endif

        /// This is a function which gets called periodically by the thread
        /// manager to allow for maintenance tasks to be executed in the
        /// scheduler. Returns true if the OS thread calling this function
        /// has to be terminated (i.e. no more work has to be done).
        bool wait_or_add_new(std::size_t num_thread, bool running,
            std::int64_t& idle_loop_count, bool enable_stealing,
            std::size_t& added) override
        {
            bool result = true;

            added = 0;

            thread_queue_type* this_high_priority_queue = nullptr;
            thread_queue_type* this_queue = queues_[num_thread].data_;

            if (num_thread < num_high_priority_queues_)
            {
                this_high_priority_queue =
                    high_priority_queues_[num_thread].data_;
                result =
                    this_high_priority_queue->wait_or_add_new(running, added) &&
                    result;
                if (0 != added)
                    return result;
            }

            result = this_queue->wait_or_add_new(running, added) && result;
            if (0 != added)
                return result;

            // Check if we have been disabled
            if (!running)
            {
                return true;
            }

            if (enable_stealing)
            {
                for (std::size_t idx : victim_threads_[num_thread].data_)
                {
                    HPX_ASSERT(idx != num_thread);

                    if (idx < num_high_priority_queues_ &&
                        num_thread < num_high_priority_queues_)
                    {
                        thread_queue_type* q = high_priority_queues_[idx].data_;
                        result = this_high_priority_queue->wait_or_add_new(
                                     true, added, q) &&
                            result;

                        if (0 != added)
                        {
                            q->increment_num_stolen_from_staged(added);
                            this_high_priority_queue
                                ->increment_num_stolen_to_staged(added);
                            return result;
                        }
                    }

                    result = this_queue->wait_or_add_new(
                                 true, added, queues_[idx].data_) &&
                        result;

                    if (0 != added)
                    {
                        queues_[idx].data_->increment_num_stolen_from_staged(
                            added);
                        this_queue->increment_num_stolen_to_staged(added);
                        return result;
                    }
                }
            }

#ifdef HPX_HAVE_THREAD_MINIMAL_DEADLOCK_DETECTION
            // no new work is available, are we deadlocked?
            if (HPX_UNLIKELY(get_minimal_deadlock_detection_enabled() &&
                    LHPX_ENABLED(error)))
            {
                bool suspended_only = true;

                for (std::size_t i = 0; suspended_only && i != num_queues_; ++i)
                {
                    suspended_only = queues_[i].data_->dump_suspended_threads(
                        i, idle_loop_count, running);
                }

                if (HPX_UNLIKELY(suspended_only))
                {
                    if (running)
                    {
                        LTM_(error)    //-V128
                            << "queue(" << num_thread << "): "
                            << "no new work available, are we "
                               "deadlocked?";
                    }
                    else
                    {
                        LHPX_CONSOLE_(
                            hpx::util::logging::level::error)    //-V128
                            << "  [TM] "                         //-V128
                            << "queue(" << num_thread << "): "
                            << "no new work available, are we "
                               "deadlocked?\n";
                    }
                }
            }
#else
            HPX_UNUSED(idle_loop_count);
#endif

            if (num_thread == num_queues_ - 1)
            {
                result = low_priority_queue_.wait_or_add_new(running, added) &&
                    result;
            }

            return result;
        }

        ///////////////////////////////////////////////////////////////////////
        void on_start_thread(std::size_t num_thread) override
        {
            hpx::threads::detail::set_local_thread_num_tss(num_thread);
            hpx::threads::detail::set_thread_pool_num_tss(
                parent_pool_->get_pool_id().index());

            if (nullptr == queues_[num_thread].data_)
            {
                queues_[num_thread].data_ =
                    new thread_queue_type(num_thread, thread_queue_init_);

                if (num_thread < num_high_priority_queues_)
                {
                    high_priority_queues_[num_thread].data_ =
                        new thread_queue_type(num_thread, thread_queue_init_);
                }
            }

            // forward this call to all queues etc.
            if (num_thread < num_high_priority_queues_)
            {
                high_priority_queues_[num_thread].data_->on_start_thread(
                    num_thread);
            }

            if (num_thread == num_queues_ - 1)
                low_priority_queue_.on_start_thread(num_thread);

            queues_[num_thread].data_->on_start_thread(num_thread);

            std::size_t num_threads = num_queues_;
            auto const& topo = create_topology();

            // get NUMA domain masks of all queues...
            std::vector<mask_type> numa_masks(num_threads);
            std::vector<mask_type> core_masks(num_threads);
            for (std::size_t i = 0; i != num_threads; ++i)
            {
                std::size_t num_pu = affinity_data_.get_pu_num(i);
                numa_masks[i] = topo.get_numa_node_affinity_mask(num_pu);
                core_masks[i] = topo.get_core_affinity_mask(num_pu);
            }

            // iterate over the number of threads again to determine where to
            // steal from
            std::ptrdiff_t radius =
                std::lround(static_cast<double>(num_threads) / 2.0);
            victim_threads_[num_thread].data_.reserve(num_threads);

            std::size_t num_pu = affinity_data_.get_pu_num(num_thread);
            mask_cref_type pu_mask = topo.get_thread_affinity_mask(num_pu);
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

            auto iterate =
                [&](hpx::util::function_nonser<bool(std::size_t)> f) {
                    // check our neighbors in a radial fashion (left and right
                    // alternating, increasing distance each iteration)
                    std::ptrdiff_t i = 1;
                    for (/**/; i < radius; ++i)
                    {
                        std::ptrdiff_t left =
                            (static_cast<std::ptrdiff_t>(num_thread) - i) %
                            static_cast<std::ptrdiff_t>(num_threads);
                        if (left < 0)
                            left = num_threads + left;

                        if (f(std::size_t(left)))
                        {
                            victim_threads_[num_thread].data_.push_back(
                                static_cast<std::size_t>(left));
                        }

                        std::size_t right = (num_thread + i) % num_threads;
                        if (f(right))
                        {
                            victim_threads_[num_thread].data_.push_back(right);
                        }
                    }
                    if ((num_threads % 2) == 0)
                    {
                        std::size_t right = (num_thread + i) % num_threads;
                        if (f(right))
                        {
                            victim_threads_[num_thread].data_.push_back(right);
                        }
                    }
                };

            // check for threads which share the same core...
            iterate([&](std::size_t other_num_thread) {
                return any(core_mask & core_masks[other_num_thread]);
            });

            // check for threads which share the same NUMA domain...
            iterate([&](std::size_t other_num_thread) {
                return !any(core_mask & core_masks[other_num_thread]) &&
                    any(numa_mask & numa_masks[other_num_thread]);
            });

            // check for the rest and if we are NUMA aware
            if (has_scheduler_mode(policies::enable_stealing_numa) &&
                any(first_mask & pu_mask))
            {
                iterate([&](std::size_t other_num_thread) {
                    return !any(numa_mask & numa_masks[other_num_thread]);
                });
            }
        }

        void on_stop_thread(std::size_t num_thread) override
        {
            if (num_thread < num_high_priority_queues_)
            {
                high_priority_queues_[num_thread].data_->on_stop_thread(
                    num_thread);
            }
            if (num_thread == num_queues_ - 1)
                low_priority_queue_.on_stop_thread(num_thread);

            queues_[num_thread].data_->on_stop_thread(num_thread);
        }

        void on_error(
            std::size_t num_thread, std::exception_ptr const& e) override
        {
            if (num_thread < num_high_priority_queues_)
            {
                high_priority_queues_[num_thread].data_->on_error(
                    num_thread, e);
            }
            if (num_thread == num_queues_ - 1)
                low_priority_queue_.on_error(num_thread, e);

            queues_[num_thread].data_->on_error(num_thread, e);
        }

        void reset_thread_distribution() override
        {
            curr_queue_.store(0, std::memory_order_release);
        }

    protected:
        std::atomic<std::size_t> curr_queue_;

        detail::affinity_data const& affinity_data_;

        std::size_t const num_queues_;
        std::size_t const num_high_priority_queues_;

        thread_queue_type low_priority_queue_;

        std::vector<util::cache_line_data<thread_queue_type*>> queues_;
        std::vector<util::cache_line_data<thread_queue_type*>>
            high_priority_queues_;
        std::vector<util::cache_line_data<std::vector<std::size_t>>>
            victim_threads_;
    };
}}}    // namespace hpx::threads::policies

#include <hpx/config/warnings_suffix.hpp>
