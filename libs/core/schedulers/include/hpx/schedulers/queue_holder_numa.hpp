//  Copyright (c) 2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/schedulers/lockfree_queue_backends.hpp>
#include <hpx/schedulers/thread_queue_mc.hpp>
#include <hpx/threading_base/print.hpp>
#include <hpx/threading_base/thread_data.hpp>
//
#include <hpx/modules/logging.hpp>
#include <hpx/thread_support/unlock_guard.hpp>
#include <hpx/type_support/unused.hpp>
//
#include <hpx/schedulers/queue_holder_thread.hpp>
//
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <list>
#include <map>
#include <unordered_set>
#include <vector>

#include <atomic>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#if !defined(QUEUE_HOLDER_NUMA_DEBUG)
#if defined(HPX_DEBUG)
#define QUEUE_HOLDER_NUMA_DEBUG false
#else
#define QUEUE_HOLDER_NUMA_DEBUG false
#endif
#endif

namespace hpx {
    static hpx::debug::enable_print<QUEUE_HOLDER_NUMA_DEBUG> nq_deb("QH_NUMA");
}

// ------------------------------------------------------------////////
namespace hpx { namespace threads { namespace policies {
    // ----------------------------------------------------------------
    // Helper class to hold a set of thread queue holders.
    // ----------------------------------------------------------------
    template <typename QueueType>
    struct queue_holder_numa
    {
        // ----------------------------------------------------------------
        using ThreadQueue = queue_holder_thread<QueueType>;
        using mutex_type = typename QueueType::mutex_type;

        // ----------------------------------------------------------------
        queue_holder_numa()
          : num_queues_(0)
          , domain_(0)
        {
        }

        // ----------------------------------------------------------------
        ~queue_holder_numa()
        {
            for (auto& q : queues_)
                delete q;
            queues_.clear();
        }

        // ----------------------------------------------------------------
        void init(std::size_t domain, std::size_t queues)
        {
            num_queues_ = queues;
            domain_ = domain;
            // start with unset queue pointers
            queues_.resize(num_queues_, nullptr);
        }

        // ----------------------------------------------------------------
        inline std::size_t size() const
        {
            return queues_.size();
        }

        // ----------------------------------------------------------------
        inline ThreadQueue* thread_queue(std::size_t id) const
        {
            return queues_[id];
        }

        // ----------------------------------------------------------------
        inline bool get_next_thread_HP(std::size_t qidx,
            threads::thread_id_ref_type& thrd, bool stealing,
            bool core_stealing)
        {
            // loop over queues and take one task,
            std::size_t q = qidx;
            for (std::size_t i = 0; i < num_queues_;
                 ++i, q = fast_mod((qidx + i), num_queues_))
            {
                if (queues_[q]->get_next_thread_HP(
                        thrd, (stealing || (i > 0)), i == 0))
                {
                    // clang-format off
                    nq_deb.debug(debug::str<>("HP/BP get_next")
                         , "D", debug::dec<2>(domain_)
                         , "Q",  debug::dec<3>(q)
                         , "Qidx",  debug::dec<3>(qidx)
                         , ((i==0 && !stealing) ? "taken" : "stolen from")
                         , typename ThreadQueue::queue_data_print(queues_[q])
                         , debug::threadinfo<threads::thread_id_ref_type*>(&thrd));
                    // clang-format on
                    return true;
                }
                // if stealing disabled, do not check other queues
                if (!core_stealing)
                    return false;
            }
            return false;
        }

        // ----------------------------------------------------------------
        inline bool get_next_thread(std::size_t qidx,
            threads::thread_id_ref_type& thrd, bool stealing,
            bool core_stealing)
        {
            // loop over queues and take one task,
            // starting with the requested queue
            std::size_t q = qidx;
            for (std::size_t i = 0; i < num_queues_;
                 ++i, q = fast_mod((qidx + i), num_queues_))
            {
                // if we got a thread, return it, only allow stealing if i>0
                if (queues_[q]->get_next_thread(thrd, (stealing || (i > 0))))
                {
                    nq_deb.debug(debug::str<>("get_next"), "D",
                        debug::dec<2>(domain_), "Q", debug::dec<3>(q), "Qidx",
                        debug::dec<3>(qidx),
                        ((i == 0 && !stealing) ? "taken" : "stolen from"),
                        typename ThreadQueue::queue_data_print(queues_[q]),
                        debug::threadinfo<threads::thread_id_ref_type*>(&thrd));
                    return true;
                }
                // if stealing disabled, do not check other queues
                if (!core_stealing)
                    return false;
            }
            return false;
        }

        // ----------------------------------------------------------------
        bool add_new_HP(ThreadQueue* receiver, std::size_t qidx,
            std::size_t& added, bool stealing, bool allow_stealing)
        {
            // loop over queues and take one task,
            std::size_t q = qidx;
            for (std::size_t i = 0; i < num_queues_;
                 ++i, q = fast_mod((qidx + i), num_queues_))
            {
                added =
                    receiver->add_new_HP(64, queues_[q], (stealing || (i > 0)));
                if (added > 0)
                {
                    // clang-format off
                    nq_deb.debug(debug::str<>("HP/BP add_new")
                        , "added", debug::dec<>(added)
                        , "D", debug::dec<2>(domain_)
                        , "Q",  debug::dec<3>(q)
                        , "Qidx",  debug::dec<3>(qidx)
                        , ((i==0 && !stealing) ? "taken" : "stolen from")
                        , typename ThreadQueue::queue_data_print(queues_[q]));
                    // clang-format on
                    return true;
                }
                // if stealing disabled, do not check other queues
                if (!allow_stealing)
                    return false;
            }
            return false;
        }

        // ----------------------------------------------------------------
        bool add_new(ThreadQueue* receiver, std::size_t qidx,
            std::size_t& added, bool stealing, bool allow_stealing)
        {
            // loop over queues and take one task,
            std::size_t q = qidx;
            for (std::size_t i = 0; i < num_queues_;
                 ++i, q = fast_mod((qidx + i), num_queues_))
            {
                added =
                    receiver->add_new(64, queues_[q], (stealing || (i > 0)));
                if (added > 0)
                {
                    // clang-format off
                    nq_deb.debug(debug::str<>("add_new")
                         , "added", debug::dec<>(added)
                         , "D", debug::dec<2>(domain_)
                         , "Q",  debug::dec<3>(q)
                         , "Qidx",  debug::dec<3>(qidx)
                         , ((i==0 && !stealing) ? "taken" : "stolen from")
                         , typename ThreadQueue::queue_data_print(queues_[q]));
                    // clang-format on
                    return true;
                }
                // if stealing disabled, do not check other queues
                if (!allow_stealing)
                    return false;
            }
            return false;
        }

        // ----------------------------------------------------------------
        inline std::size_t get_new_tasks_queue_length() const
        {
            std::size_t len = 0;
            for (auto& q : queues_)
                len += q->new_tasks_count_;
            return len;
        }

        // ----------------------------------------------------------------
        inline std::int64_t get_thread_count(
            thread_schedule_state state = thread_schedule_state::unknown,
            thread_priority priority = thread_priority::default_) const
        {
            std::size_t len = 0;
            for (auto& q : queues_)
                len += q->get_thread_count(state, priority);
            return len;
        }

        // ----------------------------------------------------------------
        void abort_all_suspended_threads()
        {
            for (auto& q : queues_)
                q->abort_all_suspended_threads();
        }

        // ----------------------------------------------------------------
        bool enumerate_threads(hpx::function<bool(thread_id_type)> const& f,
            thread_schedule_state state) const
        {
            bool result = true;
            for (auto& q : queues_)
                result = result && q->enumerate_threads(f, state);
            return result;
        }

        // ----------------------------------------------------------------
        // ----------------------------------------------------------------
        // ----------------------------------------------------------------
        std::size_t num_queues_;
        std::size_t domain_;
        std::vector<ThreadQueue*> queues_;

    public:
        //        // ------------------------------------------------------------
        //        // This returns the current length of the pending queue
        //        std::int64_t get_pending_queue_length() const
        //        {
        //            return work_items_count_;
        //        }

        //        // This returns the current length of the staged queue
        //        std::int64_t get_staged_queue_length(
        //            std::memory_order order = std::memory_order_seq_cst) const
        //        {
        //            return new_tasks_count_.load(order);
        //        }

        void increment_num_pending_misses(std::size_t /* num */ = 1) {}
        void increment_num_pending_accesses(std::size_t /* num */ = 1) {}
        void increment_num_stolen_from_pending(std::size_t /* num */ = 1) {}
        void increment_num_stolen_from_staged(std::size_t /* num */ = 1) {}
        void increment_num_stolen_to_pending(std::size_t /* num */ = 1) {}
        void increment_num_stolen_to_staged(std::size_t /* num */ = 1) {}

        // ------------------------------------------------------------
        bool dump_suspended_threads(std::size_t /* num_thread */,
            std::int64_t& /* idle_loop_count */, bool /* running */)
        {
            return false;
        }

        // ------------------------------------------------------------
        void debug_info()
        {
            for (auto& q : queues_)
                q->debug_info();
        }

        // ------------------------------------------------------------
        void on_start_thread(std::size_t /* num_thread */) {}
        void on_stop_thread(std::size_t /* num_thread */) {}
        void on_error(
            std::size_t /* num_thread */, std::exception_ptr const& /* e */)
        {
        }
    };
}}}    // namespace hpx::threads::policies
