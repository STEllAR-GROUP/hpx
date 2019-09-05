// ------------------------------------------------------------/////////
//  Copyright (c) 2017-2018 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// ------------------------------------------------------------/////////

#if !defined(HPX_THREADMANAGER_SCHEDULING_QUEUE_HELPER)
#define HPX_THREADMANAGER_SCHEDULING_QUEUE_HELPER

#include <hpx/config.hpp>
#include <hpx/runtime/threads/policies/thread_queue_mc.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/policies/lockfree_queue_backends.hpp>
//
#include <hpx/logging.hpp>
#include <hpx/type_support/unused.hpp>
#include <hpx/thread_support/unlock_guard.hpp>
//
#include <hpx/runtime/threads/policies/queue_holder_thread.hpp>
//
#include <cmath>
#include <cstddef>
#include <cstdint>
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

// ------------------------------------------------------------////////
namespace hpx { namespace threads { namespace policies
{
    // ----------------------------------------------------------------
    // Helper class to hold a set of thread queue holders.
    // ----------------------------------------------------------------
    template <typename QueueType>
    struct queue_holder_numa
    {
        // ----------------------------------------------------------------
        using ThreadQueue = queue_holder_thread<QueueType>;

        // ----------------------------------------------------------------
        queue_holder_numa() : num_queues_(0) {}

        // ----------------------------------------------------------------
        ~queue_holder_numa() {
            for (auto &q : queues_) delete q;
            queues_.clear();
        }

        // ----------------------------------------------------------------
        void init(std::size_t queues)
        {
            num_queues_ = queues;
            // start with unset queue pointers
            queues_.resize(num_queues_, nullptr);
        }

        // ----------------------------------------------------------------
        inline std::size_t size() const {
            return queues_.size();
        }

        // ----------------------------------------------------------------
        inline ThreadQueue* thread_queue(std::size_t id) const
        {
            return queues_[id];
        }

        // ----------------------------------------------------------------
        inline bool get_next_thread(std::size_t qidx,
            threads::thread_data*& thrd, bool steal)
        {
            // loop over queues and take one task,
            // starting with the requested queue
            for (std::size_t i=0; i<num_queues_; ++i) {
                std::size_t q = fast_mod((qidx + i), num_queues_);
                // if we got a thread, return it, only allow stealing if i>0
                if (queues_[q]->get_next_thread(thrd, (i>0), steal)) {
//                    if (i>0) queues_[q]->debug("stolen    ", q,
//                        queues_[q]->np_queue_->new_tasks_count_,
//                        queues_[q]->np_queue_->work_items_count_, thrd);
//                    else queues_[q]->debug("normal    ", q,
//                        queues_[q]->np_queue_->new_tasks_count_,
//                        queues_[q]->np_queue_->work_items_count_, thrd);
                    return true;
                }
//                else {
//                    queues_[q]->debug_timed(1, "empty     ", q,
//                        queues_[q]->np_queue_->new_tasks_count_,
//                        queues_[q]->np_queue_->work_items_count_, thrd);
//                }
                // if stealing disabled, do not check other queues
                if (!steal) return false;
            }
            return false;
        }

        // ----------------------------------------------------------------
        inline bool wait_or_add_new(std::size_t id, bool running,
           std::int64_t& idle_loop_count, std::size_t& added, bool steal)
        {
            // loop over all queues and take one task,
            // starting with the requested queue
            // then stealing from any other one in the container
            bool result = true;
            for (std::size_t i=0; i<num_queues_; ++i) {
                std::size_t q = fast_mod((id + i), num_queues_);
                result = queues_[q]->wait_or_add_new(running, idle_loop_count, added)
                        && result;
                if (0 != added) {
                    return result;
                }
                if (!steal) break;
            }
            return result;
        }

        // ----------------------------------------------------------------
        inline std::size_t get_new_tasks_queue_length() const
        {
            std::size_t len = 0;
            for (auto &q : queues_) len += q->new_tasks_count_;
            return len;
        }

        // ----------------------------------------------------------------
        inline std::int64_t get_thread_count(thread_state_enum state = unknown,
            thread_priority priority = thread_priority_default) const
        {
            std::size_t len = 0;
            for (auto &q : queues_) len += q->get_thread_count(state, priority);
            return len;
        }

        // ----------------------------------------------------------------
        void abort_all_suspended_threads()
        {
            for (auto &q : queues_) q->abort_all_suspended_threads();
        }

        // ----------------------------------------------------------------
        bool enumerate_threads(
            util::function_nonser<bool(thread_id_type)> const& f,
            thread_state_enum state) const
        {
            bool result = true;
            for (auto &q : queues_) result = q->enumerate_threads(f, state) && result;
            return result;
        }


        // ----------------------------------------------------------------
        // ----------------------------------------------------------------
        // ----------------------------------------------------------------
        std::size_t                         num_queues_;
        std::vector<ThreadQueue*>           queues_;

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

        void increment_num_pending_misses(std::size_t num = 1) {}
        void increment_num_pending_accesses(std::size_t num = 1) {}
        void increment_num_stolen_from_pending(std::size_t num = 1) {}
        void increment_num_stolen_from_staged(std::size_t num = 1) {}
        void increment_num_stolen_to_pending(std::size_t num = 1) {}
        void increment_num_stolen_to_staged(std::size_t num = 1) {}

        // ------------------------------------------------------------
        bool dump_suspended_threads(std::size_t num_thread
          , std::int64_t& idle_loop_count, bool running)
        {
            return false;
        }

        // ------------------------------------------------------------
        void on_start_thread(std::size_t num_thread) {}
        void on_stop_thread(std::size_t num_thread) {}
        void on_error(std::size_t num_thread, std::exception_ptr const& e) {}
    };

#ifdef HPX_HAVE_THREAD_MINIMAL_DEADLOCK_DETECTION
    // ------------------------------------------------------------////
    // We globally control whether to do minimal deadlock detection using this
    // global bool variable. It will be set once by the runtime configuration
    // startup code
    extern bool minimal_deadlock_detection;
#endif

// ------------------------------------------------------------////////

}}}

#endif // HPX_F0153C92_99B1_4F31_8FA9_4208DB2F26CE

