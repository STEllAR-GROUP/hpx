//  Copyright (c) 2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_THREADMANAGER_SCHEDULING_QUEUE_HOLDER_THREAD
#define HPX_THREADMANAGER_SCHEDULING_QUEUE_HOLDER_THREAD

#include <hpx/config.hpp>
#include <hpx/assertion.hpp>
#include <hpx/debugging/print.hpp>
#include <hpx/runtime/threads/policies/lockfree_queue_backends.hpp>
#include <hpx/runtime/threads/policies/thread_queue_init_parameters.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/thread_data_stackful.hpp>
#include <hpx/runtime/threads/thread_data_stackless.hpp>
#include <hpx/type_support/unused.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
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

#if !defined(QUEUE_HOLDER_THREAD_DEBUG)
#ifndef NDEBUG
#define QUEUE_HOLDER_THREAD_DEBUG false
#else
#if !defined(QUEUE_HOLDER_THREAD_DEBUG)
#define QUEUE_HOLDER_THREAD_DEBUG false
#endif
#endif
#endif

namespace hpx {
    static hpx::debug::enable_print<QUEUE_HOLDER_THREAD_DEBUG> tq_deb(
        "QH_THRD");
}

// ------------------------------------------------------------
namespace hpx { namespace threads { namespace policies {

    // apply the modulo operator only when needed
    // (i.e. when the input is greater than the ceiling)
    // NB: the numbers must be positive
    HPX_FORCEINLINE int fast_mod(
        const unsigned int input, const unsigned int ceil)
    {
        return input >= ceil ? input % ceil : input;
    }

    enum : std::size_t
    {
        max_thread_count = 1000
    };
    enum : std::size_t
    {
        round_robin_rollover = 1
    };

    // ----------------------------------------------------------------
    // Helper class to hold a set of queues.
    // ----------------------------------------------------------------
    template <typename QueueType>
    struct queue_holder_thread
    {
        using thread_holder_type = queue_holder_thread<QueueType>;

        // Queues that will store actual work for this thread
        // They might be shared between cores, so we use a pointer to
        // reference them
        QueueType* const bp_queue_;
        QueueType* const hp_queue_;
        QueueType* const np_queue_;
        QueueType* const lp_queue_;

        // these are the domain and local thread queue ids for the container
        const std::uint16_t domain_index_;
        const std::uint16_t queue_index_;
        const std::uint16_t thread_num_;
        // a mask that hold a bit per queue to indicate ownership of the queue
        const std::uint16_t owner_mask_;

        // we must use OS mutexes here because we cannot suspend an HPX
        // thread whilst processing the Queues for that thread, this code
        // is running at the OS level in effect.
        using mutex_type = std::mutex;
        typedef std::unique_lock<mutex_type> scoped_lock;

        // mutex protecting the thread map
        mutable util::cache_line_data<mutex_type> thread_map_mtx_;

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
        const int min_delete_count_;
        const int max_delete_count_;

        // number of terminated threads to collect before cleaning them up
        const int max_terminated_threads_;

        // these ought to be atomic, but if we get a race and assign a thread
        // to queue N instead of N+1 it doesn't really matter

        mutable util::cache_line_data<std::tuple<int, int>> rollover_counters_;

        // ----------------------------------------------------------------
        // ----------------------------------------------------------------
        // ----------------------------------------------------------------

        static util::internal_allocator<threads::thread_data> thread_alloc_;

        typedef util::tuple<thread_init_data, thread_state_enum>
            task_description;

        // -------------------------------------
        // thread map stores every task in this queue set
        // this is the type of a map holding all threads (except depleted/terminated)
        using thread_map_type = std::unordered_set<thread_id_type,
            std::hash<thread_id_type>, std::equal_to<thread_id_type>,
            util::internal_allocator<thread_id_type>>;
        thread_map_type thread_map_;

        mutable util::cache_line_data<std::atomic<std::int32_t>>
            thread_map_count_;

        // -------------------------------------
        // terminated tasks
        // completed tasks that can be reused (stack space etc)
        using terminated_items_type = lockfree_fifo::apply<thread_data*>::type;
        terminated_items_type terminated_items_;
        mutable util::cache_line_data<std::atomic<std::int32_t>>
            terminated_items_count_;

        // ------------------------------------------------------------
        struct queue_mc_print
        {
            const QueueType* const q_;
            explicit queue_mc_print(const QueueType* const q)
              : q_(q)
            {
            }
            //
            friend std::ostream& operator<<(
                std::ostream& os, const queue_mc_print& d)
            {
                os << "n " << debug::dec<3>(d.q_->new_tasks_count_.data_)
                   << " w " << debug::dec<3>(d.q_->work_items_count_.data_);
                return os;
            }
        };

        struct queue_data_print
        {
            const queue_holder_thread* q_;
            explicit queue_data_print(const queue_holder_thread* q)
              : q_(q)
            {
            }
            //
            friend std::ostream& operator<<(
                std::ostream& os, const queue_data_print& d)
            {
                os << "D " << debug::dec<2>(d.q_->domain_index_) << " Q "
                   << debug::dec<3>(d.q_->queue_index_) << " TM "
                   << debug::dec<3>(d.q_->thread_map_count_.data_) << " [BP "
                   << queue_mc_print(d.q_->bp_queue_) << "] [HP "
                   << queue_mc_print(d.q_->hp_queue_) << "] [NP "
                   << queue_mc_print(d.q_->np_queue_) << "] [LP "
                   << queue_mc_print(d.q_->lp_queue_) << "] T "
                   << debug::dec<3>(d.q_->terminated_items_count_.data_);
                return os;
            }
        };
        // ------------------------------------------------------------

        // ----------------------------------------------------------------
        // ----------------------------------------------------------------
        // ----------------------------------------------------------------
        queue_holder_thread(QueueType* bp_queue, QueueType* hp_queue,
            QueueType* np_queue, QueueType* lp_queue, std::uint16_t domain,
            std::uint16_t queue, std::uint16_t thread_num, std::uint16_t owner,
            const thread_queue_init_parameters& init)
          : bp_queue_(bp_queue)
          , hp_queue_(hp_queue)
          , np_queue_(np_queue)
          , lp_queue_(lp_queue)
          , domain_index_(domain)
          , queue_index_(queue)
          , thread_num_(thread_num)
          , owner_mask_(owner)
          , min_delete_count_(static_cast<int>(init.min_delete_count_))
          , max_delete_count_(static_cast<int>(init.max_delete_count_))
          , max_terminated_threads_(
                static_cast<int>(init.max_terminated_threads_))
          , terminated_items_(max_thread_count)
        {
            rollover_counters_.data_ =
                std::make_tuple(queue_index_, round_robin_rollover);
            tq_deb.debug(debug::str<>("construct"), "D",
                debug::dec<2>(domain_index_), "Q", debug::dec<3>(queue_index_),
                "Rollover counter",
                debug::dec<>(std::get<0>(rollover_counters_.data_)),
                debug::dec<>(std::get<1>(rollover_counters_.data_)));
            thread_map_count_.data_ = 0;
            terminated_items_count_.data_ = 0;
            if (bp_queue_)
                bp_queue_->set_holder(this);
            if (hp_queue_)
                hp_queue_->set_holder(this);
            if (np_queue_)
                np_queue_->set_holder(this);
            if (lp_queue_)
                lp_queue_->set_holder(this);
        }

        // ----------------------------------------------------------------
        ~queue_holder_thread()
        {
            if (owns_bp_queue())
                delete bp_queue_;
            if (owns_hp_queue())
                delete hp_queue_;
            if (owns_np_queue())
                delete np_queue_;
            if (owns_lp_queue())
                delete lp_queue_;
            //
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

        // ----------------------------------------------------------------
        inline bool owns_bp_queue() const
        {
            return bp_queue_ && ((owner_mask_ & 1) != 0);
        }

        // ----------------------------------------------------------------
        inline bool owns_hp_queue() const
        {
            return hp_queue_ && ((owner_mask_ & 2) != 0);
        }

        // ----------------------------------------------------------------
        inline bool owns_np_queue() const
        {
            return ((owner_mask_ & 4) != 0);
        }

        // ----------------------------------------------------------------
        inline bool owns_lp_queue() const
        {
            return lp_queue_ && ((owner_mask_ & 8) != 0);
        }

        // ------------------------------------------------------------
        // return the next round robin thread index across all workers
        // using a batching of N per worker before incrementing
        inline unsigned int worker_next(const unsigned int workers) const
        {
            tq_deb.debug(debug::str<>("worker_next"), "Rollover counter ",
                debug::dec<4>(std::get<0>(rollover_counters_.data_)),
                debug::dec<4>(std::get<1>(rollover_counters_.data_)), "workers",
                debug::dec<4>(workers));
            if (--std::get<1>(rollover_counters_.data_) == 0)
            {
                std::get<1>(rollover_counters_.data_) = round_robin_rollover;
                std::get<0>(rollover_counters_.data_) = fast_mod(
                    std::get<0>(rollover_counters_.data_) + 1, workers);
            }
            return std::get<0>(rollover_counters_.data_);
        }

        // ------------------------------------------------------------
        void schedule_thread(threads::thread_data* thrd,
            thread_priority priority, bool other_end = false)
        {
            if (bp_queue_ && (priority == thread_priority_bound))
            {
                tq_deb.debug(debug::str<>("schedule_thread"),
                    queue_data_print(this),
                    debug::threadinfo<threads::thread_data*>(thrd),
                    "queing thread_priority_bound");
                bp_queue_->schedule_work(thrd, other_end);
            }
            else if (hp_queue_ &&
                (priority == thread_priority_high ||
                    priority == thread_priority_high_recursive ||
                    priority == thread_priority_boost))
            {
                tq_deb.debug(debug::str<>("schedule_thread"),
                    queue_data_print(this),
                    debug::threadinfo<threads::thread_data*>(thrd),
                    "queing thread_priority_high");
                hp_queue_->schedule_work(thrd, other_end);
            }
            else if (lp_queue_ && (priority == thread_priority_low))
            {
                tq_deb.debug(debug::str<>("schedule_thread"),
                    queue_data_print(this),
                    debug::threadinfo<threads::thread_data*>(thrd),
                    "queing thread_priority_low");
                lp_queue_->schedule_work(thrd, other_end);
            }
            else
            {
                tq_deb.debug(debug::str<>("schedule_thread"),
                    queue_data_print(this),
                    debug::threadinfo<threads::thread_data*>(thrd),
                    "queing thread_priority_normal");
                np_queue_->schedule_work(thrd, other_end);
            }
        }

        // ----------------------------------------------------------------
        bool cleanup_terminated(std::size_t thread_num, bool delete_all)
        {
            // clang-format off
            if (thread_num!=thread_num_) {
                tq_deb.error(debug::str<>("assertion fail")
                             , "thread_num", thread_num
                             , "thread_num_", thread_num_
                             , "queue_index_", queue_index_
                             , queue_data_print(this)
                             );
            }
            // clang-format on
            HPX_ASSERT(thread_num == thread_num_);

            if (terminated_items_count_.data_.load(std::memory_order_relaxed) ==
                0)
                return true;

            scoped_lock lk(thread_map_mtx_.data_);

            if (delete_all)
            {
                // delete all threads
                thread_data* todelete;
                while (terminated_items_.pop(todelete))
                {
                    --terminated_items_count_.data_;
                    tq_deb.debug(debug::str<>("cleanup"), "delete",
                        queue_data_print(this),
                        debug::threadinfo<thread_data*>(todelete));
                    thread_id_type tid(todelete);
                    remove_from_thread_map(tid, true);
                }
            }
            else
            {
                // delete only this many threads
                std::int64_t delete_count = static_cast<std::int64_t>(
                    terminated_items_count_.data_.load(
                        std::memory_order_relaxed) /
                    2);

                tq_deb.debug(debug::str<>("cleanup"), "recycle", "delete_count",
                    debug::dec<3>(delete_count));

                thread_data* todelete;
                while (delete_count && terminated_items_.pop(todelete))
                {
                    thread_id_type tid(todelete);
                    --terminated_items_count_.data_;
                    remove_from_thread_map(tid, false);
                    tq_deb.debug(debug::str<>("cleanup"), "recycle",
                        queue_data_print(this),
                        debug::threadinfo<thread_id_type*>(&tid));
                    recycle_thread(tid);
                    --delete_count;
                }
            }
            return terminated_items_count_.data_.load(
                       std::memory_order_relaxed) == 0;
        }

        // ----------------------------------------------------------------
        void create_thread(thread_init_data& data, thread_id_type* tid,
            thread_state_enum state, bool run_now, std::uint16_t thread_num,
            error_code& ec)
        {
            if (thread_num != thread_num_)
            {
                run_now = false;
            }

            // create the thread using priority to select queue
            if (data.priority == thread_priority_normal)
            {
                tq_deb.debug(debug::str<>("create_thread "),
                    queue_data_print(this), "thread_priority_normal",
                    "run_now ", run_now);
                return np_queue_->create_thread(data, tid, state, run_now, ec);
            }
            else if (bp_queue_ && (data.priority == thread_priority_bound))
            {
                tq_deb.debug(debug::str<>("create_thread "),
                    queue_data_print(this), "thread_priority_bound", "run_now ",
                    run_now);
                return bp_queue_->create_thread(data, tid, state, run_now, ec);
            }
            else if (hp_queue_ &&
                (data.priority == thread_priority_high ||
                    data.priority == thread_priority_high_recursive ||
                    data.priority == thread_priority_boost))
            {
                // boosted threads return to normal after being queued
                if (data.priority == thread_priority_boost)
                {
                    data.priority = thread_priority_normal;
                }
                tq_deb.debug(debug::str<>("create_thread "),
                    queue_data_print(this), "thread_priority_high", "run_now ",
                    run_now);
                return hp_queue_->create_thread(data, tid, state, run_now, ec);
            }
            else if (lp_queue_ && (data.priority == thread_priority_low))
            {
                tq_deb.debug(debug::str<>("create_thread "),
                    queue_data_print(this), "thread_priority_low", "run_now ",
                    run_now);
                return lp_queue_->create_thread(data, tid, state, run_now, ec);
            }

            tq_deb.error(debug::str<>("create_thread "), "priority?");
            std::terminate();
        }

        // ----------------------------------------------------------------
        // Not thread safe. This function must only be called by the
        // thread that owns the holder object.
        // Creates a thread_data object using information from
        // thread_init_data .
        // If a thread data object is available on one of the heaps
        // it will use that, otherwise a new one is created.
        // Heaps store data ordered/sorted by stack size
        void create_thread_object(threads::thread_id_type& tid,
            threads::thread_init_data& data, thread_state_enum state)
        {
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
                tq_deb.debug(debug::str<>("create_thread_object"), "rebind",
                    queue_data_print(this),
                    debug::threadinfo<threads::thread_id_type*>(&tid));
            }
            else
            {
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
                tq_deb.debug(debug::str<>("create_thread_object"), "new",
                    queue_data_print(this),
                    debug::threadinfo<threads::thread_id_type*>(&tid));
            }
        }

        // ----------------------------------------------------------------
        void recycle_thread(thread_id_type tid)
        {
            std::ptrdiff_t stacksize =
                get_thread_id_data(tid)->get_stack_size();

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
        void add_to_thread_map(threads::thread_id_type tid)
        {
            scoped_lock lk(thread_map_mtx_.data_);

            // add a new entry in the map for this thread
            std::pair<thread_map_type::iterator, bool> p =
                thread_map_.insert(tid);

            if (/*HPX_UNLIKELY*/ (!p.second))
            {
                std::string map_size = std::to_string(thread_map_.size());
                // threads::thread_id_type tid2 = *(p.first);
                // threads::thread_data* td = get_thread_id_data(tid2);
                //std::ostringstream address;
                //address << (void const*) td;
                //std::string prev = address.str();

                tq_deb.error(debug::str<>("map add"),
                    "Couldn't add new thread to the thread map",
                    queue_data_print(this),
                    debug::threadinfo<thread_id_type*>(&tid));

                lk.unlock();
                HPX_THROW_EXCEPTION(hpx::out_of_memory,
                    "queue_holder_thread::add_to_thread_map",
                    "Couldn't add new thread to the thread map " + map_size +
                        " " /*+ prev*/);
            }

            ++thread_map_count_.data_;

            tq_deb.debug(debug::str<>("map add"), queue_data_print(this),
                debug::threadinfo<thread_id_type*>(&tid));

            // this thread has to be in the map now
            HPX_ASSERT(thread_map_.find(tid) != thread_map_.end());
        }

        // ----------------------------------------------------------------
        void remove_from_thread_map(threads::thread_id_type tid, bool dealloc)
        {
            // this thread has to be in this map
            HPX_ASSERT(thread_map_.find(tid) != thread_map_.end());

            HPX_ASSERT(thread_map_count_.data_ >= 0);

            bool deleted = thread_map_.erase(tid) != 0;
            HPX_ASSERT(deleted);
            (void) deleted;

            tq_deb.debug(debug::str<>("map remove"), queue_data_print(this),
                debug::threadinfo<thread_id_type*>(&tid));

            if (dealloc)
            {
                deallocate(get_thread_id_data(tid));
            }
            --thread_map_count_.data_;
        }

        // ----------------------------------------------------------------
        bool get_next_thread_HP(
            threads::thread_data*& thrd, bool stealing, bool check_new) HPX_HOT
        {
            // only take from BP queue if we are not stealing
            if (!stealing && bp_queue_ &&
                bp_queue_->get_next_thread(thrd, stealing, check_new))
            {
                tq_deb.debug(debug::str<>("next_thread_BP"),
                    queue_data_print(this),
                    debug::threadinfo<threads::thread_data*>(thrd),
                    "thread_priority_bound");
                return true;
            }

            if (hp_queue_ &&
                hp_queue_->get_next_thread(thrd, stealing, check_new))
            {
                tq_deb.debug(debug::str<>("get_next_thread_HP"),
                    queue_data_print(this),
                    debug::threadinfo<threads::thread_data*>(thrd),
                    "thread_priority_high");
                return true;
            }
            // if we're out of work in the main queues,
            debug_queues("get_next_thread");
            return false;
        }

        // ----------------------------------------------------------------
        bool get_next_thread(threads::thread_data*& thrd, bool stealing) HPX_HOT
        {
            if (np_queue_->get_next_thread(thrd, stealing))
            {
                tq_deb.debug(debug::str<>("next_thread_NP"),
                    queue_data_print(this),
                    debug::threadinfo<threads::thread_data*>(thrd),
                    "thread_priority_normal");
                return true;
            }

            if (lp_queue_ && lp_queue_->get_next_thread(thrd, stealing))
            {
                tq_deb.debug(debug::str<>("next_thread_LP"),
                    queue_data_print(this),
                    debug::threadinfo<threads::thread_data*>(thrd),
                    "thread_priority_low");
                return true;
            }
            // if we're out of work in the main queues,
            debug_queues("get_next_thread");
            return false;
        }

        // ----------------------------------------------------------------
        std::size_t add_new_HP(
            std::int64_t add_count, thread_holder_type* addfrom, bool stealing)
        {
            std::size_t added;
            if (owns_bp_queue())
            {
                added =
                    bp_queue_->add_new(add_count, addfrom->bp_queue_, stealing);
                if (added > 0)
                    return added;
            }

            if (owns_hp_queue())
            {
                added =
                    hp_queue_->add_new(add_count, addfrom->hp_queue_, stealing);
                if (added > 0)
                    return added;
            }
            return 0;
        }

        // ----------------------------------------------------------------
        std::size_t add_new(
            std::int64_t add_count, thread_holder_type* addfrom, bool stealing)
        {
            std::size_t added;
            if (owns_np_queue())
            {
                added =
                    np_queue_->add_new(add_count, addfrom->np_queue_, stealing);
                if (added > 0)
                    return added;
            }

            if (owns_lp_queue())
            {
                added =
                    lp_queue_->add_new(add_count, addfrom->lp_queue_, stealing);
                if (added > 0)
                    return added;
            }
            //
            static auto an_timed =
                tq_deb.make_timer(1, debug::str<>("add_new"));
            tq_deb.timed(an_timed, "add", debug::dec<3>(add_count),
                "owns bp, hp, np, lp", owns_bp_queue(), owns_hp_queue(),
                owns_np_queue(), owns_lp_queue(), "this",
                queue_data_print(this), "from", queue_data_print(addfrom));
            //
            return 0;
        }

        // ----------------------------------------------------------------
        inline std::size_t get_queue_length()
        {
            std::size_t count = 0;
            count += owns_bp_queue() ? bp_queue_->get_queue_length() : 0;
            count += owns_hp_queue() ? hp_queue_->get_queue_length() : 0;
            count += owns_np_queue() ? np_queue_->get_queue_length() : 0;
            count += owns_lp_queue() ? lp_queue_->get_queue_length() : 0;
            debug_queues("get_queue_length");
            return count;
        }

        // ----------------------------------------------------------------
        inline std::size_t get_thread_count_staged(
            thread_priority priority) const
        {
            // Return thread count of one specific queue.
            switch (priority)
            {
            case thread_priority_default:
            {
                std::int64_t count = 0;
                count +=
                    owns_bp_queue() ? bp_queue_->get_queue_length_staged() : 0;
                count +=
                    owns_hp_queue() ? hp_queue_->get_queue_length_staged() : 0;
                count +=
                    owns_np_queue() ? np_queue_->get_queue_length_staged() : 0;
                count +=
                    owns_lp_queue() ? lp_queue_->get_queue_length_staged() : 0;
                return count;
            }
            case thread_priority_bound:
            {
                return owns_bp_queue() ? bp_queue_->get_queue_length_staged() :
                                         0;
            }
            case thread_priority_low:
            {
                return owns_lp_queue() ? lp_queue_->get_queue_length_staged() :
                                         0;
            }
            case thread_priority_normal:
            {
                return owns_np_queue() ? np_queue_->get_queue_length_staged() :
                                         0;
            }
            case thread_priority_boost:
            case thread_priority_high:
            case thread_priority_high_recursive:
            {
                return owns_hp_queue() ? hp_queue_->get_queue_length_staged() :
                                         0;
            }
            default:
            case thread_priority_unknown:
            {
                HPX_THROW_EXCEPTION(bad_parameter,
                    "queue_holder_thread::get_thread_count_staged",
                    "unknown thread priority value (thread_priority_unknown)");
            }
            }
            return 0;
        }

        // ----------------------------------------------------------------
        inline std::size_t get_thread_count_pending(
            thread_priority priority) const
        {
            // Return thread count of one specific queue.
            switch (priority)
            {
            case thread_priority_default:
            {
                std::int64_t count = 0;
                count +=
                    owns_hp_queue() ? hp_queue_->get_queue_length_pending() : 0;
                count +=
                    owns_np_queue() ? np_queue_->get_queue_length_pending() : 0;
                count +=
                    owns_lp_queue() ? lp_queue_->get_queue_length_pending() : 0;
                return count;
            }
            case thread_priority_bound:
            {
                return owns_bp_queue() ? bp_queue_->get_queue_length_pending() :
                                         0;
            }
            case thread_priority_low:
            {
                return owns_lp_queue() ? lp_queue_->get_queue_length_pending() :
                                         0;
            }
            case thread_priority_normal:
            {
                return owns_np_queue() ? np_queue_->get_queue_length_pending() :
                                         0;
            }
            case thread_priority_boost:
            case thread_priority_high:
            case thread_priority_high_recursive:
            {
                return owns_hp_queue() ? hp_queue_->get_queue_length_pending() :
                                         0;
            }
            default:
            case thread_priority_unknown:
            {
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
                return terminated_items_count_.data_.load(
                    std::memory_order_relaxed);

            if (staged == state)
                return get_thread_count_staged(priority);

            if (pending == state)
                return get_thread_count_pending(priority);

            if (unknown == state)
                return thread_map_count_.data_.load(std::memory_order_relaxed) +
                    get_thread_count_staged(priority) -
                    terminated_items_count_.data_.load(
                        std::memory_order_relaxed);

            // acquire lock only if absolutely necessary
            scoped_lock lk(thread_map_mtx_.data_);

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
        void destroy_thread(
            threads::thread_data* thrd, std::uint16_t thread_num, bool xthread)
        {
            // the thread must be destroyed by the same queue holder that created it
            HPX_ASSERT(&thrd->get_queue<queue_holder_thread>() == this);
            //
            tq_deb.debug(debug::str<>("destroy"), "terminated_items push",
                "xthread", xthread, queue_data_print(this),
                debug::threadinfo<threads::thread_data*>(thrd));
            terminated_items_.push(thrd);
            std::int64_t count = ++terminated_items_count_.data_;

            if (!xthread && (count > max_terminated_threads_))
            {
                cleanup_terminated(
                    thread_num, false);    // clean up all terminated threads
            }
        }

        // ------------------------------------------------------------
        void abort_all_suspended_threads()
        {
            scoped_lock lk(thread_map_mtx_.data_);
            thread_map_type::iterator end = thread_map_.end();
            for (thread_map_type::iterator it = thread_map_.begin(); it != end;
                 ++it)
            {
                if (get_thread_id_data(*it)->get_state().state() == suspended)
                {
                    get_thread_id_data(*it)->set_state(pending, wait_abort);
                    // np queue always exists so use that as priority doesn't matter
                    np_queue_->schedule_work(get_thread_id_data(*it), true);
                }
            }
            throw std::runtime_error("This function needs to be reimplemented");
        }

        // ------------------------------------------------------------
        bool enumerate_threads(
            util::function_nonser<bool(thread_id_type)> const& f,
            thread_state_enum state = unknown) const
        {
            std::uint64_t count = thread_map_count_.data_;
            if (state == terminated)
            {
                count = terminated_items_count_.data_;
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
                scoped_lock lk(thread_map_mtx_.data_);
                thread_map_type::const_iterator end = thread_map_.end();
                for (thread_map_type::const_iterator it = thread_map_.begin();
                     it != end; ++it)
                {
                    tids.push_back(*it);
                }
            }
            else
            {
                scoped_lock lk(thread_map_mtx_.data_);
                thread_map_type::const_iterator end = thread_map_.end();
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
                    return false;    // stop iteration
            }

            return true;
        }

        // ------------------------------------------------------------
        void debug_info()
        {
            tq_deb.debug(debug::str<>("details"), "owner_mask_",
                debug::bin<8>(owner_mask_), "D", debug::dec<2>(domain_index_),
                "Q", debug::dec<3>(queue_index_));
            tq_deb.debug(debug::str<>("bp_queue"),
                debug::hex<12, void*>(bp_queue_), "holder",
                debug::hex<12, void*>(
                    bp_queue_->holder_ ? bp_queue_->holder_ : nullptr));
            tq_deb.debug(debug::str<>("hp_queue"),
                debug::hex<12, void*>(hp_queue_), "holder",
                debug::hex<12, void*>(
                    hp_queue_->holder_ ? hp_queue_->holder_ : nullptr));
            tq_deb.debug(debug::str<>("np_queue"),
                debug::hex<12, void*>(np_queue_), "holder",
                debug::hex<12, void*>(
                    np_queue_->holder_ ? np_queue_->holder_ : nullptr));
            tq_deb.debug(debug::str<>("lp_queue"),
                debug::hex<12, void*>(lp_queue_), "holder",
                debug::hex<12, void*>(
                    lp_queue_->holder_ ? lp_queue_->holder_ : nullptr));
        }

        // ------------------------------------------------------------
        void debug_queues(const char* prefix)
        {
            static auto deb_queues =
                tq_deb.make_timer(1, debug::str<>("debug_queues"));
            //
            tq_deb.timed(deb_queues, prefix, queue_data_print(this));
        }
    };

    template <typename QueueType>
    util::internal_allocator<threads::thread_data>
        queue_holder_thread<QueueType>::thread_alloc_;

#ifdef HPX_HAVE_THREAD_MINIMAL_DEADLOCK_DETECTION
    // ------------------------------------------------------------
    // We globally control whether to do minimal deadlock detection using this
    // global bool variable. It will be set once by the runtime configuration
    // startup code
    extern bool minimal_deadlock_detection;
#endif

    // ------------------------------------------------------------

}}}    // namespace hpx::threads::policies

#endif
