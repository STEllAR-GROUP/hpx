//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADMANAGER_THREAD_QUEUE_AUG_25_2009_0132PM)
#define HPX_THREADMANAGER_THREAD_QUEUE_AUG_25_2009_0132PM

#include <hpx/config.hpp>
#include <hpx/compat/mutex.hpp>
#include <hpx/error_code.hpp>
#include <hpx/runtime/config_entry.hpp>
#include <hpx/runtime/threads/policies/lockfree_queue_backends.hpp>
#include <hpx/runtime/threads/policies/queue_helpers.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/block_profiler.hpp>
#include <hpx/util/function.hpp>
#include <hpx/util/get_and_reset_value.hpp>
#include <hpx/util/high_resolution_clock.hpp>
#include <hpx/util/unlock_guard.hpp>

#ifdef HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES
#   include <hpx/util/tick_counter.hpp>
#endif

#include <boost/atomic.hpp>
#include <boost/exception_ptr.hpp>
#include <boost/lexical_cast.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace std
{
    template <>
    struct hash< ::hpx::threads::thread_id_type>
    {
        typedef ::hpx::threads::thread_id_type argument_type;
        typedef std::size_t result_type;

        std::size_t operator()(::hpx::threads::thread_id_type const& v) const
        {
            std::hash<std::size_t> hasher_;
            return hasher_(reinterpret_cast<std::size_t>(v.get()));
        }
    };
}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace policies
{
#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
    ///////////////////////////////////////////////////////////////////////////
    // We control whether to collect queue wait times using this global bool.
    // It will be set by any of the related performance counters. Once set it
    // stays set, thus no race conditions will occur.
    extern bool maintain_queue_wait_times;
#endif
#ifdef HPX_HAVE_THREAD_MINIMAL_DEADLOCK_DETECTION
    ///////////////////////////////////////////////////////////////////////////
    // We globally control whether to do minimal deadlock detection using this
    // global bool variable. It will be set once by the runtime configuration
    // startup code
    extern bool minimal_deadlock_detection;
#endif

    namespace detail
    {
        inline int get_min_tasks_to_steal_pending()
        {
            static int min_tasks_to_steal_pending =
                boost::lexical_cast<int>(hpx::get_config_entry(
                    "hpx.thread_queue.min_tasks_to_steal_pending", "0"));
            return min_tasks_to_steal_pending;
        }

        inline int get_min_tasks_to_steal_staged()
        {
            static int min_tasks_to_steal_staged =
                boost::lexical_cast<int>(hpx::get_config_entry(
                    "hpx.thread_queue.min_tasks_to_steal_staged", "10"));
            return min_tasks_to_steal_staged;
        }

        inline int get_min_add_new_count()
        {
            static int min_add_new_count =
                boost::lexical_cast<int>(hpx::get_config_entry(
                    "hpx.thread_queue.min_add_new_count", "10"));
            return min_add_new_count;
        }

        inline int get_max_add_new_count()
        {
            static int max_add_new_count =
                boost::lexical_cast<int>(hpx::get_config_entry(
                    "hpx.thread_queue.max_add_new_count", "10"));
            return max_add_new_count;
        }

        inline int get_max_delete_count()
        {
            static int max_delete_count =
                boost::lexical_cast<int>(hpx::get_config_entry(
                    "hpx.thread_queue.max_delete_count", "1000"));
            return max_delete_count;
        }
    }

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
    template <typename Mutex = compat::mutex,
        typename PendingQueuing = lockfree_lifo,
        typename StagedQueuing = lockfree_lifo,
        typename TerminatedQueuing = lockfree_fifo>
    class thread_queue
    {
    private:
        // we use a simple mutex to protect the data members for now
        typedef Mutex mutex_type;

        // don't steal if less than this amount of tasks are left
        int const min_tasks_to_steal_pending;
        int const min_tasks_to_steal_staged;

        // create at least this amount of threads from tasks
        int const min_add_new_count;

        // create not more than this amount of threads from tasks
        int const max_add_new_count;

        // number of terminated threads to discard
        int const max_delete_count;

        // this is the type of a map holding all threads (except depleted ones)
        typedef std::unordered_set<thread_id_type> thread_map_type;

#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
        typedef
            util::tuple<thread_init_data, thread_state_enum, std::uint64_t>
        task_description;
#else
        typedef util::tuple<thread_init_data, thread_state_enum> task_description;
#endif

#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
        typedef util::tuple<thread_data*, std::uint64_t> thread_description;
#else
        typedef thread_data thread_description;
#endif

        typedef typename PendingQueuing::template
            apply<thread_description*>::type work_items_type;

        typedef typename StagedQueuing::template
            apply<task_description*>::type task_items_type;

        typedef typename TerminatedQueuing::template
            apply<thread_data*>::type terminated_items_type;

    protected:
        template <typename Lock>
        void create_thread_object(threads::thread_id_type& thrd,
            threads::thread_init_data& data, thread_state_enum state, Lock& lk)
        {
            HPX_ASSERT(lk.owns_lock());
            HPX_ASSERT(data.stacksize != 0);

            std::ptrdiff_t stacksize = data.stacksize;

            std::list<thread_id_type>* heap = nullptr;

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
            else {
                switch(stacksize) {
                case thread_stacksize_small:
                    heap = &thread_heap_small_;
                    break;

                case thread_stacksize_medium:
                    heap = &thread_heap_medium_;
                    break;

                case thread_stacksize_large:
                    heap = &thread_heap_large_;
                    break;

                case thread_stacksize_huge:
                    heap = &thread_heap_huge_;
                    break;

                default:
                    break;
                }
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
                thrd->rebind(data, state);
            }

            else
            {
                // Allocate a new thread object.
                thrd = threads::thread_data::create(data, memory_pool_, state);
            }
        }

        ///////////////////////////////////////////////////////////////////////
        // add new threads if there is some amount of work available
        thread_id_type add_new(thread_queue* addfrom, bool steal = false)
        {
#ifdef HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES
            util::tick_counter tc(add_new_time_);
#endif

            task_description* task = nullptr;
            if (addfrom->new_tasks_.pop(task, steal))
            {
#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
                if (maintain_queue_wait_times) {
                    addfrom->new_tasks_wait_ +=
                        util::high_resolution_clock::now() - util::get<2>(*task);
                    ++addfrom->new_tasks_wait_count_;
                }
#endif
                --addfrom->new_tasks_count_;

                std::unique_lock<mutex_type> lk(mtx_, std::try_to_lock);
                if (!lk.owns_lock())
                {
                    ++new_tasks_count_;
                    new_tasks_.push(task, steal);
                    return nullptr;            // avoid long wait on lock
                }

                // measure thread creation time
                util::block_profiler_wrapper<add_new_tag> bp(add_new_logger_);

                // create the new thread
                threads::thread_init_data& data = util::get<0>(*task);
                thread_state_enum state = util::get<1>(*task);
                threads::thread_id_type thrd;

                create_thread_object(thrd, data, state, lk);

                delete task;

                // add the new entry to the map of all threads
                std::pair<thread_map_type::iterator, bool> p =
                    thread_map_.insert(thrd);

                if (HPX_UNLIKELY(!p.second)) {
                    lk.unlock();
                    HPX_THROW_EXCEPTION(hpx::out_of_memory,
                        "threadmanager::add_new",
                        "Couldn't add new thread to the thread map");
                    return 0;
                }
                ++thread_map_count_;

                // this thread has to be in the map now
                HPX_ASSERT(thread_map_.find(thrd.get()) != thread_map_.end());
                HPX_ASSERT(thrd->get_pool() == &memory_pool_);

                LTM_(debug) << "add_new: added new task to queues"; //-V128

                HPX_ASSERT(state == pending);
                // only return the thread into the work-items queue if it is in
                // pending state
                return thrd;
            }

            return nullptr;
        }

        void recycle_thread(thread_id_type thrd)
        {
            std::ptrdiff_t stacksize = thrd->get_stack_size();

            if (stacksize == get_stack_size(thread_stacksize_small))
            {
                thread_heap_small_.push_front(thrd);
            }
            else if (stacksize == get_stack_size(thread_stacksize_medium))
            {
                thread_heap_medium_.push_front(thrd);
            }
            else if (stacksize == get_stack_size(thread_stacksize_large))
            {
                thread_heap_large_.push_front(thrd);
            }
            else if (stacksize == get_stack_size(thread_stacksize_huge))
            {
                thread_heap_huge_.push_front(thrd);
            }
            else
            {
                switch(stacksize) {
                case thread_stacksize_small:
                    thread_heap_small_.push_front(thrd);
                    break;

                case thread_stacksize_medium:
                    thread_heap_medium_.push_front(thrd);
                    break;

                case thread_stacksize_large:
                    thread_heap_large_.push_front(thrd);
                    break;

                case thread_stacksize_huge:
                    thread_heap_huge_.push_front(thrd);
                    break;

                default:
                    HPX_ASSERT(false);
                    break;
                }
            }
        }

    public:
        /// This function makes sure all threads which are marked for deletion
        /// (state is terminated) are properly destroyed.
        ///
        /// This returns 'true' if there are no more terminated threads waiting
        /// to be deleted.
        bool cleanup_terminated_locked_helper(bool delete_all = false)
        {
#ifdef HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES
            util::tick_counter tc(cleanup_terminated_time_);
#endif

            if (terminated_items_count_ == 0 && thread_map_.empty())
                return true;

            if (delete_all) {
                // delete all threads
                thread_data* todelete;
                while (terminated_items_.pop(todelete))
                {
                    --terminated_items_count_;

                    // this thread has to be in this map
                    HPX_ASSERT(thread_map_.find(todelete) != thread_map_.end());

                    bool deleted = thread_map_.erase(todelete) != 0;
                    HPX_ASSERT(deleted);
                    if (deleted) {
                        --thread_map_count_;
                        HPX_ASSERT(thread_map_count_ >= 0);
                    }
                }
            }
            else {
                // delete only this many threads
                std::int64_t delete_count =
                    (std::max)(
                        static_cast<std::int64_t>(terminated_items_count_ / 10),
                        static_cast<std::int64_t>(max_delete_count));

                thread_data* todelete;
                while (delete_count && terminated_items_.pop(todelete))
                {
                    --terminated_items_count_;

                    thread_map_type::iterator it = thread_map_.find(todelete);

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

        bool cleanup_terminated_locked(bool delete_all = false)
        {
            return cleanup_terminated_locked_helper(delete_all) &&
                thread_map_.empty();
        }

    public:
        bool cleanup_terminated(bool delete_all = false)
        {
            if (terminated_items_count_ == 0)
                return thread_map_count_ == 0;

            if (delete_all) {
                // do not lock mutex while deleting all threads, do it piece-wise
                bool thread_map_is_empty = false;
                while (true)
                {
                    std::lock_guard<mutex_type> lk(mtx_);
                    if (cleanup_terminated_locked_helper(false))
                    {
                        thread_map_is_empty =
                            (thread_map_count_ == 0) && (new_tasks_count_ == 0);
                        break;
                    }
                }
                return thread_map_is_empty;
            }

            std::lock_guard<mutex_type> lk(mtx_);
            return cleanup_terminated_locked_helper(false) &&
                (thread_map_count_ == 0) && (new_tasks_count_ == 0);
        }

        // The maximum number of active threads this thread manager should
        // create. This number will be a constraint only as long as the work
        // items queue is not empty. Otherwise the number of active threads
        // will be incremented in steps equal to the \a min_add_new_count
        // specified above.
        enum { max_thread_count = 1000 };

        thread_queue(std::size_t queue_num = std::size_t(-1),
                std::size_t max_count = max_thread_count)
          : min_tasks_to_steal_pending(detail::get_min_tasks_to_steal_pending()),
            min_tasks_to_steal_staged(detail::get_min_tasks_to_steal_staged()),
            min_add_new_count(detail::get_min_add_new_count()),
            max_add_new_count(detail::get_max_add_new_count()),
            max_delete_count(detail::get_max_delete_count()),
            thread_map_count_(0),
            work_items_(128, queue_num),
            work_items_count_(0),
#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
            work_items_wait_(0),
            work_items_wait_count_(0),
#endif
            terminated_items_(128),
            terminated_items_count_(0),
            max_count_((0 == max_count)
                      ? static_cast<std::size_t>(max_thread_count)
                      : max_count),
            new_tasks_(128),
            new_tasks_count_(0),
#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
            new_tasks_wait_(0),
            new_tasks_wait_count_(0),
#endif
            memory_pool_(64),
            thread_heap_small_(),
            thread_heap_medium_(),
            thread_heap_large_(),
            thread_heap_huge_(),
#ifdef HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES
            add_new_time_(0),
            cleanup_terminated_time_(0),
#endif
#ifdef HPX_HAVE_THREAD_STEALING_COUNTS
            pending_misses_(0),
            pending_accesses_(0),
            stolen_from_pending_(0),
            stolen_from_staged_(0),
            stolen_to_pending_(0),
            stolen_to_staged_(0),
#endif
            add_new_logger_("thread_queue::add_new")
        {}

        void set_max_count(std::size_t max_count = max_thread_count)
        {
            max_count_ = (0 == max_count) ? max_thread_count : max_count; //-V105
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
        std::int64_t get_queue_length() const
        {
            return work_items_count_ + new_tasks_count_;
        }

        // This returns the current length of the pending queue
        std::int64_t get_pending_queue_length() const
        {
            return work_items_count_;
        }

        // This returns the current length of the staged queue
        std::int64_t get_staged_queue_length(
            boost::memory_order order = boost::memory_order_seq_cst) const
        {
            return new_tasks_count_.load(order);
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
            pending_misses_ += num;
        }

        std::int64_t get_num_pending_accesses(bool reset)
        {
            return util::get_and_reset_value(pending_accesses_, reset);
        }

        void increment_num_pending_accesses(std::size_t num = 1)
        {
            pending_accesses_ += num;
        }

        std::int64_t get_num_stolen_from_pending(bool reset)
        {
            return util::get_and_reset_value(stolen_from_pending_, reset);
        }

        void increment_num_stolen_from_pending(std::size_t num = 1)
        {
            stolen_from_pending_ += num;
        }

        std::int64_t get_num_stolen_from_staged(bool reset)
        {
            return util::get_and_reset_value(stolen_from_staged_, reset);
        }

        void increment_num_stolen_from_staged(std::size_t num = 1)
        {
            stolen_from_staged_ += num;
        }

        std::int64_t get_num_stolen_to_pending(bool reset)
        {
            return util::get_and_reset_value(stolen_to_pending_, reset);
        }

        void increment_num_stolen_to_pending(std::size_t num = 1)
        {
            stolen_to_pending_ += num;
        }

        std::int64_t get_num_stolen_to_staged(bool reset)
        {
            return util::get_and_reset_value(stolen_to_staged_, reset);
        }

        void increment_num_stolen_to_staged(std::size_t num = 1)
        {
            stolen_to_staged_ += num;
        }
#else
        void increment_num_pending_misses(std::size_t num = 1) {}
        void increment_num_pending_accesses(std::size_t num = 1) {}
        void increment_num_stolen_from_pending(std::size_t num = 1) {}
        void increment_num_stolen_from_staged(std::size_t num = 1) {}
        void increment_num_stolen_to_pending(std::size_t num = 1) {}
        void increment_num_stolen_to_staged(std::size_t num = 1) {}
#endif

        ///////////////////////////////////////////////////////////////////////
        // create a new thread and schedule it if the initial state is equal to
        // pending
        void create_thread(thread_init_data& data, thread_id_type* id,
            thread_state_enum initial_state, bool run_now, error_code& ec)
        {
            // thread has not been created yet
            if (id) *id = invalid_thread_id;

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

                    if (HPX_UNLIKELY(!p.second)) {
                        HPX_THROWS_IF(ec, hpx::out_of_memory,
                            "threadmanager::register_thread",
                            "Couldn't add new thread to the map of threads");
                        return;
                    }
                    ++thread_map_count_;

                    // this thread has to be in the map now
                    HPX_ASSERT(thread_map_.find(thrd.get()) != thread_map_.end());
                    HPX_ASSERT(thrd->get_pool() == &memory_pool_);

                    // push the new thread in the pending queue thread
                    if (initial_state == pending)
                        schedule_thread(thrd.get());

                    // return the thread_id of the newly created thread
                    if (id) *id = std::move(thrd);

                    if (&ec != &throws)
                        ec = make_success_code();
                    return;
                }
            }

            // do not execute the work, but register a task description for
            // later thread creation
            ++new_tasks_count_;

#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
            new_tasks_.push(new task_description(
                std::move(data), initial_state,
                util::high_resolution_clock::now()
            ));
#else
            new_tasks_.push(new task_description( //-V106
                std::move(data), initial_state));
#endif
            if (&ec != &throws)
                ec = make_success_code();
        }

        void move_work_items_from(thread_queue *src, std::int64_t count)
        {
            thread_description* trd;
            while (src->work_items_.pop(trd))
            {
                --src->work_items_count_;

#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
                if (maintain_queue_wait_times) {
                    std::uint64_t now = util::high_resolution_clock::now();
                    src->work_items_wait_ += now - util::get<1>(*trd);
                    ++src->work_items_wait_count_;
                    util::get<1>(*trd) = now;
                }
#endif

                bool finished = count == ++work_items_count_;
                work_items_.push(trd);
                if (finished)
                    break;
            }
        }

        void move_task_items_from(thread_queue *src,
            std::int64_t count)
        {
            task_description* task;
            while (src->new_tasks_.pop(task))
            {
                --src->new_tasks_count_;

#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
                if (maintain_queue_wait_times) {
                    std::int64_t now = util::high_resolution_clock::now();
                    src->new_tasks_wait_ += now - util::get<2>(*task);
                    ++src->new_tasks_wait_count_;
                    util::get<2>(*task) = now;
                }
#endif

                bool finish = count == ++new_tasks_count_;
                if (new_tasks_.push(task))
                {
                    if (finish)
                        break;
                }
                else
                {
                    --new_tasks_count_;
                }
            }
        }

        /// Return the next thread to be executed, return false if non is
        /// available
        bool get_next_thread(threads::thread_data*& thrd,
            bool allow_stealing = false, bool steal = false) HPX_HOT
        {
            std::int64_t work_items_count =
                work_items_count_.load(boost::memory_order_relaxed);

            if (allow_stealing && min_tasks_to_steal_pending > work_items_count)
            {
                return false;
            }

#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
            thread_description* tdesc;
            if (0 != work_items_count && work_items_.pop(tdesc, steal))
            {
                --work_items_count_;

                if (maintain_queue_wait_times) {
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
                --work_items_count_;
                return true;
            }
#endif
            return false;
        }

        /// Schedule the passed thread
        void schedule_thread(threads::thread_data* thrd, bool other_end = false)
        {
            ++work_items_count_;
#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
            work_items_.push(new thread_description(
                thrd, util::high_resolution_clock::now()), other_end);
#else
            work_items_.push(thrd, other_end);
#endif
        }

        /// Destroy the passed thread as it has been terminated
        bool destroy_thread(threads::thread_data* thrd, std::int64_t& busy_count)
        {
            if (thrd->get_pool() == &memory_pool_)
            {
                terminated_items_.push(thrd);

                std::int64_t count = ++terminated_items_count_;
                if (count > HPX_MAX_TERMINATED_THREADS)
                {
                    cleanup_terminated(true);   // clean up all terminated threads
                }
                return true;
            }
            return false;
        }

        ///////////////////////////////////////////////////////////////////////
        /// Return the number of existing threads with the given state.
        std::int64_t get_thread_count(thread_state_enum state = unknown) const
        {
            if (terminated == state)
                return terminated_items_count_;

            if (staged == state)
                return new_tasks_count_;

            if (unknown == state)
                return thread_map_count_ + new_tasks_count_ - terminated_items_count_;

            // acquire lock only if absolutely necessary
            std::lock_guard<mutex_type> lk(mtx_);

            std::int64_t num_threads = 0;
            thread_map_type::const_iterator end = thread_map_.end();
            for (thread_map_type::const_iterator it = thread_map_.begin();
                 it != end; ++it)
            {
                if ((*it)->get_state().state() == state)
                    ++num_threads;
            }
            return num_threads;
        }

        ///////////////////////////////////////////////////////////////////////
        void abort_all_suspended_threads()
        {
            std::lock_guard<mutex_type> lk(mtx_);
            thread_map_type::iterator end =  thread_map_.end();
            for (thread_map_type::iterator it = thread_map_.begin();
                 it != end; ++it)
            {
                if ((*it)->get_state().state() == suspended)
                {
                    (*it)->set_state(pending, wait_abort);
                    schedule_thread((*it).get());
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
            ids.reserve(count);

            if (state == unknown)
            {
                std::lock_guard<mutex_type> lk(mtx_);
                thread_map_type::const_iterator end =  thread_map_.end();
                for (thread_map_type::const_iterator it = thread_map_.begin();
                     it != end; ++it)
                {
                    ids.push_back(*it);
                }
            }
            else
            {
                std::lock_guard<mutex_type> lk(mtx_);
                thread_map_type::const_iterator end =  thread_map_.end();
                for (thread_map_type::const_iterator it = thread_map_.begin();
                     it != end; ++it)
                {
                    if ((*it)->get_state().state() == state)
                        ids.push_back(*it);
                }
            }

            // now invoke callback function for all matching threads
            for (thread_id_type const& id : ids)
            {
                if (!f(id))
                    return false;       // stop iteration
            }

            return true;
        }

        /// This is a function which gets called periodically by the thread
        /// manager to allow for maintenance tasks to be executed in the
        /// scheduler. Returns true if the OS thread calling this function
        /// has to be terminated (i.e. no more work has to be done).
        inline thread_id_type wait_or_add_new(bool running,
            std::int64_t& idle_loop_count, thread_queue* addfrom = nullptr,
            bool steal = false) HPX_HOT
        {
            // try to generate new threads from task lists, but only if our
            // own list of threads is empty
            if (0 == work_items_count_.load(boost::memory_order_relaxed))
            {
                // see if we can avoid grabbing the lock below
                if (addfrom)
                {
                    // don't try to steal if there are only a few tasks left on
                    // this queue
                    if (running && min_tasks_to_steal_staged >
                        addfrom->new_tasks_count_.load(boost::memory_order_relaxed))
                    {
                        return nullptr;
                    }
                }
                else
                {
                    if (running &&
                        0 == new_tasks_count_.load(boost::memory_order_relaxed))
                    {
                        return nullptr;
                    }
                    addfrom = this;
                }

                return add_new(addfrom, steal);
            }
            return nullptr;
        }

        ///////////////////////////////////////////////////////////////////////
        bool dump_suspended_threads(std::size_t num_thread
          , std::int64_t& idle_loop_count, bool running)
        {
#ifndef HPX_HAVE_THREAD_MINIMAL_DEADLOCK_DETECTION
            return false;
#else
            if (minimal_deadlock_detection) {
                std::lock_guard<mutex_type> lk(mtx_);
                return detail::dump_suspended_threads(num_thread, thread_map_
                  , idle_loop_count, running);
            }
            return false;
#endif
        }

        ///////////////////////////////////////////////////////////////////////
        void on_start_thread(std::size_t num_thread) {}
        void on_stop_thread(std::size_t num_thread) {}
        void on_error(std::size_t num_thread, boost::exception_ptr const& e) {}

    private:
        mutable mutex_type mtx_;                    ///< mutex protecting the members

        thread_map_type thread_map_;
        ///< mapping of thread id's to HPX-threads
        boost::atomic<std::int64_t> thread_map_count_;
        ///< overall count of work items

        work_items_type work_items_;
        ///< list of active work items
        boost::atomic<std::int64_t> work_items_count_;
        ///< count of active work items

#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
        boost::atomic<std::int64_t> work_items_wait_;
        ///< overall wait time of work items
        boost::atomic<std::int64_t> work_items_wait_count_;
        ///< overall number of work items in queue
#endif
        terminated_items_type terminated_items_;     ///< list of terminated threads
        boost::atomic<std::int64_t> terminated_items_count_;
        ///< count of terminated items

        std::size_t max_count_;
        ///< maximum number of existing HPX-threads
        task_items_type new_tasks_;
        ///< list of new tasks to run

        boost::atomic<std::int64_t> new_tasks_count_;
        ///< count of new tasks to run
#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
        boost::atomic<std::int64_t> new_tasks_wait_;
        ///< overall wait time of new tasks
        boost::atomic<std::int64_t> new_tasks_wait_count_;
        ///< overall number tasks waited
#endif

        threads::thread_pool memory_pool_;          ///< OS thread local memory pools for
                                                    ///< HPX-threads

        std::list<thread_id_type> thread_heap_small_;
        std::list<thread_id_type> thread_heap_medium_;
        std::list<thread_id_type> thread_heap_large_;
        std::list<thread_id_type> thread_heap_huge_;

#ifdef HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES
        std::uint64_t add_new_time_;
        std::uint64_t cleanup_terminated_time_;
#endif

#ifdef HPX_HAVE_THREAD_STEALING_COUNTS
        // # of times our associated worker-thread couldn't find work in work_items
        boost::atomic<std::int64_t> pending_misses_;

        // # of times our associated worker-thread looked for work in work_items
        boost::atomic<std::int64_t> pending_accesses_;

        boost::atomic<std::int64_t> stolen_from_pending_;
        ///< count of work_items stolen from this queue
        boost::atomic<std::int64_t> stolen_from_staged_;
        ///< count of new_tasks stolen from this queue
        boost::atomic<std::int64_t> stolen_to_pending_;
        ///< count of work_items stolen to this queue from other queues
        boost::atomic<std::int64_t> stolen_to_staged_;
        ///< count of new_tasks stolen to this queue from other queues
#endif

        util::block_profiler<add_new_tag> add_new_logger_;
    };
}}}

#endif

