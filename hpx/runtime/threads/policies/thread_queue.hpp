//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADMANAGER_THREAD_QUEUE_AUG_25_2009_0132PM)
#define HPX_THREADMANAGER_THREAD_QUEUE_AUG_25_2009_0132PM

#include <map>
#include <memory>

#include <hpx/config.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/get_and_reset_value.hpp>
#include <hpx/util/block_profiler.hpp>
#include <hpx/util/lockfree/fifo.hpp>
#include <hpx/util/high_resolution_clock.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/policies/queue_helpers.hpp>

#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>
#include <boost/atomic.hpp>
#include <boost/ptr_container/ptr_map.hpp>

#ifdef HPX_ACCEL_QUEUING
#   include <hpx/runtime/threads/policies/accel_fifo.hpp>
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace policies
{
#if HPX_THREAD_MAINTAIN_QUEUE_WAITTIME
    ///////////////////////////////////////////////////////////////////////////
    // We control whether to collect queue wait times using this global bool.
    // It will be set by any of the related performance counters. Once set it
    // stays set, thus no race conditions will occur.
    extern bool maintain_queue_wait_times;
#endif
#if HPX_THREAD_MINIMAL_DEADLOCK_DETECTION
    ///////////////////////////////////////////////////////////////////////////
    // We globally control whether to do minimal deadlock detection using this
    // global bool variable. It will be set once by the runtime configuration
    // startup code
    extern bool minimal_deadlock_detection;
#endif

    ///////////////////////////////////////////////////////////////////////////
#ifdef HPX_ACCEL_QUEUING
    // hardware accelerated queuing
    typedef accel::fifo<thread_data *> work_item_queue_type;

    template <typename ThreadData>
    inline void
    enqueue(work_item_queue_type& work_items, ThreadData* thrd,
        std::size_t num_thread)
    {
        //printf("enqueue invoked by thread %ld\n", num_thread);
        work_items.enqueue(thrd, num_thread);
    }

    template <typename ThreadData>
    inline bool
    dequeue(work_item_queue_type& work_items, ThreadData*& thrd,
        std::size_t num_thread)
    {
        return work_items.dequeue(thrd, num_thread);
    }

    inline bool
    empty(work_item_queue_type& work_items, std::size_t num_thread)
    {
        return work_items.empty(num_thread);
    }

#else
    // software queuing
#if HPX_THREAD_MAINTAIN_QUEUE_WAITTIME
    typedef HPX_STD_TUPLE<thread_data*, boost::uint64_t> thread_description;
#else
    typedef thread_data thread_description;
#endif

    typedef boost::lockfree::fifo<thread_description*> work_item_queue_type;

    template <typename ThreadData>
    inline void
    enqueue(work_item_queue_type& work_items, ThreadData* thrd,
        std::size_t num_thread)
    {
        work_items.enqueue(thrd);
    }

    template <typename ThreadData>
    inline bool
    dequeue(work_item_queue_type& work_items, ThreadData*& thrd,
        std::size_t num_thread)
    {
        return work_items.dequeue(thrd);
    }

    inline bool
    empty(work_item_queue_type& work_items, std::size_t num_thread)
    {
        return work_items.empty();
    }

#endif

    ///////////////////////////////////////////////////////////////////////////
    template <typename Mutex = boost::mutex>
    class thread_queue
    {
    private:
        // we use a simple mutex to protect the data members for now
        typedef Mutex mutex_type;

        // Add this number of threads to the work items queue each time the
        // function \a add_new() is called if the queue is empty.
        enum {
            min_add_new_count = 100,
            max_add_new_count = 100,
            max_delete_count = 1000
        };

        // this is the type of the queues of new or pending threads
        typedef work_item_queue_type work_items_type;

        // this is the type of a map holding all threads (except depleted ones)
        typedef boost::ptr_map<
            thread_id_type, thread_data, std::less<thread_id_type>
        > thread_map_type;

        // this is the type of the queue of new tasks not yet converted to
        // threads
#if HPX_THREAD_MAINTAIN_QUEUE_WAITTIME
        typedef
            HPX_STD_TUPLE<thread_init_data, thread_state_enum, boost::uint64_t>
        task_description;
#else
        typedef HPX_STD_TUPLE<thread_init_data, thread_state_enum> task_description;
#endif

        typedef boost::lockfree::fifo<task_description*> task_items_type;
        typedef boost::lockfree::fifo<thread_id_type> thread_id_queue_type;

    protected:
        ///////////////////////////////////////////////////////////////////////
        // add new threads if there is some amount of work available
        std::size_t add_new(boost::int64_t add_count, thread_queue* addfrom,
            std::size_t num_thread)
        {
#if defined(HPX_DEBUG)
            // make sure our mutex is locked at this point
            typename mutex_type::scoped_try_lock l(mtx_);
            BOOST_ASSERT(!l);
#endif

            if (HPX_UNLIKELY(0 == add_count))
                return 0;

            std::size_t added = 0;
            task_description* task = 0;
            while (add_count-- && addfrom->new_tasks_.dequeue(task))
            {
#if HPX_THREAD_MAINTAIN_QUEUE_WAITTIME
                if (maintain_queue_wait_times) {
                    addfrom->new_tasks_wait_ +=
                        util::high_resolution_clock::now() - HPX_STD_GET(2, *task);
                    ++addfrom->new_tasks_wait_count_;
                }
#endif
                --addfrom->new_tasks_count_;

                // measure thread creation time
                util::block_profiler_wrapper<add_new_tag> bp(add_new_logger_);

                // create the new thread
                thread_state_enum state = HPX_STD_GET(1, *task);
                HPX_STD_UNIQUE_PTR<threads::thread_data> thrd (
                    new (memory_pool_) threads::thread_data(
                        HPX_STD_GET(0, *task), memory_pool_, state));

                delete task;

                // add the new entry to the map of all threads
                thread_id_type id = thrd->get_thread_id();
                std::pair<thread_map_type::iterator, bool> p =
                    thread_map_.insert(id, thrd.get());

                if (HPX_UNLIKELY(!p.second)) {
                    HPX_THROW_EXCEPTION(hpx::out_of_memory,
                        "threadmanager::add_new",
                        "Couldn't add new thread to the thread map");
                    return 0;
                }

                // only insert the thread into the work-items queue if it is in
                // pending state
                if (state == pending) {
                    // pushing the new thread into the pending queue of the
                    // specified thread_queue
                    ++added;
                    schedule_thread(thrd.get(), num_thread);
                }

                // this thread has to be in the map now
                BOOST_ASSERT(thread_map_.find(id) != thread_map_.end());
                BOOST_ASSERT(thrd->is_created_from(&memory_pool_));

                // transfer ownership to map
                thrd.release();
            }

            if (added) {
                LTM_(debug) << "add_new: added " << added << " tasks to queues";
            }
            return added;
        }

        ///////////////////////////////////////////////////////////////////////
        bool add_new_if_possible(std::size_t& added, thread_queue* addfrom,
            std::size_t num_thread)
        {
            if (0 == addfrom->new_tasks_count_.load(boost::memory_order_relaxed))
                return false;

            // create new threads from pending tasks (if appropriate)
            boost::int64_t add_count = -1;                  // default is no constraint

            // if the map doesn't hold max_count threads yet add some
            // FIXME: why do we have this test? can max_count_ ever be zero?
            if (HPX_LIKELY(max_count_)) {
                std::size_t count = thread_map_.size();
                if (max_count_ >= count + min_add_new_count) { //-V104
                    BOOST_ASSERT(max_count_ - count <
                        static_cast<std::size_t>((std::numeric_limits<boost::int64_t>::max)()));
                    add_count = static_cast<boost::int64_t>(max_count_ - count);
                    if (add_count < min_add_new_count)
                        add_count = min_add_new_count;
                }
                else {
                    return false;
                }
            }

            std::size_t addednew = add_new(add_count, addfrom, num_thread);
            added += addednew;
            return addednew != 0;
        }

        ///////////////////////////////////////////////////////////////////////
        bool add_new_always(std::size_t& added, thread_queue* addfrom,
            std::size_t num_thread)
        {
            if (0 == addfrom->new_tasks_count_.load(boost::memory_order_relaxed))
                return false;

            // create new threads from pending tasks (if appropriate)
            boost::int64_t add_count = -1;                  // default is no constraint

            // if we are desperate (no work in the queues), add some even if the
            // map holds more than max_count
            if (HPX_LIKELY(max_count_)) {
                std::size_t count = thread_map_.size();
                if (max_count_ >= count + min_add_new_count) { //-V104
                    BOOST_ASSERT(max_count_ - count <
                        static_cast<std::size_t>((std::numeric_limits<boost::int64_t>::max)()));
                    add_count = static_cast<boost::int64_t>(max_count_ - count);
                    if (add_count < min_add_new_count)
                        add_count = min_add_new_count;
                    if (add_count > max_add_new_count)
                        add_count = max_add_new_count;
                }
                else if (empty(work_items_, num_thread)) {
                    add_count = min_add_new_count;    // add this number of threads
                    max_count_ += min_add_new_count;  // increase max_count //-V101
                }
                else {
                    return false;
                }
            }

            std::size_t addednew = add_new(add_count, addfrom, num_thread);
            added += addednew;
            return addednew != 0;
        }

    public:
        /// This function makes sure all threads which are marked for deletion
        /// (state is terminated) are properly destroyed
        bool cleanup_terminated_locked_helper(bool delete_all = false)
        {
            if (thread_map_.empty())
                return false;

            if (terminated_items_count_ == 0)
                return false;

            if (delete_all) {
                // delete all threads
                thread_id_type todelete;
                while (terminated_items_.dequeue(todelete))
                {
                    // this thread has to be in this map
                    BOOST_ASSERT(thread_map_.find(todelete) != thread_map_.end());

                    --terminated_items_count_;
                    bool deleted = thread_map_.erase(todelete) ? true : false;
                    (void)deleted;
                    BOOST_ASSERT(deleted);
                }
                return false;
            }
            else {
                // delete only this many threads
                boost::int64_t delete_count =
                    (std::max)(
                        static_cast<boost::int64_t>(terminated_items_count_ / 10),
                        static_cast<boost::int64_t>(max_delete_count));

                thread_id_type todelete;
                while (delete_count && terminated_items_.dequeue(todelete))
                {
                    // this thread has to be in this map
                    BOOST_ASSERT(thread_map_.find(todelete) != thread_map_.end());

                    --terminated_items_count_;
                    bool deleted = thread_map_.erase(todelete) ? true : false;
                    BOOST_ASSERT(deleted);
                    if (deleted)
                        --delete_count;
                }
                return terminated_items_count_ != 0;
            }
        }

        bool cleanup_terminated_locked(bool delete_all = false)
        {
            cleanup_terminated_locked_helper(delete_all);
            return thread_map_.empty();
        }

    public:
        bool cleanup_terminated(bool delete_all = false)
        {
            if (delete_all) {
                bool thread_map_is_empty = false;
                while (true)
                {
                    typename mutex_type::scoped_lock lk(mtx_);
                    if (/*!lk || */!cleanup_terminated_locked_helper(false))
                    {
                        thread_map_is_empty = thread_map_.empty();
                        break;
                    }
                }
                return thread_map_is_empty;
            }

            typename mutex_type::scoped_lock lk(mtx_);
//            if (!lk) return false;

            cleanup_terminated_locked_helper(false);
            return thread_map_.empty();
        }

        // The maximum number of active threads this thread manager should
        // create. This number will be a constraint only as long as the work
        // items queue is not empty. Otherwise the number of active threads
        // will be incremented in steps equal to the \a min_add_new_count
        // specified above.
        enum { max_thread_count = 1000 };

        thread_queue(std::size_t max_count = max_thread_count)
          : work_items_(128),
            work_items_count_(0),
#if HPX_THREAD_MAINTAIN_QUEUE_WAITTIME
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
#if HPX_THREAD_MAINTAIN_QUEUE_WAITTIME
            new_tasks_wait_(0),
            new_tasks_wait_count_(0),
#endif
            memory_pool_(64),
            stolen_threads_(0),
            add_new_logger_("thread_queue::add_new")
        {}

        void set_max_count(std::size_t max_count = max_thread_count)
        {
            max_count_ = (0 == max_count) ? max_thread_count : max_count; //-V105
        }

        ///////////////////////////////////////////////////////////////////////
        // This returns the current length of the queues (work items and new items)
        boost::int64_t get_queue_length() const
        {
            return work_items_count_ + new_tasks_count_;
        }

        // This returns the current length of the work queue
        boost::int64_t get_work_length() const
        {
            return work_items_count_;
        }

        // This returns the current length of the work queue
        boost::int64_t get_task_length() const
        {
            return new_tasks_count_;
        }

#if HPX_THREAD_MAINTAIN_QUEUE_WAITTIME
        boost::uint64_t get_average_task_wait_time() const
        {
            boost::uint64_t count = new_tasks_wait_count_;
            if (count == 0)
                return 0;
            return new_tasks_wait_ / count;
        }

        boost::uint64_t get_average_thread_wait_time() const
        {
            boost::uint64_t count = work_items_wait_count_;
            if (count == 0)
                return 0;
            return work_items_wait_ / count;
        }
#endif

        std::size_t get_num_stolen_threads(bool reset)
        {
            return util::get_and_reset_value(stolen_threads_, reset);
        }

        void increment_num_stolen_threads(std::size_t num = 1)
        {
            stolen_threads_ += num;
        }

        ///////////////////////////////////////////////////////////////////////
        // create a new thread and schedule it if the initial state is equal to
        // pending
        thread_id_type create_thread(thread_init_data& data,
            thread_state_enum initial_state, bool run_now,
            std::size_t num_thread, error_code& ec)
        {
            if (run_now) {
                typename mutex_type::scoped_lock lk(mtx_);

                HPX_STD_UNIQUE_PTR<threads::thread_data> thrd (
                    new (memory_pool_) threads::thread_data(
                        data, memory_pool_, initial_state));

                // add a new entry in the map for this thread
                thread_id_type id = thrd->get_thread_id();
                std::pair<thread_map_type::iterator, bool> p =
                    thread_map_.insert(id, thrd.get());

                if (HPX_UNLIKELY(!p.second)) {
                    HPX_THROWS_IF(ec, hpx::out_of_memory,
                        "threadmanager::register_thread",
                        "Couldn't add new thread to the map of threads");
                    return invalid_thread_id;
                }

                // push the new thread in the pending queue thread
                if (initial_state == pending)
                    schedule_thread(thrd.get(), num_thread);

                // this thread has to be in the map now
                BOOST_ASSERT(thread_map_.find(id) != thread_map_.end());
                BOOST_ASSERT(thrd->is_created_from(&memory_pool_));

                do_some_work();       // try to execute the new work item
                thrd.release();       // release ownership to the map

                if (&ec != &throws)
                    ec = make_success_code();

                // return the thread_id of the newly created thread
                return id;
            }

            // do not execute the work, but register a task description for
            // later thread creation
            ++new_tasks_count_;

#if HPX_THREAD_MAINTAIN_QUEUE_WAITTIME
            new_tasks_.enqueue(new task_description(
                boost::move(data), initial_state,
                util::high_resolution_clock::now()
            ));
#else
            new_tasks_.enqueue(new task_description(
                boost::move(data), initial_state));
#endif
            if (&ec != &throws)
                ec = make_success_code();

            return invalid_thread_id;     // thread has not been created yet
        }

        void move_work_items_from(thread_queue *src,
            boost::int64_t count, std::size_t num_thread)
        {
            thread_description* trd;
            while (dequeue(src->work_items_, trd, num_thread))
            {
                --src->work_items_count_;

#if HPX_THREAD_MAINTAIN_QUEUE_WAITTIME
                if (maintain_queue_wait_times) {
                    boost::uint64_t now = util::high_resolution_clock::now();
                    src->work_items_wait_ += now - HPX_STD_GET(1, *trd);
                    ++src->work_items_wait_count_;
                    HPX_STD_GET(1, *trd) = now;
                }
#endif

                bool finished = count == ++work_items_count_;
                enqueue(work_items_, trd, num_thread);
                if (finished)
                    break;
            }
        }

        void move_task_items_from(thread_queue *src,
            boost::int64_t count)
        {
            task_description* task;
            while (src->new_tasks_.dequeue(task))
            {
                --src->new_tasks_count_;

#if HPX_THREAD_MAINTAIN_QUEUE_WAITTIME
                if (maintain_queue_wait_times) {
                    boost::int64_t now = util::high_resolution_clock::now();
                    src->new_tasks_wait_ += now - HPX_STD_GET(2, *task);
                    ++src->new_tasks_wait_count_;
                    HPX_STD_GET(2, *task) = now;
                }
#endif

                bool finish = count == ++new_tasks_count_;
                if (new_tasks_.enqueue(task))
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
        bool get_next_thread(threads::thread_data*& thrd, std::size_t num_thread)
        {
#if HPX_THREAD_MAINTAIN_QUEUE_WAITTIME
            thread_description* tdesc;
            if (0 != work_items_count_.load(boost::memory_order_relaxed) &&
                dequeue(work_items_, tdesc, num_thread))
            {
                --work_items_count_;

                if (maintain_queue_wait_times) {
                    work_items_wait_ += util::high_resolution_clock::now() -
                        HPX_STD_GET(1, *tdesc);
                    ++work_items_wait_count_;
                }

                thrd = HPX_STD_GET(0, *tdesc);
                delete tdesc;

                return true;
            }
#else
            if (0 != work_items_count_.load(boost::memory_order_relaxed) &&
                dequeue(work_items_, thrd, num_thread))
            {
                --work_items_count_;
                return true;
            }
#endif
            return false;
        }

        /// Schedule the passed thread
        void schedule_thread(threads::thread_data* thrd, std::size_t num_thread)
        {
            ++work_items_count_;
#if HPX_THREAD_MAINTAIN_QUEUE_WAITTIME
            enqueue(work_items_,
                new thread_description(thrd, util::high_resolution_clock::now()),
                num_thread);
#else
            enqueue(work_items_, thrd, num_thread);
#endif
            do_some_work();         // wake up sleeping threads
        }

        /// Destroy the passed thread as it has been terminated
        bool destroy_thread(threads::thread_data* thrd, boost::int64_t& busy_count)
        {
            if (thrd->is_created_from(&memory_pool_))
            {
                thread_id_type id = thrd->get_thread_id();
                terminated_items_.enqueue(id);

                boost::int64_t count = ++terminated_items_count_;
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
        boost::int64_t get_thread_count(thread_state_enum state = unknown) const
        {
            if (terminated == state)
                return terminated_items_count_;

            if (staged == state)
                return static_cast<boost::int64_t>(new_tasks_count_);

            typename mutex_type::scoped_lock lk(mtx_);
            if (unknown == state)
            {
                BOOST_ASSERT((thread_map_.size() + new_tasks_count_) <
                    static_cast<std::size_t>((std::numeric_limits<boost::int64_t>::max)()));
                return static_cast<boost::int64_t>(thread_map_.size() + new_tasks_count_);
            }

            boost::int64_t num_threads = 0;
            thread_map_type::const_iterator end = thread_map_.end();
            for (thread_map_type::const_iterator it = thread_map_.begin();
                 it != end; ++it)
            {
                if ((*it).second->get_state() == state)
                    ++num_threads;
            }
            return num_threads;
        }

        ///////////////////////////////////////////////////////////////////////
        void abort_all_suspended_threads(std::size_t num_thread)
        {
            typename mutex_type::scoped_lock lk(mtx_);
            thread_map_type::iterator end =  thread_map_.end();
            for (thread_map_type::iterator it = thread_map_.begin();
                 it != end; ++it)
            {
                if ((*it).second->get_state() == suspended) {
                    (*it).second->set_state_ex(wait_abort);
                    (*it).second->set_state(pending);
                    schedule_thread((*it).second, num_thread);
                }
            }
        }

        /// This is a function which gets called periodically by the thread
        /// manager to allow for maintenance tasks to be executed in the
        /// scheduler. Returns true if the OS thread calling this function
        /// has to be terminated (i.e. no more work has to be done).
        inline bool wait_or_add_new(std::size_t num_thread, bool running,
            boost::int64_t& idle_loop_count, std::size_t& added,
            thread_queue* addfrom_ = 0) HPX_HOT
        {
            // try to generate new threads from task lists, but only if our
            // own list of threads is empty
            if (0 == work_items_count_.load(boost::memory_order_relaxed)) {

                // No obvious work has to be done, so a lock won't hurt too much.
                //
                // We prefer to exit this function (some kind of very short
                // busy waiting) to blocking on this lock. Locking fails either
                // when a thread is currently doing thread maintenance, which
                // means there might be new work, or the thread owning the lock
                // just falls through to the cleanup work below (no work is available)
                // in which case the current thread (which failed to acquire
                // the lock) will just retry to enter this loop.
                typename mutex_type::scoped_try_lock lk(mtx_);
                if (!lk)
                    return false;            // avoid long wait on lock

                // stop running after all HPX threads have been terminated
                thread_queue* addfrom = addfrom_ ? addfrom_ : this;
                bool added_new = add_new_always(added, addfrom, num_thread);
                if (!added_new) {
                    // Before exiting each of the OS threads deletes the
                    // remaining terminated PX threads
                    bool canexit = cleanup_terminated_locked(true);
                    if (!running && canexit) {
                        // we don't have any registered work items anymore
                        //do_some_work();       // notify possibly waiting threads
                        return true;            // terminate scheduling loop
                    }
                    return false;
                }
                cleanup_terminated_locked();
            }
            return false;
        }

        /// This function gets called by the thread-manager whenever new work
        /// has been added, allowing the scheduler to reactivate one or more of
        /// possibly idling OS threads
        inline void do_some_work() {}

        ///////////////////////////////////////////////////////////////////////
        bool dump_suspended_threads(std::size_t num_thread
          , boost::int64_t& idle_loop_count, bool running)
        {
#if !HPX_THREAD_MINIMAL_DEADLOCK_DETECTION
            return false;
#else
            if (minimal_deadlock_detection) {
                typename mutex_type::scoped_lock lk(mtx_);
                return detail::dump_suspended_threads(num_thread, thread_map_
                  , idle_loop_count, running);
            }
            return false;
#endif
        }

        ///////////////////////////////////////////////////////////////////////
        void on_start_thread(std::size_t num_thread) {}
        void on_stop_thread(std::size_t num_thread)
        {
            if (0 == num_thread) {
                // print queue statistics
                detail::log_fifo_statistics(work_items_, "thread_queue");
                detail::log_fifo_statistics(terminated_items_, "thread_queue");
                detail::log_fifo_statistics(new_tasks_, "thread_queue");
            }
        }
        void on_error(std::size_t num_thread, boost::exception_ptr const& e) {}

    private:
        mutable mutex_type mtx_;                    ///< mutex protecting the members

        thread_map_type thread_map_;                ///< mapping of thread id's to HPX-threads
        work_items_type work_items_;                ///< list of active work items

        boost::atomic<boost::int64_t> work_items_count_;       ///< count of active work items
#if HPX_THREAD_MAINTAIN_QUEUE_WAITTIME
        boost::atomic<boost::int64_t> work_items_wait_;        ///< overall wait time of work items
        boost::atomic<boost::int64_t> work_items_wait_count_;  ///< overall number of work items in queue
#endif
        thread_id_queue_type terminated_items_;     ///< list of terminated threads
        boost::atomic<boost::int64_t> terminated_items_count_; ///< count of terminated items

        std::size_t max_count_;                     ///< maximum number of existing PX-threads
        task_items_type new_tasks_;                 ///< list of new tasks to run

        boost::atomic<boost::int64_t> new_tasks_count_;        ///< count of new tasks to run
#if HPX_THREAD_MAINTAIN_QUEUE_WAITTIME
        boost::atomic<boost::int64_t> new_tasks_wait_;         ///< overall wait time of new tasks
        boost::atomic<boost::int64_t> new_tasks_wait_count_;   ///< overall number tasks waited
#endif

        threads::thread_pool memory_pool_;          ///< OS thread local memory pools for
                                                    ///< HPX-threads

        boost::atomic<boost::int64_t> stolen_threads_;        ///< count of threads stolen of this queue
        util::block_profiler<add_new_tag> add_new_logger_;
    };
}}}

#endif

