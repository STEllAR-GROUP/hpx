//  Copyright (c) 2007-2012 Hartmut Kaiser
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
#include <hpx/util/block_profiler.hpp>
#include <hpx/util/lockfree/fifo.hpp>
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
    ///////////////////////////////////////////////////////////////////////////
#ifdef HPX_ACCEL_QUEUING
    // hardware accelerated queuing
    typedef accel::fifo<thread_data *> work_item_queue_type;

    inline void
    enqueue(work_item_queue_type& work_items, thread_data* thrd,
        std::size_t num_thread)
    {
        //printf("enqueue invoked by thread %ld\n", num_thread);
        work_items.enqueue(thrd, num_thread);
    }

    inline bool
    dequeue(work_item_queue_type& work_items, thread_data*& thrd,
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
    typedef boost::lockfree::fifo<thread_data*> work_item_queue_type;

    inline void
    enqueue(work_item_queue_type& work_items, thread_data* thrd,
        std::size_t num_thread)
    {
        work_items.enqueue(thrd);
    }

    inline bool
    dequeue(work_item_queue_type& work_items, thread_data*& thrd,
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
    template <typename Fifo>
    inline void log_fifo_statistics(Fifo const& q, char const* const desc)
    {
        // FIXME
    }

    ///////////////////////////////////////////////////////////////////////////
    template <bool Global>
    class thread_queue
    {
    private:
        // we use a simple mutex to protect the data members for now
        typedef boost::mutex mutex_type;

        // Add this number of threads to the work items queue each time the
        // function \a add_new() is called if the queue is empty.
        enum {
            min_add_new_count = 100,
            max_add_new_count = 100,
            max_delete_count = 100
        };

        // this is the type of the queues of new or pending threads
        typedef work_item_queue_type work_items_type;

        // this is the type of a map holding all threads (except depleted ones)
        typedef boost::ptr_map<
            thread_id_type, thread_data, std::less<thread_id_type>, heap_clone_allocator
        > thread_map_type;

        // this is the type of the queue of new tasks not yet converted to
        // threads
        typedef HPX_STD_TUPLE<thread_init_data, thread_state_enum> task_description;
        typedef boost::lockfree::fifo<task_description*> task_items_type;

        typedef boost::lockfree::fifo<thread_id_type> thread_id_queue_type;

    protected:
        ///////////////////////////////////////////////////////////////////////
        // add new threads if there is some amount of work available
        std::size_t add_new(boost::int64_t add_count, thread_queue* addfrom,
            std::size_t num_thread)
        {
            if (HPX_UNLIKELY(0 == add_count))
                return 0;

            std::size_t added = 0;
            task_description* task = 0;
            while (add_count-- && addfrom->new_tasks_.dequeue(task))
            {
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
                    HPX_THROW_EXCEPTION(hpx::no_success,
                        "threadmanager::add_new",
                        "Couldn't add new thread to the map of threads");
                    return false;
                }

                // transfer ownership to map
                threads::thread_data* t = thrd.release();

                // only insert the thread into the work-items queue if it is in
                // pending state
                if (state == pending) {
                    // pushing the new thread into the pending queue of the
                    // specified thread_queue
                    ++added;
                    enqueue(work_items_, t, num_thread);
                    ++work_items_count_;
                    do_some_work();         // wake up sleeping threads
                }
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
                if (max_count_ >= count + min_add_new_count) {
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
                if (max_count_ >= count + min_add_new_count) {
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
                    max_count_ += min_add_new_count;  // increase max_count
                }
                else {
                    return false;
                }
            }

            std::size_t addednew = add_new(add_count, addfrom, num_thread);
            added += addednew;
            return addednew != 0;
        }

        /// This function makes sure all threads which are marked for deletion
        /// (state is terminated) are properly destroyed
        // FIXME: seg faults at shutdown on Linux.
        bool cleanup_terminated_locked()
        {
            if (thread_map_.empty())
                return true;

            {
                boost::int64_t delete_count = max_delete_count;   // delete only this many threads
                thread_id_type todelete;
                while (delete_count && terminated_items_.dequeue(todelete))
                {
                    if (thread_map_.erase(todelete))
                        --delete_count;
                }
            }
            return thread_map_.empty();
        }

    public:
        bool cleanup_terminated()
        {
            mutex_type::scoped_lock lk(mtx_);
            return cleanup_terminated_locked();
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
            terminated_items_(128),
            max_count_((0 == max_count)
                      ? static_cast<std::size_t>(max_thread_count)
                      : max_count),
            new_tasks_(128),
            new_tasks_count_(0),
            add_new_logger_("thread_queue::add_new")
        {}

        void set_max_count(std::size_t max_count = max_thread_count)
        {
            max_count_ = (0 == max_count) ? max_thread_count : max_count;
        }

        ///////////////////////////////////////////////////////////////////////
        // This returns the current length of the queues (work items and new items)
        boost::int64_t get_queue_length() const
        {
            return work_items_count_ + new_tasks_count_;
        }
        ///////////////////////////////////////////////////////////////////////
        // This returns the current length of the work queue
        boost::int64_t get_work_length() const
        {
            return work_items_count_;
        }
        ///////////////////////////////////////////////////////////////////////
        // This returns the current length of the work queue
        boost::int64_t get_task_length() const
        {
            return new_tasks_count_;
        }

        ///////////////////////////////////////////////////////////////////////
        // create a new thread and schedule it if the initial state is equal to
        // pending
        thread_id_type create_thread(thread_init_data& data,
            thread_state_enum initial_state, bool run_now,
            std::size_t num_thread, error_code& ec)
        {
            if (run_now) {
                mutex_type::scoped_lock lk(mtx_);

                HPX_STD_UNIQUE_PTR<threads::thread_data> thrd (
                    new (memory_pool_) threads::thread_data(
                        data, memory_pool_, initial_state));

                // add a new entry in the map for this thread
                thread_id_type id = thrd->get_thread_id();
                std::pair<thread_map_type::iterator, bool> p =
                    thread_map_.insert(id, thrd.get());

                if (HPX_UNLIKELY(!p.second)) {
                    HPX_THROWS_IF(ec, hpx::no_success,
                        "threadmanager::register_thread",
                        "Couldn't add new thread to the map of threads");
                    return invalid_thread_id;
                }

                // push the new thread in the pending queue thread
                if (initial_state == pending)
                    schedule_thread(thrd.get(), num_thread);

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
            new_tasks_.enqueue(
                new task_description(boost::move(data), initial_state));

            if (&ec != &throws)
                ec = make_success_code();

            return invalid_thread_id;     // thread has not been created yet
        }

        void move_work_items_from(thread_queue<Global> *src,
            boost::int64_t count, std::size_t num_thread)
        {
            threads::thread_data* trd;
            while (dequeue(src->work_items_, trd, num_thread))
            {
                --src->work_items_count_;
                enqueue(work_items_, trd, num_thread);
                {
                    ++work_items_count_;
                  if (count == work_items_count_)
                    break;
                }
            }
        }

        void move_task_items_from(thread_queue<Global> *src,
            boost::int64_t count)
        {
            task_description* td;
            while (src->new_tasks_.dequeue(td))
            {
                --src->new_tasks_count_;
                if(new_tasks_.enqueue(td))
                {
                  ++new_tasks_count_;
                  if (count == new_tasks_count_)
                    break;
                }
            }
        }

        /// Return the next thread to be executed, return false if non is
        /// available
        bool get_next_thread(threads::thread_data*& thrd, std::size_t num_thread)
        {
            if (dequeue(work_items_, thrd, num_thread)) {
                --work_items_count_;
                return true;
            }
            return false;
        }

        /// Schedule the passed thread
        void schedule_thread(threads::thread_data* thrd, std::size_t num_thread)
        {
            enqueue(work_items_, thrd, num_thread);
            ++work_items_count_;
            do_some_work();         // wake up sleeping threads
        }

        /// Destroy the passed thread as it has been terminated
        bool destroy_thread(threads::thread_data* thrd)
        {
            if (thrd->is_created_from(&memory_pool_)) {
                thread_id_type id = thrd->get_thread_id();
                terminated_items_.enqueue(id);
                return true;
            }
            return false;
        }

        ///////////////////////////////////////////////////////////////////////
        /// Return the number of existing threads with the given state.
        boost::int64_t get_thread_count(thread_state_enum state = unknown) const
        {
            mutex_type::scoped_lock lk(mtx_);
            if (unknown == state)
            {
                BOOST_ASSERT(thread_map_.size() <
                    static_cast<std::size_t>((std::numeric_limits<boost::int64_t>::max)()));
                return static_cast<boost::int64_t>(thread_map_.size());
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
            mutex_type::scoped_lock lk(mtx_);
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
            std::size_t& idle_loop_count, std::size_t& added,
            thread_queue* addfrom_ = 0) HPX_HOT;

        /// This function gets called by the threadmanager whenever new work
        /// has been added, allowing the scheduler to reactivate one or more of
        /// possibly idleing OS threads
        void do_some_work()
        {
            if (Global)
                cond_.notify_all();
        }

        ///////////////////////////////////////////////////////////////////////
        bool dump_suspended_threads(std::size_t num_thread
          , std::size_t& idle_loop_count, bool running)
        {
            mutex_type::scoped_lock lk(mtx_);
            return detail::dump_suspended_threads(num_thread, thread_map_
              , idle_loop_count, running);
        }

        ///////////////////////////////////////////////////////////////////////
        void on_start_thread(std::size_t num_thread) {}
        void on_stop_thread(std::size_t num_thread)
        {
            if (0 == num_thread) {
                // print queue statistics
                log_fifo_statistics(work_items_, "thread_queue");
                log_fifo_statistics(terminated_items_, "thread_queue");
                log_fifo_statistics(new_tasks_, "thread_queue");
            }
        }
        void on_error(std::size_t num_thread, boost::exception_ptr const& e) {}

    private:
        mutable mutex_type mtx_;            ///< mutex protecting the members
        boost::condition cond_;             ///< used to trigger some action

        thread_map_type thread_map_;        ///< mapping of thread id's to PX-threads
        work_items_type work_items_;        ///< list of active work items
        boost::atomic<boost::int64_t> work_items_count_; ///< count of active work items
        thread_id_queue_type terminated_items_;   ///< list of terminated threads

        std::size_t max_count_;             ///< maximum number of existing PX-threads
        task_items_type new_tasks_;         ///< list of new tasks to run
        boost::atomic<boost::int64_t> new_tasks_count_; ///< count of new tasks to run

        threads::thread_pool memory_pool_;  ///< OS thread local memory pools for
                                            ///< PX-threads

        util::block_profiler<add_new_tag> add_new_logger_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <>
    inline bool thread_queue<false>::wait_or_add_new(std::size_t num_thread,
        bool running, std::size_t& idle_loop_count, std::size_t& added,
        thread_queue* addfrom_)
    {
        // this thread acquired the lock, do maintenance, if needed
        if (0 == work_items_count_.load(boost::memory_order_relaxed)) {

            // No obvious work has to be done, so a lock won't hurt too much
            // but we lock only one of the threads, assuming this thread
            // will do the maintenance
            //
            // We prefer to exit this function (some kind of very short
            // busy waiting) to blocking on this lock. Locking fails either
            // when a thread is currently doing thread maintenance, which
            // means there might be new work, or the thread owning the lock
            // just falls through to the cleanup work below (no work is available)
            // in which case the current thread (which failed to acquire
            // the lock) will just retry to enter this loop.
            util::try_lock_wrapper<mutex_type> lk(mtx_);
            if (!lk)
                return false;            // avoid long wait on lock

//            LTM_(debug) << "tfunc(" << num_thread << "): queues empty"
//                        << ", threads left: " << thread_map_.size();

            // stop running after all PX threads have been terminated
            thread_queue* addfrom = addfrom_ ? addfrom_ : this;
            bool added_new = add_new_always(added, addfrom, num_thread);
            if (!added_new && !running) {
                // Before exiting each of the OS threads deletes the
                // remaining terminated PX threads
                if (cleanup_terminated_locked()) {
                    // we don't have any registered work items anymore
                    //do_some_work();       // notify possibly waiting threads
                    return true;            // terminate scheduling loop
                }
            }
            else {
                cleanup_terminated_locked();
            }
        }
        return false;
    }

    template <>
    inline bool thread_queue<true>::wait_or_add_new(std::size_t num_thread,
        bool running, std::size_t& idle_loop_count, std::size_t& added,
        thread_queue* addfrom_)
    {
        thread_queue* addfrom = addfrom_ ? addfrom_ : this;

        // only one dedicated OS thread is allowed to acquire the
        // lock for the purpose of inserting the new threads into the
        // thread-map and deleting all terminated threads
        if (0 == num_thread) {
            // first clean up terminated threads
            util::try_lock_wrapper<mutex_type> lk(mtx_);
            if (lk) {
                // no point in having a thread waiting on the lock
                // while another thread is doing the maintenance
                cleanup_terminated_locked();

                // now, add new threads from the queue of task descriptions
                add_new_if_possible(added, addfrom, num_thread);    // calls notify_all
            }
        }

        bool terminate = false;
        while (0 == work_items_count_) {
            // No obvious work has to be done, so a lock won't hurt too much
            // but we lock only one of the threads, assuming this thread
            // will do the maintenance
            //
            // We prefer to exit this while loop (some kind of very short
            // busy waiting) to blocking on this lock. Locking fails either
            // when a thread is currently doing thread maintenance, which
            // means there might be new work, or the thread owning the lock
            // just falls through to the wait below (no work is available)
            // in which case the current thread (which failed to acquire
            // the lock) will just retry to enter this loop.
            util::try_lock_wrapper<mutex_type> lk(mtx_);
            if (!lk)
                break;            // avoid long wait on lock

            // this thread acquired the lock, do maintenance and finally
            // call wait() if no work is available
//            LTM_(debug) << "tfunc(" << num_thread << "): queues empty"
//                        << ", threads left: " << thread_map_.size();

            // stop running after all PX threads have been terminated
            bool added_new = add_new_always(added, addfrom, num_thread);
            if (!added_new && !running) {
                // Before exiting each of the OS threads deletes the
                // remaining terminated PX threads
                if (cleanup_terminated_locked()) {
                    // we don't have any registered work items anymore
                    do_some_work();       // notify possibly waiting threads
                    terminate = true;
                    break;                // terminate scheduling loop
                }

                LTM_(debug) << "tfunc(" << num_thread
                            << "): threadmap not empty";
            }
            else {
                cleanup_terminated_locked();
            }

            if (added_new) {
                // dump list of suspended threads once a second
                if (HPX_UNLIKELY(LHPX_ENABLED(error) && addfrom->new_tasks_.empty())) {
                    detail::dump_suspended_threads(num_thread, thread_map_,
                        idle_loop_count, running);
                }
                break;    // we got work, exit loop
            }

            // Wait until somebody needs some action (if no new work
            // arrived in the meantime).
            // Ask again if queues are empty to avoid race conditions (we
            // needed to lock anyways...), this way no notify_all() gets lost
            if (0 == work_items_count_) {
                LTM_(debug) << "tfunc(" << num_thread
                           << "): queues empty, entering wait";

                // dump list of suspended threads once a second
                if (HPX_UNLIKELY(LHPX_ENABLED(error) && addfrom->new_tasks_.empty())) {
                    detail::dump_suspended_threads(num_thread, thread_map_,
                        idle_loop_count, running);
                }

                namespace bpt = boost::posix_time;
                BOOST_ASSERT(10*idle_loop_count <
                    static_cast<std::size_t>((std::numeric_limits<boost::int64_t>::max)()));
                bool timed_out = !cond_.timed_wait(lk,
                    bpt::microseconds(static_cast<boost::int64_t>(10*idle_loop_count)));

                LTM_(debug) << "tfunc(" << num_thread << "): exiting wait";

                // make sure all pending new threads are properly queued
                // but do that only if the lock has been acquired while
                // exiting the condition.wait() above
                if ((lk && add_new_always(added, addfrom, num_thread)) || timed_out)
                    break;
            }
        }
        return terminate;
    }
}}}

#endif

