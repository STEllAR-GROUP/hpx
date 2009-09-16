//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADMANAGER_THREAD_QUEUE_AUG_25_2009_0132PM)
#define HPX_THREADMANAGER_THREAD_QUEUE_AUG_25_2009_0132PM

#include <map>
#include <memory>

#include <hpx/config.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/block_profiler.hpp>
#include <hpx/runtime/threads/thread.hpp>

#include <boost/thread.hpp>
#include <boost/thread/condition.hpp>
#include <boost/bind.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/lockfree/fifo.hpp>
#include <boost/ptr_container/ptr_map.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace policies
{
    struct add_new_tag {};

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        // This try_lock_wrapper is essentially equivalent to the template 
        // boost::thread::detail::try_lock_wrapper with the one exception, that
        // the lock() function always calls base::try_lock(). This allows us to 
        // skip lock acquisition while exiting the condition variable.
        template<typename Mutex>
        class try_lock_wrapper
          : public boost::detail::try_lock_wrapper<Mutex>
        {
            typedef boost::detail::try_lock_wrapper<Mutex> base;

        public:
            explicit try_lock_wrapper(Mutex& m):
                base(m, boost::try_to_lock)
            {}

            void lock()
            {
                base::try_lock();       // this is different
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // debug helper function, logs all suspended threads
        inline void dump_suspended_threads(
            boost::ptr_map<thread_id_type, threads::thread>& tm)
        {
            typedef boost::ptr_map<thread_id_type, threads::thread> thread_map_type;

            thread_map_type::const_iterator end = tm.end();
            for (thread_map_type::const_iterator it = tm.begin(); it != end; ++it)
            {
                threads::thread const* thrd = (*it).second;
                thread_state state = thrd->get_state();
                if (suspended == state)
                {
                    LTM_(info) << "suspended thread(" << (*it).first << "): "
                               << thrd->get_description();
                }
                else if (marked_for_suspension == state)
                {
                    LTM_(info) << "marked_for_suspension thread(" 
                               << (*it).first << "): "
                               << thrd->get_description();
                }
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    class thread_queue
    {
    private:
        // Add this number of threads to the work items queue each time the 
        // function \a add_new() is called if the queue is empty.
        enum { 
            min_add_new_count = 100, 
            max_add_new_count = 100,
            max_delete_count = 100
        };

        // we use a simple mutex to protect the data members for now
        typedef boost::mutex mutex_type;

        // this is the type of the queues of new or pending threads
        typedef boost::lockfree::fifo<thread*> work_items_type;

        // this is the type of a map holding all threads (except depleted ones)
        typedef boost::ptr_map<thread_id_type, thread> thread_map_type;

        // this is the type of the queue of new tasks not yet converted to
        // threads
        typedef boost::tuple<
            boost::function<thread_function_type>, thread_state, char const*,
            thread_id_type, boost::uint32_t
        > task_description;
        typedef boost::lockfree::fifo<task_description> task_items_type;

        typedef boost::lockfree::fifo<thread_id_type> thread_id_queue_type;

    protected:
        ///////////////////////////////////////////////////////////////////////
        // add new threads if there is some amount of work available
        bool add_new(long add_count)
        {
            if (0 == add_count)
                return false;

            long added = 0;
            task_description task;
            while (add_count-- && new_tasks_.dequeue(&task)) 
            {
                // measure thread creation time
                util::block_profiler_wrapper<add_new_tag> bp(add_new_logger_);

                // create the new thread
                thread_state state = boost::get<1>(task);
                std::auto_ptr<threads::thread> thrd (
                    new threads::thread(boost::get<0>(task), state, 
                        boost::get<2>(task), boost::get<3>(task),
                        boost::get<4>(task)));

                // add the new entry to the map of all threads
                thread_id_type id = thrd->get_thread_id();
                std::pair<thread_map_type::iterator, bool> p =
                    thread_map_.insert(id, thrd.get());

                if (!p.second) {
                    HPX_THROW_EXCEPTION(hpx::no_success, 
                        "threadmanager::add_new", 
                        "Couldn't add new thread to the map of threads");
                    return false;
                }

                // transfer ownership to map
                threads::thread* t = thrd.release();

                // only insert the thread into the work-items queue if it is in 
                // pending state
                if (state == pending) {
                    // pushing the new thread into the pending queue 
                    ++added;
                    work_items_.enqueue(t);
                    cond_.notify_all();         // wake up sleeping threads
                }
            }

            if (added) {
                LTM_(info) << "add_new: added " << added << " tasks to queues";
            }
            return added != 0;
        }

        ///////////////////////////////////////////////////////////////////////
        bool add_new_if_possible()
        {
            if (new_tasks_.empty()) 
                return false;

            // create new threads from pending tasks (if appropriate)
            long add_count = -1;                  // default is no constraint

            // if the map doesn't hold max_count threads yet add some
            if (max_count_) {
                std::size_t count = thread_map_.size();
                if (max_count_ >= count + min_add_new_count) {
                    add_count = max_count_ - count;
                    if (add_count < min_add_new_count)
                        add_count = min_add_new_count;
                }
                else {
                    return false;
                }
            }
            return add_new(add_count);
        }

        ///////////////////////////////////////////////////////////////////////
        bool add_new_always()
        {
            if (new_tasks_.empty()) 
                return false;

            // create new threads from pending tasks (if appropriate)
            long add_count = -1;                  // default is no constraint

            // if we are desperate (no work in the queues), add some even if the 
            // map holds more than max_count
            if (max_count_) {
                std::size_t count = thread_map_.size();
                if (max_count_ >= count + min_add_new_count) {
                    add_count = max_count_ - count;
                    if (add_count < min_add_new_count)
                        add_count = min_add_new_count;
                    if (add_count > max_add_new_count)
                        add_count = max_add_new_count;
                }
                else if (work_items_.empty()) {
                    add_count = min_add_new_count;    // add this number of threads
                    max_count_ += min_add_new_count;  // increase max_count
                }
                else {
                    return false;
                }
            }
            return add_new(add_count);
        }

        /// This function makes sure all threads which are marked for deletion
        /// (state is terminated) are properly destroyed
        bool cleanup_terminated()
        {
            if (!terminated_items_.empty()) {
                long delete_count = max_delete_count;   // delete only this much threads
                thread_id_type todelete;
                while (delete_count && terminated_items_.dequeue(&todelete)) 
                {
                    if (thread_map_.erase(todelete))
                        --delete_count;
                }
            }
            return thread_map_.empty();
        }

    public:
        // The maximum number of active threads this thread manager should
        // create. This number will be a constraint only as long as the work
        // items queue is not empty. Otherwise the number of active threads 
        // will be incremented in steps equal to the \a min_add_new_count
        // specified above.
        enum { max_thread_count = 1000 };

        thread_queue(std::size_t max_count = max_thread_count)
          : work_items_("work_items"), 
            terminated_items_("terminated_items"), 
            max_count_((0 == max_count) ? max_thread_count : max_count),
            add_new_logger_("thread_queue::add_new")
        {}

        void set_max_count(std::size_t max_count = max_thread_count)
        {
            max_count_ = (0 == max_count) ? max_thread_count : max_count;
        }

        ///////////////////////////////////////////////////////////////////////
        // This returns the current length of the queues (work items and new items)
        boost::int64_t get_queue_lengths() const
        {
            return work_items_.count_ + new_tasks_.count_;
        }

        ///////////////////////////////////////////////////////////////////////
        // create a new thread and schedule it if the initial state is equal to 
        // pending
        thread_id_type create_thread(
            boost::function<thread_function_type> const& threadfunc, 
            char const* const description, thread_state initial_state,
            bool run_now, boost::uint32_t parent_prefix, 
            thread_id_type parent_id)
        {
            if (run_now) {
                std::auto_ptr<threads::thread> thrd (
                    new threads::thread(threadfunc, initial_state, description,
                        parent_id, parent_prefix));

                mutex_type::scoped_lock lk(mtx_);

                // add a new entry in the map for this thread
                thread_id_type id = thrd->get_thread_id();
                std::pair<thread_map_type::iterator, bool> p =
                    thread_map_.insert(id, thrd.get());

                if (!p.second) {
                    HPX_THROW_EXCEPTION(hpx::no_success, 
                        "threadmanager::register_thread", 
                        "Couldn't add new thread to the map of threads");
                    return invalid_thread_id;
                }

                // push the new thread in the pending queue thread
                if (initial_state == pending) 
                    schedule_thread(thrd.get());

                do_some_work();       // try to execute the new work item
                thrd.release();       // release ownership to the map

                // return the thread_id of the newly created thread
                return id;
            }

            // do not execute the work, but register a task description for 
            // later thread creation
            if (0 == parent_id) {
                threads::thread_self* self = get_self_ptr();
                if (self)
                    parent_id = self->get_thread_id();
            }
            if (0 == parent_prefix) 
                parent_prefix = applier::get_prefix_id();
            new_tasks_.enqueue(
                task_description(threadfunc, initial_state, description, 
                    parent_id, parent_prefix));

            return invalid_thread_id;     // thread has not been created yet
        }

        /// Return the next thread to be executed, return false if non is 
        /// available
        bool get_next_thread(threads::thread** thrd)
        {
            return work_items_.dequeue(thrd);
        }

        /// Schedule the passed thread
        void schedule_thread(threads::thread* thrd)
        {
            work_items_.enqueue(thrd);
        }

        /// Destroy the passed thread as it has been terminated
        void destroy_thread(threads::thread* thrd)
        {
            thread_id_type id = thrd->get_thread_id();
            terminated_items_.enqueue(id);
        }

        /// Return the number of existing threads, regardless of their state
        std::size_t get_thread_count() const
        {
            return thread_map_.size();
        }

        /// This is a function which gets called periodically by the thread 
        /// manager to allow for maintenance tasks to be executed in the 
        /// scheduler. Returns true if the OS thread calling this function
        /// has to be terminated (i.e. no more work has to be done).
        bool wait_or_add_new(std::size_t num_thread, bool running)
        {
            // only one dedicated OS thread is allowed to acquire the 
            // lock for the purpose of inserting the new threads into the 
            // thread-map and deleting all terminated threads
            {
                // first clean up terminated threads
                detail::try_lock_wrapper<mutex_type> lk(mtx_);
                if (lk) {
                    // no point in having a thread waiting on the lock 
                    // while another thread is doing the maintenance
                    cleanup_terminated();

                    // now, add new threads from the queue of task descriptions
                    add_new_if_possible();    // calls notify_all
                }
            }

            bool terminate = false;
            while (work_items_.empty()) {
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
                detail::try_lock_wrapper<mutex_type> lk(mtx_);
                if (!lk)
                    break;            // avoid long wait on lock

                // this thread acquired the lock, do maintenance and finally
                // call wait() if no work is available
                LTM_(info) << "tfunc(" << num_thread << "): queues empty"
                           << ", threads left: " << thread_map_.size();

                // stop running after all PX threads have been terminated
                if (!add_new_always() && !running) {
                    // Before exiting each of the OS threads deletes the 
                    // remaining terminated PX threads 
                    if (cleanup_terminated()) {
                        // we don't have any registered work items anymore
                        cond_.notify_all();   // notify possibly waiting threads
                        terminate = true;
                        break;                // terminate scheduling loop
                    }

                    LTM_(info) << "tfunc(" << num_thread 
                               << "): threadmap not empty";
                }
                else {
                    cleanup_terminated();
                }

                // Wait until somebody needs some action (if no new work 
                // arrived in the meantime).
                // Ask again if queues are empty to avoid race conditions (we 
                // needed to lock anyways...), this way no notify_all() gets lost
                if (work_items_.empty())
                {
                    LTM_(info) << "tfunc(" << num_thread 
                               << "): queues empty, entering wait";

                    if (LHPX_ENABLED(error) && new_tasks_.empty())
                        detail::dump_suspended_threads(thread_map_);

                    bool timed_out = false;
                    {
                        namespace bpt = boost::posix_time;
                        timed_out = !cond_.timed_wait(lk, bpt::milliseconds(5));
                    }

                    LTM_(info) << "tfunc(" << num_thread << "): exiting wait";

                    // make sure all pending new threads are properly queued
                    // but do that only if the lock has been acquired while 
                    // exiting the condition.wait() above
                    if ((lk && add_new_always()) || timed_out)
                        break;
                }
            }
            return terminate;
        }

        /// This function gets called by the threadmanager whenever new work
        /// has been added, allowing the scheduler to reactivate one or more of
        /// possibly idleing OS threads
        void do_some_work()
        {
            cond_.notify_all();
        }

        ///////////////////////////////////////////////////////////////////////
        void on_start_thread(std::size_t num_thread) {}
        void on_stop_thread(std::size_t num_thread)
        {
            if (0 == num_thread) {
                // print queue statistics
                using boost::lockfree::detail::log_fifo_statistics;
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
        thread_id_queue_type terminated_items_;  ///< list of terminated threads

        std::size_t max_count_;             ///< maximum number of existing PX-threads
        task_items_type new_tasks_;         ///< list of new tasks to run

        util::block_profiler<add_new_tag> add_new_logger_;
    };

}}}

#endif
