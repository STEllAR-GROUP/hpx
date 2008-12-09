//  Copyright (c) 2008-2009 Chirag Dekate, Hartmut Kaiser, Anshul Tandon
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/threads/thread_affinity.hpp>      // must be first header!
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/util/unlock_lock.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/block_profiler.hpp>
#include <hpx/util/time_logger.hpp>

#include <boost/assert.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/asio/deadline_timer.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads
{
    ///////////////////////////////////////////////////////////////////////////
    namespace strings
    {
        char const* const thread_state_names[] = 
        {
            "init",
            "active",
            "pending",
            "suspended",
            "depleted",
            "terminated"
        };
    }

    char const* const get_thread_state_name(thread_state state)
    {
        if (state < init || state > terminated)
            return "unknown";
        return strings::thread_state_names[state];
    }

    ///////////////////////////////////////////////////////////////////////////
    threadmanager::threadmanager(util::io_service_pool& timer_pool,
            boost::function<void()> start_thread, boost::function<void()> stop,
            std::size_t max_count)
      : max_count_(max_count), running_(false), wait_count_(0),
        timer_pool_(timer_pool), start_thread_(start_thread), stop_(stop),
        work_items_("work_items"), terminated_items_("terminated_items"), 
        active_set_state_("active_set_state"), new_tasks_("new_tasks")
#if HPX_DEBUG != 0
      , thread_count_(0)
#endif
    {
        LTM_(debug) << "threadmanager ctor";
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Fifo>
    inline void log_fifo_statistics(Fifo const& q)
    {
        LTIM_(fatal) << "~threadmanager: queue: "  << q.description_
                     << ", enqueue_spin_count: " << long(q.enqueue_spin_count_)
                     << ", dequeue_spin_count: " << long(q.dequeue_spin_count_);
    }

    threadmanager::~threadmanager() 
    {
        LTM_(debug) << "~threadmanager";
        log_fifo_statistics(work_items_);
        log_fifo_statistics(terminated_items_);
        log_fifo_statistics(active_set_state_);
        log_fifo_statistics(new_tasks_);

        if (!threads_.empty()) {
            if (running_) 
                stop();
            threads_.clear();
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    struct register_thread_tag {};

    thread_id_type threadmanager::register_thread(
        boost::function<thread_function_type> const& threadfunc, 
        char const* const description, thread_state initial_state, bool run_now)
    {
        util::block_profiler<register_thread_tag> bp("threadmanager::register_thread");

        // verify parameters
        if (initial_state != pending && initial_state != suspended)
        {
            HPX_OSSTREAM strm;
            strm << "invalid initial state: " 
                 << get_thread_state_name(initial_state);
            HPX_THROW_EXCEPTION(bad_parameter, HPX_OSSTREAM_GETSTRING(strm));
            return invalid_thread_id;
        }
        if (0 == description)
        {
            HPX_THROW_EXCEPTION(bad_parameter, "description is NULL");
            return invalid_thread_id;
        }

        LTM_(info) << "register_thread: initial_state(" 
                   << get_thread_state_name(initial_state) << "), "
                   << std::boolalpha << "run_now(" << run_now << "), "
                   << "description(" << description << ")";

        // create the new thread
        boost::shared_ptr<threads::thread> thrd (
            new threads::thread(threadfunc, initial_state, description));

        // lock data members while adding work
        mutex_type::scoped_lock lk(mtx_);

        // add a new entry in the map for this thread
        std::pair<thread_map_type::iterator, bool> p =
            thread_map_.insert(map_pair(thrd->get_thread_id(), thrd));

        if (!p.second) {
            HPX_THROW_EXCEPTION(hpx::no_success, 
                "Couldn't add new thread to the map of threads");
            return invalid_thread_id;
        }

        // only insert in the work-items queue if it is in pending state
        if (initial_state == pending) {
            // pushing the new thread in the pending queue thread
            work_items_.enqueue(thrd);
        }

        if (run_now) {
            // try to execute the new work item
            cond_.notify_all();
        }

        // return the thread_id of the newly created thread
        return thrd->get_thread_id();
    }

    ///////////////////////////////////////////////////////////////////////////
    struct register_work_tag {};

    void threadmanager::register_work(
        boost::function<thread_function_type> const& threadfunc, 
        char const* const description, thread_state initial_state, bool run_now)
    {
        util::block_profiler<register_work_tag> bp("threadmanager::register_work");

        // verify parameters
        if (initial_state != pending && initial_state != suspended)
        {
            HPX_OSSTREAM strm;
            strm << "invalid initial state: " 
                 << get_thread_state_name(initial_state);
            HPX_THROW_EXCEPTION(bad_parameter, HPX_OSSTREAM_GETSTRING(strm));
            return;
        }
        if (0 == description)
        {
            HPX_THROW_EXCEPTION(bad_parameter, "description is NULL");
            return;
        }

        LTM_(info) << "register_work: initial_state(" 
                   << get_thread_state_name(initial_state) << "), "
                   << std::boolalpha << "run_now(" << run_now << "), "
                   << "description(" << description << ")";

    // create a new task
        new_tasks_.enqueue(
            task_description(threadfunc, initial_state, description));
        if (run_now) {
            // try to execute the new work item
            cond_.notify_all();
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // add new threads if there is some amount of work available
    inline bool threadmanager::add_new(long add_count)
    {
        if (0 == add_count)
            return false;

        std::size_t num_threads = threads_.size();
        long added = 0;
        task_description task;
        while (add_count-- && new_tasks_.dequeue(&task)) 
        {
            // create the new thread
            thread_state state = boost::get<1>(task);
            boost::shared_ptr<threads::thread> thrd (
                new threads::thread(boost::get<0>(task), state, 
                    boost::get<2>(task)));

            // add the new entry to the map of all threads
            std::pair<thread_map_type::iterator, bool> p =
                thread_map_.insert(map_pair(thrd->get_thread_id(), thrd));

            if (!p.second) {
                HPX_THROW_EXCEPTION(hpx::no_success, 
                    "Couldn't add new thread to the map of threads");
                return false;
            }

            // only insert the thread into the work-items queue if it is in 
            // pending state
            if (state == pending) {
                // pushing the new thread into the pending queue 
                ++added;
                work_items_.enqueue(thrd);
                if (0 != wait_count_)
                    cond_.notify_all();         // wake up sleeping threads
            }
        }

        if (added) {
            LTM_(info) << "add_new: added " << added << " tasks to queues";
        }
        return added != 0;
    }

    inline bool threadmanager::add_new_if_possible()
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

    ///////////////////////////////////////////////////////////////////////////
    inline bool threadmanager::add_new_always()
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

    ///////////////////////////////////////////////////////////////////////////
    /// The set_state function is part of the thread related API and allows
    /// to change the state of one of the threads managed by this threadmanager
    struct set_state_tag {};

    thread_state threadmanager::set_state(thread_id_type id, 
        thread_state new_state)
    {
        util::block_profiler<set_state_tag> bp("threadmanager::set_state");

        // set_state can't be used to force a thread into active state
        if (new_state == active) {
            HPX_OSSTREAM strm;
            strm << "invalid new state: " << get_thread_state_name(new_state);
            HPX_THROW_EXCEPTION(bad_parameter, HPX_OSSTREAM_GETSTRING(strm));
            return unknown;
        }

        // lock data members while setting a thread state
        mutex_type::scoped_lock lk(mtx_);

        thread_map_type::iterator map_iter = thread_map_.find(id);

        // the id may reference a new thread not yet stored in the map
        if (map_iter == thread_map_.end())
            map_iter = thread_map_.find(id);

        if (map_iter != thread_map_.end())
        {
            boost::shared_ptr<thread> thrd = map_iter->second;

            lk.unlock();

            // action depends on the current state
            thread_state previous_state = thrd->get_state();

            // nothing to do here if the state doesn't change
            if (new_state == previous_state)
                return new_state;

            // the thread to set the state for is currently running, so we 
            // yield control for the main thread manager loop to release this
            // thread
            if (previous_state == active) {
                // if we can't suspend (because we don't know the PX thread
                // executing this function) we need to return 'unknown'
                if (NULL == get_self_ptr()) 
                    return unknown;

                LTM_(info) << "set_state: " << "thread(" << id << "), "
                           << "is currently active, yielding control...";

                do {
                    thread_self& self = get_self();
                    active_set_state_.enqueue(self.get_thread_id());
                    self.yield(suspended);
                } while ((previous_state = thrd->get_state()) == active);

                LTM_(info) << "set_state: " << "thread(" << id << "), "
                           << "reactivating..." << "current state(" 
                           << get_thread_state_name(previous_state) << ")";
            }

            // If the thread has been terminated while this set_state was 
            // waiting in the active_set_state_ queue nothing has to be done 
            // anymore.
            if (previous_state == terminated)
                return terminated;

            // If the previous state was pending we are supposed to remove the
            // thread from the queue. But in order to avoid linearly looking 
            // through the queue we defer this to the thread function, which 
            // at some point will ignore this thread by simply skipping it 
            // (if it's not pending anymore). 

            LTM_(info) << "set_state: " << "thread(" << id << "), "
                       << "description(" << thrd->get_description() << "), "
                       << "new state(" << get_thread_state_name(new_state) << ")";

            // So all what we do here is to set the new state.
            thrd->set_state(new_state);
            if (new_state == pending) {
                work_items_.enqueue(thrd);
                cond_.notify_all();
            }
            return previous_state;
        }
        return unknown;
    }

    /// The get_state function is part of the thread related API and allows
    /// to query the state of one of the threads known to the threadmanager
    thread_state threadmanager::get_state(thread_id_type id) 
    {
        // lock data members while getting a thread state
        mutex_type::scoped_lock lk(mtx_);

        thread_map_type::iterator map_iter = thread_map_.find(id);

        // the id may reference a new thread not yet stored in the map
        if (map_iter == thread_map_.end())
            map_iter = thread_map_.find(id);

        if (map_iter != thread_map_.end())
            return map_iter->second->get_state();

        return unknown;
    }

    /// This thread function is used by the at_timer thread below to trigger
    /// the required action.
    thread_state threadmanager::wake_timer_thread (thread_id_type id, 
        thread_state newstate, thread_id_type timer_id) 
    {
        // first trigger the requested set_state 
        set_state(id, newstate);

        // then re-activate the thread holding the deadline_timer
        set_state(timer_id, pending);
        return terminated;
    }

    /// This thread function initiates the required set_state action (on 
    /// behalf of one of the threadmanager#set_state functions).
    template <typename TimeType>
    thread_state threadmanager::at_timer (TimeType const& expire, 
        thread_id_type id, thread_state newstate)
    {
        // create timer firing in correspondence with given time
        boost::asio::deadline_timer t (timer_pool_.get_io_service(), expire);

        // create a new thread in suspended state, which will execute the 
        // requested set_state when timer fires and will re-awaken this thread, 
        // allowing the deadline_timer to go out of scope gracefully
        thread_self& self = get_self();
        thread_id_type wake_id = register_thread(boost::bind(
            &threadmanager::wake_timer_thread, this, id, newstate,
            self.get_thread_id()), "", suspended);

        // let the timer invoke the set_state on the new (suspended) thread
        t.async_wait(boost::bind(&threadmanager::set_state, this, wake_id, 
            pending));

        // this waits for the thread executed when the timer fired
        self.yield(suspended);
        return terminated;
    }

    /// Set a timer to set the state of the given \a thread to the given 
    /// new value after it expired (at the given time)
    thread_id_type threadmanager::set_state (time_type const& expire_at, 
        thread_id_type id, thread_state newstate)
    {
        // this creates a new thread which creates the timer and handles the
        // requested actions
        thread_state (threadmanager::*f)(time_type const&, thread_id_type, 
                thread_state)
            = &threadmanager::at_timer<time_type>;

        return register_thread(boost::bind(f, this, expire_at, id, newstate),
            "at_timer (expire at)");
    }

    /// Set a timer to set the state of the given \a thread to the given
    /// new value after it expired (after the given duration)
    thread_id_type threadmanager::set_state (duration_type const& from_now, 
        thread_id_type id, thread_state newstate)
    {
        // this creates a new thread which creates the timer and handles the
        // requested actions
        thread_state (threadmanager::*f)(duration_type const&, thread_id_type, 
                thread_state)
            = &threadmanager::at_timer<duration_type>;

        return register_thread(boost::bind(f, this, from_now, id, newstate),
            "at_timer (from now)");
    }

    /// Retrieve the global id of the given thread
    naming::id_type threadmanager::get_thread_gid(thread_id_type id) 
    {
        // lock data members while getting a thread state
        mutex_type::scoped_lock lk(mtx_);

        thread_map_type::iterator map_iter = thread_map_.find(id);

        // the id may reference a new thread not yet stored in the map
        if (map_iter == thread_map_.end())
            map_iter = thread_map_.find(id);

        if (map_iter != thread_map_.end())
            return map_iter->second->get_gid();

        return naming::invalid_id;
    }

    // helper class for switching thread state in and out during execution
    class switch_status
    {
    public:
        switch_status (boost::shared_ptr<thread> t, thread_state new_state)
            : thread_(t), prev_state_(t->set_state(new_state))
        {}

        ~switch_status ()
        {
            thread_->set_state(prev_state_);
        }

        // allow to change the state the thread will be switched to after 
        // execution
        thread_state operator=(thread_state new_state)
        {
            return prev_state_ = new_state;
        }

        // allow to compare against the previous state of the thread
        bool operator== (thread_state rhs)
        {
            return prev_state_ == rhs;
        }

    private:
        boost::shared_ptr<thread> thread_;
        thread_state prev_state_;
    };

    ///////////////////////////////////////////////////////////////////////////
    inline void handle_pending_set_state(threadmanager& tm, 
        threadmanager::set_state_queue_type& active_set_state)
    {
        if (!active_set_state.empty()) {
            threadmanager::set_state_queue_type still_active;
            thread_id_type id = 0;

            // try to reactivate the threads in the set_state queue
            while (active_set_state.dequeue(&id)) {
                // if the thread is still active, just re-queue the 
                // set_state request
                if (unknown == tm.set_state(id, pending))
                    still_active.enqueue(id);
            }

            // copy the PX threads which are still active to the main queue
            if (!still_active.empty()) {
                while (still_active.dequeue(&id)) 
                    active_set_state.enqueue(id);
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    inline bool threadmanager::cleanup_terminated()
    {
        if (!terminated_items_.empty()) {
            long delete_count = max_delete_count;   // delete only this much threads
            boost::shared_ptr<thread> todelete;
            while (--delete_count && terminated_items_.dequeue(&todelete)) 
                thread_map_.erase(todelete->get_thread_id());
        }
        return thread_map_.empty();
    }

    ///////////////////////////////////////////////////////////////////////////
    // main function executed by all OS threads managed by this threadmanager
    void threadmanager::tfunc(std::size_t num_thread)
    {
        LTM_(info) << "tfunc(" << num_thread << "): start";
        std::size_t num_px_threads = 0;
        try {
            if (start_thread_)    // notify runtime system of started thread
                start_thread_();

            num_px_threads = tfunc_impl(num_thread);
        }
        catch (hpx::exception const& e) {
            LTM_(fatal) << "tfunc(" << num_thread 
                        << "): caught hpx::exception: " 
                        << e.what() << ", aborted execution";
            if (stop_) 
                stop_();
            return;
        }
        catch (boost::system::system_error const& e) {
            LTM_(fatal) << "tfunc(" << num_thread 
                        << "): caught boost::system::system_error: " 
                        << e.what() << ", aborted execution";
            if (stop_) 
                stop_();
            return;
        }
        catch (std::exception const& e) {
            LTM_(fatal) << "tfunc(" << num_thread 
                        << "): caught std::exception: " 
                        << e.what() << ", aborted execution";
            if (stop_) 
                stop_();
            return;
        }
        catch (...) {
            LTM_(fatal) << "tfunc(" << num_thread 
                        << "): caught unexpected exception, aborted execution";
            if (stop_) 
                stop_();
            return;
        }
        LTM_(info) << "tfunc(" << num_thread << "): end, executed " 
                   << num_px_threads << " HPX threads";
    }

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
    }

    ///////////////////////////////////////////////////////////////////////////
    struct decrement_on_exit
    {
        decrement_on_exit(boost::lockfree::atomic_int<long>& count)
          : count_(count)
        {
            ++count_;
        }
        ~decrement_on_exit()
        {
            --count_;
        }

        boost::lockfree::atomic_int<long>& count_;
    };

    ///////////////////////////////////////////////////////////////////////////
    std::size_t threadmanager::tfunc_impl(std::size_t num_thread)
    {
#if HPX_DEBUG != 0
        ++thread_count_;
#endif
        std::size_t num_px_threads = 0;
        util::time_logger tl1("tfunc", num_thread);
        util::time_logger tl2("tfunc1", num_thread);

        // the thread with number zero is the master
        bool is_master_thread = (0 == num_thread) ? true : false;
        set_affinity(num_thread);     // set affinity on Linux systems

        // run the work queue
        boost::coroutines::prepare_main_thread main_thread;
        while (true) {
            // Get the next PX thread from the queue
            boost::shared_ptr<thread> thrd;
            if (work_items_.dequeue(&thrd)) {
                tl1.tick();

                // Only pending PX threads will be executed.
                // Any non-pending PX threads are leftovers from a set_state() 
                // call for a previously pending PX thread (see comments above).
                thread_state state = thrd->get_state();

                LTM_(debug) << "tfunc(" << num_thread << "): "
                           << "thread(" << thrd->get_thread_id() << "), " 
                           << "description(" << thrd->get_description() << "), "
                           << "old state(" << get_thread_state_name(state) << ")";

                if (pending == state) {
                    // switch the state of the thread to active and back to 
                    // what the thread reports as its return value

                    switch_status thrd_stat (thrd, active);
                    if (thrd_stat == pending) {
                        // thread returns new required state
                        // store the returned state in the thread
                        thrd_stat = state = (*thrd)();
                        ++num_px_threads;
                    }

                }   // this stores the new state in the PX thread

                LTM_(debug) << "tfunc(" << num_thread << "): "
                           << "thread(" << thrd->get_thread_id() << "), "
                           << "description(" << thrd->get_description() << "), "
                           << "new state(" << get_thread_state_name(state) << ")";

                // Re-add this work item to our list of work items if the PX
                // thread should be re-scheduled. If the PX thread is suspended 
                // now we just keep it in the map of threads.
                if (state == pending) {
                    work_items_.enqueue(thrd);
                    cond_.notify_all();
                }

                // Remove the mapping from thread_map_ if PX thread is depleted 
                // or terminated, this will delete the PX thread as all 
                // references go out of scope.
                // FIXME: what has to be done with depleted PX threads?
                if (state == depleted || state == terminated) {
                    // all OS threads put their terminated PX threads into a 
                    // separate queue
                    terminated_items_.enqueue(thrd);
                }

                tl1.tock();
            }

            // only one dedicated OS thread is allowed to acquire the 
            // lock for the purpose of inserting the new threads into the 
            // thread-map and deleting all terminated threads
            if (is_master_thread) {
                {
                    // first clean up terminated threads
                    mutex_type::scoped_lock lk(mtx_);
                    if (lk) {
                        // no point in having a thread waiting on the lock 
                        // while another thread is doing the maintenance
                        cleanup_terminated();

                        // now, add new threads from the queue of task descriptions
                        if (add_new_if_possible())
                            cond_.notify_all();
                    }
                }

                // make sure to handle pending set_state requests
                handle_pending_set_state(*this, active_set_state_);
            }

            // if nothing else has to be done either wait or terminate
            bool terminate = false;
            while (work_items_.empty() && active_set_state_.empty()) {
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
                // call wait() if not work is available
                LTM_(info) << "tfunc(" << num_thread << "): queues empty"
                           << ", threads left: " << thread_map_.size();

                // stop running after all PX threads have been terminated
                if (!add_new_always() && !running_) {
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
                if (work_items_.empty() && active_set_state_.empty())
                {
                    LTM_(info) << "tfunc(" << num_thread 
                               << "): queues empty, entering wait";

                    bool timed_out = false;
                    {
                        namespace bpt = boost::posix_time;
                        decrement_on_exit on_exit(wait_count_);

                        tl2.tick();
                        timed_out = cond_.timed_wait(lk, bpt::milliseconds(5));
                        tl2.tock();
                    }

                    LTM_(info) << "tfunc(" << num_thread << "): exiting wait";

                    // make sure all pending new threads are properly queued
                    // but do that only if the lock has been acquired while 
                    // exiting the condition.wait() above
                    if ((lk && add_new_always()) || timed_out)
                        break;
                }
            }
            if (terminate)
                break;
        }

#if HPX_DEBUG != 0
        // the last OS thread is allowed to exit only if no more PX threads exist
        BOOST_ASSERT(0 != --thread_count_ || thread_map_.empty());
#endif
        return num_px_threads;
    }

    ///////////////////////////////////////////////////////////////////////////
    bool threadmanager::run(std::size_t num_threads) 
    {
        LTM_(info) << "run: creating " << num_threads << " OS thread(s)";

        if (0 == num_threads) {
            HPX_THROW_EXCEPTION(bad_parameter, "number of threads is zero");
        }

        mutex_type::scoped_lock lk(mtx_);
        if (!threads_.empty() || running_) 
            return true;    // do nothing if already running

        LTM_(info) << "run: running timer pool"; 
        timer_pool_.run(false);

        running_ = false;
        try {
            // run threads and wait for initialization to complete
            running_ = true;
            while (num_threads-- != 0) {
                LTM_(info) << "run: create OS thread: " << num_threads; 

                // create a new thread
                threads_.push_back(new boost::thread(
                    boost::bind(&threadmanager::tfunc, this, num_threads)));

                // set the new threads affinity (on Windows systems)
                set_affinity(threads_.back(), num_threads);
            }

            // start timer pool as well
            timer_pool_.run(false);
        }
        catch (std::exception const& e) {
            LTM_(fatal) << "run: failed with:" << e.what(); 
            stop();
            threads_.clear();
            return false;
        }

        LTM_(info) << "run: running"; 
        return running_;
    }

    void threadmanager::stop (bool blocking)
    {
        LTM_(info) << "stop: blocking(" << std::boolalpha << blocking << ")"; 

        mutex_type::scoped_lock l(mtx_);
        if (!threads_.empty()) {
            if (running_) {
                LTM_(info) << "stop: set running_ = false"; 
                running_ = false;
                cond_.notify_all();         // make sure we're not waiting
            }

            if (blocking) {
                for (std::size_t i = 0; i < threads_.size(); ++i) 
                {
                    // make sure no OS thread is waiting
                    LTM_(info) << "stop: notify_all"; 
                    cond_.notify_all();

                    LTM_(info) << "stop(" << i << "): join"; 

                    // unlock the lock while joining
                    util::unlock_the_lock<mutex_type::scoped_lock> ul(l);
                    threads_[i].join();
                }
                threads_.clear();
            }

            LTM_(info) << "stop: stopping timer pool"; 
            timer_pool_.stop();             // stop timer pool as well
            if (blocking) 
                timer_pool_.join();
        }
    }

}}
