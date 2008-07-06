//  Copyright (c) 2008-2009 Chirag Dekate, Hartmut Kaiser, Anshul Tandon
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <boost/assert.hpp>

#include <hpx/runtime/threadmanager/threadmanager.hpp>
#include <hpx/runtime/threadmanager/px_thread.hpp>
#include <hpx/util/unlock_lock.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threadmanager
{
    ///////////////////////////////////////////////////////////////////////////
    thread_id_type threadmanager::register_work(
        boost::function<thread_function_type> threadfunc, 
        thread_state initial_state, bool run_now)
    {
        // verify parameters
        if (initial_state != pending && initial_state != suspended)
        {
            boost::throw_exception(hpx::exception(hpx::bad_parameter));
            return invalid_thread_id;
        }

        boost::intrusive_ptr<px_thread> px_t_sp (
            new px_thread(threadfunc, *this, initial_state));

        // lock data members while adding work
        mutex_type::scoped_lock lk(mtx_);

        // add a new entry in the map for this thread
        thread_map_.insert(map_pair(px_t_sp->get_thread_id(), px_t_sp));

        // only insert in the work-items queue if it is in pending state
        if (initial_state == pending)
        {
            // pushing the new thread in the pending queue thread
            work_items_.enqueue(px_t_sp);
            if (run_now) 
                cond_.notify_one();       // try to execute the new work item
        }

        // return the thread_id of the newly created thread
        return px_t_sp->get_thread_id();
    }

    /// The set_state function is part of the thread related API and allows
    /// to change the state of one of the threads managed by this threadmanager
    thread_state threadmanager::set_state(px_thread_self& self, 
        thread_id_type id, thread_state new_state)
    {
        // set_state can't be used to force a thread into active state
        if (new_state == active)
        {
            boost::throw_exception(hpx::exception(hpx::bad_parameter));
            return unknown;
        }

        // lock data members while setting a thread state
        mutex_type::scoped_lock lk(mtx_);

        thread_map_type::iterator map_iter = thread_map_.find(id);
        if (map_iter != thread_map_.end())
        {
            boost::intrusive_ptr<px_thread> px_t = map_iter->second;
            thread_state previous_state = px_t->get_state();

            // nothing to do here if the state doesn't change
            if (new_state == previous_state)
                return new_state;

            // the thread to set the state for is currently running, so we 
            // yield control for the main thread manager loop to release this
            // px_thread
            if (previous_state == active) {
                do {
                    active_set_state_.enqueue(self.get_thread_id());
                    util::unlock_the_lock<mutex_type::scoped_lock> ul(lk);
                    self.yield(suspended);
                } while ((previous_state = px_t->get_state()) == active);
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

            // So all what we do here is to set the new state.
            px_t->set_state(new_state);
            if (new_state == pending) 
            {
                work_items_.enqueue(px_t);
                cond_.notify_one();
            }
            return previous_state;
        }
        return unknown;
    }

    /// The set_state function is part of the thread related API and allows
    /// to change the state of one of the threads managed by this threadmanager
    thread_state threadmanager::set_state(thread_id_type id, 
        thread_state new_state)
    {
        // set_state can't be used to force a thread into active state
        if (new_state == active)
        {
            boost::throw_exception(hpx::exception(hpx::bad_parameter));
            return unknown;
        }

        // lock data members while setting a thread state
        mutex_type::scoped_lock lk(mtx_);

        thread_map_type::iterator map_iter = thread_map_.find(id);
        if (map_iter != thread_map_.end())
        {
            boost::intrusive_ptr<px_thread> px_t = map_iter->second;
            thread_state previous_state = px_t->get_state();

            // nothing to do here if the state doesn't change
            if (new_state == previous_state)
                return new_state;

            // if the thread to set the state for is currently active we do
            // nothing but return \a thread_state#unknown
            if (previous_state == active) 
                return unknown;

            // If the previous state was pending we are supposed to remove the
            // thread from the queue. But in order to avoid linearly looking 
            // through the queue we defer this to the thread function, which 
            // at some point will come across this thread simply skipping it 
            // (if it's not pending anymore). 

            // So all what we do here is to set the new state.
            px_t->set_state(new_state);
            if (new_state == pending)
            {
                work_items_.enqueue(px_t);
                cond_.notify_one();
            }
            return previous_state;
        }
        return unknown;
    }

    /// The set_state function is part of the thread related API and allows
    /// to query the state of one of the threads known to the threadmanager
    thread_state threadmanager::get_state(thread_id_type id) const
    {
        // lock data members while getting a thread state
        mutex_type::scoped_lock lk(mtx_);

        thread_map_type::const_iterator map_iter = thread_map_.find(id);
        if (map_iter != thread_map_.end())
        {
            return map_iter->second->get_state();
        }
        return unknown;
    }

    /// This thread function is used by the at_timer thread below to trigger
    /// the required action.
    thread_state threadmanager::wake_timer_thread (px_thread_self& self, 
        thread_id_type id, thread_state newstate, thread_id_type timer_id) 
    {
        // first trigger the requested set_state 
        set_state(self, id, newstate);

        // then re-activate the thread holding the deadline_timer
        set_state(self, timer_id, pending);
        return terminated;
    }

    /// This thread function initiates the required set_state action (on 
    /// behalf of one of the threadmanager#timed_set_state functions).
    template <typename TimeType>
    thread_state threadmanager::at_timer (px_thread_self& self, 
        TimeType const& expire, thread_id_type id, thread_state newstate)
    {
        // create timer firing in correspondence with given time
        boost::asio::deadline_timer t (timer_pool_.get_io_service(), expire);

        // create a new thread in suspended state, which will execute the 
        // requested set_state when timer fires and will re-awaken this thread, 
        // allowing the deadline_timer to go out of scope gracefully
        thread_id_type wake_id = register_work(boost::bind(
            &threadmanager::wake_timer_thread, this, _1, id, newstate,
            self.get_thread_id()), suspended);

        // let the timer invoke the set_state on the new (suspended) thread
        t.async_wait(boost::bind(
            &threadmanager::set_state, this, wake_id, pending));

        // this waits for the thread executed when the timer fired
        self.yield(suspended);
        return terminated;
    }

    /// Set a timer to set the state of the given \a px_thread to the given 
    /// new value after it expired (at the given time)
    thread_id_type threadmanager::timed_set_state (time_type const& expire_at, 
        thread_id_type id, thread_state newstate)
    {
        // this creates a new px_thread which creates the timer and handles the
        // requested actions
        thread_state (threadmanager::*f)(px_thread_self&, time_type const&,
                thread_id_type, thread_state)
            = &threadmanager::at_timer<time_type>;

        return register_work(boost::bind(f, this, _1, expire_at, id, newstate));
    }

    /// Set a timer to set the state of the given \a px_thread to the given
    /// new value after it expired (after the given duration)
    thread_id_type threadmanager::timed_set_state (
        duration_type const& from_now, thread_id_type id, 
        thread_state newstate)
    {
        // this creates a new px_thread which creates the timer and handles the
        // requested actions
        thread_state (threadmanager::*f)(px_thread_self&, duration_type const&,
                thread_id_type, thread_state)
            = &threadmanager::at_timer<duration_type>;

        return register_work(boost::bind(f, this, _1, from_now, id, newstate));
    }

    /// Retrieve the global id of the given px_thread
    naming::id_type threadmanager::get_thread_gid(thread_id_type id, 
        applier::applier& appl) const
    {
        // lock data members while getting a thread state
        mutex_type::scoped_lock lk(mtx_);

        thread_map_type::const_iterator map_iter = thread_map_.find(id);
        if (map_iter != thread_map_.end())
        {
            return map_iter->second->get_gid(appl);
        }
        return naming::invalid_id;
    }

    // helper class for switching thread state in and out during execution
    class switch_status
    {
    public:
        switch_status (px_thread* t, thread_state new_state)
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
        // it is safe to store a plain pointer here since this class will be 
        // used inside a block holding a intrusive_ptr to this thread
        px_thread* thread_;
        thread_state prev_state_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Mutex, typename Queue, typename Map>
    inline bool cleanup(Mutex& mtx, Queue& terminated_items, Map& thread_map)
    {
        typename Mutex::scoped_lock lk(mtx);
        boost::intrusive_ptr<px_thread> todelete;
        while (terminated_items.dequeue(&todelete))
            thread_map.erase(todelete->get_thread_id());
        return thread_map.empty();
    }

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

            // copy the threads which are still active to the main queue
            if (!still_active.empty()) {
                while (still_active.dequeue(&id)) 
                    active_set_state.enqueue(id);
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // main function executed by a OS thread
    void threadmanager::tfunc(bool is_master_thread)
    {
#if HPX_DEBUG != 0
        ++thread_count_;
#endif
        // run the work queue
        boost::coroutines::prepare_main_thread main_thread;
        while (true) {
            // Get the next thread from the queue
            boost::intrusive_ptr<px_thread> thrd;
            if (work_items_.dequeue(&thrd)) {
                // Only pending threads will be executed.
                // Any non-pending threads are leftovers from a set_state() 
                // call for a previously pending thread (see comments above).
                thread_state state = thrd->get_state();
                if (pending == state) {
                    // switch the state of the thread to active and back to 
                    // what the thread reports as its return value

                    switch_status thrd_stat (thrd.get(), active);
                    if (thrd_stat == pending) {
                        // thread returns new required state
                        // store the returned state in the thread
                        thrd_stat = state = (*thrd)();
                    }

                }   // this stores the new state in the thread

                // Re-add this work item to our list of work items if thread
                // should be re-scheduled. If the thread is suspended now we
                // just keep it in the map of threads.
                if (state == pending) 
                {
                    work_items_.enqueue(thrd);
                    cond_.notify_one();
                }

                // Remove the mapping from thread_map_ if thread is depleted or 
                // terminated, this will delete the px_thread as all references
                // go out of scope.
                // FIXME: what has to be done with depleted threads?
                if (state == depleted || state == terminated) {
                    // all OS threads put their terminated threads into a 
                    // separate queue
                    terminated_items_.enqueue(thrd);
                }

                // only one dedicated OS thread is allowed to acquire the 
                // lock for the purpose of deleting all terminated threads
                if (is_master_thread) 
                    cleanup(mtx_, terminated_items_, thread_map_);
            }

            // make sure to handle pending set_state requests
            if (is_master_thread) 
                handle_pending_set_state(*this, active_set_state_);

            // if nothing else has to be done either wait or terminate
            if (work_items_.empty() && active_set_state_.empty()) {
                // stop running after all px_threads have been terminated
                if (!running_) {
                    // Before exiting each of the threads deletes the remaining 
                    // terminated px_threads 
                    if (cleanup(mtx_, terminated_items_, thread_map_)) {
                        cond_.notify_all();   // notify possibly waiting threads
                        break;                // terminate scheduling loop
                    }
                }

                // Wait until somebody needs some action (if no new work 
                // arrived in the meantime).
                // Ask again if queues are empty to avoid race conditions (we 
                // need to lock anyways...), this way no notify_one() gets lost
                mutex_type::scoped_lock lk(mtx_);
                if (work_items_.empty() && active_set_state_.empty())
                    cond_.wait(lk);
            }
        }

#if HPX_DEBUG != 0
        // the last thread is allowed to exit only if no more px_threads exist
        BOOST_ASSERT(0 != --thread_count_ || thread_map_.empty());
#endif
    }

#if defined(_WIN32) || defined(_WIN64)
    void set_affinity(boost::thread& thrd, unsigned int affinity)
    {
        DWORD_PTR process_affinity = 0, system_affinity = 0;
        if (GetProcessAffinityMask(GetCurrentProcess(), &process_affinity, 
              &system_affinity))
        {
            DWORD_PTR mask = 0x1 << affinity;
            while (!(mask & process_affinity)) {
                mask <<= 1;
                if (0 == mask)
                    mask = 0x01;
            }
            SetThreadAffinityMask(thrd.native_handle(), mask);
        }
    }
#else
    void set_affinity(boost::thread& thrd, unsigned int affinity)
    {
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    bool threadmanager::run(std::size_t num_threads) 
    {
        if (0 == num_threads) {
            boost::throw_exception(hpx::exception(
                bad_parameter, "Number of threads shouldn't be zero"));
        }

        mutex_type::scoped_lock lk(mtx_);
        if (!threads_.empty() || running_) 
            return true;    // do nothing if already running

        timer_pool_.run(false);
        running_ = false;
        try {
            // run threads and wait for initialization to complete
            unsigned int num_of_cores = boost::thread::hardware_concurrency();
            if (0 == num_of_cores)
                num_of_cores = 1;     // assume one core

            running_ = true;
            while (num_threads-- != 0) {
                // create a new thread and set its affinity, the last thread 
                // is the master
                threads_.push_back(new boost::thread(
                    boost::bind(&threadmanager::tfunc, this, !num_threads)));
                set_affinity(threads_.back(), num_threads % num_of_cores);
            }

            // start timer pool as well
            timer_pool_.run(false);
        }
        catch (std::exception const& /*e*/) {
            stop();
            threads_.clear();
        }
        return running_;
    }

    void threadmanager::stop (bool blocking)
    {
        if (!threads_.empty()) {
            if (running_) {
                running_ = false;
                cond_.notify_all();         // make sure we're not waiting
            }

            if (blocking) {
                for (std::size_t i = 0; i < threads_.size(); ++i) 
                {
                    cond_.notify_all();     // make sure we're not waiting
                    threads_[i].join();
                }
            }

            timer_pool_.stop();             // stop timer pool as well
            if (blocking) 
                timer_pool_.join();
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    thread_state set_thread_state(thread_id_type id, thread_state new_state)
    {
        px_thread* t = static_cast<px_thread*>(id);
        return t->get_thread_manager().set_state(id, new_state);
    }

    ///////////////////////////////////////////////////////////////////////////
    thread_state set_thread_state(px_thread_self& self, thread_id_type id, 
        thread_state new_state)
    {
        px_thread* t = static_cast<px_thread*>(id);
        return t->get_thread_manager().set_state(self, id, new_state);
    }

}}
