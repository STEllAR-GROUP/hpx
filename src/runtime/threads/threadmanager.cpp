//  Copyright (c) 2008-2009 Chirag Dekate, Hartmut Kaiser, Anshul Tandon
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <boost/assert.hpp>

#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/util/unlock_lock.hpp>
#include <hpx/util/logging.hpp>

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
    struct init_logging
    {
        init_logging()
        {
            util::init_threadmanager_logs();
        }
    };
    
    init_logging const init_tm_logging;

    ///////////////////////////////////////////////////////////////////////////
    threadmanager::threadmanager(util::io_service_pool& timer_pool)
      : running_(false), timer_pool_(timer_pool)
#if HPX_DEBUG != 0
      , thread_count_(0)
#endif
    {
        LTM_(info) << "threadmanager ctor";
    }

    ///////////////////////////////////////////////////////////////////////////
    threadmanager::~threadmanager() 
    {
        LTM_(info) << "~threadmanager";
        if (!threads_.empty()) {
            if (running_) 
                stop();
            threads_.clear();
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    thread_id_type threadmanager::register_work(
        boost::function<thread_function_type> threadfunc, 
        thread_state initial_state, bool run_now)
    {
        LTM_(info) << "register_work: initial_state(" 
                   << get_thread_state_name(initial_state) << "), "
                   << std::boolalpha << "run_now(" << run_now << ")";

        // verify parameters
        if (initial_state != pending && initial_state != suspended)
        {
            boost::throw_exception(hpx::exception(hpx::bad_parameter));
            return invalid_thread_id;
        }

        // create the new thread
        boost::intrusive_ptr<thread> thrd (
            new thread(threadfunc, *this, initial_state));

        // lock data members while adding work
        mutex_type::scoped_lock lk(mtx_);

        // add a new entry in the map for this thread
        thread_map_.insert(map_pair(thrd->get_thread_id(), thrd));

        // only insert in the work-items queue if it is in pending state
        if (initial_state == pending)
        {
            // pushing the new thread in the pending queue thread
            work_items_.enqueue(thrd);
            if (run_now) 
                cond_.notify_all();       // try to execute the new work item
        }

        // return the thread_id of the newly created thread
        return thrd->get_thread_id();
    }

    /// The set_state function is part of the thread related API and allows
    /// to change the state of one of the threads managed by this threadmanager
    inline thread_state threadmanager::set_state(thread_self& self, 
        thread_id_type id, thread_state newstate)
    {
        return set_state(&self, id, newstate);
    }

    /// The set_state function is part of the thread related API and allows
    /// to change the state of one of the threads managed by this threadmanager
    inline thread_state threadmanager::set_state(thread_id_type id, 
        thread_state newstate)
    {
        return set_state(NULL, id, newstate);
    }

    thread_state threadmanager::set_state(thread_self* self, thread_id_type id, 
        thread_state new_state)
    {
        // set_state can't be used to force a thread into active state
        if (new_state == active) {
            boost::throw_exception(hpx::exception(hpx::bad_parameter));
            return unknown;
        }

        // lock data members while setting a thread state
        mutex_type::scoped_lock lk(mtx_);

        thread_map_type::iterator map_iter = thread_map_.find(id);
        if (map_iter != thread_map_.end())
        {
            boost::intrusive_ptr<thread> thrd = map_iter->second;
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
                if (NULL == self) 
                    return unknown;

                do {
                    active_set_state_.enqueue(self->get_thread_id());
                    util::unlock_the_lock<mutex_type::scoped_lock> ul(lk);
                    self->yield(suspended);
                } while ((previous_state = thrd->get_state()) == active);
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
            thrd->set_state(new_state);
            if (new_state == pending) {
                work_items_.enqueue(thrd);
                cond_.notify_all();
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
    thread_state threadmanager::wake_timer_thread (thread_self& self, 
        thread_id_type id, thread_state newstate, thread_id_type timer_id) 
    {
        // first trigger the requested set_state 
        set_state(self, id, newstate);

        // then re-activate the thread holding the deadline_timer
        set_state(self, timer_id, pending);
        return terminated;
    }

    /// This thread function initiates the required set_state action (on 
    /// behalf of one of the threadmanager#set_state functions).
    template <typename TimeType>
    thread_state threadmanager::at_timer (thread_self& self, 
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

    /// Set a timer to set the state of the given \a thread to the given 
    /// new value after it expired (at the given time)
    thread_id_type threadmanager::set_state (time_type const& expire_at, 
        thread_id_type id, thread_state newstate)
    {
        // this creates a new thread which creates the timer and handles the
        // requested actions
        thread_state (threadmanager::*f)(thread_self&, time_type const&,
                thread_id_type, thread_state)
            = &threadmanager::at_timer<time_type>;

        return register_work(boost::bind(f, this, _1, expire_at, id, newstate));
    }

    /// Set a timer to set the state of the given \a thread to the given
    /// new value after it expired (after the given duration)
    thread_id_type threadmanager::set_state (
        duration_type const& from_now, thread_id_type id, 
        thread_state newstate)
    {
        // this creates a new thread which creates the timer and handles the
        // requested actions
        thread_state (threadmanager::*f)(thread_self&, duration_type const&,
                thread_id_type, thread_state)
            = &threadmanager::at_timer<duration_type>;

        return register_work(boost::bind(f, this, _1, from_now, id, newstate));
    }

    /// Retrieve the global id of the given thread
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
        switch_status (thread* t, thread_state new_state)
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
        // used inside a block holding a intrusive_ptr to this PX thread
        thread* thread_;
        thread_state prev_state_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Queue, typename Map>
    inline bool cleanup(Queue& terminated_items, Map& thread_map)
    {
        boost::intrusive_ptr<thread> todelete;
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

            // copy the PX threads which are still active to the main queue
            if (!still_active.empty()) {
                while (still_active.dequeue(&id)) 
                    active_set_state.enqueue(id);
            }
        }
    }

#if defined(_WIN32) || defined(_WIN64)
    bool set_affinity(boost::thread& thrd, unsigned int num_thread)
    {
        unsigned int num_of_cores = boost::thread::hardware_concurrency();
        if (0 == num_of_cores)
            num_of_cores = 1;     // assume one core
        unsigned int affinity = num_thread % num_of_cores;

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
            return SetThreadAffinityMask(thrd.native_handle(), mask) != 0;
        }
        return false;
    }
    inline bool set_affinity(unsigned int affinity)
    {
        return true;
    }
#else
    #include <sched.h>    // declares the scheduling interface

    inline bool set_affinity(boost::thread& thrd, unsigned int affinity)
    {
        return true;
    }
    bool set_affinity(unsigned int num_thread)
    {
        unsigned int num_of_cores = boost::thread::hardware_concurrency();
        if (0 == num_of_cores)
            num_of_cores = 1;     // assume one core
        unsigned int affinity = num_thread % num_of_cores;

        cpu_set_t cpu;
        CPU_ZERO(&cpu);
        CPU_SET(affinity, &cpu);
        if (sched_setaffinity (0, sizeof(cpu), &cpu) == 0) {
            sleep(0);   // allow the OS to pick up the change
            return true;
        }
        return false;
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    // main function executed by all OS threads managed by this threadmanager
    void threadmanager::tfunc(std::size_t num_thread)
    {
        LTM_(info) << "tfunc(" << num_thread << "): start";

#if HPX_DEBUG != 0
        ++thread_count_;
#endif
        std::size_t num_px_threads = 0;

        // the thread with number zero is the master
        bool is_master_thread = (0 == num_thread) ? true : false;
        set_affinity(num_thread);     // set affinity on Linux systems

        // run the work queue
        boost::coroutines::prepare_main_thread main_thread;
        while (true) {
            // Get the next PX thread from the queue
            boost::intrusive_ptr<thread> thrd;
            if (work_items_.dequeue(&thrd)) {
                // Only pending PX threads will be executed.
                // Any non-pending PX threads are leftovers from a set_state() 
                // call for a previously pending PX thread (see comments above).
                thread_state state = thrd->get_state();

//                 LTM_(info) << "tfunc(" << num_thread << "): "
//                            << "thread(" << thrd->get_thread_id() << "), "
//                            << "old state(" << get_thread_state_name(state) << ")";

                if (pending == state) {
                    // switch the state of the thread to active and back to 
                    // what the thread reports as its return value

                    switch_status thrd_stat (thrd.get(), active);
                    if (thrd_stat == pending) {
                        // thread returns new required state
                        // store the returned state in the thread
                        thrd_stat = state = (*thrd)();
                        ++num_px_threads;
                    }

                }   // this stores the new state in the PX thread

//                 LTM_(info) << "tfunc(" << num_thread << "): "
//                            << "thread(" << thrd->get_thread_id() << "), "
//                            << "new state(" << get_thread_state_name(state) << ")";

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

                // only one dedicated OS thread is allowed to acquire the 
                // lock for the purpose of deleting all terminated threads
                if (is_master_thread && !terminated_items_.empty()) {
                    mutex_type::scoped_lock lk(mtx_);
                    cleanup(terminated_items_, thread_map_);
                }
            }

            // make sure to handle pending set_state requests
            if (is_master_thread) 
                handle_pending_set_state(*this, active_set_state_);

            // if nothing else has to be done either wait or terminate
            if (work_items_.empty() && active_set_state_.empty()) {
                // no obvious work has to be done, so a lock won't hurt too much
                mutex_type::scoped_lock lk(mtx_);

                LTM_(info) << "tfunc(" << num_thread << "): "
                           << "queues empty";

                // stop running after all PX threads have been terminated
                if (!running_) {
                    // Before exiting each of the OS threads deletes the 
                    // remaining terminated PX threads 
                    if (cleanup(terminated_items_, thread_map_)) {
                        cond_.notify_all();   // notify possibly waiting threads
                        break;                // terminate scheduling loop
                    }

                    LTM_(info) << "tfunc(" << num_thread << "): "
                               << "threadmap not empty";
                }

                // Wait until somebody needs some action (if no new work 
                // arrived in the meantime).
                // Ask again if queues are empty to avoid race conditions (we 
                // need to lock anyways...), this way no notify_all() gets lost
                if (work_items_.empty() && active_set_state_.empty())
                {
                    LTM_(info) << "tfunc(" << num_thread << "): "
                               << "queues empty, entering wait";
                    cond_.wait(lk);
                    LTM_(info) << "tfunc(" << num_thread << "): exiting wait";
                }
            }
        }

#if HPX_DEBUG != 0
        // the last OS thread is allowed to exit only if no more PX threads exist
        BOOST_ASSERT(0 != --thread_count_ || thread_map_.empty());
#endif
        LTM_(info) << "tfunc(" << num_thread << "): end, executed " 
                   << num_px_threads << " HPX threads";
    }

    ///////////////////////////////////////////////////////////////////////////
    bool threadmanager::run(std::size_t num_threads) 
    {
        LTM_(info) << "run: creating " << num_threads << " threads";

        if (0 == num_threads) {
            boost::throw_exception(hpx::exception(
                bad_parameter, "Number of threads shouldn't be zero"));
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
            LTM_(error) << "run: failed with:" << e.what(); 
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
            }
            threads_.clear();

            LTM_(info) << "stop: stopping timer pool"; 
            timer_pool_.stop();             // stop timer pool as well
            if (blocking) 
                timer_pool_.join();
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    thread_state set_thread_state(thread_id_type id, thread_state new_state)
    {
        thread* t = static_cast<thread*>(id);
        return t->get_thread_manager().set_state(id, new_state);
    }

    ///////////////////////////////////////////////////////////////////////////
    thread_state set_thread_state(thread_self& self, thread_id_type id, 
        thread_state new_state)
    {
        thread* t = static_cast<thread*>(id);
        return t->get_thread_manager().set_state(self, id, new_state);
    }

    ///////////////////////////////////////////////////////////////////////
    thread_id_type set_thread_state(thread_id_type id, thread_state state, 
        boost::posix_time::ptime const& at_time)
    {
        thread* t = static_cast<thread*>(id);
        return t->get_thread_manager().set_state(at_time, id, state);
    }

    ///////////////////////////////////////////////////////////////////////
    thread_id_type set_thread_state(thread_id_type id, thread_state state,
        boost::posix_time::time_duration const& after)
    {
        thread* t = static_cast<thread*>(id);
        return t->get_thread_manager().set_state(after, id, state);
    }

}}
