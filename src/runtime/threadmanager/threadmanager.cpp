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
#if HPX_USE_LOCKFREE != 0
            work_items_.enqueue(px_t_sp);
#else
            work_items_.push(px_t_sp);
#endif
            if (running_ && run_now) 
                cond_.notify_one();       // try to execute the new work item
        }

        // return the thread_id of the newly created thread
        return px_t_sp->get_thread_id();
    }

    /// The set_state function is part of the thread related API and allows
    /// to change the state of one of the threads managed by this 
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
        map_iter = thread_map_.find(id);
        if (map_iter != thread_map_.end())
        {
            boost::intrusive_ptr<px_thread> px_t = map_iter->second;
            thread_state previous_state = px_t->get_state();

            // nothing to do here if the state doesn't change
            if (new_state == previous_state)
                return new_state;

            if (previous_state == active) {
            // do some juggling 
            // need to set the state of the thread to new_state,
            // not to what is returned by the active thread
            }
            else {
            // If the previous state was pending we are supposed to remove the
            // thread from the queue. But in order to avoid linearly looking 
            // through the queue we defer this to the thread function, which 
            // at some point will come across this thread simply skipping it 
            // (if it's not pending anymore). 
            // So all what we do here is to set the new state.
                px_t->set_state(new_state);
                if (new_state == pending)
                {
#if HPX_USE_LOCKFREE != 0
                    work_items_.enqueue(px_t);
#else
                    work_items_.push(px_t);
#endif
                }
                if (running_) 
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
        map_iter = thread_map_.find(id);
        if (map_iter != thread_map_.end())
        {
            return map_iter->second->get_state();
        }
        return unknown;
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
    // main function executed by a OS thread
    void threadmanager::tfunc()
    {
#if HPX_USE_LOCKFREE != 0
        // run the work queue
        boost::coroutines::prepare_main_thread main_thread;
        while (running_ || !work_items_.empty()) 
        {
            // Get the next thread from the queue
            boost::intrusive_ptr<px_thread> thrd;
            if (work_items_.dequeue(&thrd))
            {
                // Only pending threads will be executed.
                // Any non-pending threads are leftovers from a set_state() 
                // call for a previously pending thread (see comments above).
                thread_state state = thrd->get_state();
                if (pending == state) 
                {
                    // switch the state of the thread to active and back to 
                    // what the thread reports as its return value

                    switch_status thrd_stat (thrd.get(), active);
                    if (thrd_stat == pending) 
                    {
                        // thread returns new required state
                        // store the returned state in the thread
                        thrd_stat = state = (*thrd)();
                    }

                }   // this stores the new state in the thread

                // Re-add this work item to our list of work items if thread
                // should be re-scheduled. If the thread is suspended now we
                // just keep it in the map of threads.
                if (state == pending) 
                    work_items_.enqueue(thrd);

                // Remove the mapping from thread_map_ if thread is depleted or 
                // terminated, this will delete the px_thread as all references
                // go out of scope.
                // FIXME: what has to be done with depleted threads?
                if (state == depleted || state == terminated) 
                {
                    // all OS threads put their terminated threads into a 
                    // separate queue
                    terminated_items_.enqueue(thrd);

                    // only one dedicated OS thread is allowed to acquire the 
                    // lock for the purpose of deleting all terminated threads
                    if (boost::this_thread::get_id() == threads_[0].get_id()) 
                    {
                        mutex_type::scoped_lock lk(mtx_);
                        boost::intrusive_ptr<px_thread> todelete;
                        while (terminated_items_.dequeue(&todelete))
                            thread_map_.erase(todelete->get_thread_id());
                    }
                }
            }

            if (work_items_.empty()) {
                // stop running after all px_threads have been terminated
                if (!running_)
                    break;

                // wait until somebody needs some action (if no new work 
                // arrived in the meantime)
                mutex_type::scoped_lock lk(mtx_);
                cond_.wait(lk);
            }
        }
#else
        mutex_type::scoped_lock lk(mtx_);

        // run the work queue
        boost::coroutines::prepare_main_thread main_thread;
        while (running_ || !work_items_.empty()) 
        {
            if (!work_items_.empty())
            {
                // Get the next thread from the queue
                boost::intrusive_ptr<px_thread> thrd (work_items_.front());
                work_items_.pop();

                // Only pending threads will be executed.
                // Any non-pending threads are leftovers from a set_state() 
                // call for a previously pending thread (see comments above).
                thread_state state = thrd->get_state();
                if (state == pending)
                {
                    // switch the state of the thread to active and back to 
                    // what the thread reports as its return value
                    switch_status thrd_stat (thrd.get(), active);

                    {
                        // make sure lock is unlocked during execution of the 
                        // work item
                        util::unlock_the_lock<mutex_type::scoped_lock> l(lk);
                        state = (*thrd)();  // thread returns new required state
                    } // the lock gets locked again here!

                    // store the returned state in the thread
                    thrd_stat = state;
                }   // this stores the new state in the thread

                // Re-add this work item to our list of work items if thread
                // should be re-scheduled. If the thread is suspended now we
                // just keep it in the map of threads.
                if (state == pending) 
                    work_items_.push(thrd);

                // Remove the mapping from thread_map_ if thread is depleted or 
                // terminated, this will delete the px_thread as all references
                // go out of scope.
                // FIXME: what has to be done with depleted threads?
                if (state == depleted || state == terminated)
                    thread_map_.erase(thrd->get_thread_id());
            }

            if (work_items_.empty()) {
                // stop running after all px_threads have been terminated
                if (!running_)
                    break;

                // wait until somebody needs some action (if no new work 
                // arrived in the meantime)
                cond_.wait(lk);
            }
        }
#endif
    }

    // 
    boost::intrusive_ptr<px_thread> 
    threadmanager::get_thread(thread_id_type id) const
    {
        // lock data members while getting a thread state
        mutex_type::scoped_lock lk(mtx_);

        thread_map_type::const_iterator map_iter = thread_map_.find(id);
        map_iter = thread_map_.find(id);
        if (map_iter != thread_map_.end())
        {
            return map_iter->second;
        }
        return boost::intrusive_ptr<px_thread>();
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

        running_ = false;
        try {
            // run threads and wait for initialization to complete
            unsigned int num_of_cores = boost::thread::hardware_concurrency();
            if (0 == num_of_cores)
                num_of_cores = 1;     // assume one core

            running_ = true;
            while (num_threads-- != 0) {
                threads_.push_back(new boost::thread(
                    boost::bind(&threadmanager::tfunc, this)));
                set_affinity(threads_.back(), num_threads % num_of_cores);
            }
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
                cond_.notify_all();     // make sure we're not waiting
            }

            if (blocking) {
                for (std::size_t i = 0; i < threads_.size(); ++i) 
                {
                    threads_[i].join();
                    cond_.notify_all();     // make sure we're not waiting
                }
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    void set_thread_state(thread_id_type id, thread_state new_state)
    {
        px_thread* t = static_cast<px_thread*>(id);
        t->get_thread_manager().set_state(id, new_state);
    }

}}
