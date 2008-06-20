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
    px_thread::thread_id_type threadmanager::register_work(
        boost::function<thread_function_type> threadfunc, 
        thread_state initial_state)
    {
        // verify parameters
        if (initial_state != pending && initial_state != suspended)
        {
            boost::throw_exception(hpx::exception(hpx::bad_parameter));
            return invalid_thread_id;
        }

        boost::shared_ptr<px_thread> px_t_sp (
            new px_thread(threadfunc, *this, initial_state));

        // lock data members while adding work
        mutex_type::scoped_lock lk(mtx_);

        // add a new entry in the map for this thread
        thread_map_.insert(map_pair(px_t_sp->get_thread_id(), px_t_sp));

        // only insert in the work-items queue if it is in pending state
        if (initial_state == pending)
        {
            // pushing the new thread in the pending queue thread
            work_items_.push(px_t_sp);
            if (running_) 
                cond_.notify_one();       // try to execute the new work item
        }

        // return the thread_id of the newly created thread
        return px_t_sp->get_thread_id();
    }

#if defined(BOOST_WINDOWS)
    ///////////////////////////////////////////////////////////////////////////
    // On Windows we need a special preparation for the main coroutines thread 
    struct prepare_main_thread
    {
        prepare_main_thread() 
        {
            ConvertThreadToFiber(0);
        }
        ~prepare_main_thread() 
        {
            ConvertFiberToThread();
        }
    };
#else
    // All other platforms do not need any special treatment of the main thread
    struct prepare_main_thread
    {
        prepare_main_thread() {}
        ~prepare_main_thread() {}
    };
#endif

    /// The set_state function is part of the thread related API and allows
    /// to change the state of one of the threads managed by this 
    thread_state threadmanager::set_state(px_thread::thread_id_type id, 
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
            boost::shared_ptr<px_thread> px_t = map_iter->second;
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
                    work_items_.push(px_t);
            }
            return previous_state;
        }
        return unknown;
    }

    /// The set_state function is part of the thread related API and allows
    /// to query the state of one of the threads known to the threadmanager
    thread_state threadmanager::get_state(px_thread::thread_id_type id) const
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

    ///////////////////////////////////////////////////////////////////////////
    // main function executed by a OS thread
    void threadmanager::tfunc()
    {
        mutex_type::scoped_lock lk(mtx_);

        // run the work queue
        prepare_main_thread main_thread;
        while (running_ || !work_items_.empty()) 
        {
            if (!work_items_.empty())
            {
                // Get the next thread from the queue
                boost::shared_ptr<px_thread> thrd (work_items_.front());
                work_items_.pop();

                // Only pending threads will be executed.
                // Any non-pending threads are leftovers from a set_state() 
                // call for a previously pending thread (see comments above).
                thread_state state = thrd->get_state();
                if (state == pending)
                {
                    // make sure lock is unlocked during execution of work item
                    util::unlock_the_lock<mutex_type::scoped_lock> l(lk);    
                    state = (*thrd)();    // thread returns new required state
                }   // the lock gets locked again here!

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
    }

    // 
    boost::shared_ptr<px_thread> 
    threadmanager::get_thread(px_thread::thread_id_type id) const
    {
        // lock data members while getting a thread state
        mutex_type::scoped_lock lk(mtx_);

        thread_map_type::const_iterator map_iter = thread_map_.find(id);
        map_iter = thread_map_.find(id);
        if (map_iter != thread_map_.end())
        {
            return map_iter->second;
        }
        return boost::shared_ptr<px_thread>();
    }

    ///////////////////////////////////////////////////////////////////////////
    void set_thread_state(thread_id_type id, thread_state new_state)
    {
        components::wrapper<detail::px_thread>* t =
            static_cast<components::wrapper<detail::px_thread>*>(id);
        (*t)->get_thread_manager().set_state(id, new_state);
    }

}}
