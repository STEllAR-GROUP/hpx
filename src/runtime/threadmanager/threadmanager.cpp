//  Copyright (c) 2008-2009 Chirag Dekate, Hartmut Kaiser, Anshul Tandon
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/asio.hpp>
#include <boost/assert.hpp>

#include <hpx/runtime/threadmanager/threadmanager.hpp>
#include <hpx/util/generate_unique_ids.hpp>

namespace hpx { namespace threadmanager
{
    ///////////////////////////////////////////////////////////////////////////
    px_thread::thread_id_type threadmanager::register_work( 
        boost::function<thread_function_type> threadfunc, thread_state initial_state)
    {
        // throw an error if the initial state is neither pending nor suspended
        if (!(initial_state == pending || initial_state == suspended))
        {
            throw hpx::exception(hpx::bad_parameter);
        }

        boost::shared_ptr<px_thread> px_t_sp(new px_thread(threadfunc, initial_state));

        // lock queue when adding work
        mutex_type::scoped_lock lk(mtx_);

        // add a new entry in the std::map for this thread
        thread_map_.insert(map_pair(px_t_sp->get_thread_id(), px_t_sp));

        // only insert in the work-items queue if it is pending
        if (initial_state == pending)
        {
            // pushing the new thread in the pending queue thread
            work_items_.push(px_t_sp);

            if (running_) 
                cond_.notify_one();           // try to execute the new work item
        }

        // return the thread_id of the newly created thread
        return px_t_sp->get_thread_id();
    }

    ///////////////////////////////////////////////////////////////////////////
    // This is a helper structure to make sure a lock get's unlocked and locked
    // again in a scope.
    struct unlock_the_lock
    {
        unlock_the_lock(threadmanager::mutex_type::scoped_lock& l) : l_(l) 
        {
            l_.unlock();
        }
        ~unlock_the_lock()
        {
            l_.lock();
        }
        threadmanager::mutex_type::scoped_lock& l_;
    };
        
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
    thread_state threadmanager::set_state(px_thread::thread_id_type id, thread_state new_state)
    {
        mutex_type::scoped_lock lk(mtx_);
        thread_map_type :: iterator map_iter;
        map_iter = thread_map_.find(id);
        if (map_iter != thread_map_.end())
        {
            boost::shared_ptr<px_thread> px_t = map_iter->second;
            thread_state previous_state = px_t->get_state();

            if (previous_state == active);
            // do some juggling 
            // need to set the state of the thread to new_state,
            // not to what is returned by the active thread
            else
                px_t->set_state(new_state);

            if (new_state == pending)
                work_items_.push(px_t);

            return previous_state;
        }
        return unknown;
    }

    /// The set_state function is part of the thread related API and allows
    /// to query the state of one of the threads known to the threadmanager
    thread_state threadmanager::get_state(px_thread::thread_id_type id) const
    {
        mutex_type::scoped_lock lk(mtx_);
        thread_map_type :: const_iterator map_iter;
        map_iter = thread_map_.find(id);
        if (map_iter != thread_map_.end())
        {
            boost::shared_ptr<px_thread> px_t = map_iter->second;
            return px_t->get_state();
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
        while (running_) 
        {
            thread_state new_state = unknown;
            if (!(work_items_.empty()))
            {
                boost::shared_ptr<px_thread> thrd (work_items_.front());
                work_items_.pop();

                // make sure lock is unlocked during execution of work item
                BOOST_ASSERT(thrd->get_state() == pending);
                {
                    unlock_the_lock l(lk);    
                    new_state = (*thrd)();
                }   // the lock gets locked here!
                
                // re-add this work item to our list of work items if appropriate
                if (new_state == pending) 
                    work_items_.push(thrd);

                // remove the mapping from thread_set_ if thread is depleted or terminated
                if (new_state == depleted || new_state == terminated)
                    thread_map_.erase(thrd->get_thread_id());
            }

            // try to execute as much work as available, but try not to 
            // schedule a certain component more than once
            if (work_items_.empty()) {
                // wait until somebody needs some action (if no new work 
                // arrived in the meantime)
                cond_.wait(lk);
            }
        }
    }

}}
