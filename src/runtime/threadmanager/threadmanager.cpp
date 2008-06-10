//  Copyright (c) 2008-2009 Chirag Dekate, Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/asio.hpp>

#include <hpx/runtime/threadmanager/threadmanager.hpp>
#include <hpx/util/generate_unique_ids.hpp>

namespace hpx { namespace threadmanager
{
    ///////////////////////////////////////////////////////////////////////////
    void threadmanager::register_work( 
        boost::function<thread_function_type> threadfunc)
    {
        // lock queue when adding work
        mutex_type::scoped_lock lk(mtx_);
        work_items_.push(boost::shared_ptr<px_thread>(
            new px_thread(threadfunc, pending)));

        if (running_) 
            cond_.notify_one();           // try to execute the new work item
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
                boost::shared_ptr <hpx::threadmanager::px_thread> thrd (work_items_.front());
                work_items_.pop();

                // make sure lock is unlocked during execution of work item
                if (thrd->get_state() == pending)
                {
                    unlock_the_lock l(lk);    
                    new_state = (*thrd)();
                }
                
                // re-add this work item to our list of work items if appropriate
                if (new_state == pending) 
                    work_items_.push(thrd);
            }
            // if thread is suspended  then push to suspended map
            //CND

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
