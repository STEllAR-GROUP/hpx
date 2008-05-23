//  Copyright (c) 2008-2009 Chirag Dekate, Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/asio.hpp>

#include <hpx/threadmanager/threadmanager.hpp>
#include <hpx/util/generate_unique_ids.hpp>
#include <hpx/naming/namespace.hpp>

namespace hpx { namespace threadmanager
{
    ///////////////////////////////////////////////////////////////////////////
    void threadmanager::register_work( 
		boost::function<bool (hpx::threadmanager::px_thread_self&)> threadfunc)
    {
        // create a new name for this new thread
//        naming::id_type n = ps_.get_prefix() | 
//             util::unique_ids::instance->get_thread_id();

        // lock queue when adding work
        mutex_type::scoped_lock lk(mtx_);
		work_items_.push(hpx::threadmanager::px_thread(threadfunc));
            
        if (running_) {
            do_some_work_ = true;
            cond_.notify_one();           // try to execute the new work item
        }
//        return n;
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
        
    ///////////////////////////////////////////////////////////////////////////
    // On Windows we need a special preparation for the main coroutines thread 
#if defined(BOOST_WINDOWS)
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
            bool exited = true;
			hpx::threadmanager::px_thread thrd (work_items_.front());
            work_items_.pop();

            // make sure lock is unlocked during execution of work item
            {
                unlock_the_lock l(lk);    
                exited = thrd();
            }
            
            // re-add this work item to our list of work items if appropriate
            if (!exited) 
                work_items_.push(thrd);
            
            // try to execute as much work as available, but try not to 
            // schedule a certain component more than once
            if (work_items_.empty()) {
                // wait until somebody needs some action (if no new work 
                // arrived in the meantime)
                if (!do_some_work_)
                    cond_.wait(lk);
                do_some_work_ = false;
            }
        }
    }

}}
