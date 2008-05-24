//  Copyright (c) 2007-2008 Chirag Dekate, Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADMANAGER_MAY_20_2008_845AM)
#define HPX_THREADMANAGER_MAY_20_2008_845AM

#include <queue>

#include <boost/thread.hpp>
#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <hpx/hpx_fwd.hpp>
#include <hpx/naming/name.hpp>
#include <hpx/threadmanager/px_thread.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threadmanager
{
    ///////////////////////////////////////////////////////////////////////////
    struct unlock_the_lock;
    
    ///////////////////////////////////////////////////////////////////////////
    class threadmanager : private boost::noncopyable
    {
    private:
        typedef std::queue<hpx::threadmanager::px_thread> work_items_type;
        typedef boost::mutex mutex_type;

        friend struct unlock_the_lock;

    public:
        threadmanager() 
          : run_thread_(NULL), running_(false), do_some_work_(false)
        {}
        ~threadmanager() 
        {
            if(run_thread_) {
                if (running_) 
                    stop();
                delete run_thread_;
            }
        }

        /// This adds a new work item to the thread manager
        void register_work(
            boost::function<bool (hpx::threadmanager::px_thread_self&)> threadfunc);
        
        /// run the threadmanager's work queue
        bool run() 
        {
            mutex_type::scoped_lock lk(mtx_);
            if (run_thread_ || running_) 
                return true;    // do nothing if already running

            running_ = false;
            do_some_work_ = false;
            try {
                // run thread and wait for initialization to complete
                run_thread_ = new boost::thread(
                    boost::bind(&threadmanager::tfunc, this));
                running_ = true;
            }
            catch (std::exception const& /*e*/) {
                delete run_thread_;
                run_thread_ = NULL;
            }
            return running_;
        }

        /// forcefully stop the threadmanager
        void stop()
        {
            if (run_thread_ && running_) {
                {
                    mutex_type::scoped_lock lk(mtx_);
                    running_ = false;
                    cond_.notify_one();     // make sure we're not waiting
                }
                run_thread_->join();
            }
        }
        
        void wait()
        {
            if (run_thread_ && running_) 
                run_thread_->join();
        }
        
    public:
        /// this notifies the thread manager that there is some more work 
        /// available 
        void do_some_work()
        {
            mutex_type::scoped_lock lk(mtx_);
            if (running_) {
                cond_.notify_one();
                do_some_work_ = true;
            }
        }
        
    protected:
        // this is the thread function executing the work items in the queue
        void tfunc();

    private:
        boost::thread *run_thread_;         /// this thread manager has exactly one thread
        
        work_items_type work_items_;        /// list of active work items
        bool running_;                      /// thread manager has bee started
        bool do_some_work_;                 /// new work item(s) arrived
        mutex_type mtx_;                    /// mutex protecting the members
        boost::condition cond_;             /// used to trigger some action
    };
    
///////////////////////////////////////////////////////////////////////////////
}}

#endif 
