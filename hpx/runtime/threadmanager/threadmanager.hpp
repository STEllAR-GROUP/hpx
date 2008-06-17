//  Copyright (c) 2007-2008 Chirag Dekate, Hartmut Kaiser, Anshul Tandon
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADMANAGER_MAY_20_2008_845AM)
#define HPX_THREADMANAGER_MAY_20_2008_845AM

#include <queue>
#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>
#include <boost/thread/condition.hpp>
#include <boost/function.hpp>
#include <boost/bind.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/threadmanager/px_thread.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threadmanager
{
    ///////////////////////////////////////////////////////////////////////////
    struct unlock_the_lock;

    ///////////////////////////////////////////////////////////////////////////
    class threadmanager : private boost::noncopyable
    {
    private:
        typedef 
            std::queue <boost::shared_ptr<hpx::threadmanager::px_thread> > 
        work_items_type;

        typedef
            std::map <px_thread::thread_id_type, boost::shared_ptr<px_thread>>
        thread_map_type;

        typedef std::pair <px_thread::thread_id_type, boost::shared_ptr<px_thread>> map_pair;

        typedef boost::mutex mutex_type;
        friend struct unlock_the_lock;

    public:
        threadmanager() 
          : run_thread_(NULL), running_(false)
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
        void register_work(boost::function<thread_function_type> threadfunc);
        
        /// run the threadmanager's work queue
        bool run() 
        {
            mutex_type::scoped_lock lk(mtx_);
            if (run_thread_ || running_) 
                return true;    // do nothing if already running

            running_ = false;
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
        void stop(bool blocking = true)
        {
            if (run_thread_) {
                if (running_) {
                    mutex_type::scoped_lock lk(mtx_);
                    running_ = false;
                    cond_.notify_all();     // make sure we're not waiting
                }
                if (blocking)
                    run_thread_->join();
            }
        }

        void wait()
        {
            if (run_thread_ && running_) 
                run_thread_->join();
        }

        /// The set_state function is part of the thread related API and allows
        /// to change the state of one of the threads managed by this 
        /// threadmanager.
        ///
        /// \param id       [in] The thread id of the thread the state should 
        ///                 be modified for.
        /// \param newstate [in] The new state to be set for the thread 
        ///                 referenced by the \a id parameter.
        ///
        /// \returns        This function returns the previous state of the 
        ///                 thread referenced by the \a id parameter. It will 
        ///                 return one of the values as defined by the 
        ///                 \a thread_state enumeration. If the 
        ///                 thread is not known to the threadmanager the return 
        ///                 value will be \a thread_state#unknown.
        thread_state set_state(px_thread::thread_id_type id, 
            thread_state new_state)
        {
            map_iter_ = thread_map_.find(id);
            if (map_iter_ != thread_map_.end())
            {
                boost::shared_ptr<px_thread> px_t = map_iter_->second;
                thread_state previous_state = px_t->get_state();

                if (previous_state == active);
                    // do some juggling
                else
                    px_t->set_state(new_state);
                return previous_state;
            }
            return unknown;
        }

        /// The set_state function is part of the thread related API and allows
        /// to query the state of one of the threads known to the threadmanager
        ///
        /// \param id       [in] The thread id of the thread the state should 
        ///                 be returned for.
        ///
        /// \returns        This function returns the current state of the 
        ///                 thread referenced by the \a id parameter. It will 
        ///                 return one of the values as defined by the 
        ///                 \a thread_state enumeration. If the 
        ///                 thread is not known to the threadmanager the return 
        ///                 value will be \a thread_state#unknown.
        thread_state get_state(px_thread::thread_id_type id)
        {
            map_iter_ = thread_map_.find(id);
            if (map_iter_ != thread_map_.end())
            {
                boost::shared_ptr<px_thread> px_t = map_iter_->second;
                return px_t->get_state();
            }
            return unknown;
        }

    public:
        /// this notifies the thread manager that there is some more work 
        /// available 
        void do_some_work()
       { 
            mutex_type::scoped_lock lk(mtx_);
            if (running_) 
                cond_.notify_one();
        }

    protected:
        // this is the thread function executing the work items in the queue
        void tfunc();

    private:
        boost::thread *run_thread_;         /// this thread manager has exactly one thread
        
        thread_map_type thread_map_;        /// mapping of LVAs of threads
        std::map <px_thread::thread_id_type, boost::shared_ptr<px_thread>> :: const_iterator map_iter_;

        work_items_type work_items_;        /// list of active work items
        bool running_;                      /// thread manager has bee started
        mutex_type mtx_;                    /// mutex protecting the members
        boost::condition cond_;             /// used to trigger some action
    };

///////////////////////////////////////////////////////////////////////////////
}}

#endif 
