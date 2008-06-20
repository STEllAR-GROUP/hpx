//  Copyright (c) 2007-2008 Chirag Dekate, Hartmut Kaiser, Anshul Tandon
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADMANAGER_MAY_20_2008_845AM)
#define HPX_THREADMANAGER_MAY_20_2008_845AM

#include <queue>
#include <map>

#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>
#include <boost/thread/condition.hpp>
#include <boost/function.hpp>
#include <boost/bind.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/name.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threadmanager
{
    ///////////////////////////////////////////////////////////////////////////
    class threadmanager : private boost::noncopyable
    {
    private:
        // this is the type of the queue of pending threads
        typedef std::queue <boost::shared_ptr<px_thread> > work_items_type;

        // this is the type of a map holding all threads (except depleted ones)
        typedef
            std::map<thread_id_type, boost::shared_ptr<px_thread> >
        thread_map_type;
        typedef thread_map_type::value_type map_pair;

        // we use a simple mutex to protect the data members of the 
        // threadmanager for now
        typedef boost::mutex mutex_type;

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

        /// The function \a register_work adds a new work item to the thread 
        /// manager. It creates a new \a px_thread, adds it to the internal
        /// management data structures, and schedules the new thread, if 
        /// appropriate.
        ///
        /// \param func   [in] The function or function object to execute as 
        ///               the thread's function. This must have a signature as
        ///               defined by \a thread_function_type.
        /// \param initial_state
        ///               [in] The value of this parameter defines the initial 
        ///               state of the newly created \a px_thread. This must be
        ///               one of the values as defined by the \a thread_state 
        ///               enumeration (thread_state#pending, or 
        ///               thread_state#suspended, any other value will throw a
        ///               hpx#bad_parameter exception).
        ///
        /// \returns      The function retunrs the thread id of the newly 
        ///               created thread. 
        thread_id_type 
        register_work(boost::function<thread_function_type> func,
            thread_state initial_state = pending);

        /// \brief Run the thread manager's work queue
        ///
        /// \returns      The function returns \a true if the thread manager
        ///               has been started successfully, otherwise it returns 
        ///               \a false.
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

        /// \brief Forcefully stop the threadmanager
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
        thread_state set_state(thread_id_type id, thread_state new_state);

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
        thread_state get_state(thread_id_type id) const;

        ///
        boost::shared_ptr<px_thread> get_thread(thread_id_type id) const;

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

        thread_map_type thread_map_;        /// mapping of thread id's to threads
        work_items_type work_items_;        /// list of active work items

        bool running_;                      /// thread manager has bee started
        mutable mutex_type mtx_;            /// mutex protecting the members
        boost::condition cond_;             /// used to trigger some action
    };

    ///////////////////////////////////////////////////////////////////////////
    void set_thread_state(thread_id_type id, thread_state new_state);

///////////////////////////////////////////////////////////////////////////////
}}

#endif 
