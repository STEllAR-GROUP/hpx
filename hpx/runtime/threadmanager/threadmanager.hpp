//  Copyright (c) 2007-2008 Chirag Dekate, Hartmut Kaiser, Anshul Tandon
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADMANAGER_MAY_20_2008_845AM)
#define HPX_THREADMANAGER_MAY_20_2008_845AM

#include <queue>
#include <map>
#include <vector>
#include <memory>

#include <hpx/config.hpp>

#include <boost/intrusive_ptr.hpp>
#include <boost/thread.hpp>
#include <boost/thread/condition.hpp>
#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#if HPX_USE_LOCKFREE != 0
#include <boost/lockfree/fifo.hpp>
#endif

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/naming/name.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threadmanager
{
    // forward declaration only (needed for intrusive_ptr<px_thread>)
    void intrusive_ptr_add_ref(px_thread* p);
    void intrusive_ptr_release(px_thread* p);

    ///////////////////////////////////////////////////////////////////////////
    /// \class threadmanager threadmanager.hpp hpx/runtime/threadmanager/threadmanager.hpp
    ///
    /// The \a threadmanager class is the central instance of management for
    /// all (non-depleted) \a px_thread's
    class threadmanager : private boost::noncopyable
    {
    private:
        // this is the type of the queue of pending threads
#if HPX_USE_LOCKFREE != 0
        typedef boost::lockfree::fifo<boost::intrusive_ptr<px_thread> > work_items_type;
#else
        typedef std::queue <boost::intrusive_ptr<px_thread> > work_items_type;
#endif

        // this is the type of a map holding all threads (except depleted ones)
        typedef
            std::map<thread_id_type, boost::intrusive_ptr<px_thread> >
        thread_map_type;
        typedef thread_map_type::value_type map_pair;

        // we use a simple mutex to protect the data members of the 
        // threadmanager for now
        typedef boost::mutex mutex_type;

    public:
        threadmanager() 
          : running_(false)
        {}
        ~threadmanager() 
        {
            if (!threads_.empty()) {
                if (running_) 
                    stop();
                threads_.clear();
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
        ///               enumeration (thread_state#pending, or \a
        ///               thread_state#suspended, any other value will throw a
        ///               hpx#bad_parameter exception).
        /// \param run_now [in] If this parameter is \a true and the initial 
        ///               state is given as \a thread_state#pending the thread 
        ///               will be run immediately, otherwise it will be 
        ///               scheduled to run later (either this function is 
        ///               called for another thread using \a true for the
        ///               parameter \a run_now or the function \a 
        ///               threadmanager#do_some_work is called). This parameter
        ///               is optional and defaults to \a true.
        ///
        /// \returns      The function returns the thread id of the newly 
        ///               created thread. 
        thread_id_type 
        register_work(boost::function<thread_function_type> func,
            thread_state initial_state = pending, bool run_now = true);

        /// \brief  Run the thread manager's work queue. This function 
        ///         instantiates the specified number of OS threads. All OS
        ///         threads are started to execute the function \a tfunc.
        ///
        /// \param num_threads
        ///               [in] The initial number of threads to be started by
        ///               this thread manager instance. This parameter is 
        ///               optional and defaults to 1 (one).
        ///
        /// \returns      The function returns \a true if the thread manager
        ///               has been started successfully, otherwise it returns 
        ///               \a false.
        bool run(std::size_t num_threads = 1);

        /// \brief Forcefully stop the threadmanager
        ///
        /// \param blocking
        ///
        void stop (bool blocking = true);

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
        thread_state set_state(thread_id_type id, thread_state newstate);

        /// The get_state function is part of the thread related API and allows
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
        boost::intrusive_ptr<px_thread> get_thread(thread_id_type id) const;

    protected:
        // this is the thread function executing the work items in the queue
        void tfunc();

    public:
        /// this notifies the thread manager that there is some more work 
        /// available 
        void do_some_work(bool runall = true)
        {
            if (running_) {
                if (runall)
                    cond_.notify_all();
                else
                    cond_.notify_one();
            }
        }

    private:
        /// this thread manager has exactly as much threads as requested
        boost::ptr_vector<boost::thread> threads_;

        thread_map_type thread_map_;        ///< mapping of thread id's to threads
        work_items_type work_items_;        ///< list of active work items
        work_items_type terminated_items_;  ///< list of terminated threads

        bool running_;                      ///< thread manager has bee started
        mutable mutex_type mtx_;            ///< mutex protecting the members
        boost::condition cond_;             ///< used to trigger some action
    };

    ///////////////////////////////////////////////////////////////////////////
    void set_thread_state(thread_id_type id, thread_state new_state);

///////////////////////////////////////////////////////////////////////////////
}}

#endif 
