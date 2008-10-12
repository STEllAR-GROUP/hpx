//  Copyright (c) 2007-2008 Chirag Dekate, Hartmut Kaiser, Anshul Tandon
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADMANAGER_MAY_20_2008_845AM)
#define HPX_THREADMANAGER_MAY_20_2008_845AM

#include <map>
#include <vector>
#include <memory>

#include <hpx/config.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>
#include <boost/thread/condition.hpp>
#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/lockfree/fifo.hpp>
#if defined(HPX_DEBUG)
#include <boost/detail/atomic_count.hpp>
#endif

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/util/io_service_pool.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads
{
    // forward declaration only (needed for intrusive_ptr<thread>)
    void intrusive_ptr_add_ref(thread* p);
    void intrusive_ptr_release(thread* p);

    ///////////////////////////////////////////////////////////////////////////
    /// \class threadmanager threadmanager.hpp hpx/runtime/threads/threadmanager.hpp
    ///
    /// The \a threadmanager class is the central instance of management for
    /// all (non-depleted) \a thread's
    class threadmanager : private boost::noncopyable
    {
    private:
        // this is the type of the queue of pending threads
        typedef 
            boost::lockfree::fifo<boost::shared_ptr<thread> > 
        work_items_type;

        // this is the type of a map holding all threads (except depleted ones)
        typedef
            std::map<thread_id_type, boost::shared_ptr<thread> >
        thread_map_type;
        typedef thread_map_type::value_type map_pair;

        // we use a simple mutex to protect the data members of the 
        // threadmanager for now
        typedef boost::mutex mutex_type;

        // we use the boost::posix_time::ptime type for time representation
        typedef boost::posix_time::ptime time_type;

        // we use the boost::posix_time::time_duration type as the duration 
        // representation
        typedef boost::posix_time::time_duration duration_type;

    public:
        ///
        threadmanager(util::io_service_pool& timer_pool, 
            boost::function<void()> stop = boost::function<void()>());
        ~threadmanager();

        typedef boost::lockfree::fifo<thread_id_type> set_state_queue_type;

        /// The function \a register_work adds a new work item to the thread 
        /// manager. It creates a new \a thread, adds it to the internal
        /// management data structures, and schedules the new thread, if 
        /// appropriate.
        ///
        /// \param func   [in] The function or function object to execute as 
        ///               the thread's function. This must have a signature as
        ///               defined by \a thread_function_type.
        /// \param description [in] The value of this parameter allows to 
        ///               specify a description of the thread to create. This 
        ///               information is used for logging purposes mainly, but 
        ///               might be useful for debugging as well. This parameter 
        ///               is optional and defaults to an empty string.
        /// \param initial_state
        ///               [in] The value of this parameter defines the initial 
        ///               state of the newly created \a thread. This must be
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
            char const* const description = "", 
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
        /// \param self     [in] A reference to the thread executing this 
        ///                 function. 
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
        ///
        /// \note           This function yields the \a thread specified by
        ///                 the parameter \a self if the thread referenced by 
        ///                 the parameter \a id is in \a thread_state#active 
        ///                 state.
        thread_state set_state(thread_self& self, thread_id_type id, 
            thread_state newstate);

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
        ///
        /// \note           If the thread referenced by the parameter \a id is 
        ///                 in \a thread_state#active state this function does 
        ///                 nothing except returning thread_state#unknown. 
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
        naming::id_type get_thread_gid(thread_id_type id, 
            applier::applier& appl) const;

        /// Set a timer to set the state of the given \a thread to the given 
        /// new value after it expired (at the given time)
        thread_id_type set_state (time_type const& expire_at, 
            thread_id_type id, thread_state newstate = pending);

        /// Set a timer to set the state of the given \a thread to the given
        /// new value after it expired (after the given duration)
        thread_id_type set_state (duration_type const& expire_from_now, 
            thread_id_type id, thread_state newstate = pending);

    protected:
        // this is the thread function executing the work items in the queue
        void tfunc(std::size_t num_thread);
        std::size_t tfunc_impl(std::size_t num_thread);

    public:
        /// this notifies the thread manager that there is some more work 
        /// available 
        void do_some_work()
        {
            cond_.notify_all();
        }

    protected:
        /// This thread function is used by the at_timer thread below to trigger
        /// the required action.
        thread_state wake_timer_thread (thread_self& self, 
            thread_id_type id, thread_state newstate, thread_id_type timer_id);

        /// This thread function initiates the required set_state action (on 
        /// behalf of one of the threadmanager#set_state functions).
        template <typename TimeType>
        thread_state at_timer (thread_self& self, TimeType const& expire, 
            thread_id_type id, thread_state newstate);

        /// This function is the workhorse behind the two public set_state 
        /// functions 
        thread_state set_state(thread_self* self, thread_id_type id, 
            thread_state new_state);

    private:
        /// this thread manager has exactly as much threads as requested
        boost::ptr_vector<boost::thread> threads_;

        thread_map_type thread_map_;        ///< mapping of thread id's to threads
        work_items_type work_items_;        ///< list of active work items
        work_items_type terminated_items_;  ///< list of terminated threads
        set_state_queue_type active_set_state_;  ///< list of threads waiting for 
                                            ///< set_state on an active thread

        bool running_;                      ///< thread manager has bee started
        mutable mutex_type mtx_;            ///< mutex protecting the members
        boost::condition cond_;             ///< used to trigger some action

        util::io_service_pool& timer_pool_; ///< used for timed set_state
        boost::function<void()> stop_;  ///< function to call in case of error

#if HPX_DEBUG != 0
        boost::detail::atomic_count thread_count_;
#endif
    };

///////////////////////////////////////////////////////////////////////////////
}}

#include <hpx/config/warnings_suffix.hpp>

#endif 
