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
#include <boost/tuple/tuple.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/lockfree/fifo.hpp>
#if defined(HPX_DEBUG)
#include <boost/lockfree/atomic_int.hpp>
#endif

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/util/io_service_pool.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads
{
    ///////////////////////////////////////////////////////////////////////////
    /// \class threadmanager threadmanager.hpp hpx/runtime/threads/threadmanager.hpp
    ///
    /// The \a threadmanager class is the central instance of management for
    /// all (non-depleted) \a thread's
    class threadmanager : private boost::noncopyable
    {
    private:
        // this is the type of the queues of new or pending threads
        typedef 
            boost::lockfree::fifo<boost::shared_ptr<thread> > 
        work_items_type;

        // this is the type of the queue of new tasks not yet converted to
        // threads
        typedef boost::tuple<
            boost::function<thread_function_type>, thread_state, char const*
        > task_description;
        typedef boost::lockfree::fifo<task_description> task_items_type;

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

        // Add this number of threads to the work items queue each time the 
        // function \a add_new() is called if the queue is empty.
        enum { 
            min_add_new_count = 100, 
            max_add_new_count = 100,
            max_delete_count = 100
        };

        // The maximum number of active threads this thread manager should
        // create. This number will be a constraint only as long as the work
        // items queue is not empty. Otherwise the number of active threads 
        // will be incremented in steps equal to the \a min_add_new_count
        // specified above.
        enum { max_thread_count = 1000 };

    public:
        ///
        threadmanager(util::io_service_pool& timer_pool, 
            boost::function<void()> start_thread = boost::function<void()>(),
            boost::function<void()> stop = boost::function<void()>(),
            boost::function<void(boost::exception_ptr const&)> on_error =
                boost::function<void(boost::exception_ptr const&)>(),
            std::size_t max_count = max_thread_count);
        ~threadmanager();

        typedef boost::lockfree::fifo<thread_id_type> set_state_queue_type;

        /// The function \a register_work adds a new work item to the thread 
        /// manager. It doesn't immediately create a new \a thread, it just adds 
        /// the task parameters (function, initial state and description) to 
        /// the internal management data structures. The thread itself will be 
        /// created when the number of existing threads drops below the number
        /// of threads specified by the constructors max_count parameter.
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
        void
        register_work(boost::function<thread_function_type> const& func,
            char const* const description = "", 
            thread_state initial_state = pending, bool run_now = true);

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
        register_thread(boost::function<thread_function_type> const& threadfunc, 
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
        /// \param id       [in] The thread id of the thread the state should 
        ///                 be modified for.
        /// \param newstate [in] The new state to be set for the thread 
        ///                 referenced by the \a id parameter.
        /// \param newstate_ex [in] The new extended state to be set for the 
        ///                 thread referenced by the \a id parameter.
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
        thread_state set_state(thread_id_type id, thread_state newstate,
            thread_state_ex newstate_ex = wait_signaled);

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
        thread_state get_state(thread_id_type id);

        ///
        naming::id_type get_thread_gid(thread_id_type id);

        /// Set a timer to set the state of the given \a thread to the given 
        /// new value after it expired (at the given time)
        thread_id_type set_state (time_type const& expire_at, 
            thread_id_type id, thread_state newstate = pending,
            thread_state_ex newstate_ex = wait_timeout);

        /// Set a timer to set the state of the given \a thread to the given
        /// new value after it expired (after the given duration)
        thread_id_type set_state (duration_type const& expire_from_now, 
            thread_id_type id, thread_state newstate = pending,
            thread_state_ex newstate_ex = wait_timeout);

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

        /// 
        void report_error(boost::exception_ptr const& e)
        {
            if (on_error_)
                on_error_(e);
        }

    protected:
        /// This thread function is used by the at_timer thread below to trigger
        /// the required action.
        thread_state wake_timer_thread (thread_id_type id, 
            thread_state newstate, thread_state_ex newstate_ex, 
            thread_id_type timer_id);

        /// This thread function initiates the required set_state action (on 
        /// behalf of one of the threadmanager#set_state functions).
        template <typename TimeType>
        thread_state at_timer (TimeType const& expire, thread_id_type id, 
            thread_state newstate, thread_state_ex newstate_ex);

        /// This function adds threads stored in the new_items queue to the 
        /// thread map and the work_items queue (if appropriate)
        bool add_new(long add_count);
        bool add_new_if_possible();
        bool add_new_always();

        /// This function makes sure all threads which are marked for deletion
        /// (state is terminated) are properly destroyed
        bool cleanup_terminated();

    private:
        /// this thread manager has exactly as much threads as requested
        boost::ptr_vector<boost::thread> threads_;

        std::size_t max_count_;             ///< maximum number of existing PX-threads
        thread_map_type thread_map_;        ///< mapping of thread id's to PX-threads

        work_items_type work_items_;        ///< list of active work items
        work_items_type terminated_items_;  ///< list of terminated threads
        set_state_queue_type active_set_state_;  ///< list of threads waiting for 
                                            ///< set_state on an active thread

        task_items_type new_tasks_;         ///< list of new tasks to run

        bool running_;                      ///< thread manager has bee started
        mutable mutex_type mtx_;            ///< mutex protecting the members
        boost::condition cond_;             ///< used to trigger some action
        boost::lockfree::atomic_int<long> wait_count_;  ///< count waiting threads

        util::io_service_pool& timer_pool_; ///< used for timed set_state

        boost::function<void()> start_thread_;    ///< function to call for each created thread
        boost::function<void()> stop_;            ///< function to call in case of unexpected stop
        boost::function<void(boost::exception_ptr)> on_error_;  ///< function to call in case of error

#if HPX_DEBUG != 0
        boost::lockfree::atomic_int<long> thread_count_;
#endif
    };

///////////////////////////////////////////////////////////////////////////////
}}

#include <hpx/config/warnings_suffix.hpp>

#endif 
