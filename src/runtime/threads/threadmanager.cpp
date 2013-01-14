//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach, Katelyn Kufahl
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/runtime/threads/threadmanager_impl.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/threads/detail/scheduling_loop.hpp>
#include <hpx/runtime/threads/detail/create_thread.hpp>
#include <hpx/runtime/threads/detail/create_work.hpp>
#include <hpx/runtime/threads/detail/set_thread_state.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/performance_counters/counter_creators.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/util/unlock_lock.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/block_profiler.hpp>
#include <hpx/util/itt_notify.hpp>
#include <hpx/util/stringstream.hpp>
#include <hpx/util/hardware/timestamp.hpp>

#include <boost/assert.hpp>
#include <boost/make_shared.hpp>
#include <boost/bind.hpp>
#include <boost/asio/deadline_timer.hpp>
#include <boost/cstdint.hpp>
#include <boost/format.hpp>

#include <numeric>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads
{
    ///////////////////////////////////////////////////////////////////////////
    namespace strings
    {
        char const* const thread_state_names[] =
        {
            "unknown",
            "active",
            "pending",
            "suspended",
            "depleted",
            "terminated"
        };
    }

    char const* get_thread_state_name(thread_state_enum state)
    {
        if (state < unknown || state > terminated)
            return "unknown";
        return strings::thread_state_names[state];
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace strings
    {
        char const* const thread_priority_names[] =
        {
            "default",
            "low",
            "normal",
            "critical"
        };
    }

    char const* get_thread_priority_name(thread_priority priority)
    {
        if (priority < thread_priority_default || priority > thread_priority_critical)
            return "unknown";
        return strings::thread_priority_names[priority];
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy>
    threadmanager_impl<SchedulingPolicy, NotificationPolicy>::threadmanager_impl(
            util::io_service_pool& timer_pool,
            scheduling_policy_type& scheduler,
            notification_policy_type& notifier,
            std::size_t num_threads)
      : startup_(NULL),
        thread_count_(0),
        state_(starting),
        timer_pool_(timer_pool),
        thread_logger_("threadmanager_impl::register_thread"),
        work_logger_("threadmanager_impl::register_work"),
        set_state_logger_("threadmanager_impl::set_state"),
        scheduler_(scheduler),
        notifier_(notifier),
        used_processing_units_(0)
    {
        for (std::size_t i = 0; i < num_threads; ++i)
            used_processing_units_ |= scheduler_.get_pu_mask(get_topology(), i);
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    threadmanager_impl<SchedulingPolicy, NotificationPolicy>::~threadmanager_impl()
    {
        //LTM_(debug) << "~threadmanager_impl";
        if (!threads_.empty()) {
            if (state_.load() == running)
                stop();
            threads_.clear();
        }
        delete startup_;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy>
    boost::int64_t threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        get_thread_count(thread_state_enum state) const
    {
        mutex_type::scoped_lock lk(mtx_);
        return scheduler_.get_thread_count(state);
    }

    ///////////////////////////////////////////////////////////////////////////
    // \brief Abort all threads which are in suspended state. This will set
    //        the state of all suspended threads to \a pending while
    //        supplying the wait_abort extended state flag
    template <typename SchedulingPolicy, typename NotificationPolicy>
    void threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        abort_all_suspended_threads()
    {
        mutex_type::scoped_lock lk(mtx_);
        scheduler_.abort_all_suspended_threads();
    }

    ///////////////////////////////////////////////////////////////////////////
    // \brief Clean up terminated threads. This deletes all threads which
    //        have been terminated but which are still held in the queue
    //        of terminated threads. Some schedulers might not do anything
    //        here.
    template <typename SchedulingPolicy, typename NotificationPolicy>
    bool threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        cleanup_terminated(bool delete_all)
    {
        mutex_type::scoped_lock lk(mtx_);
        return scheduler_.cleanup_terminated(delete_all);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy>
    thread_id_type threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        register_thread(thread_init_data& data, thread_state_enum initial_state,
            bool run_now, error_code& ec)
    {
        util::block_profiler_wrapper<register_thread_tag> bp(thread_logger_);

        // verify state
        if (thread_count_ == 0 && state_ != running)
        {
            // thread-manager is not currently running
            HPX_THROWS_IF(ec, invalid_status,
                "threadmanager_impl::register_thread",
                "invalid state: thread manager is not running");
            return invalid_thread_id;
        }

        return detail::create_thread(scheduler_, data, initial_state, run_now, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy>
    void threadmanager_impl<SchedulingPolicy, NotificationPolicy>::register_work(
        thread_init_data& data, thread_state_enum initial_state, error_code& ec)
    {
        util::block_profiler_wrapper<register_work_tag> bp(work_logger_);

        // verify state
        if (thread_count_ == 0 && state_ != running)
        {
            // thread-manager is not currently running
            HPX_THROWS_IF(ec, invalid_status,
                "threadmanager_impl::register_work",
                "invalid state: thread manager is not running");
            return;
        }

        detail::create_work(scheduler_, data, initial_state, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    /// The set_state function is part of the thread related API and allows
    /// to change the state of one of the threads managed by this threadmanager_impl
    template <typename SchedulingPolicy, typename NotificationPolicy>
    thread_state threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        set_state(thread_id_type id, thread_state_enum new_state,
            thread_state_ex_enum new_state_ex, thread_priority priority,
            error_code& ec)
    {
        return detail::set_thread_state(scheduler_, id, new_state, 
            new_state_ex, priority, get_worker_thread_num(), ec);
    }

    /// The get_state function is part of the thread related API. It
    /// queries the state of one of the threads known to the threadmanager_impl
    template <typename SchedulingPolicy, typename NotificationPolicy>
    thread_state threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        get_state(thread_id_type id)
    {
        // we know that the id is actually the pointer to the thread
        thread_data* thrd = reinterpret_cast<thread_data*>(id);
        return thrd ? thrd->get_state() : thread_state(terminated);
    }

    /// The get_phase function is part of the thread related API. It
    /// queries the phase of one of the threads known to the threadmanager_impl
    template <typename SchedulingPolicy, typename NotificationPolicy>
    std::size_t threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        get_phase(thread_id_type id)
    {
        // we know that the id is actually the pointer to the thread
        thread_data* thrd = reinterpret_cast<thread_data*>(id);
        return thrd ? thrd->get_thread_phase() : std::size_t(~0);
    }

    /// The get_description function is part of the thread related API and
    /// allows to query the description of one of the threads known to the
    /// threadmanager_impl
    template <typename SchedulingPolicy, typename NotificationPolicy>
    char const* threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        get_description(thread_id_type id) const
    {
        // we know that the id is actually the pointer to the thread
        thread_data* thrd = reinterpret_cast<thread_data*>(id);
        return thrd ? thrd->get_description() : "<unknown>";
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    char const* threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        set_description(thread_id_type id, char const* desc)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROW_EXCEPTION(null_thread_id,
                "threadmanager_impl::set_description",
                "NULL thread id encountered");
            return NULL;
        }

        // we know that the id is actually the pointer to the thread
        thread_data* thrd = reinterpret_cast<thread_data*>(id);
        if (thrd)
            return thrd->set_description(desc);
        return NULL;
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    char const* threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        get_lco_description(thread_id_type id) const
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROW_EXCEPTION(null_thread_id,
                "threadmanager_impl::get_lco_description",
                "NULL thread id encountered");
            return NULL;
        }

        // we know that the id is actually the pointer to the thread
        thread_data* thrd = reinterpret_cast<thread_data*>(id);
        return thrd ? thrd->get_lco_description() : "<unknown>";
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    char const* threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        set_lco_description(thread_id_type id, char const* desc)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROW_EXCEPTION(null_thread_id,
                "threadmanager_impl::set_lco_description",
                "NULL thread id encountered");
            return NULL;
        }

        // we know that the id is actually the pointer to the thread
        thread_data* thrd = reinterpret_cast<thread_data*>(id);
        if (thrd)
            return thrd->set_lco_description(desc);
        return NULL;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy>
    util::backtrace const* threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        get_backtrace(thread_id_type id) const
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROW_EXCEPTION(null_thread_id,
                "threadmanager_impl::get_backtrace",
                "NULL thread id encountered");
            return NULL;
        }

        // we know that the id is actually the pointer to the thread
        thread_data* thrd = reinterpret_cast<thread_data*>(id);
        return thrd ? thrd->get_backtrace() : 0;
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    util::backtrace const* threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        set_backtrace(thread_id_type id, util::backtrace const* bt)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROW_EXCEPTION(null_thread_id,
                "threadmanager_impl::set_backtrace",
                "NULL thread id encountered");
            return NULL;
        }

        // we know that the id is actually the pointer to the thread
        thread_data* thrd = reinterpret_cast<thread_data*>(id);
        return thrd ? thrd->set_backtrace(bt) : 0;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy>
    bool threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        get_interruption_enabled(thread_id_type id, error_code& ec)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROW_EXCEPTION(null_thread_id,
                "threadmanager_impl::get_interruption_enabled",
                "NULL thread id encountered");
            return false;
        }

        if (&ec != &throws)
            ec = make_success_code();

        // we know that the id is actually the pointer to the thread
        thread_data* thrd = reinterpret_cast<thread_data*>(id);
        return thrd ? thrd->interruption_enabled() : false;
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    bool threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        set_interruption_enabled(thread_id_type id, bool enable, error_code& ec)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROW_EXCEPTION(null_thread_id,
                "threadmanager_impl::set_interruption_enabled",
                "NULL thread id encountered");
        }

        if (&ec != &throws)
            ec = make_success_code();

        // we know that the id is actually the pointer to the thread
        thread_data* thrd = reinterpret_cast<thread_data*>(id);
        if (thrd)
            return thrd->set_interruption_enabled(enable);
        return false;
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    bool threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        get_interruption_requested(thread_id_type id, error_code& ec)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "threadmanager_impl::get_interruption_requested",
                "NULL thread id encountered");
            return false;
        }

        if (&ec != &throws)
            ec = make_success_code();

        // we know that the id is actually the pointer to the thread
        thread_data* thrd = reinterpret_cast<thread_data*>(id);
        return thrd ? thrd->interruption_requested() : false;
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    void threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        interrupt(thread_id_type id, bool flag, error_code& ec)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "threadmanager_impl::interrupt",
                "NULL thread id encountered");
            return;
        }

        if (&ec != &throws)
            ec = make_success_code();

        // we know that the id is actually the pointer to the thread
        thread_data* thrd = reinterpret_cast<thread_data*>(id);
        if (thrd) {
            thrd->interrupt(flag);      // notify thread

            // set thread state to pending, if the thread is currently active,
            // this will be rescheduled until it calls an interruption point
            set_thread_state(id, pending, wait_abort,
                thread_priority_normal, ec);
        }
    }

#if HPX_THREAD_MAINTAIN_THREAD_DATA
    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy>
    std::size_t threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        get_thread_data(thread_id_type id, error_code& ec) const
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "threadmanager_impl::get_thread_data",
                "NULL thread id encountered");
            return 0;
        }

        // we know that the id is actually the pointer to the thread
        thread_data* thrd = reinterpret_cast<thread_data*>(id);
        return thrd ? thrd->get_thread_data() : 0;
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    std::size_t threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        set_thread_data(thread_id_type id, std::size_t data,
            error_code& ec)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "threadmanager_impl::set_thread_data",
                "NULL thread id encountered");
            return 0;
        }

        // we know that the id is actually the pointer to the thread
        thread_data* thrd = reinterpret_cast<thread_data*>(id);
        return thrd ? thrd->set_thread_data(data) : 0;
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy>
    void threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
    run_thread_exit_callbacks(thread_id_type id, error_code& ec)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "threadmanager_impl::run_thread_exit_callbacks",
                "NULL thread id encountered");
            return;
        }

        if (&ec != &throws)
            ec = make_success_code();

        thread_data* thrd = reinterpret_cast<thread_data*>(id);
        if (thrd)
            thrd->run_thread_exit_callbacks();
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    bool threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
    add_thread_exit_callback(thread_id_type id, HPX_STD_FUNCTION<void()> const& f,
        error_code& ec)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "threadmanager_impl::add_thread_exit_callback",
                "NULL thread id encountered");
            return false;
        }

        if (&ec != &throws)
            ec = make_success_code();

        thread_data* thrd = reinterpret_cast<thread_data*>(id);
        return (0 != thrd) ? thrd->add_thread_exit_callback(f) : false;
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    void threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        free_thread_exit_callbacks(thread_id_type id, error_code& ec)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "threadmanager_impl::free_thread_exit_callbacks",
                "NULL thread id encountered");
            return;
        }

        if (&ec != &throws)
            ec = make_success_code();

        thread_data* thrd = reinterpret_cast<thread_data*>(id);
        if (0 != thrd)
            thrd->free_thread_exit_callbacks();
    }

    ///////////////////////////////////////////////////////////////////////////
    /// This thread function is used by the at_timer thread below to trigger
    /// the required action.
    template <typename SchedulingPolicy, typename NotificationPolicy>
    thread_state_enum threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        wake_timer_thread(thread_id_type id,
            thread_state_enum newstate, thread_state_ex_enum newstate_ex,
            thread_priority priority, thread_id_type timer_id,
            boost::shared_ptr<boost::atomic<bool> > triggered)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROW_EXCEPTION(null_thread_id,
                "threadmanager_impl::wake_timer_thread",
                "NULL thread id encountered (id)");
            return terminated;
        }
        if (HPX_UNLIKELY(!timer_id)) {
            HPX_THROW_EXCEPTION(null_thread_id,
                "threadmanager_impl::wake_timer_thread",
                "NULL thread id encountered (timer_id)");
            return terminated;
        }

        bool oldvalue = false;
        if (triggered->compare_exchange_strong(oldvalue, true)) //-V601
        {
            // timer has not been canceled yet, trigger the requested set_state
            set_state(id, newstate, newstate_ex, priority);
        }

        // then re-activate the thread holding the deadline_timer
        // REVIEW: Why do we ignore errors here?
        error_code ec(lightweight);    // do not throw
        set_state(timer_id, pending, wait_timeout, thread_priority_normal, ec);
        return terminated;
    }

    /// This thread function initiates the required set_state action (on
    /// behalf of one of the threadmanager_impl#set_state functions).
    template <typename SchedulingPolicy, typename NotificationPolicy>
    template <typename TimeType>
    thread_state_enum threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        at_timer(TimeType const& expire, thread_id_type id,
            thread_state_enum newstate, thread_state_ex_enum newstate_ex,
            thread_priority priority)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROW_EXCEPTION(null_thread_id,
                "threadmanager_impl::at_timer",
                "NULL thread id encountered");
            return terminated;
        }

        // create a new thread in suspended state, which will execute the
        // requested set_state when timer fires and will re-awaken this thread,
        // allowing the deadline_timer to go out of scope gracefully
        thread_self& self = get_self();
        thread_id_type self_id = self.get_thread_id();

        boost::shared_ptr<boost::atomic<bool> > triggered(
            boost::make_shared<boost::atomic<bool> >(false));

        thread_init_data data(
            boost::bind(&threadmanager_impl::wake_timer_thread, this, id,
                newstate, newstate_ex, priority, self_id, triggered),
            "wake_timer", 0, priority);
        thread_id_type wake_id = register_thread(data, suspended, true);

        // create timer firing in correspondence with given time
        boost::asio::deadline_timer t (timer_pool_.get_io_service(), expire);

        // let the timer invoke the set_state on the new (suspended) thread
        t.async_wait(boost::bind(&threadmanager_impl::set_state, this, wake_id,
            pending, wait_timeout, priority, boost::ref(throws)));

        // this waits for the thread to be reactivated when the timer fired
        // if it returns signaled the timer has been canceled, otherwise
        // the timer fired and the wake_timer_thread above has been executed
        bool oldvalue = false;
        thread_state_ex_enum statex = self.yield(suspended);

        if (wait_timeout != statex &&
            triggered->compare_exchange_strong(oldvalue, true)) //-V601
        {
            // wake_timer_thread has not been executed yet, cancel timer
            t.cancel();
        }

        return terminated;
    }

    /// Set a timer to set the state of the given \a thread to the given
    /// new value after it expired (at the given time)
    template <typename SchedulingPolicy, typename NotificationPolicy>
    thread_id_type threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        set_state(time_type const& expire_at, thread_id_type id,
            thread_state_enum newstate, thread_state_ex_enum newstate_ex,
            thread_priority priority, error_code& ec)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "threadmanager_impl::set_state",
                "NULL thread id encountered");
            return 0;
        }

        // this creates a new thread which creates the timer and handles the
        // requested actions
        thread_state_enum (threadmanager_impl::*f)(time_type const&,
                thread_id_type, thread_state_enum, thread_state_ex_enum,
                thread_priority)
            = &threadmanager_impl::template at_timer<time_type>;

        thread_init_data data(
            boost::bind(f, this, expire_at, id, newstate, newstate_ex, priority),
            "at_timer (expire at)", 0, priority);
        return register_thread(data, pending, true, ec);
    }

    /// Set a timer to set the state of the given \a thread to the given
    /// new value after it expired (after the given duration)
    template <typename SchedulingPolicy, typename NotificationPolicy>
    thread_id_type threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        set_state(duration_type const& from_now, thread_id_type id,
            thread_state_enum newstate, thread_state_ex_enum newstate_ex,
            thread_priority priority, error_code& ec)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "threadmanager_impl::set_state",
                "NULL thread id encountered");
            return 0;
        }

        // this creates a new thread which creates the timer and handles the
        // requested actions
        thread_state_enum (threadmanager_impl::*f)(duration_type const&,
                thread_id_type, thread_state_enum, thread_state_ex_enum,
                thread_priority)
            = &threadmanager_impl::template at_timer<duration_type>;

        thread_init_data data(
            boost::bind(f, this, from_now, id, newstate, newstate_ex, priority),
            "at_timer (from now)", 0, priority);
        return register_thread(data, pending, true, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    // main function executed by all OS threads managed by this threadmanager_impl
    template <typename SP, typename NP>
    struct init_tss_helper
    {
        typedef threadmanager_impl<SP, NP> threadmanager_type;

        init_tss_helper(threadmanager_type& tm, std::size_t thread_num,
                bool numa_sensitive)
          : tm_(tm)
        {
            tm_.init_tss(thread_num, numa_sensitive);
        }
        ~init_tss_helper()
        {
            tm_.deinit_tss();
        }

        threadmanager_type& tm_;
    };

    struct manage_active_thread_count
    {
        manage_active_thread_count(boost::atomic<long>& counter)
          : counter_(counter)
        {
            ++counter_;
        }
        ~manage_active_thread_count()
        {
            --counter_;
        }

        boost::atomic<long>& counter_;
    };

    template <typename SchedulingPolicy, typename NotificationPolicy>
    void threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        tfunc(std::size_t num_thread)
    {
        // wait for all threads to start up before before starting px work
        startup_->wait();

        // manage the number of this thread in its TSS
        init_tss_helper<SchedulingPolicy, NotificationPolicy>
            tss_helper(*this, num_thread, scheduler_.numa_sensitive());

        // needs to be done as the first thing, otherwise logging won't work
        notifier_.on_start_thread(num_thread);       // notify runtime system of started thread
        scheduler_.on_start_thread(num_thread);

        {
            LTM_(info) << "tfunc(" << num_thread << "): starting OS thread";
            try {
                try {
                    tfunc_impl(num_thread);
                }
                catch (hpx::exception const& e) {
                    LFATAL_ << "tfunc(" << num_thread
                            << "): caught hpx::exception: "
                            << e.what() << ", aborted thread execution";
                    report_error(num_thread, boost::current_exception());
                    return;
                }
                catch (boost::system::system_error const& e) {
                    LFATAL_ << "tfunc(" << num_thread
                            << "): caught boost::system::system_error: "
                            << e.what() << ", aborted thread execution";
                    report_error(num_thread, boost::current_exception());
                    return;
                }
                catch (std::exception const& e) {
                    // Repackage exceptions to avoid slicing.
                    boost::throw_exception(boost::enable_error_info(
                        hpx::exception(unhandled_exception, e.what())));
                }
            }
            catch (...) {
                LFATAL_ << "tfunc(" << num_thread << "): caught unexpected "
                    "exception, aborted thread execution";
                report_error(num_thread, boost::current_exception());
                return;
            }

            LTM_(info) << "tfunc(" << num_thread << "): ending OS thread, "
                "executed " << executed_threads_[num_thread] << " HPX threads";
        }

        notifier_.on_stop_thread(num_thread);
        scheduler_.on_stop_thread(num_thread);
    }

    ///////////////////////////////////////////////////////////////////////////
    // counter creator and discovery functions

    // queue length(s) counter creation function
    template <typename SchedulingPolicy, typename NotificationPolicy>
    naming::gid_type threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        queue_length_counter_creator(
            performance_counters::counter_info const& info, error_code& ec)
    {
        // verify the validity of the counter instance name
        performance_counters::counter_path_elements paths;
        performance_counters::get_counter_path_elements(info.fullname_, paths, ec);
        if (ec) return naming::invalid_gid;

        // /threadqueue{locality#%d/total}/length
        // /threadqueue{locality#%d/worker-thread%d}/length
        if (paths.parentinstance_is_basename_) {
            HPX_THROWS_IF(ec, bad_parameter, "queue_length_counter_creator",
                "invalid counter instance parent name: " +
                    paths.parentinstancename_);
            return naming::invalid_gid;
        }

        typedef scheduling_policy_type spt;

        if (paths.instancename_ == "total" && paths.instanceindex_ == -1)
        {
            // overall counter
            using performance_counters::detail::create_raw_counter;
            return create_raw_counter(info,
                HPX_STD_BIND(&spt::get_queue_length, &scheduler_, -1), ec);
        }
        else if (paths.instancename_ == "worker-thread" &&
            paths.instanceindex_ >= 0 &&
            std::size_t(paths.instanceindex_) < threads_.size())
        {
            // specific counter
            using performance_counters::detail::create_raw_counter;
            return create_raw_counter(info,
                HPX_STD_BIND(&spt::get_queue_length, &scheduler_,
                    static_cast<std::size_t>(paths.instanceindex_)), ec);
        }

        HPX_THROWS_IF(ec, bad_parameter, "queue_length_counter_creator",
            "invalid counter instance name: " + paths.instancename_);
        return naming::invalid_gid;
    }

    bool locality_allocator_counter_discoverer(
        performance_counters::counter_info const& info,
        HPX_STD_FUNCTION<performance_counters::discover_counter_func> const& f,
        performance_counters::discover_counters_mode mode, error_code& ec)
    {
        performance_counters::counter_info i = info;

        // compose the counter name templates
        performance_counters::counter_path_elements p;
        performance_counters::counter_status status =
            get_counter_path_elements(info.fullname_, p, ec);
        if (!status_is_valid(status)) return false;

        if (mode == performance_counters::discover_counters_minimal ||
            p.parentinstancename_.empty() || p.instancename_.empty())
        {
            if (p.parentinstancename_.empty())
            {
                p.parentinstancename_ = "locality#*";
                p.parentinstanceindex_ = -1;
            }

            if (p.instancename_.empty())
            {
                p.instancename_ = "allocator#*";
                p.instanceindex_ = -1;
            }

            status = get_counter_name(p, i.fullname_, ec);
            if (!status_is_valid(status) || !f(i, ec) || ec)
                return false;
        }
        else if (p.instancename_ == "allocator#*") {
            BOOST_ASSERT(mode == performance_counters::discover_counters_full);

            for (std::size_t t = 0; t < HPX_COROUTINE_NUM_ALL_HEAPS; ++t)
            {
                p.instancename_ = "allocator";
                p.instanceindex_ = static_cast<boost::int32_t>(t);
                status = get_counter_name(p, i.fullname_, ec);
                if (!status_is_valid(status) || !f(i, ec) || ec)
                    return false;
            }
        }
        else if (!f(i, ec) || ec) {
            return false;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return true;
    }

    ///////////////////////////////////////////////////////////////////////////
    // idle rate counter creation function
    template <typename SchedulingPolicy, typename NotificationPolicy>
    naming::gid_type threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        idle_rate_counter_creator(
            performance_counters::counter_info const& info, error_code& ec)
    {
        // verify the validity of the counter instance name
        performance_counters::counter_path_elements paths;
        performance_counters::get_counter_path_elements(info.fullname_, paths, ec);
        if (ec) return naming::invalid_gid;

        // /threads{locality#%d/total}/idle-rate
        // /threads{locality#%d/worker-thread%d}/idle-rate
        if (paths.parentinstance_is_basename_) {
            HPX_THROWS_IF(ec, bad_parameter, "idle_rate_counter_creator",
                "invalid counter instance parent name: " +
                    paths.parentinstancename_);
            return naming::invalid_gid;
        }

        typedef threadmanager_impl ti;

        if (paths.instancename_ == "total" && paths.instanceindex_ == -1)
        {
            // overall counter
            using performance_counters::detail::create_raw_counter;
            return create_raw_counter(info,
                HPX_STD_BIND(&ti::avg_idle_rate, this), ec);
        }
        else if (paths.instancename_ == "worker-thread" &&
            paths.instanceindex_ >= 0 &&
            std::size_t(paths.instanceindex_) < threads_.size())
        {
            // specific counter
            using performance_counters::detail::create_raw_counter;
            return create_raw_counter(info,
                HPX_STD_BIND(&ti::avg_idle_rate, this,
                    static_cast<std::size_t>(paths.instanceindex_)), ec);
        }

        HPX_THROWS_IF(ec, bad_parameter, "idle_rate_counter_creator",
            "invalid counter instance name: " + paths.instancename_);
        return naming::invalid_gid;
    }

    ///////////////////////////////////////////////////////////////////////////
    naming::gid_type
    counter_creator(performance_counters::counter_info const& info,
        performance_counters::counter_path_elements const& paths,
        HPX_STD_FUNCTION<boost::uint64_t()> const& total_creator,
        HPX_STD_FUNCTION<boost::uint64_t()> const& individual_creator,
        char const* individual_name, std::size_t individual_count,
        error_code& ec)
    {
        if (paths.parentinstance_is_basename_) {
            HPX_THROWS_IF(ec, bad_parameter, "counter_creator",
                "invalid counter instance parent name: " +
                    paths.parentinstancename_);
            return naming::invalid_gid;
        }

        if (!total_creator.empty() &&
            paths.instancename_ == "total" && paths.instanceindex_ == -1)
        {
            // overall counter
            using performance_counters::detail::create_raw_counter;
            return create_raw_counter(info, total_creator, ec);
        }
        else if (!individual_creator.empty() &&
            paths.instancename_ == individual_name &&
            paths.instanceindex_ >= 0 &&
            std::size_t(paths.instanceindex_) < individual_count)
        {
            // specific counter
            using performance_counters::detail::create_raw_counter;
            return create_raw_counter(info, individual_creator, ec);
        }

        HPX_THROWS_IF(ec, bad_parameter, "counter_creator",
            "invalid counter instance name: " + paths.instancename_);
        return naming::invalid_gid;
    }

    ///////////////////////////////////////////////////////////////////////////
    // thread counts counter creation function
    template <typename SchedulingPolicy, typename NotificationPolicy>
    naming::gid_type threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        thread_counts_counter_creator(
            performance_counters::counter_info const& info, error_code& ec)
    {
        // verify the validity of the counter instance name
        performance_counters::counter_path_elements paths;
        performance_counters::get_counter_path_elements(info.fullname_, paths, ec);
        if (ec) return naming::invalid_gid;

        struct creator_data
        {
            char const* const countername;
            HPX_STD_FUNCTION<boost::uint64_t()> total_func;
            HPX_STD_FUNCTION<boost::uint64_t()> individual_func;
            char const* const individual_name;
            std::size_t individual_count;
        };

        typedef scheduling_policy_type spt;
        typedef threadmanager_impl ti;

        std::size_t shepherd_count = threads_.size();
        creator_data data[] =
        {
            // /threads{locality#%d/total}/count/cumulative
            // /threads{locality#%d/worker-thread%d}/count/cumulative
            { "count/cumulative",
              HPX_STD_BIND(&ti::get_executed_threads, this, -1),
              HPX_STD_BIND(&ti::get_executed_threads, this,
                  static_cast<std::size_t>(paths.instanceindex_)),
              "worker-thread", shepherd_count
            },
            // /threads(locality#%d/total}/count/instantaneous/all
            // /threads(locality#%d/worker-thread%d}/count/instantaneous/all
            { "count/instantaneous/all",
              HPX_STD_BIND(&spt::get_thread_count, &scheduler_, unknown, -1),
              HPX_STD_BIND(&spt::get_thread_count, &scheduler_, unknown,
                  static_cast<std::size_t>(paths.instanceindex_)),
              "worker-thread", shepherd_count
            },
            // /threads(locality#%d/total}/count/instantaneous/active
            // /threads(locality#%d/worker-thread%d}/count/instantaneous/active
            { "count/instantaneous/active",
              HPX_STD_BIND(&spt::get_thread_count, &scheduler_, active, -1),
              HPX_STD_BIND(&spt::get_thread_count, &scheduler_, active,
                  static_cast<std::size_t>(paths.instanceindex_)),
              "worker-thread", shepherd_count
            },
            // /threads(locality#%d/total}/count/instantaneous/pending
            // /threads(locality#%d/worker-thread%d}/count/instantaneous/pending
            { "count/instantaneous/pending",
              HPX_STD_BIND(&spt::get_thread_count, &scheduler_, pending, -1),
              HPX_STD_BIND(&spt::get_thread_count, &scheduler_, pending,
                  static_cast<std::size_t>(paths.instanceindex_)),
              "worker-thread", shepherd_count
            },
            // /threads(locality#%d/total}/count/instantaneous/suspended
            // /threads(locality#%d/worker-thread%d}/count/instantaneous/suspended
            { "count/instantaneous/suspended",
              HPX_STD_BIND(&spt::get_thread_count, &scheduler_, suspended, -1),
              HPX_STD_BIND(&spt::get_thread_count, &scheduler_, suspended,
                  static_cast<std::size_t>(paths.instanceindex_)),
              "worker-thread", shepherd_count
            },
            // /threads(locality#%d/total}/count/instantaneous/terminated
            // /threads(locality#%d/worker-thread%d}/count/instantaneous/terminated
            { "count/instantaneous/terminated",
              HPX_STD_BIND(&spt::get_thread_count, &scheduler_, terminated, -1),
              HPX_STD_BIND(&spt::get_thread_count, &scheduler_, terminated,
                  static_cast<std::size_t>(paths.instanceindex_)),
              "worker-thread", shepherd_count
            },
            // /threads(locality#%d/total}/count/stack-recycles
            { "count/stack-recycles",
              &coroutine_type::impl_type::get_stack_recycle_count,
              HPX_STD_FUNCTION<boost::uint64_t()>(), "", 0
            },
#if !defined(BOOST_WINDOWS) && !defined(HPX_COROUTINE_USE_GENERIC_CONTEXT)
            // /threads(locality#%d/total}/count/stack-unbinds
            { "count/stack-unbinds",
              &coroutine_type::impl_type::get_stack_unbind_count,
              HPX_STD_FUNCTION<boost::uint64_t()>(), "", 0
            },
#endif
            // /threads(locality#%d/total}/count/objects
            // /threads(locality#%d/allocator%d)/count/objects
            { "count/objects",
              &coroutine_type::impl_type::get_allocation_count_all,
              HPX_STD_BIND(&coroutine_type::impl_type::get_allocation_count,
                  static_cast<std::size_t>(paths.instanceindex_)),
              "allocator", HPX_COROUTINE_NUM_ALL_HEAPS
            },
        };
        std::size_t const data_size = sizeof(data)/sizeof(data[0]);

        for (creator_data const* d = data; d < &d[data_size]; ++d)
        {
            if (paths.countername_ == d->countername)
            {
                return counter_creator(info, paths, d->total_func,
                    d->individual_func, d->individual_name,
                    d->individual_count, ec);
            }
        }

        HPX_THROWS_IF(ec, bad_parameter, "thread_counts_counter_creator",
            "invalid counter instance name: " + paths.instancename_);
        return naming::invalid_gid;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy>
    void threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        register_counter_types()
    {
        typedef threadmanager_impl ti;
        HPX_STD_FUNCTION<performance_counters::create_counter_func> counts_creator(
            boost::bind(&ti::thread_counts_counter_creator, this, _1, _2));

        performance_counters::generic_counter_type_data counter_types[] =
        {
            // length of thread queue(s)
            { "/threadqueue/length", performance_counters::counter_raw,
              "returns the current queue length for the referenced queue",
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&ti::queue_length_counter_creator, this, _1, _2),
              &performance_counters::locality_thread_counter_discoverer,
              ""
            },
            // idle rate
            { "/threads/idle-rate", performance_counters::counter_raw,
              "returns the idle rate for the referenced object [0.1%]",
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&ti::idle_rate_counter_creator, this, _1, _2),
              &performance_counters::locality_thread_counter_discoverer,
              "0.1%"
            },
            // thread counts
            { "/threads/count/cumulative", performance_counters::counter_raw,
              "returns the overall number of executed (retired) HPX-threads for "
              "the referenced locality", HPX_PERFORMANCE_COUNTER_V1, counts_creator,
              &performance_counters::locality_thread_counter_discoverer,
              ""
            },
            { "/threads/count/instantaneous/all", performance_counters::counter_raw,
              "returns the overall current number of HPX-threads instantiated at the "
              "referenced locality", HPX_PERFORMANCE_COUNTER_V1, counts_creator,
              &performance_counters::locality_thread_counter_discoverer,
              ""
            },
            { "/threads/count/instantaneous/active", performance_counters::counter_raw,
              "returns the current number of active HPX-threads at the referenced locality",
              HPX_PERFORMANCE_COUNTER_V1, counts_creator,
              &performance_counters::locality_thread_counter_discoverer,
              ""
            },
            { "/threads/count/instantaneous/pending", performance_counters::counter_raw,
              "returns the current number of pending HPX-threads at the referenced locality",
              HPX_PERFORMANCE_COUNTER_V1, counts_creator,
              &performance_counters::locality_thread_counter_discoverer,
              ""
            },
            { "/threads/count/instantaneous/suspended", performance_counters::counter_raw,
              "returns the current number of suspended HPX-threads at the referenced locality",
              HPX_PERFORMANCE_COUNTER_V1, counts_creator,
              &performance_counters::locality_thread_counter_discoverer,
              ""
            },
            { "/threads/count/instantaneous/terminated", performance_counters::counter_raw,
              "returns the current number of terminated HPX-threads at the referenced locality",
              HPX_PERFORMANCE_COUNTER_V1, counts_creator,
              &performance_counters::locality_thread_counter_discoverer,
              ""
            },
            { "/threads/count/stack-recycles", performance_counters::counter_raw,
              "returns the total number of HPX-thread recycling operations performed "
              "for the referenced locality", HPX_PERFORMANCE_COUNTER_V1,
              counts_creator, &performance_counters::locality_counter_discoverer,
              ""
            },
#if !defined(BOOST_WINDOWS) && !defined(HPX_COROUTINE_USE_GENERIC_CONTEXT)
            { "/threads/count/stack-unbinds", performance_counters::counter_raw,
              "returns the total number of HPX-thread unbind (madvise) operations "
              "performed for the referenced locality", HPX_PERFORMANCE_COUNTER_V1,
              counts_creator, &performance_counters::locality_counter_discoverer,
              ""
            },
#endif
            { "/threads/count/objects", performance_counters::counter_raw,
              "returns the overall number of created HPX-threads objects for "
              "the referenced locality", HPX_PERFORMANCE_COUNTER_V1,
              counts_creator,
              &locality_allocator_counter_discoverer,
              ""
            }
        };
        performance_counters::install_counter_types(
            counter_types, sizeof(counter_types)/sizeof(counter_types[0]));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy>
    void threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        tfunc_impl(std::size_t num_thread)
    {
        manage_active_thread_count count(thread_count_);

        // set affinity on Linux systems or when using HWLOC
        topology const& topology_ = get_topology();
        std::size_t mask = get_pu_mask(topology_, num_thread);

        LTM_(info) << "tfunc(" << num_thread
            << "): will run on one processing unit within this mask: "
            << std::hex << "0x" << mask;

        error_code ec(lightweight);
        topology_.set_thread_affinity_mask(mask, ec);
        if (ec) {
            LTM_(warning) << "run: setting thread affinity on OS thread "
                << num_thread << " failed with: " << ec.get_message();
        }

        // run the work queue
        hpx::util::coroutines::prepare_main_thread main_thread;

        // run main scheduling loop until terminated
        detail::scheduling_loop(num_thread, scheduler_, state_,
            executed_threads_[num_thread], tfunc_times[num_thread],
            exec_times[num_thread]);

#if HPX_DEBUG != 0
        // the last OS thread is allowed to exit only if no more PX threads exist
        BOOST_ASSERT(!scheduler_.get_thread_count(unknown, num_thread));
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy>
    bool threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        run(std::size_t num_threads)
    {
        LTM_(info) << "run: creating " << num_threads << " OS thread(s)";

        if (0 == num_threads) {
            HPX_THROW_EXCEPTION(bad_parameter,
                "threadmanager_impl::run", "number of threads is zero");
        }

        mutex_type::scoped_lock lk(mtx_);
        if (!threads_.empty() || (state_.load() == running))
            return true;    // do nothing if already running

        LTM_(info) << "run: running timer pool";
        timer_pool_.run(false);

        executed_threads_.resize(num_threads);
        tfunc_times.resize(num_threads);
        exec_times.resize(num_threads);

        try {
            // run threads and wait for initialization to complete
            BOOST_ASSERT (NULL == startup_);
            startup_ = new boost::barrier(static_cast<unsigned>(num_threads+1));

            state_.store(running);

            topology const& topology_ = get_topology();

            std::size_t thread_num = num_threads;
            while (thread_num-- != 0) {
                std::size_t mask = get_pu_mask(topology_, thread_num);

                LTM_(info) << "run: create OS thread " << thread_num
                    << ": will run on one processing unit within this mask: "
                    << std::hex << "0x" << mask;

                // create a new thread
                threads_.push_back(new boost::thread(boost::bind(
                    &threadmanager_impl::tfunc, this, thread_num)));

                // set the new threads affinity (on Windows systems)
                error_code ec(lightweight);
                topology_.set_thread_affinity_mask(threads_.back(), mask, ec);

                if (ec)
                {
                    LTM_(warning) << "run: setting thread affinity on OS "
                                     "thread " << thread_num << " failed with: "
                                  << ec.get_message();
                }
            }

            // start timer pool as well
            timer_pool_.run(false);

            // the main thread needs to have a unique thread_num
            init_tss(thread_num, scheduler_.numa_sensitive());
            startup_->wait();
        }
        catch (std::exception const& e) {
            LTM_(always) << "run: failed with: " << e.what();

            // trigger the barrier
            while (num_threads-- != 0 && !startup_->wait())
                ;

            stop();
            threads_.clear();

            return false;
        }

        LTM_(info) << "run: running";
        return true;
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    void threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        stop (bool blocking)
    {
        LTM_(info) << "stop: blocking(" << std::boolalpha << blocking << ")";

        deinit_tss();

        mutex_type::scoped_lock l(mtx_);
        if (!threads_.empty()) {
            if (state_.load() == running) {
                state_.store(stopping);
                do_some_work();         // make sure we're not waiting
            }

            if (blocking) {
                for (std::size_t i = 0; i < threads_.size(); ++i)
                {
                    // make sure no OS thread is waiting
                    LTM_(info) << "stop: notify_all";
                    do_some_work();

                    LTM_(info) << "stop(" << i << "): join";

                    // unlock the lock while joining
                    util::unlock_the_lock<mutex_type::scoped_lock> ul(l);
                    threads_[i].join();
                }
                threads_.clear();

                LTM_(info) << "stop: stopping timer pool";
                timer_pool_.stop();             // stop timer pool as well
                if (blocking) {
                    timer_pool_.join();
                    timer_pool_.clear();
                }
            }
        }
        delete startup_;
        startup_ = NULL;
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    boost::int64_t threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        get_executed_threads(std::size_t num) const
    {
        if (num != std::size_t(-1))
            return executed_threads_[num];

        return std::accumulate(executed_threads_.begin(),
            executed_threads_.end(), 0LL);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy>
    boost::int64_t threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        avg_idle_rate() const
    {
        double const exec_total =
            std::accumulate(exec_times.begin(), exec_times.end(), 0.);
        double const tfunc_total =
            std::accumulate(tfunc_times.begin(), tfunc_times.end(), 0.);

        if (std::abs(tfunc_total) < 1e-16)   // avoid division by zero
            return 1000LL;

        double const percent = 1. - (exec_total / tfunc_total);
        return boost::int64_t(1000. * percent);    // 0.1 percent
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    boost::int64_t threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        avg_idle_rate(std::size_t num_thread) const
    {
        if (0 == tfunc_times[num_thread])   // avoid division by zero
            return 1000LL;

        double const exec_time = static_cast<double>(exec_times[num_thread]);
        double const tfunc_time = static_cast<double>(tfunc_times[num_thread]);
        double const percent =
            1. - (exec_time / tfunc_time);
        return boost::int64_t(1000. * percent);   // 0.1 percent
    }
}}

///////////////////////////////////////////////////////////////////////////////
/// explicit template instantiation for the thread manager of our choice
#include <hpx/runtime/threads/policies/callback_notifier.hpp>

#if defined(HPX_GLOBAL_SCHEDULER)
#include <hpx/runtime/threads/policies/global_queue_scheduler.hpp>

template class HPX_EXPORT hpx::threads::threadmanager_impl<
    hpx::threads::policies::global_queue_scheduler,
    hpx::threads::policies::callback_notifier>;
#endif

#if defined(HPX_LOCAL_SCHEDULER)
#include <hpx/runtime/threads/policies/local_queue_scheduler.hpp>

template class HPX_EXPORT hpx::threads::threadmanager_impl<
    hpx::threads::policies::local_queue_scheduler<>,
    hpx::threads::policies::callback_notifier>;
#endif

#if defined(HPX_ABP_SCHEDULER)
#include <hpx/runtime/threads/policies/abp_queue_scheduler.hpp>

template class HPX_EXPORT hpx::threads::threadmanager_impl<
    hpx::threads::policies::abp_queue_scheduler,
    hpx::threads::policies::callback_notifier>;
#endif

#if defined(HPX_ABP_PRIORITY_SCHEDULER)
#include <hpx/runtime/threads/policies/abp_priority_queue_scheduler.hpp>

template class HPX_EXPORT hpx::threads::threadmanager_impl<
    hpx::threads::policies::abp_priority_queue_scheduler,
    hpx::threads::policies::callback_notifier>;
#endif

#include <hpx/runtime/threads/policies/local_priority_queue_scheduler.hpp>

template class HPX_EXPORT hpx::threads::threadmanager_impl<
    hpx::threads::policies::local_priority_queue_scheduler,
    hpx::threads::policies::callback_notifier>;

#if defined(HPX_HIERARCHY_SCHEDULER)
#include <hpx/runtime/threads/policies/hierarchy_scheduler.hpp>

template class HPX_EXPORT hpx::threads::threadmanager_impl<
    hpx::threads::policies::hierarchy_scheduler,
    hpx::threads::policies::callback_notifier>;
#endif

#if defined(HPX_PERIODIC_PRIORITY_SCHEDULER)
#include <hpx/runtime/threads/policies/periodic_priority_scheduler.hpp>

template class HPX_EXPORT hpx::threads::threadmanager_impl<
    hpx::threads::policies::local_periodic_priority_scheduler,
    hpx::threads::policies::callback_notifier>;
#endif

