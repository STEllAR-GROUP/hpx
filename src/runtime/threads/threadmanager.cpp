//  Copyright (c) 2008-2009 Chirag Dekate, Hartmut Kaiser, Anshul Tandon
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threads/thread_affinity.hpp>      // must be first header!
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/util/unlock_lock.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/block_profiler.hpp>
#include <hpx/util/time_logger.hpp>
#include <hpx/util/performance_counters.hpp>

#include <boost/assert.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/bind.hpp>
#include <boost/asio/deadline_timer.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads
{
    ///////////////////////////////////////////////////////////////////////////
    namespace strings
    {
        char const* const thread_state_names[] = 
        {
            "unknown",
            "init",
            "active",
            "pending",
            "suspended",
            "depleted",
            "terminated"
        };
    }

    char const* const get_thread_state_name(thread_state_enum state)
    {
        if (state < init || state > terminated)
            return "unknown";
        return strings::thread_state_names[state];
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy>
    threadmanager_impl<SchedulingPolicy, NotificationPolicy>::threadmanager_impl(
            util::io_service_pool& timer_pool, 
            scheduling_policy_type& scheduler,
            notification_policy_type& notifier)
      : running_(false),
        timer_pool_(timer_pool), 
        scheduler_(scheduler),
        notifier_(notifier),

        thread_logger_("threadmanager_impl::register_thread"),
        work_logger_("threadmanager_impl::register_work"),
        set_state_logger_("threadmanager_impl::set_state")
#if HPX_DEBUG != 0
      , thread_count_(0)
#endif
    {
        LTM_(debug) << "threadmanager_impl ctor";
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    threadmanager_impl<SchedulingPolicy, NotificationPolicy>::~threadmanager_impl() 
    {
        LTM_(debug) << "~threadmanager_impl";
        if (!threads_.empty()) {
            if (running_) 
                stop();
            threads_.clear();
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy>
    thread_id_type threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        register_thread(thread_init_data& data, thread_state_enum initial_state, 
            bool run_now, error_code& ec)
    {
        util::block_profiler_wrapper<register_thread_tag> bp(thread_logger_);

        // verify state
        if (!running_) 
        {
            // threadmanager is not currently running
            HPX_THROWS_IF(ec, invalid_status,
                "threadmanager_impl::register_thread",
                "invalid state: thread manager is not running");
            return invalid_thread_id;
        }

        // verify parameters
        if (initial_state != pending && initial_state != suspended)
        {
            HPX_OSSTREAM strm;
            strm << "invalid initial state: " 
                 << get_thread_state_name(initial_state);
            HPX_THROWS_IF(ec, bad_parameter,
                "threadmanager_impl::register_thread",
                HPX_OSSTREAM_GETSTRING(strm));
            return invalid_thread_id;
        }
        if (0 == data.description)
        {
            HPX_THROWS_IF(ec, bad_parameter,
                "threadmanager_impl::register_thread", "description is NULL");
            return invalid_thread_id;
        }

        if (0 == data.parent_id) {
            thread_self* self = get_self_ptr();
            if (self)
                data.parent_id = self->get_thread_id();
        }
        if (0 == data.parent_prefix) 
            data.parent_prefix = applier::get_prefix_id();

        // create the new thread
        thread_id_type newid = scheduler_.create_thread(
            data, initial_state, run_now, ec, get_thread_num());

        LTM_(info) << "register_thread(" << newid << "): initial_state(" 
                   << get_thread_state_name(initial_state) << "), "
                   << std::boolalpha << "run_now(" << run_now << "), "
                   << "description(" << data.description << ")";

        return newid;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy>
    void threadmanager_impl<SchedulingPolicy, NotificationPolicy>::register_work(
        thread_init_data& data, thread_state_enum initial_state, error_code& ec)
    {
        util::block_profiler_wrapper<register_work_tag> bp(work_logger_);

        // verify state
        if (!running_) 
        {
            // threadmanager is not currently running
            HPX_THROWS_IF(ec, invalid_status,
                "threadmanager_impl::register_work",
                "invalid state: thread manager is not running");
            return;
        }

        // verify parameters
        if (initial_state != pending && initial_state != suspended)
        {
            HPX_OSSTREAM strm;
            strm << "invalid initial state: " 
                 << get_thread_state_name(initial_state);
            HPX_THROWS_IF(ec, bad_parameter,
                "threadmanager_impl::register_work",
                HPX_OSSTREAM_GETSTRING(strm));
            return;
        }
        if (0 == data.description)
        {
            HPX_THROWS_IF(ec, bad_parameter,
                "threadmanager_impl::register_work", "description is NULL");
            return;
        }

        LTM_(info) << "register_work: initial_state(" 
                   << get_thread_state_name(initial_state) << "), "
                   << "description(" << data.description << ")";

        if (0 == data.parent_id) {
            thread_self* self = get_self_ptr();
            if (self)
                data.parent_id = self->get_thread_id();
        }
        if (0 == data.parent_prefix) 
            data.parent_prefix = applier::get_prefix_id();

        // create the new thread
        scheduler_.create_thread(data, initial_state, false, ec, get_thread_num());
    }

    ///////////////////////////////////////////////////////////////////////////
    // thread function registered for set_state if thread is currently active
    template <typename SchedulingPolicy, typename NotificationPolicy>
    thread_state threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        set_active_state(thread_id_type id, 
            thread_state_enum newstate, thread_state_ex_enum newstate_ex)
    {
        // just retry, set_state will create new thread if target is still active
        set_state(id, newstate, newstate_ex);
        return thread_state(terminated);
    }

    ///////////////////////////////////////////////////////////////////////////
    /// The set_state function is part of the thread related API and allows
    /// to change the state of one of the threads managed by this threadmanager_impl
    template <typename SchedulingPolicy, typename NotificationPolicy>
    thread_state threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        set_state(thread_id_type id, thread_state_enum new_state, 
            thread_state_ex_enum new_state_ex)
    {
        util::block_profiler_wrapper<set_state_tag> bp(set_state_logger_);

        // set_state can't be used to force a thread into active state
        if (new_state == active) {
            HPX_OSSTREAM strm;
            strm << "invalid new state: " << get_thread_state_name(new_state);
            HPX_THROW_EXCEPTION(bad_parameter, 
                "threadmanager_impl::set_state", HPX_OSSTREAM_GETSTRING(strm));
            return thread_state(unknown);
        }

        // we know that the id is actually the pointer to the thread
        thread* thrd = reinterpret_cast<thread*>(id);
        if (NULL == thrd->get())
            return thread_state(terminated);     // this thread has already been terminated 

        // action depends on the current state
        thread_state previous_state = thrd->get_state();
        thread_state_enum previous_state_val = previous_state;

        // nothing to do here if the state doesn't change
        if (new_state == previous_state_val)
            return thread_state(new_state);

        // the thread to set the state for is currently running, so we 
        // schedule another thread to execute the pending set_state
        if (previous_state_val == active) 
        {
            // schedule a new thread to set the state
            LTM_(warning) << "set_state: " << "thread(" << id << "), "
                          << "is currently active, scheduling new thread...";

            thread_init_data data(
                boost::bind(&threadmanager_impl::set_active_state, this, 
                    id, new_state, new_state_ex), 
                "set state for active thread");
            register_work(data);

            return previous_state;     // done
        }
        else if (previous_state_val == terminated) {
            // If the thread has been terminated while this set_state was 
            // pending nothing has to be done anymore.
            return previous_state;
        }

        // If the previous state was pending we are supposed to remove the
        // thread from the queue. But in order to avoid linearly looking 
        // through the queue we defer this to the thread function, which 
        // at some point will ignore this thread by simply skipping it 
        // (if it's not pending anymore). 

        LTM_(info) << "set_state: " << "thread(" << id << "), "
                   << "description(" << thrd->get_description() << "), "
                   << "new state(" << get_thread_state_name(new_state) << ")";

        // So all what we do here is to set the new state.
        thrd->set_state(new_state);
        thrd->set_state_ex(new_state_ex);

        if (new_state == pending) {
            scheduler_.schedule_thread(thrd);
            do_some_work();
        }

        return previous_state;
    }

    /// The get_state function is part of the thread related API and allows
    /// to query the state of one of the threads known to the threadmanager_impl
    template <typename SchedulingPolicy, typename NotificationPolicy>
    thread_state threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        get_state(thread_id_type id) 
    {
        // we know that the id is actually the pointer to the thread
        thread* thrd = reinterpret_cast<thread*>(id);
        return thrd->get() ? thrd->get_state() : thread_state(terminated);
    }

    /// The get_description function is part of the thread related API and 
    /// allows to query the description of one of the threads known to the 
    /// threadmanager_impl
    template <typename SchedulingPolicy, typename NotificationPolicy>
    std::string threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        get_description(thread_id_type id)
    {
        // we know that the id is actually the pointer to the thread
        thread* thrd = reinterpret_cast<thread*>(id);
        return thrd->get() ? thrd->get_description() : "<unknown>";
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    std::string threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        get_lco_description(thread_id_type id)
    {
        // we know that the id is actually the pointer to the thread
        thread* thrd = reinterpret_cast<thread*>(id);
        return thrd->get() ? thrd->get_lco_description() : "<unknown>";
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    void threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        set_lco_description(thread_id_type id, char const* desc)
    {
        // we know that the id is actually the pointer to the thread
        thread* thrd = reinterpret_cast<thread*>(id);
        return thrd->get() ? thrd->set_lco_description(desc) : "<unknown>";
    }

    /// This thread function is used by the at_timer thread below to trigger
    /// the required action.
    template <typename SchedulingPolicy, typename NotificationPolicy>
    thread_state_enum threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        wake_timer_thread (thread_id_type id, 
            thread_state_enum newstate, thread_state_ex_enum newstate_ex, 
            thread_id_type timer_id) 
    {
        // first trigger the requested set_state 
        set_state(id, newstate, newstate_ex);

        // then re-activate the thread holding the deadline_timer
        set_state(timer_id, pending, wait_timeout);
        return terminated;
    }

    /// This thread function initiates the required set_state action (on 
    /// behalf of one of the threadmanager_impl#set_state functions).
    template <typename SchedulingPolicy, typename NotificationPolicy>
    template <typename TimeType>
    thread_state_enum threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        at_timer (TimeType const& expire, thread_id_type id, 
            thread_state_enum newstate, thread_state_ex_enum newstate_ex)
    {
        // create a new thread in suspended state, which will execute the 
        // requested set_state when timer fires and will re-awaken this thread, 
        // allowing the deadline_timer to go out of scope gracefully
        thread_self& self = get_self();
        thread_id_type self_id = self.get_thread_id();

        thread_init_data data(
            boost::bind(&threadmanager_impl::wake_timer_thread, this, id, 
                newstate, newstate_ex, self_id), 
            "wake_timer");
        thread_id_type wake_id = register_thread(data, suspended);

        // create timer firing in correspondence with given time
        boost::asio::deadline_timer t (timer_pool_.get_io_service(), expire);

        // let the timer invoke the set_state on the new (suspended) thread
        t.async_wait(boost::bind(&threadmanager_impl::set_state, this, wake_id, 
            thread_state(pending), thread_state_ex(wait_timeout)));

        // this waits for the thread to be reactivated when the timer fired
        self.yield(suspended);
        return terminated;
    }

    /// Set a timer to set the state of the given \a thread to the given 
    /// new value after it expired (at the given time)
    template <typename SchedulingPolicy, typename NotificationPolicy>
    thread_id_type threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        set_state (time_type const& expire_at, thread_id_type id, 
            thread_state_enum newstate, thread_state_ex_enum newstate_ex)
    {
        // this creates a new thread which creates the timer and handles the
        // requested actions
        thread_state_enum (threadmanager_impl::*f)(time_type const&, 
                thread_id_type, thread_state_enum, thread_state_ex_enum)
            = &threadmanager_impl::template at_timer<time_type>;

        thread_init_data data(
            boost::bind(f, this, expire_at, id, newstate, newstate_ex),
            "at_timer (expire at)");
        return register_thread(data);
    }

    /// Set a timer to set the state of the given \a thread to the given
    /// new value after it expired (after the given duration)
    template <typename SchedulingPolicy, typename NotificationPolicy>
    thread_id_type threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        set_state (duration_type const& from_now, thread_id_type id, 
            thread_state_enum newstate, thread_state_ex_enum newstate_ex)
    {
        // this creates a new thread which creates the timer and handles the
        // requested actions
        thread_state_enum (threadmanager_impl::*f)(duration_type const&, 
                thread_id_type, thread_state_enum, thread_state_ex_enum)
            = &threadmanager_impl::template at_timer<duration_type>;

        thread_init_data data(
            boost::bind(f, this, from_now, id, newstate, newstate_ex),
            "at_timer (from now)");
        return register_thread(data);
    }

    /// Retrieve the global id of the given thread
    template <typename SchedulingPolicy, typename NotificationPolicy>
    naming::id_type const& 
    threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        get_thread_gid(thread_id_type id) 
    {
        // we know that the id is actually the pointer to the thread
        thread* thrd = reinterpret_cast<thread*>(id);
        return thrd->get() ? thrd->get_gid() : naming::invalid_id;
    }

    // helper class for switching thread state in and out during execution
    class switch_status
    {
    public:
        switch_status (thread* t, thread_state prev_state)
          : thread_(t), prev_state_(prev_state),
            need_restore_state_(t->set_state_tagged(active, prev_state_, orig_state_))
        {}

        ~switch_status ()
        {
            if (need_restore_state_)
                store_state(prev_state_);
        }

        bool is_valid() const { return need_restore_state_; }

        // allow to change the state the thread will be switched to after 
        // execution
        thread_state operator=(thread_state_enum new_state)
        {
            return prev_state_ = thread_state(new_state);
        }

        // Get the state this thread was in before execution (usually pending),
        // this helps making sure no other OS-thread is started to execute this
        // PX-thread in the meantime.
        thread_state get_previous() const
        {
            return prev_state_;
        }

        // This restores the previous state, while making sure that the 
        // original state has not been changed since we started executing this
        // thread. The function returns true if the state has been set, false
        // otherwise.
        bool store_state(thread_state& newstate)
        {
            disable_restore();
            if (thread_->restore_state(prev_state_, orig_state_)) {
                newstate = prev_state_;
                return true;
            }
            return false;
        }

        // disable default handling in destructor
        void disable_restore() { need_restore_state_ = false; }

    private:
        thread* thread_;
        thread_state prev_state_;
        thread_state orig_state_;
        bool need_restore_state_;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct manage_counter_type
    {
        manage_counter_type()
          : status_(performance_counters::status_invalid_data)
        {
        }
        ~manage_counter_type()
        {
            if (performance_counters::status_invalid_data != status_) {
                error_code ec;
                util::remove_counter_type(info_, ec);   // ignore errors
            }
        }

        performance_counters::counter_status install(
            std::string const& name, performance_counters::counter_type type, 
            error_code& ec = throws)
        {
            BOOST_ASSERT(performance_counters::status_invalid_data == status_);
            info_.fullname_ = name;
            info_.type_ = type;
            status_ = util::add_counter_type(info_, ec);
            return status_;
        }

        performance_counters::counter_status status_;
        performance_counters::counter_info info_;
    };

    ///////////////////////////////////////////////////////////////////////////
    // main function executed by all OS threads managed by this threadmanager_impl
    template <typename SP, typename NP>
    struct init_tss_helper
    {
        typedef threadmanager_impl<SP, NP> threadmanager_type;

        init_tss_helper(threadmanager_type& tm, std::size_t thread_num)
          : tm_(tm)
        {
            tm_.init_tss(thread_num);
        }
        ~init_tss_helper()
        {
            tm_.deinit_tss();
        }

        threadmanager_type& tm_;
    };

    template <typename SchedulingPolicy, typename NotificationPolicy>
    void threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        tfunc(std::size_t num_thread)
    {
        // manage the number of this thread in its TSS
        init_tss_helper<SchedulingPolicy, NotificationPolicy> 
            tss_helper(*this, num_thread);

        // needs to be done as the first thing, otherwise logging won't work
        notifier_.on_start_thread(num_thread);       // notify runtime system of started thread
        scheduler_.on_start_thread(num_thread);

        {
            manage_counter_type counter_type;
            if (0 == num_thread) {
                // register counter types
                error_code ec;
                counter_type.install("/queue/length", 
                    performance_counters::counter_raw, ec);   // doesn't throw
                if (ec) {
                    LTM_(info) << "tfunc(" << num_thread << "): failed to install "
                        "counter type '/queue/length': " << ec.get_message();
                }
            }

            LTM_(info) << "tfunc(" << num_thread << "): start";
            std::size_t num_px_threads = 0;
            try {
                num_px_threads = tfunc_impl(num_thread);
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
                LFATAL_ << "tfunc(" << num_thread 
                        << "): caught std::exception: " 
                        << e.what() << ", aborted thread execution";
                report_error(num_thread, boost::current_exception());
                return;
            }
            catch (...) {
                LFATAL_ << "tfunc(" << num_thread << "): caught unexpected "
                    "exception, aborted thread execution";
                report_error(num_thread, boost::current_exception());
                return;
            }

            LTM_(fatal) << "tfunc(" << num_thread << "): end, executed " 
                       << num_px_threads << " HPX threads";
        }

        notifier_.on_stop_thread(num_thread);
        scheduler_.on_stop_thread(num_thread);
    }

    ///////////////////////////////////////////////////////////////////////////
    struct manage_counter
    {
        ~manage_counter()
        {
            uninstall();
        }

        performance_counters::counter_status install(
            std::string const& name, boost::function<boost::int64_t()> f, 
            error_code& ec = throws)
        {
            BOOST_ASSERT(!counter_);
            info_.fullname_ = name;
            return util::add_counter(info_, f, counter_, ec);
        }

        void uninstall()
        {
            if (counter_) {
                error_code ec;
                util::remove_counter(info_, counter_, ec);
                counter_ = naming::invalid_id;
            }
        }

        performance_counters::counter_info info_;
        naming::id_type counter_;
    };

    ///////////////////////////////////////////////////////////////////////////
    inline void write_old_state_log(std::size_t num_thread, thread* thrd, 
        thread_state_enum state)
    {
        LTM_(debug) << "tfunc(" << num_thread << "): "
                   << "thread(" << thrd->get_thread_id() << "), " 
                   << "description(" << thrd->get_description() << "), "
                   << "old state(" << get_thread_state_name(state) << ")";
    }

    inline void write_new_state_log_debug(std::size_t num_thread, thread* thrd, 
        thread_state_enum state, char const* info)
    {
        LTM_(debug) << "tfunc(" << num_thread << "): "
                   << "thread(" << thrd->get_thread_id() << "), "
                   << "description(" << thrd->get_description() << "), "
                   << "new state(" << get_thread_state_name(state) << "), "
                   << info;
    }
    inline void write_new_state_log_warning(std::size_t num_thread, thread* thrd, 
        thread_state_enum state, char const* info)
    {
        LTM_(debug) << "tfunc(" << num_thread << "): "
                   << "thread(" << thrd->get_thread_id() << "), "
                   << "description(" << thrd->get_description() << "), "
                   << "new state(" << get_thread_state_name(state) << "), "
                   << info;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy>
    std::size_t threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        tfunc_impl(std::size_t num_thread)
    {
#if HPX_DEBUG != 0
        ++thread_count_;
#endif
        std::size_t num_px_threads = 0;
        util::time_logger tl1("tfunc", num_thread);
        util::time_logger tl2("tfunc2", num_thread);

        // the thread with number zero is the master
        bool is_master_thread = (0 == num_thread) ? true : false;
        set_affinity(num_thread);     // set affinity on Linux systems

        // register performance counters
        manage_counter queue_length_counter; 
        if (is_master_thread) {
            std::string name("/queue(threadmanager)/length");
            queue_length_counter.install(name, 
                boost::bind(&scheduling_policy_type::get_queue_lengths, 
                    &scheduler_, num_thread));
        }

        std::size_t idle_loop_count = 0;
        
        // run the work queue
        boost::coroutines::prepare_main_thread main_thread;
        while (true) {
            // Get the next PX thread from the queue
            thread* thrd = NULL;
            if (scheduler_.get_next_thread(num_thread, &thrd)) {
                idle_loop_count = 0;
                tl1.tick();

                // Only pending PX threads will be executed.
                // Any non-pending PX threads are leftovers from a set_state() 
                // call for a previously pending PX thread (see comments above).
                thread_state state = thrd->get_state();
                thread_state_enum state_val = state;

                write_old_state_log(num_thread, thrd, state_val);

                if (pending == state_val) {
                    // switch the state of the thread to active and back to 
                    // what the thread reports as its return value

                    {
                        // tries to set state to active (only if state is still 
                        // the same as 'state')
                        switch_status thrd_stat (thrd, state);
                        if (thrd_stat.is_valid() && thrd_stat.get_previous() == pending) {
                            // thread returns new required state
                            // store the returned state in the thread
                            tl2.tick();
                            thrd_stat = (*thrd)();
                            tl2.tock();
                            ++num_px_threads;
                        }
                        else {
                            // some other OS-thread got in between and started 
                            // executing this PX-thread, we just continue with 
                            // the next one
                            thrd_stat.disable_restore();
                            write_new_state_log_warning(
                                num_thread, thrd, state_val, "no execution");
                            tl1.tock();
                            continue;
                        }

                        // store and retrieve the new state in the thread 
                        if (!thrd_stat.store_state(state)) {
                            // some other OS-thread got in between and changed 
                            // the state of this thread, we just continue with 
                            // the next one
                            write_new_state_log_warning(
                                num_thread, thrd, state_val, "no state change");
                            tl1.tock();
                            continue;
                        }
                        state_val = state;

                        // any exception thrown from the thread will reset its 
                        // state at this point
                    }

                    write_new_state_log_debug(num_thread, thrd, state_val, "normal");

                    // Re-add this work item to our list of work items if the PX
                    // thread should be re-scheduled. If the PX thread is suspended 
                    // now we just keep it in the map of threads.
                    if (state_val == pending) {
                        scheduler_.schedule_thread(thrd, num_thread);
                        do_some_work(num_thread);
                    }
                }
                else if (active == state_val) {
                    // re-schedule thread, if it is still marked as active
                    // this might happen, if some thread has been added to the
                    // scheduler queue already but the state has not been reset 
                    // yet
                    scheduler_.schedule_thread(thrd, num_thread);
                }

                // Remove the mapping from thread_map_ if PX thread is depleted 
                // or terminated, this will delete the PX thread as all 
                // references go out of scope.
                // FIXME: what has to be done with depleted PX threads?
                if (state_val == depleted || state_val == terminated) 
                    scheduler_.destroy_thread(thrd);

                tl1.tock();
            }

            // if we need to terminate, unregister the counter first
            if (!running_)
                queue_length_counter.uninstall();

            // if nothing else has to be done either wait or terminate
            if (scheduler_.wait_or_add_new(num_thread, running_, idle_loop_count))
                break;
        }

#if HPX_DEBUG != 0
        // the last OS thread is allowed to exit only if no more PX threads exist
        BOOST_ASSERT(0 != --thread_count_ || !scheduler_.get_thread_count(num_thread));
#endif
        return num_px_threads;
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
        if (!threads_.empty() || running_) 
            return true;    // do nothing if already running

        LTM_(info) << "run: running timer pool"; 
        timer_pool_.run(false);

        running_ = false;
        try {
            // run threads and wait for initialization to complete
            running_ = true;
            while (num_threads-- != 0) {
                LTM_(info) << "run: create OS thread: " << num_threads; 

                // create a new thread
                threads_.push_back(new boost::thread(
                    boost::bind(&threadmanager_impl::tfunc, this, num_threads)));

                // set the new threads affinity (on Windows systems)
                set_affinity(threads_.back(), num_threads);
            }

            // start timer pool as well
            timer_pool_.run(false);
        }
        catch (std::exception const& e) {
            LTM_(fatal) << "run: failed with:" << e.what(); 
            stop();
            threads_.clear();
            return false;
        }

        LTM_(info) << "run: running"; 
        return running_;
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    void threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        stop (bool blocking)
    {
        LTM_(info) << "stop: blocking(" << std::boolalpha << blocking << ")"; 

        mutex_type::scoped_lock l(mtx_);
        if (!threads_.empty()) {
            if (running_) {
                LTM_(info) << "stop: set running_ = false"; 
                running_ = false;
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
            }

            LTM_(info) << "stop: stopping timer pool"; 
            timer_pool_.stop();             // stop timer pool as well
            if (blocking) {
                timer_pool_.join();
                timer_pool_.clear();
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy>
    boost::thread_specific_ptr<std::size_t> 
    threadmanager_impl<SchedulingPolicy, NotificationPolicy>::thread_num_;

    template <typename SchedulingPolicy, typename NotificationPolicy>
    void threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        init_tss(std::size_t thread_num)
    {
        BOOST_ASSERT(NULL == threadmanager_impl::thread_num_.get());    // shouldn't be initialized yet
        threadmanager_impl::thread_num_.reset(new std::size_t(thread_num));
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    void threadmanager_impl<SchedulingPolicy, NotificationPolicy>::deinit_tss()
    {
        threadmanager_impl::thread_num_.reset();
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    std::size_t 
    threadmanager_impl<SchedulingPolicy, NotificationPolicy>::get_thread_num()
    {
        if (NULL != threadmanager_impl::thread_num_.get())
            return *threadmanager_impl::thread_num_;

        // some OS threads are not managed by the threadmanager
        return std::size_t(-1);
    }

}}

///////////////////////////////////////////////////////////////////////////////
/// explicit template instantiation for the thread manager of our choice
#include <hpx/runtime/threads/policies/global_queue_scheduler.hpp>
#include <hpx/runtime/threads/policies/local_queue_scheduler.hpp>
#include <hpx/runtime/threads/policies/callback_notifier.hpp>

template HPX_EXPORT class hpx::threads::threadmanager_impl<
    hpx::threads::policies::global_queue_scheduler, 
    hpx::threads::policies::callback_notifier>;

template HPX_EXPORT class hpx::threads::threadmanager_impl<
    hpx::threads::policies::local_queue_scheduler, 
    hpx::threads::policies::callback_notifier>;
