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
#include <hpx/include/performance_counters.hpp>
#include <hpx/performance_counters/counter_creators.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/util/unlock_lock.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/block_profiler.hpp>
#include <hpx/util/itt_notify.hpp>
#include <hpx/util/apex.hpp>
#include <hpx/util/stringstream.hpp>
#include <hpx/util/hardware/timestamp.hpp>

#include <boost/assert.hpp>
#include <boost/make_shared.hpp>
#include <boost/bind.hpp>
#include <boost/asio/deadline_timer.hpp>
#include <boost/cstdint.hpp>
#include <boost/format.hpp>

#if defined(_POSIX_VERSION)
#include <sys/syscall.h>
#include <sys/resource.h>
#endif

#include <numeric>

#if HPX_THREAD_MAINTAIN_QUEUE_WAITTIME
///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace policies
{
    ///////////////////////////////////////////////////////////////////////////
    // We control whether to collect queue wait times using this global bool.
    // It will be set by any of the related performance counters. Once set it
    // stays set, thus no race conditions will occur.
    bool maintain_queue_wait_times = false;
}}}
#endif

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
            "terminated",
            "staged"
        };
    }

    char const* get_thread_state_name(thread_state_enum state)
    {
        if (state < unknown || state > staged)
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
        used_processing_units_(hardware_concurrency())
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
        get_thread_count(thread_state_enum state, thread_priority priority) const
    {
        mutex_type::scoped_lock lk(mtx_);
        return scheduler_.get_thread_count(state, priority);
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
        if (thread_count_ == 0 && state_.load() != running)
        {
            // thread-manager is not currently running
            HPX_THROWS_IF(ec, invalid_status,
                "threadmanager_impl::register_thread",
                "invalid state: thread manager is not running");
            return invalid_thread_id;
        }

        // verify parameters
        switch (initial_state) {
        case pending:
        case suspended:
            break;

        default:
            {
                hpx::util::osstream strm;
                strm << "invalid initial state: "
                     << get_thread_state_name(initial_state);
                HPX_THROWS_IF(ec, bad_parameter,
                    "threadmanager_impl::register_thread",
                    hpx::util::osstream_get_string(strm));
                return invalid_thread_id;
            }
        }

#if HPX_THREAD_MAINTAIN_DESCRIPTION
        if (0 == data.description)
        {
            HPX_THROWS_IF(ec, bad_parameter,
                "threadmanager_impl::register_thread", "description is NULL");
            return invalid_thread_id;
        }
#endif

#if HPX_THREAD_MAINTAIN_PARENT_REFERENCE
        if (0 == data.parent_id) {
            thread_self* self = get_self_ptr();
            if (self)
            {
                data.parent_id = self->get_thread_id();
                data.parent_phase = self->get_thread_phase();
            }
        }
        if (0 == data.parent_locality_id)
            data.parent_locality_id = get_locality_id();
#endif

        // NOTE: This code overrides a request to schedule a thread on a scheduler
        // selected queue. The schedulers are written to select a queue to put
        // a thread in if the OS thread number is -1. Not only does overriding this
        // prevent extensibility by forcing a certain queuing behavior, but it also
        // schedules unfairly. A px thread is always put into the queue of the
        // OS thread that it's producer is currently running on. In a single
        // producer environment, this can lead to unexpected imbalances and
        // work only gets distributed by work stealing.
        //if (std::size_t(-1)  == data.num_os_thread)
        //    data.num_os_thread = get_worker_thread_num();

        // create the new thread
        thread_id_type newid = scheduler_.create_thread(
            data, initial_state, run_now, ec, data.num_os_thread);

        LTM_(info) << "register_thread(" << newid << "): initial_state("
                   << get_thread_state_name(initial_state) << "), "
                   << "run_now(" << (run_now ? "true" : "false")
#if HPX_THREAD_MAINTAIN_DESCRIPTION
                   << "), description(" << data.description
#endif
                   << ")";

        return newid;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy>
    void threadmanager_impl<SchedulingPolicy, NotificationPolicy>::register_work(
        thread_init_data& data, thread_state_enum initial_state, error_code& ec)
    {
        util::block_profiler_wrapper<register_work_tag> bp(work_logger_);

        // verify state
        if (thread_count_ == 0 && state_.load() != running)
        {
            // thread-manager is not currently running
            HPX_THROWS_IF(ec, invalid_status,
                "threadmanager_impl::register_work",
                "invalid state: thread manager is not running");
            return;
        }

        // verify parameters
        switch (initial_state) {
        case pending:
        case suspended:
            break;

        default:
            {
                hpx::util::osstream strm;
                strm << "invalid initial state: "
                     << get_thread_state_name(initial_state);
                HPX_THROWS_IF(ec, bad_parameter,
                    "threadmanager_impl::register_work",
                    hpx::util::osstream_get_string(strm));
                return;
            }
        }

#if HPX_THREAD_MAINTAIN_DESCRIPTION
        if (0 == data.description)
        {
            HPX_THROWS_IF(ec, bad_parameter,
                "threadmanager_impl::register_work", "description is NULL");
            return;
        }
#endif

        LTM_(info) << "register_work: initial_state("
                   << get_thread_state_name(initial_state) << "), thread_priority("
                   << get_thread_priority_name(data.priority)
#if HPX_THREAD_MAINTAIN_DESCRIPTION
                   << "), description(" << data.description
#endif
                   << ")";

#if HPX_THREAD_MAINTAIN_PARENT_REFERENCE
        if (0 == data.parent_id) {
            thread_self* self = get_self_ptr();
            if (self)
            {
                data.parent_id = self->get_thread_id();
                data.parent_phase = self->get_thread_phase();
            }
        }
        if (0 == data.parent_locality_id)
            data.parent_locality_id = get_locality_id();
#endif

        // NOTE: This code overrides a request to schedule a thread on a scheduler
        // selected queue. The schedulers are written to select a queue to put
        // a thread in if the OS thread number is -1. Not only does overriding this
        // prevent extensibility by forcing a certain queuing behavior, but it also
        // schedules unfairly. A px thread is always put into the queue of the
        // OS thread that it's producer is currently running on. In a single
        // producer environment, this can lead to unexpected imbalances and
        // work only gets distributed by work stealing.
        //if (std::size_t(-1) == data.num_os_thread)
        //    data.num_os_thread = get_worker_thread_num();

        if (thread_priority_critical == data.priority) {
            // For critical priority threads, create the thread immediately.
            scheduler_.create_thread(data, initial_state, true, ec, data.num_os_thread);
        }
        else {
            // Create a task description for the new thread.
            scheduler_.create_thread(data, initial_state, false, ec, data.num_os_thread);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // thread function registered for set_state if thread is currently active
    template <typename SchedulingPolicy, typename NotificationPolicy>
    thread_state_enum threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        set_active_state(thread_id_type id,
            thread_state_enum newstate, thread_state_ex_enum newstate_ex,
            thread_priority priority, thread_state previous_state)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROW_EXCEPTION(null_thread_id,
                "threadmanager_impl::set_active_state",
                "NULL thread id encountered");
            return terminated;
        }

        // make sure that the thread has not been suspended and set active again
        // in the mean time
        thread_data* thrd = reinterpret_cast<thread_data*>(id);
        thread_state current_state = thrd->get_state();

        if (thread_state_enum(current_state) == thread_state_enum(previous_state) &&
            current_state != previous_state)
        {
            LTM_(warning)
                << "set_active_state: thread is still active, however "
                      "it was non-active since the original set_state "
                      "request was issued, aborting state change, thread("
                << id << "), description("
                << thrd->get_description() << "), new state("
                << get_thread_state_name(newstate) << ")";
            return terminated;
        }

        // just retry, set_state will create new thread if target is still active
        error_code ec(lightweight);      // do not throw
        set_state(id, newstate, newstate_ex, priority, ec);
        return terminated;
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
        if (HPX_UNLIKELY(!id)) {
            HPX_THROWS_IF(ec, null_thread_id, "threadmanager_impl::set_state",
                "NULL thread id encountered");
            return thread_state(unknown);
        }

        util::block_profiler_wrapper<set_state_tag> bp(set_state_logger_);

        // set_state can't be used to force a thread into active state
        if (new_state == active) {
            hpx::util::osstream strm;
            strm << "invalid new state: " << get_thread_state_name(new_state);
            HPX_THROWS_IF(ec, bad_parameter, "threadmanager_impl::set_state",
                hpx::util::osstream_get_string(strm));
            return thread_state(unknown);
        }

        // we know that the id is actually the pointer to the thread
        thread_data* thrd = reinterpret_cast<thread_data*>(id);
        if (!thrd) {
            if (&ec != &throws)
                ec = make_success_code();
            return thread_state(terminated);     // this thread has already been terminated
        }

        // handle priority, restore original priority of thread, if needed
        if (priority == thread_priority_default)
            priority = thrd->get_priority();
        else if (new_state == pending)
            thrd->set_priority(priority);

        BOOST_ASSERT(priority != thread_priority_default);

        thread_state previous_state;
        do {
            // action depends on the current state
            previous_state = thrd->get_state();
            thread_state_enum previous_state_val = previous_state;

            // nothing to do here if the state doesn't change
            if (new_state == previous_state_val) {
                LTM_(warning) << "set_state: old thread state is the same as new "
                               "thread state, aborting state change, thread("
                            << id << "), description("
                            << thrd->get_description() << "), new state("
                            << get_thread_state_name(new_state) << ")";

                if (&ec != &throws)
                    ec = make_success_code();

                return thread_state(new_state);
            }

            // the thread to set the state for is currently running, so we
            // schedule another thread to execute the pending set_state
            if (active == previous_state_val) {
                // schedule a new thread to set the state
                LTM_(warning)
                    << "set_state: thread is currently active, scheduling "
                        "new thread, thread(" << id << "), description("
                    << thrd->get_description() << "), new state("
                    << get_thread_state_name(new_state) << ")";

                thread_init_data data(
                    boost::bind(&threadmanager_impl::set_active_state, this,
                        id, new_state, new_state_ex, priority, previous_state),
                    "set state for active thread", 0, priority);
                register_work(data);

                if (&ec != &throws)
                    ec = make_success_code();

                return previous_state;     // done
            }
            else if (terminated == previous_state_val) {
                LTM_(warning)
                    << "set_state: thread is terminated, aborting state "
                        "change, thread(" << id << "), description("
                    << thrd->get_description() << "), new state("
                    << get_thread_state_name(new_state) << ")";

                if (&ec != &throws)
                    ec = make_success_code();

                // If the thread has been terminated while this set_state was
                // pending nothing has to be done anymore.
                return previous_state;
            }
            else if (pending == previous_state_val && suspended == new_state) {
                // we do not allow explicit resetting of a state to suspended
                // without the thread being executed.
                hpx::util::osstream strm;
                strm << "set_state: invalid new state, can't demote a pending thread, "
                     << "thread(" << id << "), description("
                     << thrd->get_description() << "), new state("
                     << get_thread_state_name(new_state) << ")";

                LTM_(fatal) << hpx::util::osstream_get_string(strm);

                HPX_THROWS_IF(ec, bad_parameter, "threadmanager_impl::set_state",
                    hpx::util::osstream_get_string(strm));
                return thread_state(unknown);
            }

            // If the previous state was pending we are supposed to remove the
            // thread from the queue. But in order to avoid linearly looking
            // through the queue we defer this to the thread function, which
            // at some point will ignore this thread by simply skipping it
            // (if it's not pending anymore).

            LTM_(info) << "set_state: thread(" << id << "), "
                          "description(" << thrd->get_description() << "), "
                          "new state(" << get_thread_state_name(new_state) << "), "
                          "old state(" << get_thread_state_name(previous_state_val)
                       << ")";

            // So all what we do here is to set the new state.
            if (thrd->restore_state(new_state, previous_state)) {
                thrd->set_state_ex(new_state_ex);
                break;
            }

            // state has changed since we fetched it from the thread, retry
            LTM_(error) << "set_state: state has been changed since it was fetched, "
                          "retrying, thread(" << id << "), "
                          "description(" << thrd->get_description() << "), "
                          "new state(" << get_thread_state_name(new_state) << "), "
                          "old state(" << get_thread_state_name(previous_state_val)
                       << ")";
        } while (true);

        if (new_state == pending) {
            // REVIEW: Passing a specific target thread may interfere with the
            // round robin queuing.
            scheduler_.schedule_thread(thrd, get_worker_thread_num(), priority);
            do_some_work();
        }

        if (&ec != &throws)
            ec = make_success_code();

        return previous_state;
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

    /// The get_priority function is part of the thread related API. It
    /// queries the priority of one of the threads known to the threadmanager_impl
    template <typename SchedulingPolicy, typename NotificationPolicy>
    thread_priority threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        get_priority(thread_id_type id)
    {
        // we know that the id is actually the pointer to the thread
        thread_data* thrd = reinterpret_cast<thread_data*>(id);
        return thrd ? thrd->get_priority() : thread_priority_unknown;
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

    template <typename SchedulingPolicy, typename NotificationPolicy>
    void threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        interruption_point(thread_id_type id, error_code& ec)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "threadmanager_impl::interruption_point",
                "NULL thread id encountered");
            return;
        }

        if (&ec != &throws)
            ec = make_success_code();

        // we know that the id is actually the pointer to the thread
        thread_data* thrd = reinterpret_cast<thread_data*>(id);
        if (thrd)
            thrd->interruption_point();      // notify thread
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

        // handle priority, restore original priority of thread, if needed
        if (priority == thread_priority_default)
        {
            // we know that the id is actually the pointer to the thread
            thread_data* thrd = reinterpret_cast<thread_data*>(id);
            priority = thrd->get_priority();
        }
        BOOST_ASSERT(priority != thread_priority_default);

        bool oldvalue = false;
        if (triggered->compare_exchange_strong(oldvalue, true)) //-V601
        {
            // timer has not been canceled yet, trigger the requested set_state
            set_state(id, newstate, newstate_ex, priority);
        }

        // then re-activate the thread holding the deadline_timer
        // REVIEW: Why do we ignore errors here?
        error_code ec(lightweight);    // do not throw
        set_state(timer_id, pending, wait_timeout, priority, ec);
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
                "threadmanager_impl::at_timer", "NULL thread id encountered");
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
            "wake_timer", 0, thread_priority_critical);
        thread_id_type wake_id = register_thread(data, suspended, true);

        // create timer firing in correspondence with given time
        boost::asio::deadline_timer t (timer_pool_.get_io_service(), expire);

        // let the timer invoke the set_state on the new (suspended) thread
        t.async_wait(boost::bind(&threadmanager_impl::set_state, this, wake_id,
            pending, wait_timeout, thread_priority_critical, boost::ref(throws)));

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
            "at_timer (expire at)", 0, thread_priority_critical);
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
            "at_timer (from now)", 0, thread_priority_critical);
        return register_thread(data, pending, true, ec);
    }

    // helper class for switching thread state in and out during execution
    class switch_status
    {
    public:
        switch_status (thread_data* t, thread_state prev_state)
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
            return prev_state_ = thread_state(new_state, prev_state_.get_tag() + 1);
        }

        // Get the state this thread was in before execution (usually pending),
        // this helps making sure no other worker-thread is started to execute this
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
        thread_data* thread_;
        thread_state prev_state_;
        thread_state orig_state_;
        bool need_restore_state_;
    };

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
          : has_exited_(false), counter_(counter)
        {
            ++counter_;
        }
        ~manage_active_thread_count()
        {
            if (!has_exited_)
                --counter_;
        }

        void exit()
        {
            has_exited_ = true;
            --counter_;
        }

        bool has_exited_;
        boost::atomic<long>& counter_;
    };

    template <typename SchedulingPolicy, typename NotificationPolicy>
    void threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        tfunc(std::size_t num_thread)
    {
        
        // Setting priority of worker threads to a lower priority, this needs to
        // be done in order to give the parcel pool threads higher priority
#if defined(_POSIX_VERSION)
        {
            pid_t tid;
            tid = syscall(SYS_gettid);
            int ret = setpriority(PRIO_PROCESS, tid, 19);
            if(ret != 0)
            {
                HPX_THROW_EXCEPTION(no_success,
                    "threadmanager_impl::run", "setpriority returned an error");
            }
        }
#endif

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
    inline void write_old_state_log(std::size_t num_thread, thread_data* thrd,
        thread_state_enum state)
    {
        LTM_(debug) << "tfunc(" << num_thread << "): "
                   << "thread(" << thrd->get_thread_id() << "), "
                   << "description(" << thrd->get_description() << "), "
                   << "old state(" << get_thread_state_name(state) << ")";
    }

    inline void write_new_state_log_debug(std::size_t num_thread, thread_data* thrd,
        thread_state_enum state, char const* info)
    {
        LTM_(debug) << "tfunc(" << num_thread << "): "
            << "thread(" << thrd->get_thread_id() << "), "
            << "description(" << thrd->get_description() << "), "
            << "new state(" << get_thread_state_name(state) << "), "
            << info;
    }
    inline void write_new_state_log_warning(std::size_t num_thread, thread_data* thrd,
        thread_state_enum state, char const* info)
    {
        // log this in any case
        LTM_(warning) << "tfunc(" << num_thread << "): "
            << "thread(" << thrd->get_thread_id() << "), "
            << "description(" << thrd->get_description() << "), "
            << "new state(" << get_thread_state_name(state) << "), "
            << info;
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

        using HPX_STD_PLACEHOLDERS::_1;
        if (paths.instancename_ == "total" && paths.instanceindex_ == -1)
        {
            // overall counter
            using performance_counters::detail::create_raw_counter;
            HPX_STD_FUNCTION<boost::int64_t()> f =
                HPX_STD_BIND(&spt::get_queue_length, &scheduler_, -1);
            return create_raw_counter(info, f, ec);
        }
        else if (paths.instancename_ == "worker-thread" &&
            paths.instanceindex_ >= 0 &&
            std::size_t(paths.instanceindex_) < threads_.size())
        {
            // specific counter
            using performance_counters::detail::create_raw_counter;
            HPX_STD_FUNCTION<boost::int64_t()> f =
                HPX_STD_BIND(&spt::get_queue_length, &scheduler_,
                    static_cast<std::size_t>(paths.instanceindex_));
            return create_raw_counter(info, f, ec);
        }

        HPX_THROWS_IF(ec, bad_parameter, "queue_length_counter_creator",
            "invalid counter instance name: " + paths.instancename_);
        return naming::invalid_gid;
    }

#if HPX_THREAD_MAINTAIN_QUEUE_WAITTIME
    // average pending thread wait time
    template <typename SchedulingPolicy, typename NotificationPolicy>
    naming::gid_type threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        thread_wait_time_counter_creator(
            performance_counters::counter_info const& info, error_code& ec)
    {
        // verify the validity of the counter instance name
        performance_counters::counter_path_elements paths;
        performance_counters::get_counter_path_elements(info.fullname_, paths, ec);
        if (ec) return naming::invalid_gid;

        // /threads{locality#%d/total}/wait-time/pending
        // /threads{locality#%d/worker-thread%d}/wait-time/pending
        if (paths.parentinstance_is_basename_) {
            HPX_THROWS_IF(ec, bad_parameter,
                "thread_wait_time_counter_creator",
                "invalid counter instance parent name: " +
                    paths.parentinstancename_);
            return naming::invalid_gid;
        }

        typedef scheduling_policy_type spt;

        using HPX_STD_PLACEHOLDERS::_1;
        if (paths.instancename_ == "total" && paths.instanceindex_ == -1)
        {
            policies::maintain_queue_wait_times = true;

            // overall counter
            using performance_counters::detail::create_raw_counter;
            HPX_STD_FUNCTION<boost::int64_t()> f =
                HPX_STD_BIND(&spt::get_average_thread_wait_time, &scheduler_, -1);
            return create_raw_counter(info, f, ec);
        }
        else if (paths.instancename_ == "worker-thread" &&
            paths.instanceindex_ >= 0 &&
            std::size_t(paths.instanceindex_) < threads_.size())
        {
            policies::maintain_queue_wait_times = true;

            // specific counter
            using performance_counters::detail::create_raw_counter;
            HPX_STD_FUNCTION<boost::int64_t()> f =
                HPX_STD_BIND(&spt::get_average_thread_wait_time, &scheduler_,
                    static_cast<std::size_t>(paths.instanceindex_));
            return create_raw_counter(info, f, ec);
        }

        HPX_THROWS_IF(ec, bad_parameter, "thread_wait_time_counter_creator",
            "invalid counter instance name: " + paths.instancename_);
        return naming::invalid_gid;
    }

    // average pending task wait time
    template <typename SchedulingPolicy, typename NotificationPolicy>
    naming::gid_type threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        task_wait_time_counter_creator(
            performance_counters::counter_info const& info, error_code& ec)
    {
        // verify the validity of the counter instance name
        performance_counters::counter_path_elements paths;
        performance_counters::get_counter_path_elements(info.fullname_, paths, ec);
        if (ec) return naming::invalid_gid;

        // /threads{locality#%d/total}/wait-time/pending
        // /threads{locality#%d/worker-thread%d}/wait-time/pending
        if (paths.parentinstance_is_basename_) {
            HPX_THROWS_IF(ec, bad_parameter,
                "task_wait_time_counter_creator",
                "invalid counter instance parent name: " +
                    paths.parentinstancename_);
            return naming::invalid_gid;
        }

        typedef scheduling_policy_type spt;

        using HPX_STD_PLACEHOLDERS::_1;
        if (paths.instancename_ == "total" && paths.instanceindex_ == -1)
        {
            policies::maintain_queue_wait_times = true;

            // overall counter
            using performance_counters::detail::create_raw_counter;
            HPX_STD_FUNCTION<boost::int64_t()> f =
                HPX_STD_BIND(&spt::get_average_task_wait_time, &scheduler_, -1);
            return create_raw_counter(info, f, ec);
        }
        else if (paths.instancename_ == "worker-thread" &&
            paths.instanceindex_ >= 0 &&
            std::size_t(paths.instanceindex_) < threads_.size())
        {
            policies::maintain_queue_wait_times = true;

            // specific counter
            using performance_counters::detail::create_raw_counter;
            HPX_STD_FUNCTION<boost::int64_t()> f =
                HPX_STD_BIND(&spt::get_average_task_wait_time, &scheduler_,
                    static_cast<std::size_t>(paths.instanceindex_));
            return create_raw_counter(info, f, ec);
        }

        HPX_THROWS_IF(ec, bad_parameter, "task_wait_time_counter_creator",
            "invalid counter instance name: " + paths.instancename_);
        return naming::invalid_gid;
    }
#endif

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
                p.instancename_ = "total";
                p.instanceindex_ = -1;
            }

            status = get_counter_name(p, i.fullname_, ec);
            if (!status_is_valid(status) || !f(i, ec) || ec)
                return false;

            p.instancename_ = "allocator#*";
            p.instanceindex_ = -1;

            status = get_counter_name(p, i.fullname_, ec);
            if (!status_is_valid(status) || !f(i, ec) || ec)
                return false;
        }
        if (p.instancename_ == "total" && p.instanceindex_ == -1)
        {
            // overall counter
            status = get_counter_name(p, i.fullname_, ec);
            if (!status_is_valid(status) || !f(i, ec) || ec)
                return false;
        }
        else if (p.instancename_ == "allocator#*") 
        {
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

        using HPX_STD_PLACEHOLDERS::_1;
        if (paths.instancename_ == "total" && paths.instanceindex_ == -1)
        {
            // overall counter
            using performance_counters::detail::create_raw_counter;
            HPX_STD_FUNCTION<boost::int64_t(bool)> f =
                 HPX_STD_BIND(&ti::avg_idle_rate, this, _1);
            return create_raw_counter(info, f, ec);
        }
        else if (paths.instancename_ == "worker-thread" &&
            paths.instanceindex_ >= 0 &&
            std::size_t(paths.instanceindex_) < threads_.size())
        {
            // specific counter
            using performance_counters::detail::create_raw_counter;
            HPX_STD_FUNCTION<boost::int64_t(bool)> f =
                HPX_STD_BIND(&ti::avg_idle_rate, this,
                    static_cast<std::size_t>(paths.instanceindex_), _1);
            return create_raw_counter(info, f, ec);
        }

        HPX_THROWS_IF(ec, bad_parameter, "idle_rate_counter_creator",
            "invalid counter instance name: " + paths.instancename_);
        return naming::invalid_gid;
    }

    ///////////////////////////////////////////////////////////////////////////
    naming::gid_type
    counter_creator(performance_counters::counter_info const& info,
        performance_counters::counter_path_elements const& paths,
        HPX_STD_FUNCTION<boost::int64_t(bool)> const& total_creator,
        HPX_STD_FUNCTION<boost::int64_t(bool)> const& individual_creator,
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
            HPX_STD_FUNCTION<boost::int64_t(bool)> total_func;
            HPX_STD_FUNCTION<boost::int64_t(bool)> individual_func;
            char const* const individual_name;
            std::size_t individual_count;
        };

        typedef scheduling_policy_type spt;
        typedef threadmanager_impl ti;

        using HPX_STD_PLACEHOLDERS::_1;

        std::size_t shepherd_count = threads_.size();
        creator_data data[] =
        {
            // /threads{locality#%d/total}/count/cumulative
            // /threads{locality#%d/worker-thread%d}/count/cumulative
            { "count/cumulative",
              HPX_STD_BIND(&ti::get_executed_threads, this, -1, _1),
              HPX_STD_BIND(&ti::get_executed_threads, this,
                  static_cast<std::size_t>(paths.instanceindex_), _1),
              "worker-thread", shepherd_count
            },
            // /threads{locality#%d/total}/count/instantaneous/all
            // /threads{locality#%d/worker-thread%d}/count/instantaneous/all
            { "count/instantaneous/all",
              HPX_STD_BIND(&spt::get_thread_count, &scheduler_, unknown,
                  thread_priority_default, std::size_t(-1), _1),
              HPX_STD_BIND(&spt::get_thread_count, &scheduler_, unknown,
                  thread_priority_default,
                  static_cast<std::size_t>(paths.instanceindex_), _1),
              "worker-thread", shepherd_count
            },
            // /threads{locality#%d/total}/count/instantaneous/active
            // /threads{locality#%d/worker-thread%d}/count/instantaneous/active
            { "count/instantaneous/active",
              HPX_STD_BIND(&spt::get_thread_count, &scheduler_, active,
                  thread_priority_default, std::size_t(-1), _1),
              HPX_STD_BIND(&spt::get_thread_count, &scheduler_, active,
                  thread_priority_default,
                  static_cast<std::size_t>(paths.instanceindex_), _1),
              "worker-thread", shepherd_count
            },
            // /threads{locality#%d/total}/count/instantaneous/pending
            // /threads{locality#%d/worker-thread%d}/count/instantaneous/pending
            { "count/instantaneous/pending",
              HPX_STD_BIND(&spt::get_thread_count, &scheduler_, pending,
                  thread_priority_default, std::size_t(-1), _1),
              HPX_STD_BIND(&spt::get_thread_count, &scheduler_, pending,
                  thread_priority_default,
                  static_cast<std::size_t>(paths.instanceindex_), _1),
              "worker-thread", shepherd_count
            },
            // /threads{locality#%d/total}/count/instantaneous/suspended
            // /threads{locality#%d/worker-thread%d}/count/instantaneous/suspended
            { "count/instantaneous/suspended",
              HPX_STD_BIND(&spt::get_thread_count, &scheduler_, suspended,
                  thread_priority_default, std::size_t(-1), _1),
              HPX_STD_BIND(&spt::get_thread_count, &scheduler_, suspended,
                  thread_priority_default,
                  static_cast<std::size_t>(paths.instanceindex_), _1),
              "worker-thread", shepherd_count
            },
            // /threads(locality#%d/total}/count/instantaneous/terminated
            // /threads(locality#%d/worker-thread%d}/count/instantaneous/terminated
            { "count/instantaneous/terminated",
              HPX_STD_BIND(&spt::get_thread_count, &scheduler_, terminated,
                  thread_priority_default, std::size_t(-1), _1),
              HPX_STD_BIND(&spt::get_thread_count, &scheduler_, terminated,
                  thread_priority_default,
                  static_cast<std::size_t>(paths.instanceindex_), _1),
              "worker-thread", shepherd_count
            },
            // /threads{locality#%d/total}/count/instantaneous/staged
            // /threads{locality#%d/worker-thread%d}/count/instantaneous/staged
            { "count/instantaneous/staged",
              HPX_STD_BIND(&spt::get_thread_count, &scheduler_, staged,
                  thread_priority_default, std::size_t(-1), _1),
              HPX_STD_BIND(&spt::get_thread_count, &scheduler_, staged,
                  thread_priority_default,
                  static_cast<std::size_t>(paths.instanceindex_), _1),
              "worker-thread", shepherd_count
            },
            // /threads{locality#%d/total}/count/stack-recycles
            { "count/stack-recycles",
              HPX_STD_BIND(&coroutine_type::impl_type::get_stack_recycle_count, _1),
              HPX_STD_FUNCTION<boost::uint64_t(bool)>(), "", 0
            },
#if !defined(BOOST_WINDOWS) && !defined(HPX_COROUTINE_USE_GENERIC_CONTEXT)
            // /threads{locality#%d/total}/count/stack-unbinds
            { "count/stack-unbinds",
              HPX_STD_BIND(&coroutine_type::impl_type::get_stack_unbind_count, _1),
              HPX_STD_FUNCTION<boost::uint64_t(bool)>(), "", 0
            },
#endif
            // /threads{locality#%d/total}/count/objects
            // /threads{locality#%d/allocator%d}/count/objects
            { "count/objects",
              &coroutine_type::impl_type::get_allocation_count_all,
              HPX_STD_BIND(&coroutine_type::impl_type::get_allocation_count,
                  static_cast<std::size_t>(paths.instanceindex_), _1),
              "allocator", HPX_COROUTINE_NUM_ALL_HEAPS
            },
            // /threads{locality#%d/total}/count/stolen
            // /threads{locality#%d/worker-thread%d}/count/stolen
            { "count/stolen",
              HPX_STD_BIND(&spt::get_num_stolen_threads, &scheduler_,
                  std::size_t(-1), _1),
              HPX_STD_BIND(&spt::get_num_stolen_threads, &scheduler_,
                  static_cast<std::size_t>(paths.instanceindex_), _1),
              "worker-thread", shepherd_count
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
#if HPX_THREAD_MAINTAIN_QUEUE_WAITTIME
            // average thread wait time for queue(s)
            { "/threads/wait-time/pending", performance_counters::counter_raw,
              "returns the average wait time of pending threads for the referenced queue",
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&ti::thread_wait_time_counter_creator, this, _1, _2),
              &performance_counters::locality_thread_counter_discoverer,
              "ns"
            },
            // average task wait time for queue(s)
            { "/threads/wait-time/staged", performance_counters::counter_raw,
              "returns the average wait time of staged threads (task descriptions) "
              "for the referenced queue",
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&ti::task_wait_time_counter_creator, this, _1, _2),
              &performance_counters::locality_thread_counter_discoverer,
              "ns"
            },
#endif
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
            { "/threads/count/instantaneous/staged", performance_counters::counter_raw,
              "returns the current number of staged HPX-threads (task descriptions) "
              "at the referenced locality",
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
              "returns the overall number of created HPX-threads for "
              "the referenced locality", HPX_PERFORMANCE_COUNTER_V1,
              counts_creator,
              &locality_allocator_counter_discoverer,
              ""
            },
            { "/threads/count/stolen", performance_counters::counter_raw,
              "returns the overall number of HPX-threads stolen from neighboring"
              "schedulers for the referenced locality", HPX_PERFORMANCE_COUNTER_V1,
              counts_creator,
              &performance_counters::locality_thread_counter_discoverer,
              ""
            },
        };
        performance_counters::install_counter_types(
            counter_types, sizeof(counter_types)/sizeof(counter_types[0]));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy>
    void threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        tfunc_impl(std::size_t num_thread)
    {
        util::itt::stack_context ctx;        // helper for itt support
        util::itt::domain domain(get_thread_name()->data());
//         util::itt::id threadid(domain, this);
        util::itt::frame_context fctx(domain);

        manage_active_thread_count count(thread_count_);

        boost::int64_t idle_loop_count = 0;
        boost::int64_t busy_loop_count = 0;

        // set affinity on Linux systems or when using HWLOC
        topology const& topology_ = get_topology();
        threads::mask_cref_type mask = get_pu_mask(topology_, num_thread);

        LTM_(info) << "tfunc(" << num_thread
            << "): will run on one processing unit within this mask: "
            << std::hex << "0x" << mask;

        error_code ec(lightweight);
        topology_.set_thread_affinity_mask(mask, ec);

        if (ec)
        {
            LTM_(warning) << "run: setting thread affinity on OS thread "
                << num_thread << " failed with: " << ec.get_message();
        }

        // run the work queue
        hpx::util::coroutines::prepare_main_thread main_thread;

        boost::int64_t& executed_threads = executed_threads_[num_thread];
        boost::uint64_t& tfunc_time = tfunc_times[num_thread];
        boost::uint64_t& exec_time = exec_times[num_thread];
        boost::uint64_t overall_timestamp = util::hardware::timestamp();

        start_periodic_maintenance<SchedulingPolicy>(
            typename SchedulingPolicy::has_periodic_maintenance());

        util::apex_wrapper apex("hpx-thread-scheduler-loop");
        while (true) {
            // Get the next PX thread from the queue
            thread_data* thrd = NULL;
            if (scheduler_.get_next_thread(num_thread,
                    state_.load() == running, idle_loop_count, thrd))
            {
                idle_loop_count = 0;
                ++busy_loop_count;

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
                            {
#if defined(HPX_USE_ITTNOTIFY)
                                util::itt::caller_context cctx(ctx);
                                util::itt::undo_frame_context undoframe(fctx);
                                util::itt::task task(domain, thrd->get_description());
#endif
#if defined(HPX_HAVE_APEX)
                                util::apex_wrapper apex("hpx-user-level-thread");
#endif

                                // Record time elapsed in thread changing state
                                // and add to aggregate execution time.
                                boost::uint64_t timestamp = util::hardware::timestamp();
                                thrd_stat = (*thrd)();
                                exec_time += util::hardware::timestamp() - timestamp;
                            }

                            tfunc_time = util::hardware::timestamp() - overall_timestamp;
                            ++executed_threads;
                        }
                        else {
                            // some other worker-thread got in between and started
                            // executing this PX-thread, we just continue with
                            // the next one
                            thrd_stat.disable_restore();
                            write_new_state_log_warning(
                                num_thread, thrd, state_val, "no execution");
                            continue;
                        }

                        // store and retrieve the new state in the thread
                        if (!thrd_stat.store_state(state)) {
                            // some other worker-thread got in between and changed
                            // the state of this thread, we just continue with
                            // the next one
                            write_new_state_log_warning(
                                num_thread, thrd, state_val, "no state change");
                            continue;
                        }
                        state_val = state;

                        // any exception thrown from the thread will reset its
                        // state at this point
                    }

                    write_new_state_log_debug(num_thread, thrd, state_val, "normal");

                    // Re-add this work item to our list of work items if the HPX
                    // thread should be re-scheduled. If the HPX thread is suspended
                    // now we just keep it in the map of threads.
                    if (state_val == pending) {
                        // schedule other work
                        scheduler_.wait_or_add_new(num_thread,
                            state_.load() == running, idle_loop_count);

                        // schedule this thread again, make sure it ends up at
                        // the end of the queue
                        // REVIEW: Passing a specific target thread may mess
                        //         with the round robin queuing.
                        scheduler_.schedule_thread_last(thrd, num_thread);
                        do_some_work(num_thread);
                    }
                }
                else if (active == state_val) {
                    LTM_(warning) << "tfunc(" << num_thread << "): "
                        "thread(" << thrd->get_thread_id() << "), "
                        "description(" << thrd->get_description() << "), "
                        "rescheduling";

                    // re-schedule thread, if it is still marked as active
                    // this might happen, if some thread has been added to the
                    // scheduler queue already but the state has not been reset
                    // yet
                    // REVIEW: Passing a specific target thread may screw
                    // with the round robin queuing.
                    scheduler_.schedule_thread(thrd, num_thread);
                }

                // Remove the mapping from thread_map_ if HPX thread is depleted
                // or terminated, this will delete the HPX thread as all
                // references go out of scope.
                // REVIEW: what has to be done with depleted HPX threads?
                if (state_val == depleted || state_val == terminated)
                    scheduler_.destroy_thread(thrd, busy_loop_count);

                tfunc_time = util::hardware::timestamp() - overall_timestamp;
                // If we idle for some time, yield control to the OS scheduler
                // so other threads (like for example the parcelpool threads)
                // may be scheduled
            }

            // if nothing else has to be done either wait or terminate
            else {
                // create new threads from task descriptions, if available
                if (scheduler_.wait_or_add_new(num_thread,
                        state_.load() == running, idle_loop_count))
                {
                    // if we need to terminate, unregister the counter first
                    count.exit();
                    break;
                }

                if (0 == num_thread) {
                    // do background work in parcel layer
                    hpx::parcelset::flush_buffers();
                }
            }

            // Clean up all terminated threads for all thread queues once in a
            // while.
            if (busy_loop_count > HPX_BUSY_LOOP_COUNT_MAX) {
                // do background work in the scheduler
                busy_loop_count = 0;
                scheduler_.cleanup_terminated(true);
            }
        }

        // after tfunc loop broke, record total time elapsed
        tfunc_time = util::hardware::timestamp() - overall_timestamp;

#if HPX_DEBUG != 0
        // the last OS thread is allowed to exit only if no more PX threads exist
        BOOST_ASSERT(!scheduler_.get_thread_count(
            unknown, thread_priority_default, num_thread));
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy>
    bool threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        run(std::size_t num_threads)
    {
        LTM_(info) << "run: " << threads::hardware_concurrency() << " number of cores available";
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
                threads::mask_cref_type mask = get_pu_mask(topology_, thread_num);

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
        get_executed_threads(std::size_t num, bool reset)
    {
        boost::int64_t result = 0;
        if (num != std::size_t(-1)) {
            result = executed_threads_[num];
            if (reset)
                executed_threads_[num] = 0;
            return result;
        }

        result = std::accumulate(executed_threads_.begin(),
            executed_threads_.end(), 0LL);
        if (reset)
            std::fill(executed_threads_.begin(), executed_threads_.end(), 0LL);
        return result;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy>
    boost::int64_t threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        avg_idle_rate(bool reset)
    {
        double const exec_total =
            std::accumulate(exec_times.begin(), exec_times.end(), 0.);
        double const tfunc_total =
            std::accumulate(tfunc_times.begin(), tfunc_times.end(), 0.);

        if (reset) {
            std::fill(exec_times.begin(), exec_times.end(), 0);
            std::fill(tfunc_times.begin(), tfunc_times.end(), 0);
        }

        if (std::abs(tfunc_total) < 1e-16)   // avoid division by zero
            return 1000LL;

        double const percent = 1. - (exec_total / tfunc_total);
        return boost::int64_t(1000. * percent);    // 0.1 percent
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    boost::int64_t threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        avg_idle_rate(std::size_t num_thread, bool reset)
    {
        double const exec_time = static_cast<double>(exec_times[num_thread]);
        double const tfunc_time = static_cast<double>(tfunc_times[num_thread]);
        double const percent = (tfunc_time != 0.) ? 1. - (exec_time / tfunc_time) : 1.; //-V550

        if (reset) {
            exec_times[num_thread] = 0;
            tfunc_times[num_thread] = 0;
        }

        return boost::int64_t(1000. * percent);   // 0.1 percent
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy>
    template <typename C>
    void threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        start_periodic_maintenance(boost::mpl::true_)
    {
        scheduler_.periodic_maintenance(state_.load() == running);

        boost::posix_time::milliseconds expire(1000);
        // create timer firing in correspondence with given time
        boost::asio::deadline_timer t (timer_pool_.get_io_service(), expire);

        void (threadmanager_impl::*handler)(boost::mpl::true_)
            = &threadmanager_impl::periodic_maintenance_handler<SchedulingPolicy>;

        t.async_wait(boost::bind(handler, this, boost::mpl::true_()));
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    template <typename C>
    void threadmanager_impl<SchedulingPolicy, NotificationPolicy>::
        periodic_maintenance_handler(boost::mpl::true_)
    {
        scheduler_.periodic_maintenance(state_.load() == running);

        if(state_.load() == running)
        {
            boost::posix_time::milliseconds expire(1000);
            // create timer firing in correspondence with given time
            boost::asio::deadline_timer t (timer_pool_.get_io_service(), expire);

            void (threadmanager_impl::*handler)(boost::mpl::true_)
                = &threadmanager_impl::periodic_maintenance_handler<SchedulingPolicy>;

            t.async_wait(boost::bind(handler, this, boost::mpl::true_()));
        }
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
    hpx::threads::policies::local_queue_scheduler,
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

