//  Copyright (c) 2015 Patricia Grubel
//  Copyright (c) 2007-2015 Hartmut Kaiser
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
#include <hpx/runtime/threads/detail/set_thread_state.hpp>
#include <hpx/runtime/threads/executors/generic_thread_pool_executor.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/performance_counters/counter_creators.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/unlock_guard.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/block_profiler.hpp>
#include <hpx/util/itt_notify.hpp>
#include <hpx/util/hardware/timestamp.hpp>
#include <hpx/util/runtime_configuration.hpp>

#include <boost/make_shared.hpp>
#include <boost/bind.hpp>
#include <boost/cstdint.hpp>
#include <boost/format.hpp>
#include <boost/thread/locks.hpp>

#include <numeric>
#include <sstream>

#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
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
            "critical",
            "boost"
        };
    }

    char const* get_thread_priority_name(thread_priority priority)
    {
        if (priority < thread_priority_default || priority > thread_priority_boost)
            return "unknown";
        return strings::thread_priority_names[priority];
    }

    namespace strings
    {
        char const* const stack_size_names[] =
        {
            "small",
            "medium",
            "large",
            "huge",
            "stack-less"
        };
    }

    char const* get_stack_size_name(std::ptrdiff_t size)
    {
        if (size == thread_stacksize_unknown)
            return "unknown";

        util::runtime_configuration const& rtcfg = hpx::get_config();
        if (rtcfg.get_stack_size(thread_stacksize_small) == size)
            size = thread_stacksize_small;
        else if (rtcfg.get_stack_size(thread_stacksize_medium) == size)
            size = thread_stacksize_medium;
        else if (rtcfg.get_stack_size(thread_stacksize_large) == size)
            size = thread_stacksize_large;
        else if (rtcfg.get_stack_size(thread_stacksize_huge) == size)
            size = thread_stacksize_huge;
        else if (rtcfg.get_stack_size(thread_stacksize_nostack) == size)
            size = thread_stacksize_nostack;

        if (size < thread_stacksize_small || size > thread_stacksize_nostack)
            return "custom";

        return strings::stack_size_names[size-1];
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy>
    threadmanager_impl<SchedulingPolicy>::threadmanager_impl(
            util::io_service_pool& timer_pool,
            scheduling_policy_type& scheduler,
            notification_policy_type& notifier,
            std::size_t num_threads)
      : num_threads_(num_threads),
        timer_pool_(timer_pool),
        thread_logger_("threadmanager_impl::register_thread"),
        work_logger_("threadmanager_impl::register_work"),
        set_state_logger_("threadmanager_impl::set_state"),
        pool_(scheduler, notifier, "main_thread_scheduling_pool", true),
        notifier_(notifier)
    {}

    template <typename SchedulingPolicy>
    threadmanager_impl<SchedulingPolicy>::~threadmanager_impl()
    {
    }

    template <typename SchedulingPolicy>
    std::size_t threadmanager_impl<SchedulingPolicy>::init(
        policies::init_affinity_data const& data)
    {
        return pool_.init(num_threads_, data);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy>
    boost::int64_t threadmanager_impl<SchedulingPolicy>::
        get_thread_count(thread_state_enum state, thread_priority priority,
            std::size_t num_thread, bool reset) const
    {
        boost::lock_guard<mutex_type> lk(mtx_);
        return pool_.get_thread_count(state, priority, num_thread, reset);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Abort all threads which are in suspended state. This will set
    // the state of all suspended threads to \a pending while
    // supplying the wait_abort extended state flag
    template <typename SchedulingPolicy>
    void threadmanager_impl<SchedulingPolicy>::
        abort_all_suspended_threads()
    {
        boost::lock_guard<mutex_type> lk(mtx_);
        pool_.abort_all_suspended_threads();
    }

    ///////////////////////////////////////////////////////////////////////////
    // Clean up terminated threads. This deletes all threads which
    // have been terminated but which are still held in the queue
    // of terminated threads. Some schedulers might not do anything
    // here.
    template <typename SchedulingPolicy>
    bool threadmanager_impl<SchedulingPolicy>::
        cleanup_terminated(bool delete_all)
    {
        boost::lock_guard<mutex_type> lk(mtx_);
        return pool_.cleanup_terminated(delete_all);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy>
    void threadmanager_impl<SchedulingPolicy>::
        register_thread(thread_init_data& data, thread_id_type& id,
            thread_state_enum initial_state, bool run_now, error_code& ec)
    {
        util::block_profiler_wrapper<register_thread_tag> bp(thread_logger_);
        pool_.create_thread(data, id, initial_state, run_now, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy>
    void threadmanager_impl<SchedulingPolicy>::register_work(
        thread_init_data& data, thread_state_enum initial_state, error_code& ec)
    {
        util::block_profiler_wrapper<register_work_tag> bp(work_logger_);
        pool_.create_work(data, initial_state, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    /// The set_state function is part of the thread related API and allows
    /// to change the state of one of the threads managed by this threadmanager_impl
    template <typename SchedulingPolicy>
    thread_state threadmanager_impl<SchedulingPolicy>::
        set_state(thread_id_type const& id, thread_state_enum new_state,
            thread_state_ex_enum new_state_ex, thread_priority priority,
            error_code& ec)
    {
        return detail::set_thread_state(id, new_state, //-V107
            new_state_ex, priority, get_worker_thread_num(), ec);
    }

    /// The get_state function is part of the thread related API. It
    /// queries the state of one of the threads known to the threadmanager_impl
    template <typename SchedulingPolicy>
    thread_state threadmanager_impl<SchedulingPolicy>::
        get_state(thread_id_type const& thrd) const
    {
        return thrd ? thrd->get_state() : thread_state(terminated);
    }

    /// The get_phase function is part of the thread related API. It
    /// queries the phase of one of the threads known to the threadmanager_impl
    template <typename SchedulingPolicy>
    std::size_t threadmanager_impl<SchedulingPolicy>::
        get_phase(thread_id_type const& thrd) const
    {
        return thrd ? thrd->get_thread_phase() : std::size_t(~0);
    }

    /// The get_priority function is part of the thread related API. It
    /// queries the priority of one of the threads known to the threadmanager_impl
    template <typename SchedulingPolicy>
    thread_priority threadmanager_impl<SchedulingPolicy>::
        get_priority(thread_id_type const& thrd) const
    {
        return thrd ? thrd->get_priority() : thread_priority_unknown;
    }

    template <typename SchedulingPolicy>
    std::ptrdiff_t threadmanager_impl<SchedulingPolicy>::
        get_stack_size(thread_id_type const& thrd) const
    {
        return thrd ? thrd->get_stack_size() :
            static_cast<std::ptrdiff_t>(thread_stacksize_unknown);
    }

    /// The get_description function is part of the thread related API and
    /// allows to query the description of one of the threads known to the
    /// threadmanager_impl
    template <typename SchedulingPolicy>
    char const* threadmanager_impl<SchedulingPolicy>::
        get_description(thread_id_type const& thrd) const
    {
        return thrd ? thrd->get_description() : "<unknown>";
    }

    template <typename SchedulingPolicy>
    char const* threadmanager_impl<SchedulingPolicy>::
        set_description(thread_id_type const& thrd, char const* desc)
    {
        if (HPX_UNLIKELY(!thrd)) {
            HPX_THROW_EXCEPTION(null_thread_id,
                "threadmanager_impl::set_description",
                "NULL thread id encountered");
            return NULL;
        }

        if (thrd)
            return thrd->set_description(desc);
        return NULL;
    }

    template <typename SchedulingPolicy>
    char const* threadmanager_impl<SchedulingPolicy>::
        get_lco_description(thread_id_type const& thrd) const
    {
        if (HPX_UNLIKELY(!thrd)) {
            HPX_THROW_EXCEPTION(null_thread_id,
                "threadmanager_impl::get_lco_description",
                "NULL thread id encountered");
            return NULL;
        }

        return thrd ? thrd->get_lco_description() : "<unknown>";
    }

    template <typename SchedulingPolicy>
    char const* threadmanager_impl<SchedulingPolicy>::
        set_lco_description(thread_id_type const& thrd, char const* desc)
    {
        if (HPX_UNLIKELY(!thrd)) {
            HPX_THROW_EXCEPTION(null_thread_id,
                "threadmanager_impl::set_lco_description",
                "NULL thread id encountered");
            return NULL;
        }

        if (thrd)
            return thrd->set_lco_description(desc);
        return NULL;
    }

    ///////////////////////////////////////////////////////////////////////////
#ifdef HPX_HAVE_THREAD_FULLBACKTRACE_ON_SUSPENSION
    template <typename SchedulingPolicy>
    char const* threadmanager_impl<SchedulingPolicy>::
        get_backtrace(thread_id_type const& thrd) const
#else
    template <typename SchedulingPolicy>
    util::backtrace const* threadmanager_impl<SchedulingPolicy>::
        get_backtrace(thread_id_type const& thrd) const
#endif
    {
        if (HPX_UNLIKELY(!thrd)) {
            HPX_THROW_EXCEPTION(null_thread_id,
                "threadmanager_impl::get_backtrace",
                "NULL thread id encountered");
            return NULL;
        }

        return thrd ? thrd->get_backtrace() : 0;
    }

#ifdef HPX_HAVE_THREAD_FULLBACKTRACE_ON_SUSPENSION
    template <typename SchedulingPolicy>
    char const* threadmanager_impl<SchedulingPolicy>::
        set_backtrace(thread_id_type const& thrd, char const* bt)
#else
    template <typename SchedulingPolicy>
    util::backtrace const* threadmanager_impl<SchedulingPolicy>::
        set_backtrace(thread_id_type const& thrd, util::backtrace const* bt)
#endif
    {
        if (HPX_UNLIKELY(!thrd)) {
            HPX_THROW_EXCEPTION(null_thread_id,
                "threadmanager_impl::set_backtrace",
                "NULL thread id encountered");
            return NULL;
        }

        return thrd ? thrd->set_backtrace(bt) : 0;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy>
    bool threadmanager_impl<SchedulingPolicy>::
        get_interruption_enabled(thread_id_type const& thrd, error_code& ec)
    {
        if (HPX_UNLIKELY(!thrd)) {
            HPX_THROW_EXCEPTION(null_thread_id,
                "threadmanager_impl::get_interruption_enabled",
                "NULL thread id encountered");
            return false;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return thrd ? thrd->interruption_enabled() : false;
    }

    template <typename SchedulingPolicy>
    bool threadmanager_impl<SchedulingPolicy>::
        set_interruption_enabled(thread_id_type const& thrd, bool enable, error_code& ec)
    {
        if (HPX_UNLIKELY(!thrd)) {
            HPX_THROW_EXCEPTION(null_thread_id,
                "threadmanager_impl::set_interruption_enabled",
                "NULL thread id encountered");
        }

        if (&ec != &throws)
            ec = make_success_code();

        if (thrd)
            return thrd->set_interruption_enabled(enable);
        return false;
    }

    template <typename SchedulingPolicy>
    bool threadmanager_impl<SchedulingPolicy>::
        get_interruption_requested(thread_id_type const& thrd, error_code& ec)
    {
        if (HPX_UNLIKELY(!thrd)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "threadmanager_impl::get_interruption_requested",
                "NULL thread id encountered");
            return false;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return thrd ? thrd->interruption_requested() : false;
    }

    template <typename SchedulingPolicy>
    void threadmanager_impl<SchedulingPolicy>::
        interrupt(thread_id_type const& thrd, bool flag, error_code& ec)
    {
        if (HPX_UNLIKELY(!thrd)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "threadmanager_impl::interrupt",
                "NULL thread id encountered");
            return;
        }

        if (&ec != &throws)
            ec = make_success_code();

        if (thrd) {
            thrd->interrupt(flag);      // notify thread

            // set thread state to pending, if the thread is currently active,
            // this will be rescheduled until it calls an interruption point
            set_thread_state(thrd, pending, wait_abort,
                thread_priority_normal, ec);
        }
    }

    template <typename SchedulingPolicy>
    void threadmanager_impl<SchedulingPolicy>::
        interruption_point(thread_id_type const& thrd, error_code& ec)
    {
        if (HPX_UNLIKELY(!thrd)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "threadmanager_impl::interruption_point",
                "NULL thread id encountered");
            return;
        }

        if (&ec != &throws)
            ec = make_success_code();

        if (thrd)
            thrd->interruption_point();      // notify thread
    }

#ifdef HPX_HAVE_THREAD_LOCAL_STORAGE
    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy>
    std::size_t threadmanager_impl<SchedulingPolicy>::
        get_thread_data(thread_id_type const& thrd, error_code& ec) const
    {
        if (HPX_UNLIKELY(!thrd)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "threadmanager_impl::get_thread_data",
                "NULL thread id encountered");
            return 0;
        }

        return thrd ? thrd->get_thread_data() : 0;
    }

    template <typename SchedulingPolicy>
    std::size_t threadmanager_impl<SchedulingPolicy>::
        set_thread_data(thread_id_type const& thrd, std::size_t data,
            error_code& ec)
    {
        if (HPX_UNLIKELY(!thrd)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "threadmanager_impl::set_thread_data",
                "NULL thread id encountered");
            return 0;
        }

        return thrd ? thrd->set_thread_data(data) : 0;
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy>
    void threadmanager_impl<SchedulingPolicy>::
    run_thread_exit_callbacks(thread_id_type const& thrd, error_code& ec)
    {
        if (HPX_UNLIKELY(!thrd)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "threadmanager_impl::run_thread_exit_callbacks",
                "NULL thread id encountered");
            return;
        }

        if (&ec != &throws)
            ec = make_success_code();

        if (thrd)
            thrd->run_thread_exit_callbacks();
    }

    template <typename SchedulingPolicy>
    bool threadmanager_impl<SchedulingPolicy>::
    add_thread_exit_callback(thread_id_type const& thrd,
        util::function_nonser<void()> const& f, error_code& ec)
    {
        if (HPX_UNLIKELY(!thrd)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "threadmanager_impl::add_thread_exit_callback",
                "NULL thread id encountered");
            return false;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return (0 != thrd) ? thrd->add_thread_exit_callback(f) : false;
    }

    template <typename SchedulingPolicy>
    void threadmanager_impl<SchedulingPolicy>::
        free_thread_exit_callbacks(thread_id_type const& thrd, error_code& ec)
    {
        if (HPX_UNLIKELY(!thrd)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "threadmanager_impl::free_thread_exit_callbacks",
                "NULL thread id encountered");
            return;
        }

        if (&ec != &throws)
            ec = make_success_code();

        if (0 != thrd)
            thrd->free_thread_exit_callbacks();
    }

    // Return the executor associated with the given thread
    template <typename SchedulingPolicy>
    executors::generic_thread_pool_executor threadmanager_impl<SchedulingPolicy>::
        get_executor(thread_id_type const& thrd, error_code& ec) const
    {
        if (HPX_UNLIKELY(!thrd)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "threadmanager_impl::get_executor",
                "NULL thread id encountered");
            return executors::generic_thread_pool_executor(0);
        }

        if (&ec != &throws)
            ec = make_success_code();

        return executors::generic_thread_pool_executor(thrd->get_scheduler_base());
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Set a timer to set the state of the given \a thread to the given
    /// new value after it expired (at the given time)
    template <typename SchedulingPolicy>
    thread_id_type threadmanager_impl<SchedulingPolicy>::
        set_state(util::steady_time_point const& abs_time,
            thread_id_type const& id, thread_state_enum newstate,
            thread_state_ex_enum newstate_ex, thread_priority priority,
            error_code& ec)
    {
        return pool_.set_state(abs_time, id, newstate, newstate_ex, priority, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    // counter creator and discovery functions

    // queue length(s) counter creation function
    template <typename SchedulingPolicy>
    naming::gid_type threadmanager_impl<SchedulingPolicy>::
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

        typedef detail::thread_pool<scheduling_policy_type> spt;

        using util::placeholders::_1;
        if (paths.instancename_ == "total" && paths.instanceindex_ == -1)
        {
            // overall counter
            using performance_counters::detail::create_raw_counter;
            util::function_nonser<boost::int64_t()> f =
                util::bind(&spt::get_queue_length, &pool_, -1);
            return create_raw_counter(info, std::move(f), ec);
        }
        else if (paths.instancename_ == "worker-thread" &&
            paths.instanceindex_ >= 0 &&
            std::size_t(paths.instanceindex_) < pool_.get_os_thread_count())
        {
            // specific counter
            using performance_counters::detail::create_raw_counter;
            util::function_nonser<boost::int64_t()> f =
                util::bind(&spt::get_queue_length, &pool_, //-V107
                    static_cast<std::size_t>(paths.instanceindex_));
            return create_raw_counter(info, std::move(f), ec);
        }

        HPX_THROWS_IF(ec, bad_parameter, "queue_length_counter_creator",
            "invalid counter instance name: " + paths.instancename_);
        return naming::invalid_gid;
    }

#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
    // average pending thread wait time
    template <typename SchedulingPolicy>
    naming::gid_type threadmanager_impl<SchedulingPolicy>::
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

        using util::placeholders::_1;
        if (paths.instancename_ == "total" && paths.instanceindex_ == -1)
        {
            policies::maintain_queue_wait_times = true;

            // overall counter
            using performance_counters::detail::create_raw_counter;
            util::function_nonser<boost::int64_t()> f =
                util::bind(&spt::get_average_thread_wait_time, &pool_, -1);
            return create_raw_counter(info, std::move(f), ec);
        }
        else if (paths.instancename_ == "worker-thread" &&
            paths.instanceindex_ >= 0 &&
            std::size_t(paths.instanceindex_) < threads_.size())
        {
            policies::maintain_queue_wait_times = true;

            // specific counter
            using performance_counters::detail::create_raw_counter;
            util::function_nonser<boost::int64_t()> f =
                util::bind(&spt::get_average_thread_wait_time, &pool_,
                    static_cast<std::size_t>(paths.instanceindex_));
            return create_raw_counter(info, std::move(f), ec);
        }

        HPX_THROWS_IF(ec, bad_parameter, "thread_wait_time_counter_creator",
            "invalid counter instance name: " + paths.instancename_);
        return naming::invalid_gid;
    }

    // average pending task wait time
    template <typename SchedulingPolicy>
    naming::gid_type threadmanager_impl<SchedulingPolicy>::
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

        using util::placeholders::_1;
        if (paths.instancename_ == "total" && paths.instanceindex_ == -1)
        {
            policies::maintain_queue_wait_times = true;

            // overall counter
            using performance_counters::detail::create_raw_counter;
            util::function_nonser<boost::int64_t()> f =
                util::bind(&spt::get_average_task_wait_time, &pool_, -1);
            return create_raw_counter(info, std::move(f), ec);
        }
        else if (paths.instancename_ == "worker-thread" &&
            paths.instanceindex_ >= 0 &&
            std::size_t(paths.instanceindex_) < threads_.size())
        {
            policies::maintain_queue_wait_times = true;

            // specific counter
            using performance_counters::detail::create_raw_counter;
            util::function_nonser<boost::int64_t()> f =
                util::bind(&spt::get_average_task_wait_time, &pool_,
                    static_cast<std::size_t>(paths.instanceindex_));
            return create_raw_counter(info, std::move(f), ec);
        }

        HPX_THROWS_IF(ec, bad_parameter, "task_wait_time_counter_creator",
            "invalid counter instance name: " + paths.instancename_);
        return naming::invalid_gid;
    }
#endif

    bool locality_allocator_counter_discoverer(
        performance_counters::counter_info const& info,
        performance_counters::discover_counter_func const& f,
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

            if (mode == performance_counters::discover_counters_full) {
                for (std::size_t t = 0; t != HPX_COROUTINE_NUM_ALL_HEAPS; ++t)
                {
                    p.instancename_ = "allocator";
                    p.instanceindex_ = static_cast<boost::int32_t>(t);
                    status = get_counter_name(p, i.fullname_, ec);
                    if (!status_is_valid(status) || !f(i, ec) || ec)
                        return false;
                }
            }
            else {
                status = get_counter_name(p, i.fullname_, ec);
                if (!status_is_valid(status) || !f(i, ec) || ec)
                    return false;
            }
        }
        else if (p.instancename_ == "total" && p.instanceindex_ == -1) {
            // overall counter
            status = get_counter_name(p, i.fullname_, ec);
            if (!status_is_valid(status) || !f(i, ec) || ec)
                return false;
        }
        else if (p.instancename_ == "allocator#*") {
            for (std::size_t t = 0; t != HPX_COROUTINE_NUM_ALL_HEAPS; ++t)
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

#ifdef HPX_HAVE_THREAD_IDLE_RATES
    ///////////////////////////////////////////////////////////////////////////
    // idle rate counter creation function
    template <typename SchedulingPolicy>
    naming::gid_type threadmanager_impl<SchedulingPolicy>::
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

        using util::placeholders::_1;
        if (paths.instancename_ == "total" && paths.instanceindex_ == -1)
        {
            // overall counter
            using performance_counters::detail::create_raw_counter;
            boost::int64_t (threadmanager_impl::*avg_idle_rate_ptr)(
                bool
            ) = &ti::avg_idle_rate;
            util::function_nonser<boost::int64_t(bool)> f =
                 util::bind(avg_idle_rate_ptr, this, _1);
            return create_raw_counter(info, std::move(f), ec);
        }
        else if (paths.instancename_ == "worker-thread" &&
            paths.instanceindex_ >= 0 &&
            std::size_t(paths.instanceindex_) < pool_.get_os_thread_count())
        {
            // specific counter
            using performance_counters::detail::create_raw_counter;
            boost::int64_t (threadmanager_impl::*avg_idle_rate_ptr)(
                std::size_t, bool
            ) = &ti::avg_idle_rate;
            using performance_counters::detail::create_raw_counter;
            util::function_nonser<boost::int64_t(bool)> f =
                util::bind(avg_idle_rate_ptr, this,
                    static_cast<std::size_t>(paths.instanceindex_), _1);
            return create_raw_counter(info, std::move(f), ec);
        }

        HPX_THROWS_IF(ec, bad_parameter, "idle_rate_counter_creator",
            "invalid counter instance name: " + paths.instancename_);
        return naming::invalid_gid;
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    naming::gid_type
    counter_creator(performance_counters::counter_info const& info,
        performance_counters::counter_path_elements const& paths,
        util::function_nonser<boost::int64_t(bool)> const& total_creator,
        util::function_nonser<boost::int64_t(bool)> const& individual_creator,
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
    template <typename SchedulingPolicy>
    naming::gid_type threadmanager_impl<SchedulingPolicy>::
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
            util::function_nonser<boost::int64_t(bool)> total_func;
            util::function_nonser<boost::int64_t(bool)> individual_func;
            char const* const individual_name;
            std::size_t individual_count;
        };

        typedef detail::thread_pool<scheduling_policy_type> spt;
        typedef threadmanager_impl ti;

        using util::placeholders::_1;

        std::size_t shepherd_count = pool_.get_os_thread_count();
        creator_data data[] =
        {
#if defined(HPX_HAVE_THREAD_IDLE_RATES) && \
    defined(HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES)
            // /threads{locality#%d/total}/creation-idle-rate
            // /threads{locality#%d/worker-thread%d}/creation-idle-rate
            { "creation-idle-rate",
              util::bind(&ti::avg_creation_idle_rate, this, _1),
              util::function_nonser<boost::uint64_t(bool)>(),
              "", 0
            },
            // /threads{locality#%d/total}/cleanup-idle-rate
            // /threads{locality#%d/worker-thread%d}/cleanup-idle-rate
            { "cleanup-idle-rate",
              util::bind(&ti::avg_cleanup_idle_rate, this, _1),
              util::function_nonser<boost::uint64_t(bool)>(),
              "", 0
            },
#endif
#ifdef HPX_HAVE_THREAD_CUMULATIVE_COUNTS
            // /threads{locality#%d/total}/count/cumulative
            // /threads{locality#%d/worker-thread%d}/count/cumulative
            { "count/cumulative",
              util::bind(&ti::get_executed_threads, this, -1, _1),
              util::bind(&ti::get_executed_threads, this,
                  static_cast<std::size_t>(paths.instanceindex_), _1),
              "worker-thread", shepherd_count
            },
            // /threads{locality#%d/total}/count/cumulative-phases
            // /threads{locality#%d/worker-thread%d}/count/cumulative-phases
            { "count/cumulative-phases",
              util::bind(&ti::get_executed_thread_phases, this, -1, _1),
              util::bind(&ti::get_executed_thread_phases, this,
                  static_cast<std::size_t>(paths.instanceindex_), _1),
              "worker-thread", shepherd_count
            },
#ifdef HPX_HAVE_THREAD_IDLE_RATES
            // /threads{locality#%d/total}/time/average
            // /threads{locality#%d/worker-thread%d}/time/average
            { "time/average",
              util::bind(&ti::get_thread_duration, this, -1, _1),
              util::bind(&ti::get_thread_duration, this,
                  static_cast<std::size_t>(paths.instanceindex_), _1),
              "worker-thread", shepherd_count
            },
            // /threads{locality#%d/total}/time/average-phase
            // /threads{locality#%d/worker-thread%d}/time/average-phase
            { "time/average-phase",
              util::bind(&ti::get_thread_phase_duration, this, -1, _1),
              util::bind(&ti::get_thread_phase_duration, this,
                  static_cast<std::size_t>(paths.instanceindex_), _1),
              "worker-thread", shepherd_count
            },
            // /threads{locality#%d/total}/time/average-overhead
            // /threads{locality#%d/worker-thread%d}/time/average-overhead
            { "time/average-overhead",
              util::bind(&ti::get_thread_overhead, this, -1, _1),
              util::bind(&ti::get_thread_overhead, this,
                  static_cast<std::size_t>(paths.instanceindex_), _1),
              "worker-thread", shepherd_count
            },
            // /threads{locality#%d/total}/time/average-phase-overhead
            // /threads{locality#%d/worker-thread%d}/time/average-phase-overhead
            { "time/average-phase-overhead",
              util::bind(&ti::get_thread_phase_overhead, this, -1, _1),
              util::bind(&ti::get_thread_phase_overhead, this,
                  static_cast<std::size_t>(paths.instanceindex_), _1),
              "worker-thread", shepherd_count
            },
            // /threads{locality#%d/total}/time/cumulative
            // /threads{locality#%d/worker-thread%d}/time/cumulative
            { "time/cumulative",
              util::bind(&ti::get_cumulative_thread_duration, this, -1, _1),
              util::bind(&ti::get_cumulative_thread_duration, this,
                  static_cast<std::size_t>(paths.instanceindex_), _1),
              "worker-thread", shepherd_count
            },
            // /threads{locality#%d/total}/time/cumulative-overhead
            // /threads{locality#%d/worker-thread%d}/time/cumulative-overhead
            { "time/cumulative-overhead",
              util::bind(&ti::get_cumulative_thread_overhead, this, -1, _1),
              util::bind(&ti::get_cumulative_thread_overhead, this,
                  static_cast<std::size_t>(paths.instanceindex_), _1),
              "worker-thread", shepherd_count
            },
#endif
#endif
            // /threads{locality#%d/total}/count/instantaneous/all
            // /threads{locality#%d/worker-thread%d}/count/instantaneous/all
            { "count/instantaneous/all",
              util::bind(&ti::get_thread_count, this, unknown,
                  thread_priority_default, std::size_t(-1), _1),
              util::bind(&ti::get_thread_count, this, unknown,
                  thread_priority_default,
                  static_cast<std::size_t>(paths.instanceindex_), _1),
              "worker-thread", shepherd_count
            },
            // /threads{locality#%d/total}/count/instantaneous/active
            // /threads{locality#%d/worker-thread%d}/count/instantaneous/active
            { "count/instantaneous/active",
              util::bind(&ti::get_thread_count, this, active,
                  thread_priority_default, std::size_t(-1), _1),
              util::bind(&ti::get_thread_count, this, active,
                  thread_priority_default,
                  static_cast<std::size_t>(paths.instanceindex_), _1),
              "worker-thread", shepherd_count
            },
            // /threads{locality#%d/total}/count/instantaneous/pending
            // /threads{locality#%d/worker-thread%d}/count/instantaneous/pending
            { "count/instantaneous/pending",
              util::bind(&ti::get_thread_count, this, pending,
                  thread_priority_default, std::size_t(-1), _1),
              util::bind(&ti::get_thread_count, this, pending,
                  thread_priority_default,
                  static_cast<std::size_t>(paths.instanceindex_), _1),
              "worker-thread", shepherd_count
            },
            // /threads{locality#%d/total}/count/instantaneous/suspended
            // /threads{locality#%d/worker-thread%d}/count/instantaneous/suspended
            { "count/instantaneous/suspended",
              util::bind(&ti::get_thread_count, this, suspended,
                  thread_priority_default, std::size_t(-1), _1),
              util::bind(&ti::get_thread_count, this, suspended,
                  thread_priority_default,
                  static_cast<std::size_t>(paths.instanceindex_), _1),
              "worker-thread", shepherd_count
            },
            // /threads(locality#%d/total}/count/instantaneous/terminated
            // /threads(locality#%d/worker-thread%d}/count/instantaneous/terminated
            { "count/instantaneous/terminated",
              util::bind(&ti::get_thread_count, this, terminated,
                  thread_priority_default, std::size_t(-1), _1),
              util::bind(&ti::get_thread_count, this, terminated,
                  thread_priority_default,
                  static_cast<std::size_t>(paths.instanceindex_), _1),
              "worker-thread", shepherd_count
            },
            // /threads{locality#%d/total}/count/instantaneous/staged
            // /threads{locality#%d/worker-thread%d}/count/instantaneous/staged
            { "count/instantaneous/staged",
              util::bind(&ti::get_thread_count, this, staged,
                  thread_priority_default, std::size_t(-1), _1),
              util::bind(&ti::get_thread_count, this, staged,
                  thread_priority_default,
                  static_cast<std::size_t>(paths.instanceindex_), _1),
              "worker-thread", shepherd_count
            },
            // /threads{locality#%d/total}/count/stack-recycles
            { "count/stack-recycles",
              util::bind(&coroutine_type::impl_type::get_stack_recycle_count, _1),
              util::function_nonser<boost::uint64_t(bool)>(), "", 0
            },
#if !defined(BOOST_WINDOWS) && !defined(HPX_HAVE_GENERIC_CONTEXT_COROUTINES)
            // /threads{locality#%d/total}/count/stack-unbinds
            { "count/stack-unbinds",
              util::bind(&coroutine_type::impl_type::get_stack_unbind_count, _1),
              util::function_nonser<boost::uint64_t(bool)>(), "", 0
            },
#endif
            // /threads{locality#%d/total}/count/objects
            // /threads{locality#%d/allocator%d}/count/objects
            { "count/objects",
              &coroutine_type::impl_type::get_allocation_count_all,
              util::bind(&coroutine_type::impl_type::get_allocation_count,
                  static_cast<std::size_t>(paths.instanceindex_), _1),
              "allocator", HPX_COROUTINE_NUM_ALL_HEAPS
            },
#ifdef HPX_HAVE_THREAD_STEALING_COUNTS
            // /threads{locality#%d/total}/count/pending-misses
            // /threads{locality#%d/worker-thread%d}/count/pending-misses
            { "count/pending-misses",
              util::bind(&spt::get_num_pending_misses, &pool_,
                  std::size_t(-1), _1),
              util::bind(&spt::get_num_pending_misses, &pool_,
                  static_cast<std::size_t>(paths.instanceindex_), _1),
              "worker-thread", shepherd_count
            },
            // /threads{locality#%d/total}/count/pending-accesses
            // /threads{locality#%d/worker-thread%d}/count/pending-accesses
            { "count/pending-accesses",
              util::bind(&spt::get_num_pending_accesses, &pool_,
                  std::size_t(-1), _1),
              util::bind(&spt::get_num_pending_accesses, &pool_,
                  static_cast<std::size_t>(paths.instanceindex_), _1),
              "worker-thread", shepherd_count
            },
            // /threads{locality#%d/total}/count/stolen-from-pending
            // /threads{locality#%d/worker-thread%d}/count/stolen-from-pending
            { "count/stolen-from-pending",
              util::bind(&spt::get_num_stolen_from_pending, &pool_,
                  std::size_t(-1), _1),
              util::bind(&spt::get_num_stolen_from_pending, &pool_,
                  static_cast<std::size_t>(paths.instanceindex_), _1),
              "worker-thread", shepherd_count
            },
            // /threads{locality#%d/total}/count/stolen-from-staged
            // /threads{locality#%d/worker-thread%d}/count/stolen-from-staged
            { "count/stolen-from-staged",
              util::bind(&spt::get_num_stolen_from_staged, &pool_,
                  std::size_t(-1), _1),
              util::bind(&spt::get_num_stolen_from_staged, &pool_,
                  static_cast<std::size_t>(paths.instanceindex_), _1),
              "worker-thread", shepherd_count
            },
            // /threads{locality#%d/total}/count/stolen-to-pending
            // /threads{locality#%d/worker-thread%d}/count/stolen-to-pending
            { "count/stolen-to-pending",
              util::bind(&spt::get_num_stolen_to_pending, &pool_,
                  std::size_t(-1), _1),
              util::bind(&spt::get_num_stolen_to_pending, &pool_,
                  static_cast<std::size_t>(paths.instanceindex_), _1),
              "worker-thread", shepherd_count
            },
            // /threads{locality#%d/total}/count/stolen-to-staged
            // /threads{locality#%d/worker-thread%d}/count/stolen-to-staged
            { "count/stolen-to-staged",
              util::bind(&spt::get_num_stolen_to_staged, &pool_,
                  std::size_t(-1), _1),
              util::bind(&spt::get_num_stolen_to_staged, &pool_,
                  static_cast<std::size_t>(paths.instanceindex_), _1),
              "worker-thread", shepherd_count
            }
#endif
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
    template <typename SchedulingPolicy>
    void threadmanager_impl<SchedulingPolicy>::
        register_counter_types()
    {
        typedef threadmanager_impl ti;
        performance_counters::create_counter_func counts_creator(
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
#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
            // average thread wait time for queue(s)
            { "/threads/wait-time/pending", performance_counters::counter_raw,
              "returns the average wait time of \
                 pending threads for the referenced queue",
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
#ifdef HPX_HAVE_THREAD_IDLE_RATES
            // idle rate
            { "/threads/idle-rate", performance_counters::counter_raw,
              "returns the idle rate for the referenced object",
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&ti::idle_rate_counter_creator, this, _1, _2),
              &performance_counters::locality_thread_counter_discoverer,
              "0.01%"
            },
#ifdef HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES
            { "/threads/creation-idle-rate", performance_counters::counter_raw,
              "returns the % of idle-rate spent creating HPX-threads for the "
              "referenced object", HPX_PERFORMANCE_COUNTER_V1, counts_creator,
              &performance_counters::locality_thread_counter_discoverer,
              "0.01%"
            },
            { "/threads/cleanup-idle-rate", performance_counters::counter_raw,
              "returns the % of time spent cleaning up terminated HPX-threads "
              "for the referenced object", HPX_PERFORMANCE_COUNTER_V1, counts_creator,
              &performance_counters::locality_thread_counter_discoverer,
              "0.01%"
            },
#endif
#endif
#ifdef HPX_HAVE_THREAD_CUMULATIVE_COUNTS
            // thread counts
            { "/threads/count/cumulative", performance_counters::counter_raw,
              "returns the overall number of executed (retired) HPX-threads for "
              "the referenced locality", HPX_PERFORMANCE_COUNTER_V1, counts_creator,
              &performance_counters::locality_thread_counter_discoverer,
              ""
            },
            { "/threads/count/cumulative-phases", performance_counters::counter_raw,
              "returns the overall number of HPX-thread phases executed for "
              "the referenced locality", HPX_PERFORMANCE_COUNTER_V1, counts_creator,
              &performance_counters::locality_thread_counter_discoverer,
              ""
            },
#ifdef HPX_HAVE_THREAD_IDLE_RATES
            { "/threads/time/average", performance_counters::counter_raw,
              "returns the average time spent executing one HPX-thread",
              HPX_PERFORMANCE_COUNTER_V1, counts_creator,
              &performance_counters::locality_thread_counter_discoverer,
              "ns"
            },
            { "/threads/time/average-phase", performance_counters::counter_raw,
              "returns the average time spent executing one HPX-thread phase",
              HPX_PERFORMANCE_COUNTER_V1, counts_creator,
              &performance_counters::locality_thread_counter_discoverer,
              "ns"
            },
            { "/threads/time/average-overhead", performance_counters::counter_raw,
              "returns average overhead time executing one HPX-thread",
              HPX_PERFORMANCE_COUNTER_V1, counts_creator,
              &performance_counters::locality_thread_counter_discoverer,
              "ns"
            },
            { "/threads/time/average-phase-overhead", performance_counters::counter_raw,
              "returns average overhead time executing one HPX-thread phase",
              HPX_PERFORMANCE_COUNTER_V1, counts_creator,
              &performance_counters::locality_thread_counter_discoverer,
              "ns"
            },
            { "/threads/time/cumulative", performance_counters::counter_raw,
              "returns the cumulative time spent executing HPX-threads",
              HPX_PERFORMANCE_COUNTER_V1, counts_creator,
              &performance_counters::locality_thread_counter_discoverer,
              "ns"
            },
            { "/threads/time/cumulative-overhead", performance_counters::counter_raw,
              "returns the cumulative overhead time incurred by executing HPX threads",
              HPX_PERFORMANCE_COUNTER_V1, counts_creator,
              &performance_counters::locality_thread_counter_discoverer,
              "ns"
            },
#endif
#endif
            { "/threads/count/instantaneous/all", performance_counters::counter_raw,
              "returns the overall current number of HPX-threads instantiated at the "
              "referenced locality", HPX_PERFORMANCE_COUNTER_V1, counts_creator,
              &performance_counters::locality_thread_counter_discoverer,
              ""
            },
            { "/threads/count/instantaneous/active", performance_counters::counter_raw,
              "returns the current number of active \
                 HPX-threads at the referenced locality",
              HPX_PERFORMANCE_COUNTER_V1, counts_creator,
              &performance_counters::locality_thread_counter_discoverer,
              ""
            },
            { "/threads/count/instantaneous/pending", performance_counters::counter_raw,
              "returns the current number of pending \
                 HPX-threads at the referenced locality",
              HPX_PERFORMANCE_COUNTER_V1, counts_creator,
              &performance_counters::locality_thread_counter_discoverer,
              ""
            },
            { "/threads/count/instantaneous/suspended",
                  performance_counters::counter_raw,
              "returns the current number of suspended \
                 HPX-threads at the referenced locality",
              HPX_PERFORMANCE_COUNTER_V1, counts_creator,
              &performance_counters::locality_thread_counter_discoverer,
              ""
            },
            { "/threads/count/instantaneous/terminated",
                performance_counters::counter_raw,
              "returns the current number of terminated \
                 HPX-threads at the referenced locality",
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
#if !defined(BOOST_WINDOWS) && !defined(HPX_HAVE_GENERIC_CONTEXT_COROUTINES)
            { "/threads/count/stack-unbinds", performance_counters::counter_raw,
              "returns the total number of HPX-thread unbind (madvise) operations "
              "performed for the referenced locality", HPX_PERFORMANCE_COUNTER_V1,
              counts_creator, &performance_counters::locality_counter_discoverer,
              ""
            },
#endif
            { "/threads/count/objects", performance_counters::counter_raw,
              "returns the overall number of created HPX-thread objects for "
              "the referenced locality", HPX_PERFORMANCE_COUNTER_V1,
              counts_creator,
              &locality_allocator_counter_discoverer,
              ""
            },
#ifdef HPX_HAVE_THREAD_STEALING_COUNTS
            { "/threads/count/pending-misses", performance_counters::counter_raw,
              "returns the number of times that the referenced worker-thread "
              "on the referenced locality failed to find pending HPX-threads "
              "in its associated queue",
              HPX_PERFORMANCE_COUNTER_V1, counts_creator,
              &performance_counters::locality_thread_counter_discoverer,
              ""
            },
            { "/threads/count/pending-accesses", performance_counters::counter_raw,
              "returns the number of times that the referenced worker-thread "
              "on the referenced locality looked for pending HPX-threads "
              "in its associated queue",
              HPX_PERFORMANCE_COUNTER_V1, counts_creator,
              &performance_counters::locality_thread_counter_discoverer,
              ""
            },
            { "/threads/count/stolen-from-pending", performance_counters::counter_raw,
              "returns the overall number of pending HPX-threads stolen by neighboring"
              "schedulers from this scheduler for the referenced locality",
              HPX_PERFORMANCE_COUNTER_V1, counts_creator,
              &performance_counters::locality_thread_counter_discoverer,
              ""
            },
            { "/threads/count/stolen-from-staged", performance_counters::counter_raw,
              "returns the overall number of task descriptions stolen by neighboring"
              "schedulers from this scheduler for the referenced locality",
              HPX_PERFORMANCE_COUNTER_V1, counts_creator,
              &performance_counters::locality_thread_counter_discoverer,
              ""
            },
            { "/threads/count/stolen-to-pending", performance_counters::counter_raw,
              "returns the overall number of pending HPX-threads stolen from neighboring"
              "schedulers for the referenced locality", HPX_PERFORMANCE_COUNTER_V1,
              counts_creator,
              &performance_counters::locality_thread_counter_discoverer,
              ""
            },
            { "/threads/count/stolen-to-staged", performance_counters::counter_raw,
              "returns the overall number of task descriptions stolen from neighboring"
              "schedulers for the referenced locality", HPX_PERFORMANCE_COUNTER_V1,
              counts_creator,
              &performance_counters::locality_thread_counter_discoverer,
              ""
            }
#endif
        };
        performance_counters::install_counter_types(
            counter_types, sizeof(counter_types)/sizeof(counter_types[0]));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy>
    bool threadmanager_impl<SchedulingPolicy>::
        run(std::size_t num_threads)
    {
        boost::unique_lock<mutex_type> lk(mtx_);

        if (pool_.get_os_thread_count() != 0 ||
            pool_.has_reached_state(state_running))
        {
            return true;    // do nothing if already running
        }

        LTM_(info) << "run: running timer pool";
        timer_pool_.run(false);

        if (!pool_.run(lk, num_threads))
        {
            timer_pool_.stop();
            return false;
        }

        LTM_(info) << "run: running";
        return true;
    }

    template <typename SchedulingPolicy>
    void threadmanager_impl<SchedulingPolicy>::
        stop (bool blocking)
    {
        LTM_(info) << "stop: blocking(" << std::boolalpha << blocking << ")";

        boost::unique_lock<mutex_type> lk(mtx_);
        pool_.stop(lk, blocking);

        LTM_(info) << "stop: stopping timer pool";
        timer_pool_.stop();             // stop timer pool as well
        if (blocking) {
            timer_pool_.join();
            timer_pool_.clear();
        }
    }

#ifdef HPX_HAVE_THREAD_CUMULATIVE_COUNTS
    template <typename SchedulingPolicy>
    boost::int64_t threadmanager_impl<SchedulingPolicy>::
        get_executed_threads(std::size_t num, bool reset)
    {
        return pool_.get_executed_threads(num, reset);
    }

    template <typename SchedulingPolicy>
    boost::int64_t threadmanager_impl<SchedulingPolicy>::
        get_executed_thread_phases(std::size_t num, bool reset)
    {
        return pool_.get_executed_thread_phases(num, reset);
    }

#ifdef HPX_HAVE_THREAD_IDLE_RATES
    template <typename SchedulingPolicy>
    boost::int64_t threadmanager_impl<SchedulingPolicy>::
        get_thread_phase_duration(std::size_t num, bool reset)
    {
        return pool_.get_thread_phase_duration(num, reset);
    }

    template <typename SchedulingPolicy>
    boost::int64_t threadmanager_impl<SchedulingPolicy>::
        get_thread_duration(std::size_t num, bool reset)
    {
        return pool_.get_thread_duration(num, reset);
    }

    template <typename SchedulingPolicy>
    boost::int64_t threadmanager_impl<SchedulingPolicy>::
        get_thread_phase_overhead(std::size_t num, bool reset)
    {
        return pool_.get_thread_phase_overhead(num, reset);
    }

    template <typename SchedulingPolicy>
    boost::int64_t threadmanager_impl<SchedulingPolicy>::
        get_thread_overhead(std::size_t num, bool reset)
    {
        return pool_.get_thread_overhead(num, reset);
    }

    template <typename SchedulingPolicy>
    boost::int64_t threadmanager_impl<SchedulingPolicy>::
        get_cumulative_thread_duration(std::size_t num, bool reset)
    {
        return pool_.get_cumulative_thread_duration(num, reset);
    }

    template <typename SchedulingPolicy>
    boost::int64_t threadmanager_impl<SchedulingPolicy>::
        get_cumulative_thread_overhead(std::size_t num, bool reset)
    {
        return pool_.get_cumulative_thread_overhead(num, reset);
    }
#endif
#endif

#ifdef HPX_HAVE_THREAD_IDLE_RATES
    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy>
    boost::int64_t threadmanager_impl<SchedulingPolicy>::
        avg_idle_rate(bool reset)
    {
        return pool_.avg_idle_rate(reset);
    }

    template <typename SchedulingPolicy>
    boost::int64_t threadmanager_impl<SchedulingPolicy>::
        avg_idle_rate(std::size_t num_thread, bool reset)
    {
        return pool_.avg_idle_rate(num_thread, reset);
    }

#if defined(HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES)
    template <typename SchedulingPolicy>
    boost::int64_t threadmanager_impl<SchedulingPolicy>::
        avg_creation_idle_rate(bool reset)
    {
        return pool_.avg_creation_idle_rate(reset);
    }

    template <typename SchedulingPolicy>
    boost::int64_t threadmanager_impl<SchedulingPolicy>::
        avg_cleanup_idle_rate(bool reset)
    {
        return pool_.avg_cleanup_idle_rate(reset);
    }
#endif
#endif
}}

///////////////////////////////////////////////////////////////////////////////
/// explicit template instantiation for the thread manager of our choice
#include <hpx/runtime/threads/policies/callback_notifier.hpp>

#if defined(HPX_HAVE_LOCAL_SCHEDULER)
#include <hpx/runtime/threads/policies/local_queue_scheduler.hpp>
template class HPX_EXPORT hpx::threads::threadmanager_impl<
    hpx::threads::policies::local_queue_scheduler<> >;
#endif

#if defined(HPX_HAVE_STATIC_SCHEDULER)
#include <hpx/runtime/threads/policies/static_queue_scheduler.hpp>
template class HPX_EXPORT hpx::threads::threadmanager_impl<
    hpx::threads::policies::static_queue_scheduler<> >;
#endif

#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
#include <hpx/runtime/threads/policies/static_priority_queue_scheduler.hpp>
template class HPX_EXPORT hpx::threads::threadmanager_impl<
    hpx::threads::policies::static_priority_queue_scheduler<> >;
#endif

#include <hpx/runtime/threads/policies/local_priority_queue_scheduler.hpp>
template class HPX_EXPORT hpx::threads::threadmanager_impl<
    hpx::threads::policies::local_priority_queue_scheduler<> >;

#if defined(HPX_HAVE_ABP_SCHEDULER)
template class HPX_EXPORT hpx::threads::threadmanager_impl<
    hpx::threads::policies::abp_fifo_priority_queue_scheduler>;
#endif

#if defined(HPX_HAVE_HIERARCHY_SCHEDULER)
#include <hpx/runtime/threads/policies/hierarchy_scheduler.hpp>
template class HPX_EXPORT hpx::threads::threadmanager_impl<
    hpx::threads::policies::hierarchy_scheduler<> >;
#endif

#if defined(HPX_HAVE_PERIODIC_PRIORITY_SCHEDULER)
#include <hpx/runtime/threads/policies/periodic_priority_queue_scheduler.hpp>
template class HPX_EXPORT hpx::threads::threadmanager_impl<
    hpx::threads::policies::periodic_priority_queue_scheduler<> >;
#endif

