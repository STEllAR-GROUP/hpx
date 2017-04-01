//  Copyright (c) 2015 Patricia Grubel
//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach, Katelyn Kufahl
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/counter_creators.hpp>
#include <hpx/performance_counters/manage_counter_type.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/runtime/threads/threadmanager_impl.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/threads/thread_init_data.hpp>
#include <hpx/runtime/threads/detail/set_thread_state.hpp>
#include <hpx/runtime/threads/executors/current_executor.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/block_profiler.hpp>
#include <hpx/util/itt_notify.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/hardware/timestamp.hpp>
#include <hpx/util/runtime_configuration.hpp>

#include <boost/format.hpp>
#include <boost/thread/mutex.hpp>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <numeric>
#include <sstream>
#include <utility>

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

    char const* get_thread_state_name(thread_state state)
    {
        return get_thread_state_name(state.state());
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace strings
    {
        char const* const thread_state_ex_names[] =
        {
            "wait_unknown",
            "wait_signaled",
            "wait_timeout",
            "wait_terminate",
            "wait_abort"
        };
    }

    char const* get_thread_state_ex_name(thread_state_ex_enum state_ex)
    {
        if (state_ex < wait_unknown || state_ex > wait_abort)
            return "wait_unknown";
        return strings::thread_state_ex_names[state_ex];
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

        if (size < thread_stacksize_small || size > thread_stacksize_huge)
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
        pool_(scheduler, notifier, "main_thread_scheduling_pool",
            policies::scheduler_mode(
                policies::do_background_work | policies::reduce_thread_priority |
                policies::delay_exit)),
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
    std::int64_t threadmanager_impl<SchedulingPolicy>::
        get_thread_count(thread_state_enum state, thread_priority priority,
            std::size_t num_thread, bool reset) const
    {
        std::lock_guard<mutex_type> lk(mtx_);
        return pool_.get_thread_count(state, priority, num_thread, reset);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Enumerate all matching threads
    template <typename SchedulingPolicy>
    bool threadmanager_impl<SchedulingPolicy>::enumerate_threads(
        util::function_nonser<bool(thread_id_type)> const& f,
        thread_state_enum state) const
    {
        std::lock_guard<mutex_type> lk(mtx_);
        return pool_.enumerate_threads(f, state);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Abort all threads which are in suspended state. This will set
    // the state of all suspended threads to \a pending while
    // supplying the wait_abort extended state flag
    template <typename SchedulingPolicy>
    void threadmanager_impl<SchedulingPolicy>::
        abort_all_suspended_threads()
    {
        std::lock_guard<mutex_type> lk(mtx_);
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
        std::lock_guard<mutex_type> lk(mtx_);
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
            util::function_nonser<std::int64_t()> f =
                util::bind(&spt::get_queue_length, &pool_, -1);
            return create_raw_counter(info, std::move(f), ec);
        }
        else if (paths.instancename_ == "worker-thread" &&
            paths.instanceindex_ >= 0 &&
            std::size_t(paths.instanceindex_) < pool_.get_os_thread_count())
        {
            // specific counter
            using performance_counters::detail::create_raw_counter;
            util::function_nonser<std::int64_t()> f =
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

        typedef detail::thread_pool<scheduling_policy_type> spt;

        using util::placeholders::_1;
        if (paths.instancename_ == "total" && paths.instanceindex_ == -1)
        {
            policies::maintain_queue_wait_times = true;

            // overall counter
            using performance_counters::detail::create_raw_counter;
            util::function_nonser<std::int64_t()> f =
                util::bind(&spt::get_average_thread_wait_time, &pool_, -1);
            return create_raw_counter(info, std::move(f), ec);
        }
        else if (paths.instancename_ == "worker-thread" &&
            paths.instanceindex_ >= 0 &&
            std::size_t(paths.instanceindex_) < pool_.get_os_thread_count())
        {
            policies::maintain_queue_wait_times = true;

            // specific counter
            using performance_counters::detail::create_raw_counter;
            util::function_nonser<std::int64_t()> f =
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

        typedef detail::thread_pool<scheduling_policy_type> spt;

        using util::placeholders::_1;
        if (paths.instancename_ == "total" && paths.instanceindex_ == -1)
        {
            policies::maintain_queue_wait_times = true;

            // overall counter
            using performance_counters::detail::create_raw_counter;
            util::function_nonser<std::int64_t()> f =
                util::bind(&spt::get_average_task_wait_time, &pool_, -1);
            return create_raw_counter(info, std::move(f), ec);
        }
        else if (paths.instancename_ == "worker-thread" &&
            paths.instanceindex_ >= 0 &&
            std::size_t(paths.instanceindex_) < pool_.get_os_thread_count())
        {
            policies::maintain_queue_wait_times = true;

            // specific counter
            using performance_counters::detail::create_raw_counter;
            util::function_nonser<std::int64_t()> f =
                util::bind(&spt::get_average_task_wait_time, &pool_,
                    static_cast<std::size_t>(paths.instanceindex_));
            return create_raw_counter(info, std::move(f), ec);
        }

        HPX_THROWS_IF(ec, bad_parameter, "task_wait_time_counter_creator",
            "invalid counter instance name: " + paths.instancename_);
        return naming::invalid_gid;
    }
#endif

    // scheduler utilization counter creation function
    template <typename SchedulingPolicy>
    naming::gid_type threadmanager_impl<SchedulingPolicy>::
        scheduler_utilization_counter_creator(
            performance_counters::counter_info const& info, error_code& ec)
    {
        // verify the validity of the counter instance name
        performance_counters::counter_path_elements paths;
        performance_counters::get_counter_path_elements(info.fullname_, paths, ec);
        if (ec) return naming::invalid_gid;

        // /scheduler{locality#%d/total}/utilization/instantaneous
        if (paths.parentinstance_is_basename_) {
            HPX_THROWS_IF(ec, bad_parameter, "scheduler_utilization_creator",
                "invalid counter instance parent name: " +
                    paths.parentinstancename_);
            return naming::invalid_gid;
        }

        typedef detail::thread_pool<scheduling_policy_type> spt;

        if (paths.instancename_ == "total" && paths.instanceindex_ == -1)
        {
            // overall counter
            using performance_counters::detail::create_raw_counter;
            util::function_nonser<std::int64_t()> f =
                util::bind(&spt::get_scheduler_utilization, &pool_);
            return create_raw_counter(info, std::move(f), ec);
        }

        HPX_THROWS_IF(ec, bad_parameter, "scheduler_utilization_creator",
            "invalid counter instance name: " + paths.instancename_);
        return naming::invalid_gid;
    }

    // scheduler utilization counter creation function
    template <typename SchedulingPolicy>
    naming::gid_type threadmanager_impl<SchedulingPolicy>::
        idle_loop_count_counter_creator(
            performance_counters::counter_info const& info, error_code& ec)
    {
        // verify the validity of the counter instance name
        performance_counters::counter_path_elements paths;
        performance_counters::get_counter_path_elements(info.fullname_, paths, ec);
        if (ec) return naming::invalid_gid;

        // /scheduler{locality#%d/total}/utilization/instantaneous
        if (paths.parentinstance_is_basename_) {
            HPX_THROWS_IF(ec, bad_parameter, "idle_loop_count_counter_creator",
                "invalid counter instance parent name: " +
                    paths.parentinstancename_);
            return naming::invalid_gid;
        }

        typedef detail::thread_pool<scheduling_policy_type> spt;

        if (paths.instancename_ == "total" && paths.instanceindex_ == -1)
        {
            // overall counter
            using performance_counters::detail::create_raw_counter;
            util::function_nonser<std::int64_t()> f =
                util::bind(&spt::get_idle_loop_count, &pool_, -1);
            return create_raw_counter(info, std::move(f), ec);
        }
        else if (paths.instancename_ == "worker-thread" &&
            paths.instanceindex_ >= 0 &&
            std::size_t(paths.instanceindex_) < pool_.get_os_thread_count())
        {
            // specific counter
            using performance_counters::detail::create_raw_counter;
            util::function_nonser<std::int64_t()> f =
                util::bind(&spt::get_idle_loop_count, &pool_,
                    static_cast<std::size_t>(paths.instanceindex_));
            return create_raw_counter(info, std::move(f), ec);
        }

        HPX_THROWS_IF(ec, bad_parameter, "idle_loop_count_counter_creator",
            "invalid counter instance name: " + paths.instancename_);
        return naming::invalid_gid;
    }

    // scheduler utilization counter creation function
    template <typename SchedulingPolicy>
    naming::gid_type threadmanager_impl<SchedulingPolicy>::
        busy_loop_count_counter_creator(
            performance_counters::counter_info const& info, error_code& ec)
    {
        // verify the validity of the counter instance name
        performance_counters::counter_path_elements paths;
        performance_counters::get_counter_path_elements(info.fullname_, paths, ec);
        if (ec) return naming::invalid_gid;

        // /scheduler{locality#%d/total}/utilization/instantaneous
        if (paths.parentinstance_is_basename_) {
            HPX_THROWS_IF(ec, bad_parameter, "busy_loop_count_counter_creator",
                "invalid counter instance parent name: " +
                    paths.parentinstancename_);
            return naming::invalid_gid;
        }

        typedef detail::thread_pool<scheduling_policy_type> spt;

        if (paths.instancename_ == "total" && paths.instanceindex_ == -1)
        {
            // overall counter
            using performance_counters::detail::create_raw_counter;
            util::function_nonser<std::int64_t()> f =
                util::bind(&spt::get_busy_loop_count, &pool_, -1);
            return create_raw_counter(info, std::move(f), ec);
        }
        else if (paths.instancename_ == "worker-thread" &&
            paths.instanceindex_ >= 0 &&
            std::size_t(paths.instanceindex_) < pool_.get_os_thread_count())
        {
            // specific counter
            using performance_counters::detail::create_raw_counter;
            util::function_nonser<std::int64_t()> f =
                util::bind(&spt::get_busy_loop_count, &pool_,
                    static_cast<std::size_t>(paths.instanceindex_));
            return create_raw_counter(info, std::move(f), ec);
        }

        HPX_THROWS_IF(ec, bad_parameter, "busy_loop_count_counter_creator",
            "invalid counter instance name: " + paths.instancename_);
        return naming::invalid_gid;
    }

    ///////////////////////////////////////////////////////////////////////////
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
                    p.instanceindex_ = static_cast<std::int32_t>(t);
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
                p.instanceindex_ = static_cast<std::int32_t>(t);
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
            std::int64_t (threadmanager_impl::*avg_idle_rate_ptr)(
                bool
            ) = &ti::avg_idle_rate;
            util::function_nonser<std::int64_t(bool)> f =
                 util::bind(avg_idle_rate_ptr, this, _1);
            return create_raw_counter(info, std::move(f), ec);
        }
        else if (paths.instancename_ == "worker-thread" &&
            paths.instanceindex_ >= 0 &&
            std::size_t(paths.instanceindex_) < pool_.get_os_thread_count())
        {
            // specific counter
            using performance_counters::detail::create_raw_counter;
            std::int64_t (threadmanager_impl::*avg_idle_rate_ptr)(
                std::size_t, bool
            ) = &ti::avg_idle_rate;
            using performance_counters::detail::create_raw_counter;
            util::function_nonser<std::int64_t(bool)> f =
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
        util::function_nonser<std::int64_t(bool)> const& total_creator,
        util::function_nonser<std::int64_t(bool)> const& individual_creator,
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
            util::function_nonser<std::int64_t(bool)> total_func;
            util::function_nonser<std::int64_t(bool)> individual_func;
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
              util::function_nonser<std::uint64_t(bool)>(),
              "", 0
            },
            // /threads{locality#%d/total}/cleanup-idle-rate
            // /threads{locality#%d/worker-thread%d}/cleanup-idle-rate
            { "cleanup-idle-rate",
              util::bind(&ti::avg_cleanup_idle_rate, this, _1),
              util::function_nonser<std::uint64_t(bool)>(),
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
            // /threads{locality#%d/total}/time/overall
            // /threads{locality#%d/worker-thread%d}/time/overall
            { "time/overall",
              util::bind(&ti::get_cumulative_duration, this, -1, _1),
              util::bind(&ti::get_cumulative_duration, this,
                  static_cast<std::size_t>(paths.instanceindex_), _1),
              "worker-thread", shepherd_count
            },
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
              util::function_nonser<std::uint64_t(bool)>(), "", 0
            },
#if !defined(HPX_WINDOWS) && !defined(HPX_HAVE_GENERIC_CONTEXT_COROUTINES)
            // /threads{locality#%d/total}/count/stack-unbinds
            { "count/stack-unbinds",
              util::bind(&coroutine_type::impl_type::get_stack_unbind_count, _1),
              util::function_nonser<std::uint64_t(bool)>(), "", 0
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
        using util::placeholders::_1;
        using util::placeholders::_2;

        typedef threadmanager_impl ti;
        performance_counters::create_counter_func counts_creator(
            util::bind(&ti::thread_counts_counter_creator, this, _1, _2));

        performance_counters::generic_counter_type_data counter_types[] =
        {
            // length of thread queue(s)
            { "/threadqueue/length", performance_counters::counter_raw,
              "returns the current queue length for the referenced queue",
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&ti::queue_length_counter_creator, this, _1, _2),
              &performance_counters::locality_thread_counter_discoverer,
              ""
            },
#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
            // average thread wait time for queue(s)
            { "/threads/wait-time/pending", performance_counters::counter_raw,
              "returns the average wait time of \
                 pending threads for the referenced queue",
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&ti::thread_wait_time_counter_creator, this, _1, _2),
              &performance_counters::locality_thread_counter_discoverer,
              "ns"
            },
            // average task wait time for queue(s)
            { "/threads/wait-time/staged", performance_counters::counter_raw,
              "returns the average wait time of staged threads (task descriptions) "
              "for the referenced queue",
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&ti::task_wait_time_counter_creator, this, _1, _2),
              &performance_counters::locality_thread_counter_discoverer,
              "ns"
            },
#endif
#ifdef HPX_HAVE_THREAD_IDLE_RATES
            // idle rate
            { "/threads/idle-rate", performance_counters::counter_raw,
              "returns the idle rate for the referenced object",
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&ti::idle_rate_counter_creator, this, _1, _2),
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
            { "/threads/time/overall", performance_counters::counter_raw,
              "returns the overall time spent running the scheduler on a core",
              HPX_PERFORMANCE_COUNTER_V1, counts_creator,
              &performance_counters::locality_thread_counter_discoverer,
              "ns"
            },
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
#if !defined(HPX_WINDOWS) && !defined(HPX_HAVE_GENERIC_CONTEXT_COROUTINES)
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
            },
#endif
            // scheduler utilization
            { "/scheduler/utilization/instantaneous", performance_counters::counter_raw,
              "returns the current scheduler utilization",
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&ti::scheduler_utilization_counter_creator, this, _1, _2),
              &performance_counters::locality_counter_discoverer,
              "%"
            },
            // idle-loop count
            { "/scheduler/idle-loop-count/instantaneous",
                    performance_counters::counter_raw,
              "returns the current value of the scheduler idle-loop count",
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&ti::idle_loop_count_counter_creator, this, _1, _2),
              &performance_counters::locality_thread_counter_discoverer,
              ""
            },
            // busy-loop count
            { "/scheduler/busy-loop-count/instantaneous",
                    performance_counters::counter_raw,
              "returns the current value of the scheduler busy-loop count",
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&ti::busy_loop_count_counter_creator, this, _1, _2),
              &performance_counters::locality_thread_counter_discoverer,
              ""
            }
        };
        performance_counters::install_counter_types(
            counter_types, sizeof(counter_types)/sizeof(counter_types[0]));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy>
    bool threadmanager_impl<SchedulingPolicy>::
        run(std::size_t num_threads)
    {
        std::unique_lock<mutex_type> lk(mtx_);

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

        std::unique_lock<mutex_type> lk(mtx_);
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
    std::int64_t threadmanager_impl<SchedulingPolicy>::
        get_executed_threads(std::size_t num, bool reset)
    {
        return pool_.get_executed_threads(num, reset);
    }

    template <typename SchedulingPolicy>
    std::int64_t threadmanager_impl<SchedulingPolicy>::
        get_executed_thread_phases(std::size_t num, bool reset)
    {
        return pool_.get_executed_thread_phases(num, reset);
    }

#ifdef HPX_HAVE_THREAD_IDLE_RATES
    template <typename SchedulingPolicy>
    std::int64_t threadmanager_impl<SchedulingPolicy>::
        get_thread_phase_duration(std::size_t num, bool reset)
    {
        return pool_.get_thread_phase_duration(num, reset);
    }

    template <typename SchedulingPolicy>
    std::int64_t threadmanager_impl<SchedulingPolicy>::
        get_thread_duration(std::size_t num, bool reset)
    {
        return pool_.get_thread_duration(num, reset);
    }

    template <typename SchedulingPolicy>
    std::int64_t threadmanager_impl<SchedulingPolicy>::
        get_thread_phase_overhead(std::size_t num, bool reset)
    {
        return pool_.get_thread_phase_overhead(num, reset);
    }

    template <typename SchedulingPolicy>
    std::int64_t threadmanager_impl<SchedulingPolicy>::
        get_thread_overhead(std::size_t num, bool reset)
    {
        return pool_.get_thread_overhead(num, reset);
    }

    template <typename SchedulingPolicy>
    std::int64_t threadmanager_impl<SchedulingPolicy>::
        get_cumulative_thread_duration(std::size_t num, bool reset)
    {
        return pool_.get_cumulative_thread_duration(num, reset);
    }

    template <typename SchedulingPolicy>
    std::int64_t threadmanager_impl<SchedulingPolicy>::
        get_cumulative_thread_overhead(std::size_t num, bool reset)
    {
        return pool_.get_cumulative_thread_overhead(num, reset);
    }
#endif
#endif

    template <typename SchedulingPolicy>
    std::int64_t threadmanager_impl<SchedulingPolicy>::
        get_cumulative_duration(std::size_t num, bool reset)
    {
        return pool_.get_cumulative_duration(num, reset);
    }

#ifdef HPX_HAVE_THREAD_IDLE_RATES
    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy>
    std::int64_t threadmanager_impl<SchedulingPolicy>::
        avg_idle_rate(bool reset)
    {
        return pool_.avg_idle_rate(reset);
    }

    template <typename SchedulingPolicy>
    std::int64_t threadmanager_impl<SchedulingPolicy>::
        avg_idle_rate(std::size_t num_thread, bool reset)
    {
        return pool_.avg_idle_rate(num_thread, reset);
    }

#if defined(HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES)
    template <typename SchedulingPolicy>
    std::int64_t threadmanager_impl<SchedulingPolicy>::
        avg_creation_idle_rate(bool reset)
    {
        return pool_.avg_creation_idle_rate(reset);
    }

    template <typename SchedulingPolicy>
    std::int64_t threadmanager_impl<SchedulingPolicy>::
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

#if defined(HPX_HAVE_THROTTLE_SCHEDULER) && defined(HPX_HAVE_APEX)
#include <hpx/runtime/threads/policies/throttle_queue_scheduler.hpp>
template class HPX_EXPORT hpx::threads::threadmanager_impl<
    hpx::threads::policies::throttle_queue_scheduler<> >;
#endif

#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
#include <hpx/runtime/threads/policies/static_priority_queue_scheduler.hpp>
template class HPX_EXPORT hpx::threads::threadmanager_impl<
    hpx::threads::policies::static_priority_queue_scheduler<> >;
#endif

#include <hpx/runtime/threads/policies/local_priority_queue_scheduler.hpp>
template class HPX_EXPORT hpx::threads::threadmanager_impl<
    hpx::threads::policies::local_priority_queue_scheduler<
        boost::mutex, hpx::threads::policies::lockfree_fifo
    > >;
template class HPX_EXPORT hpx::threads::threadmanager_impl<
    hpx::threads::policies::local_priority_queue_scheduler<
        boost::mutex, hpx::threads::policies::lockfree_lifo
    > >;

#if defined(HPX_HAVE_ABP_SCHEDULER)
template class HPX_EXPORT hpx::threads::threadmanager_impl<
    hpx::threads::policies::local_priority_queue_scheduler<
        boost::mutex, hpx::threads::policies::lockfree_abp_fifo
    > >;
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

