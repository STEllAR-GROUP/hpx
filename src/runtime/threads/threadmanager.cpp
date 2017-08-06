//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach, Katelyn Kufahl
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//  Copyright (c) 2015 Patricia Grubel
//  Copyright (c) 2017 Shoshana Jakobovits
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/compat/mutex.hpp>
#include <hpx/exception.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/counter_creators.hpp>
#include <hpx/performance_counters/manage_counter_type.hpp>
#include <hpx/runtime/resource_partitioner.hpp>
#include <hpx/runtime/thread_pool_helpers.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/threads/thread_init_data.hpp>
#include <hpx/runtime/threads/detail/set_thread_state.hpp>
#include <hpx/runtime/threads/detail/scheduled_thread_pool.hpp>
#include <hpx/runtime/threads/executors/current_executor.hpp>
#include <hpx/runtime/threads/policies/schedulers.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/block_profiler.hpp>
#include <hpx/util/itt_notify.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/hardware/timestamp.hpp>
#include <hpx/util/runtime_configuration.hpp>

#include <boost/format.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>

#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace policies
{
    ///////////////////////////////////////////////////////////////////////////
    // We control whether to collect queue wait times using this global bool.
    // It will be set by any of the related performance counters. Once set it
    // stays set, thus no race conditions will occur.
    HPX_EXPORT bool maintain_queue_wait_times = false;
}}}
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx {

    namespace detail {

        // helper functions testing option compatibility
        void ensure_high_priority_compatibility(
            boost::program_options::variables_map const& vm)
        {
            if (vm.count("hpx:high-priority-threads"))
            {
                throw detail::command_line_error(
                    "Invalid command line option "
                    "--hpx:high-priority-threads, valid for "
                    "--hpx:queuing=local-priority and "
                    "--hpx:queuing=abp-priority only");
            }
        }

        void ensure_numa_sensitivity_compatibility(
            boost::program_options::variables_map const& vm)
        {
            if (vm.count("hpx:numa-sensitive"))
            {
                throw detail::command_line_error(
                    "Invalid command line option "
                    "--hpx:numa-sensitive, valid for "
                    "--hpx:queuing=local, --hpx:queuing=local-priority, or "
                    "--hpx:queuing=abp-priority only");
            }
        }

        void ensure_hierarchy_arity_compatibility(
            boost::program_options::variables_map const& vm)
        {
            if (vm.count("hpx:hierarchy-arity"))
            {
                throw detail::command_line_error(
                    "Invalid command line option "
                    "--hpx:hierarchy-arity, valid for "
                    "--hpx:queuing=hierarchy only.");
            }
        }

        void ensure_queuing_option_compatibility(
            boost::program_options::variables_map const& vm)
        {
            ensure_high_priority_compatibility(vm);
            ensure_numa_sensitivity_compatibility(vm);
            ensure_hierarchy_arity_compatibility(vm);
        }

        void ensure_hwloc_compatibility(
            boost::program_options::variables_map const& vm)
        {
#if defined(HPX_HAVE_HWLOC)
            // pu control is available for HWLOC only
            if (vm.count("hpx:pu-offset"))
            {
                throw detail::command_line_error(
                    "Invalid command line option "
                    "--hpx:pu-offset, valid for --hpx:queuing=priority or "
                    "--hpx:queuing=local-priority only.");
            }
            if (vm.count("hpx:pu-step"))
            {
                throw detail::command_line_error(
                    "Invalid command line option "
                    "--hpx:pu-step, valid for --hpx:queuing=abp-priority, "
                    "--hpx:queuing=periodic-priority, or "
                    "--hpx:queuing=local-priority only.");
            }
#endif
#if defined(HPX_HAVE_HWLOC)
            // affinity control is available for HWLOC only
            if (vm.count("hpx:affinity"))
            {
                throw detail::command_line_error(
                    "Invalid command line option "
                    "--hpx:affinity, valid for --hpx:queuing=abp-priority, "
                    "--hpx:queuing=periodic-priority, or "
                    "--hpx:queuing=local-priority only.");
            }
            if (vm.count("hpx:bind"))
            {
                throw detail::command_line_error(
                    "Invalid command line option "
                    "--hpx:bind, valid for --hpx:queuing=abp-priority, "
                    "--hpx:queuing=periodic-priority, or "
                    "--hpx:queuing=local-priority only.");
            }
            if (vm.count("hpx:print-bind"))
            {
                throw detail::command_line_error(
                    "Invalid command line option "
                    "--hpx:print-bind, valid for --hpx:queuing=abp-priority, "
                    "--hpx:queuing=periodic-priority, or "
                    "--hpx:queuing=local-priority only.");
            }
#endif
        }

        ///////////////////////////////////////////////////////////////////////
        std::size_t get_num_high_priority_queues(
            util::command_line_handling const& cfg, std::size_t num_threads)
        {
            std::size_t num_high_priority_queues = num_threads;
            if (cfg.vm_.count("hpx:high-priority-threads"))
            {
                num_high_priority_queues =
                    cfg.vm_["hpx:high-priority-threads"].as<std::size_t>();
                if (num_high_priority_queues > num_threads)
                {
                    throw detail::command_line_error(
                        "Invalid command line option: "
                        "number of high priority threads ("
                        "--hpx:high-priority-threads), should not be larger "
                        "than number of threads (--hpx:threads)");
                }
            }
            return num_high_priority_queues;
        }

        ///////////////////////////////////////////////////////////////////////
        std::string get_affinity_domain(util::command_line_handling const& cfg)
        {
            std::string affinity_domain("pu");
#if defined(HPX_HAVE_HWLOC)
            if (cfg.affinity_domain_ != "pu")
            {
                affinity_domain = cfg.affinity_domain_;
                if (0 != std::string("pu").find(affinity_domain) &&
                    0 != std::string("core").find(affinity_domain) &&
                    0 != std::string("numa").find(affinity_domain) &&
                    0 != std::string("machine").find(affinity_domain))
                {
                    throw detail::command_line_error(
                        "Invalid command line option "
                        "--hpx:affinity, value must be one of: pu, core, numa, "
                        "or machine.");
                }
            }
#endif
            return affinity_domain;
        }

        std::size_t get_affinity_description(
            util::command_line_handling const& cfg, std::string& affinity_desc)
        {
#if defined(HPX_HAVE_HWLOC)
            if (cfg.affinity_bind_.empty())
                return cfg.numa_sensitive_;

            if (!(cfg.pu_offset_ == std::size_t(-1) ||
                    cfg.pu_offset_ == std::size_t(0)) ||
                cfg.pu_step_ != 1 || cfg.affinity_domain_ != "pu")
            {
                throw detail::command_line_error(
                    "Command line option --hpx:bind "
                    "should not be used with --hpx:pu-step, --hpx:pu-offset, "
                    "or --hpx:affinity.");
            }

            affinity_desc = cfg.affinity_bind_;
#endif
            return cfg.numa_sensitive_;
        }
    }    // namespace detail
}

namespace hpx { namespace threads
{
    ///////////////////////////////////////////////////////////////////////
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
        "staged",
        "pending_do_not_schedule",
        "pending_boost"
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

    ///////////////////////////////////////////////////////////////////////
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
    } // samespace strings

    char const* get_thread_state_ex_name(thread_state_ex_enum state_ex)
    {
        if (state_ex < wait_unknown || state_ex > wait_abort)
            return "wait_unknown";
        return strings::thread_state_ex_names[state_ex];
    }

    ///////////////////////////////////////////////////////////////////////
    namespace strings
    {
        char const* const thread_priority_names[] =
        {
            "default",
            "low",
            "normal",
            "high (recursive)"
            "boost",
            "high (non-recursive)",
        };
    }

    char const* get_thread_priority_name(thread_priority priority)
    {
        if (priority < thread_priority_default ||
            priority > thread_priority_high)
        {
            return "unknown";
        }
        return strings::thread_priority_names[priority];
    }

    namespace strings {
        char const* const stack_size_names[] = {
            "small", "medium", "large", "huge",
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

        return strings::stack_size_names[size - 1];
    }

    ///////////////////////////////////////////////////////////////////////////
    threadmanager::threadmanager(
#ifdef HPX_HAVE_TIMER_POOL
            util::io_service_pool& timer_pool,
#endif
            notification_policy_type& notifier)
      : num_threads_(hpx::get_resource_partitioner().get_num_distinct_pus()),
#ifdef HPX_HAVE_TIMER_POOL
        timer_pool_(timer_pool),
#endif
        notifier_(notifier)
    {
        auto& rp = hpx::get_resource_partitioner();
        size_t num_pools = rp.get_num_pools();
        util::command_line_handling const& cfg_ = rp.get_command_line_switches();
        std::size_t thread_offset = 0;

        // instantiate the pools
        for (size_t i = 0; i != num_pools; i++)
        {
            std::string name = rp.get_pool_name(i);
            resource::scheduling_policy sched_type = rp.which_scheduler(name);
            std::size_t num_threads_in_pool = rp.get_num_threads(i);

            // make sure the first thread-pool that gets instantiated is the default one
            if (i == 0)
            {
                if (name != "default")
                {
                    throw std::invalid_argument("Trying to instantiate pool " +
                        name +
                        " as first thread pool, but first thread pool must be "
                        "named default");
                }
            }

            switch (sched_type)
            {
            case resource::user_defined:
            {
                auto const& pool_func = rp.get_pool_creator(i);
                std::unique_ptr<detail::thread_pool_base> pool(
                    pool_func(notifier, num_threads_in_pool,
                        thread_offset, i, name));
                pools_.push_back(std::move(pool));
                break;
            }
            case resource::unspecified:
            {
                throw std::invalid_argument(
                    "cannot instantiate a thread-manager if the thread-pool" +
                    name + " has an unspecified scheduler type");
            }
            case resource::local:
            {
#if defined(HPX_HAVE_LOCAL_SCHEDULER)
                // set parameters for scheduler and pool instantiation and
                // perform compatibility checks
                hpx::detail::ensure_high_priority_compatibility(cfg_.vm_);
                hpx::detail::ensure_hierarchy_arity_compatibility(cfg_.vm_);
                std::string affinity_desc;
                std::size_t numa_sensitive =
                    hpx::detail::get_affinity_description(cfg_, affinity_desc);

                // instantiate the scheduler
                typedef hpx::threads::policies::local_queue_scheduler<>
                    local_sched_type;
                local_sched_type::init_parameter_type init(num_threads_in_pool,
                    1000, numa_sensitive, "core-local_queue_scheduler");
                std::unique_ptr<local_sched_type> sched(
                    new local_sched_type(init));

                // instantiate the pool
                std::unique_ptr<detail::thread_pool_base> pool(
                    new hpx::threads::detail::scheduled_thread_pool<
                            local_sched_type
                        >(std::move(sched),
                        notifier, i, name.c_str(),
                        policies::scheduler_mode(policies::do_background_work |
                            policies::reduce_thread_priority |
                            policies::delay_exit),
                        thread_offset));
                pools_.push_back(std::move(pool));

#else
                throw detail::command_line_error(
                    "Command line option "
                    "--hpx:queuing=local "
                    "is not configured in this build. Please rebuild with "
                    "'cmake -DHPX_WITH_THREAD_SCHEDULERS=local'.");
#endif
                break;
            }

            case resource::local_priority_fifo:
            {
                // set parameters for scheduler and pool instantiation and
                // perform compatibility checks
                hpx::detail::ensure_hierarchy_arity_compatibility(cfg_.vm_);
                std::size_t num_high_priority_queues =
                    hpx::detail::get_num_high_priority_queues(
                        cfg_, rp.get_num_threads(name));
                std::string affinity_desc;
                std::size_t numa_sensitive =
                    hpx::detail::get_affinity_description(cfg_, affinity_desc);

                // instantiate the scheduler
                typedef hpx::threads::policies::local_priority_queue_scheduler<
                    compat::mutex, hpx::threads::policies::lockfree_fifo>
                    local_sched_type;
                local_sched_type::init_parameter_type init(num_threads_in_pool,
                    num_high_priority_queues, 1000, numa_sensitive,
                    "core-local_priority_queue_scheduler");
                std::unique_ptr<local_sched_type> sched(
                    new local_sched_type(init));

                // instantiate the pool
                std::unique_ptr<detail::thread_pool_base> pool(
                    new hpx::threads::detail::scheduled_thread_pool<
                            local_sched_type
                        >(std::move(sched),
                        notifier, i, name.c_str(),
                        policies::scheduler_mode(policies::do_background_work |
                            policies::reduce_thread_priority |
                            policies::delay_exit),
                        thread_offset));
                pools_.push_back(std::move(pool));

                break;
            }

            case resource::local_priority_lifo:
            {
                // set parameters for scheduler and pool instantiation and
                // perform compatibility checks
                hpx::detail::ensure_hierarchy_arity_compatibility(cfg_.vm_);
                std::size_t num_high_priority_queues =
                    hpx::detail::get_num_high_priority_queues(
                        cfg_, rp.get_num_threads(name));
                std::string affinity_desc;
                std::size_t numa_sensitive =
                    hpx::detail::get_affinity_description(cfg_, affinity_desc);

                // instantiate the scheduler
                typedef hpx::threads::policies::local_priority_queue_scheduler<
                    compat::mutex, hpx::threads::policies::lockfree_lifo>
                    local_sched_type;
                local_sched_type::init_parameter_type init(num_threads_in_pool,
                    num_high_priority_queues, 1000, numa_sensitive,
                    "core-local_priority_queue_scheduler");
                std::unique_ptr<local_sched_type> sched(
                    new local_sched_type(init));

                // instantiate the pool
                std::unique_ptr<detail::thread_pool_base> pool(
                    new hpx::threads::detail::scheduled_thread_pool<
                            local_sched_type
                        >(std::move(sched),
                        notifier, i, name.c_str(),
                        policies::scheduler_mode(policies::do_background_work |
                            policies::reduce_thread_priority |
                            policies::delay_exit),
                        thread_offset));
                pools_.push_back(std::move(pool));

                break;
            }

            case resource::static_:
            {
#if defined(HPX_HAVE_STATIC_SCHEDULER)
                // set parameters for scheduler and pool instantiation and
                // perform compatibility checks
                hpx::detail::ensure_high_priority_compatibility(cfg_.vm_);
                hpx::detail::ensure_hierarchy_arity_compatibility(cfg_.vm_);
                std::string affinity_domain =
                    hpx::detail::get_affinity_domain(cfg_);
                std::string affinity_desc;
                std::size_t numa_sensitive =
                    hpx::detail::get_affinity_description(cfg_, affinity_desc);

                // instantiate the scheduler
                typedef hpx::threads::policies::static_queue_scheduler<>
                    local_sched_type;
                local_sched_type::init_parameter_type init(num_threads_in_pool,
                    1000, numa_sensitive, "core-static_queue_scheduler");
                std::unique_ptr<local_sched_type> sched(
                    new local_sched_type(init));

                // instantiate the pool
                std::unique_ptr<detail::thread_pool_base> pool(
                    new hpx::threads::detail::scheduled_thread_pool<
                            local_sched_type
                        >(std::move(sched),
                        notifier, i, name.c_str(),
                        policies::scheduler_mode(policies::do_background_work |
                            policies::reduce_thread_priority |
                            policies::delay_exit),
                        thread_offset));
                pools_.push_back(std::move(pool));

#else
                throw detail::command_line_error(
                    "Command line option "
                    "--hpx:queuing=static "
                    "is not configured in this build. Please rebuild with "
                    "'cmake -DHPX_WITH_THREAD_SCHEDULERS=static'.");
#endif
                break;
            }

            case resource::static_priority:
            {
#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
                // set parameters for scheduler and pool instantiation and
                // perform compatibility checks
                hpx::detail::ensure_hierarchy_arity_compatibility(cfg_.vm_);
                std::size_t num_high_priority_queues =
                    hpx::detail::get_num_high_priority_queues(
                        cfg_, rp.get_num_threads(name));
                std::string affinity_domain =
                    hpx::detail::get_affinity_domain(cfg_);
                std::string affinity_desc;
                std::size_t numa_sensitive =
                    hpx::detail::get_affinity_description(cfg_, affinity_desc);

                // instantiate the scheduler
                typedef hpx::threads::policies::
                    static_priority_queue_scheduler<>
                        local_sched_type;
                local_sched_type::init_parameter_type init(num_threads_in_pool,
                    num_high_priority_queues, 1000, numa_sensitive,
                    "core-static_priority_queue_scheduler");
                std::unique_ptr<local_sched_type> sched(
                    new local_sched_type(init));

                // instantiate the pool
                std::unique_ptr<detail::thread_pool_base> pool(
                    new hpx::threads::detail::scheduled_thread_pool<
                            local_sched_type
                        >(std::move(sched),
                        notifier, i, name.c_str(),
                        policies::scheduler_mode(policies::do_background_work |
                            policies::reduce_thread_priority |
                            policies::delay_exit),
                        thread_offset));
                pools_.push_back(std::move(pool));

#else
                throw detail::command_line_error(
                    "Command line option "
                    "--hpx:queuing=static-priority "
                    "is not configured in this build. Please rebuild with "
                    "'cmake -DHPX_WITH_THREAD_SCHEDULERS=static-priority'.");
#endif
                break;
            }

            case resource::abp_priority:
            {
#if defined(HPX_HAVE_ABP_SCHEDULER)
                // set parameters for scheduler and pool instantiation and
                // perform compatibility checks
                hpx::detail::ensure_hierarchy_arity_compatibility(cfg_.vm_);
                hpx::detail::ensure_hwloc_compatibility(cfg_.vm_);
                std::size_t num_high_priority_queues =
                    hpx::detail::get_num_high_priority_queues(
                        cfg_, rp.get_num_threads(name));

                // instantiate the scheduler
                typedef hpx::threads::policies::local_priority_queue_scheduler<
                    compat::mutex, hpx::threads::policies::lockfree_fifo>
                    local_sched_type;
                local_sched_type::init_parameter_type init(num_threads_in_pool,
                    num_high_priority_queues, 1000, cfg_.numa_sensitive_,
                    "core-abp_fifo_priority_queue_scheduler");
                std::unique_ptr<local_sched_type> sched(
                    new local_sched_type(init));

                // instantiate the pool
                std::unique_ptr<detail::thread_pool_base> pool(
                    new hpx::threads::detail::scheduled_thread_pool<
                            local_sched_type
                        >(std::move(sched),
                        notifier, i, name.c_str(),
                        policies::scheduler_mode(policies::do_background_work |
                            policies::reduce_thread_priority |
                            policies::delay_exit),
                        thread_offset));
                pools_.push_back(std::move(pool));
#else
                throw detail::command_line_error(
                    "Command line option "
                    "--hpx:queuing=abp-priority "
                    "is not configured in this build. Please rebuild with "
                    "'cmake -DHPX_WITH_THREAD_SCHEDULERS=abp-priority'.");
#endif
                break;
            }

            case resource::hierarchy:
            {
#if defined(HPX_HAVE_HIERARCHY_SCHEDULER)
                // set parameters for scheduler and pool instantiation and
                // perform compatibility checks
                hpx::detail::ensure_high_priority_compatibility(cfg_.vm_);
                hpx::detail::ensure_numa_sensitivity_compatibility(cfg_.vm_);
                hpx::detail::ensure_hwloc_compatibility(cfg_.vm_);

                // instantiate the pool
                typedef hpx::threads::policies::hierarchy_scheduler<>
                    local_sched_type;
                std::size_t arity = 2;
                if (cfg_.vm_.count("hpx:hierarchy-arity"))
                    arity = cfg_.vm_["hpx:hierarchy-arity"].as<std::size_t>();
                local_sched_type::init_parameter_type init(num_threads_in_pool,
                    arity, 1000, 0, "core-hierarchy_scheduler");
                std::unique_ptr<local_sched_type> sched(
                    new local_sched_type(init));

                // instantiate the pool
                std::unique_ptr<detail::thread_pool_base> pool(
                    new hpx::threads::detail::scheduled_thread_pool<
                            local_sched_type
                        >(std::move(sched),
                        notifier, i, name.c_str(),
                        policies::scheduler_mode(policies::do_background_work |
                            policies::reduce_thread_priority |
                            policies::delay_exit),
                        thread_offset));
                pools_.push_back(std::move(pool));
#else
                throw detail::command_line_error(
                    "Command line option "
                    "--hpx:queuing=hierarchy "
                    "is not configured in this build. Please rebuild with "
                    "'cmake -DHPX_WITH_THREAD_SCHEDULERS=hierarchy'.");
#endif
                break;
            }

            case resource::periodic_priority:
            {
                // set parameters for scheduler and pool instantiation and
                // perform compatibility checks
                hpx::detail::ensure_hierarchy_arity_compatibility(cfg_.vm_);
                hpx::detail::ensure_hwloc_compatibility(cfg_.vm_);
                std::size_t num_high_priority_queues =
                    hpx::detail::get_num_high_priority_queues(
                        cfg_, rp.get_num_threads(name));

                // instantiate the scheduler
                typedef hpx::threads::policies::
                    periodic_priority_queue_scheduler<>
                        local_sched_type;
                local_sched_type::init_parameter_type init(num_threads_in_pool,
                    num_high_priority_queues, 1000, cfg_.numa_sensitive_,
                    "core-periodic_priority_queue_scheduler");
                std::unique_ptr<local_sched_type> sched(
                    new local_sched_type(init));

                // instantiate the pool
                std::unique_ptr<detail::thread_pool_base> pool(
                    new hpx::threads::detail::scheduled_thread_pool<
                            local_sched_type
                        >(std::move(sched),
                        notifier, i, name.c_str(),
                        policies::scheduler_mode(policies::do_background_work |
                            policies::reduce_thread_priority |
                            policies::delay_exit),
                        thread_offset));
                pools_.push_back(std::move(pool));

                break;
            }

            case resource::throttle:
            {
#if !defined(HPX_HAVE_THROTTLING_SCHEDULER)
                throw hpx::detail::command_line_error(
                    "Command line option "
                    "--hpx:queuing=throttle "
                    "is not configured in this build. Please rebuild with "
                    "'cmake -DHPX_WITH_THREAD_SCHEDULERS=throttle "
                    "-DHPX_WITH_APEX'.");
#endif
                break;
            }
            }
            // update the thread_offset for the next pool
            thread_offset += num_threads_in_pool;
        }

        // fill the thread-lookup table
        for (auto& pool_iter : pools_)
        {
            std::size_t nt = rp.get_num_threads(pool_iter->get_pool_name());
            for (std::size_t i = 0; i < nt; i++)
            {
                threads_lookup_.push_back(pool_iter->get_pool_id());
            }
        }
    }

    threadmanager::~threadmanager()
    {
    }

    void threadmanager::init()
    {
        auto& rp = hpx::get_resource_partitioner();
        std::size_t threads_offset = 0;

        // initialize all pools
        for (auto && pool_iter : pools_)
        {
            std::size_t num_threads_in_pool =
                rp.get_num_threads(pool_iter->get_pool_name());
            pool_iter->init(num_threads_in_pool, threads_offset);
            threads_offset += num_threads_in_pool;
        }
    }

    void threadmanager::print_pools(std::ostream& os)
    {
        os << "The thread-manager owns "
           << pools_.size() << " pool(s) : \n";

        for (auto && pool_iter : pools_)
        {
            pool_iter->print_pool(os);
        }
    }

    detail::thread_pool_base& threadmanager::default_pool() const
    {
        HPX_ASSERT(!pools_.empty());
        return *pools_[0];
    }

//     threadmanager::scheduler_type& threadmanager::get_scheduler(
//         std::string const& pool_name) const
//     {
//         // if the given pool_name is default, we don't need to look for it
//         if (pool_name == "default")
//         {
//             return default_scheduler();
//         }
//
//         // don't start at begin() since the first one is the default, start one
//         // further
//         auto pool =
//             std::find_if(
//                 ++pools_.begin(), pools_.end(),
//                 [&pool_name](pool_type const& itp) -> bool
//                 {
//                     return (itp->get_pool_name() == pool_name);
//                 });
//
//         if (pool != pools_.end())
//         {
//             return pool->get_scheduler();
//         }
//
//         throw std::invalid_argument(
//                 "the resource partitioner does not own a thread pool named \""
//                 + pool_name + "\". \n");
//     }

    detail::thread_pool_base& threadmanager::get_pool(
        std::string const& pool_name) const
    {
        // if the given pool_name is default, we don't need to look for it
        if (pool_name == "default")
        {
            return default_pool();
        }

        // don't start at begin() since the first one is the default,
        // start one further
        auto pool = std::find_if(
            ++pools_.begin(), pools_.end(),
            [&pool_name](pool_type const& itp) -> bool
            {
                return (itp->get_pool_name() == pool_name);
            });

        if (pool != pools_.end())
        {
            return **pool;
        }

        //! FIXME Add names of available pools?
        HPX_THROW_EXCEPTION(bad_parameter, "threadmanager::get_pool",
            "the resource partitioner does not own a thread pool named '" +
            pool_name + "'. \n");
    }

    detail::thread_pool_base& threadmanager::get_pool(
        detail::pool_id_type pool_id) const
    {
        return get_pool(pool_id.name_);
    }

    detail::thread_pool_base& threadmanager::get_pool(
        std::size_t thread_index) const
    {
        return get_pool(threads_lookup_[thread_index]);
    }

    ///////////////////////////////////////////////////////////////////////////
    std::int64_t threadmanager::get_thread_count(thread_state_enum state,
        thread_priority priority, std::size_t num_thread, bool reset)
    {
        std::int64_t total_count = 0;
        std::lock_guard<mutex_type> lk(mtx_);

        for (auto& pool_iter : pools_)
        {
            total_count +=
                pool_iter->get_thread_count(state, priority, num_thread, reset);
        }

        return total_count;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Enumerate all matching threads
    bool threadmanager::enumerate_threads(
        util::function_nonser<bool(thread_id_type)> const& f,
        thread_state_enum state) const
    {
        std::lock_guard<mutex_type> lk(mtx_);
        bool result = true;

        for (auto& pool_iter : pools_)
        {
            result = result && pool_iter->enumerate_threads(f, state);
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Abort all threads which are in suspended state. This will set
    // the state of all suspended threads to \a pending while
    // supplying the wait_abort extended state flag
    void threadmanager::abort_all_suspended_threads()
    {
        std::lock_guard<mutex_type> lk(mtx_);
        for (auto& pool_iter : pools_)
        {
            pool_iter->abort_all_suspended_threads();
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Clean up terminated threads. This deletes all threads which
    // have been terminated but which are still held in the queue
    // of terminated threads. Some schedulers might not do anything
    // here.
    bool threadmanager::cleanup_terminated(bool delete_all)
    {
        std::lock_guard<mutex_type> lk(mtx_);
        bool result = true;

        for (auto& pool_iter : pools_)
        {
            result = result && pool_iter->cleanup_terminated(delete_all);
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////
    void threadmanager::register_thread(thread_init_data& data,
        thread_id_type& id, thread_state_enum initial_state, bool run_now,
        error_code& ec)
    {
        default_pool().create_thread(data, id, initial_state, run_now, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    void threadmanager::register_work(
        thread_init_data& data, thread_state_enum initial_state, error_code& ec)
    {
        default_pool().create_work(data, initial_state, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    HPX_CONSTEXPR std::size_t all_threads = std::size_t(-1);

    std::int64_t threadmanager::get_queue_length(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->get_queue_length(all_threads, reset);
        return result;
    }

#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
    std::int64_t threadmanager::get_average_thread_wait_time(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result +=
                pool_iter->get_average_thread_wait_time(all_threads, reset);
        return result;
    }

    std::int64_t threadmanager::get_average_task_wait_time(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->get_average_task_wait_time(all_threads, reset);
        return result;
    }
#endif

    std::int64_t threadmanager::get_cumulative_duration(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->get_cumulative_duration(all_threads, reset);
        return result;
    }

#ifdef HPX_HAVE_THREAD_IDLE_RATES
    std::int64_t threadmanager::avg_idle_rate(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->avg_idle_rate(all_threads, reset);
        return result;
    }

#ifdef HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES
    std::int64_t threadmanager::avg_creation_idle_rate(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->avg_creation_idle_rate(all_threads, reset);
        return result;
    }

    std::int64_t threadmanager::avg_cleanup_idle_rate(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->avg_cleanup_idle_rate(all_threads, reset);
        return result;
    }
#endif
#endif

#ifdef HPX_HAVE_THREAD_CUMULATIVE_COUNTS
    std::int64_t threadmanager::get_executed_threads(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->get_executed_threads(all_threads, reset);
        return result;
    }

    std::int64_t threadmanager::get_executed_thread_phases(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->get_executed_thread_phases(all_threads, reset);
        return result;
    }

#ifdef HPX_HAVE_THREAD_IDLE_RATES
    std::int64_t threadmanager::get_thread_duration(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->get_thread_duration(all_threads, reset);
        return result;
    }

    std::int64_t threadmanager::get_thread_phase_duration(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->get_thread_phase_duration(all_threads, reset);
        return result;
    }

    std::int64_t threadmanager::get_thread_overhead(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->get_thread_overhead(all_threads, reset);
        return result;
    }

    std::int64_t threadmanager::get_thread_phase_overhead(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->get_thread_phase_overhead(all_threads, reset);
        return result;
    }

    std::int64_t threadmanager::get_cumulative_thread_duration(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result +=
                pool_iter->get_cumulative_thread_duration(all_threads, reset);
        return result;
    }

    std::int64_t threadmanager::get_cumulative_thread_overhead(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result +=
                pool_iter->get_cumulative_thread_overhead(all_threads, reset);
        return result;
    }
#endif
#endif

#ifdef HPX_HAVE_THREAD_STEALING_COUNTS
    std::int64_t threadmanager::get_num_pending_misses(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->get_num_pending_misses(all_threads, reset);
        return result;
    }

    std::int64_t threadmanager::get_num_pending_accesses(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->get_num_pending_accesses(all_threads, reset);
        return result;
    }

    std::int64_t threadmanager::get_num_stolen_from_pending(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->get_num_stolen_from_pending(all_threads, reset);
        return result;
    }

    std::int64_t threadmanager::get_num_stolen_from_staged(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->get_num_stolen_from_staged(all_threads, reset);
        return result;
    }

    std::int64_t threadmanager::get_num_stolen_to_pending(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->get_num_stolen_to_pending(all_threads, reset);
        return result;
    }

    std::int64_t threadmanager::get_num_stolen_to_staged(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->get_num_stolen_to_staged(all_threads, reset);
        return result;
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    // counter creator and discovery functions

#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
    naming::gid_type threadmanager::queue_wait_time_counter_creator(
        threadmanager_counter_func total_func,
        threadpool_counter_func pool_func,
        performance_counters::counter_info const& info, error_code& ec)
    {
        naming::gid_type gid = locality_pool_thread_counter_creator(
            total_func, pool_func, info, ec);

        if (!ec)
            policies::maintain_queue_wait_times = true;

        return gid;
    }
#endif

    naming::gid_type threadmanager::locality_pool_thread_counter_creator(
        threadmanager_counter_func total_func,
        threadpool_counter_func pool_func,
        performance_counters::counter_info const& info, error_code& ec)
    {
        // verify the validity of the counter instance name
        performance_counters::counter_path_elements paths;
        performance_counters::get_counter_path_elements(
            info.fullname_, paths, ec);
        if (ec)
            return naming::invalid_gid;

        if (paths.parentinstance_is_basename_)
        {
            HPX_THROWS_IF(ec, bad_parameter, "queue_length_counter_creator",
                "invalid counter instance parent name: " +
                    paths.parentinstancename_);
            return naming::invalid_gid;
        }

        using hpx::util::placeholders::_1;

        detail::thread_pool_base& pool = default_pool();
        if (paths.instancename_ == "total" && paths.instanceindex_ == -1)
        {
            // overall counter
            using performance_counters::detail::create_raw_counter;
            util::function_nonser<std::int64_t(bool)> f =
                util::bind(total_func, this, _1);
            return create_raw_counter(info, std::move(f), ec);
        }
        else if (paths.instancename_ == "pool")
        {
            if (paths.instanceindex_ >= 0 &&
                std::size_t(paths.instanceindex_) <
                    hpx::resource::get_num_thread_pools())
            {
                // specific for given pool counter
                detail::thread_pool_base& pool_instance =
                    hpx::resource::get_thread_pool(paths.instanceindex_);

                using performance_counters::detail::create_raw_counter;
                util::function_nonser<std::int64_t(bool)> f =
                    util::bind(pool_func, &pool_instance,
                        static_cast<std::size_t>(paths.subinstanceindex_), _1);
                return create_raw_counter(info, std::move(f), ec);
            }
        }
        else if (paths.instancename_ == "worker-thread" &&
            paths.instanceindex_ >= 0 &&
            std::size_t(paths.instanceindex_) < pool.get_os_thread_count())
        {
            // specific counter from default
            using performance_counters::detail::create_raw_counter;
            util::function_nonser<std::int64_t(bool)> f = util::bind(pool_func,
                &pool, static_cast<std::size_t>(paths.instanceindex_), _1);
            return create_raw_counter(info, std::move(f), ec);
        }

        HPX_THROWS_IF(ec, bad_parameter, "locality_pool_thread_counter_creator",
            "invalid counter instance name: " + paths.instancename_);
        return naming::invalid_gid;
    }

    // scheduler utilization counter creation function
    naming::gid_type threadmanager::scheduler_utilization_counter_creator(
        performance_counters::counter_info const& info, error_code& ec)
    {
        // verify the validity of the counter instance name
        performance_counters::counter_path_elements paths;
        performance_counters::get_counter_path_elements(
            info.fullname_, paths, ec);
        if (ec)
            return naming::invalid_gid;

        // /scheduler{locality#%d/total}/utilization/instantaneous
        // /scheduler{locality#%d/pool#%s/total}/utilization/instantaneous
        if (paths.parentinstance_is_basename_)
        {
            HPX_THROWS_IF(ec, bad_parameter, "scheduler_utilization_creator",
                "invalid counter instance parent name: " +
                    paths.parentinstancename_);
            return naming::invalid_gid;
        }

        detail::thread_pool_base& pool = default_pool();

        if (paths.instancename_ == "total" && paths.instanceindex_ == -1)
        {
            // overall counter
            using performance_counters::detail::create_raw_counter;
            util::function_nonser<std::int64_t()> f = util::bind(
                &detail::thread_pool_base::get_scheduler_utilization, &pool);
            return create_raw_counter(info, std::move(f), ec);
        }

        HPX_THROWS_IF(ec, bad_parameter, "scheduler_utilization_creator",
            "invalid counter instance name: " + paths.instancename_);
        return naming::invalid_gid;
    }

    ///////////////////////////////////////////////////////////////////////////
    // locality/pool/worker-thread counter creation function with no total
    // /threads{locality#%d/worker-thread#%d}/idle-loop-count/instantaneous
    // /threads{locality#%d/pool#%s/worker-thread#%d}/idle-loop-count/instantaneous
    naming::gid_type
    threadmanager::locality_pool_thread_no_total_counter_creator(
        threadpool_counter_func pool_func,
        performance_counters::counter_info const& info, error_code& ec)
    {
        // verify the validity of the counter instance name
        performance_counters::counter_path_elements paths;
        performance_counters::get_counter_path_elements(
            info.fullname_, paths, ec);
        if (ec)
            return naming::invalid_gid;

        if (paths.parentinstance_is_basename_)
        {
            HPX_THROWS_IF(ec, bad_parameter,
                "locality_pool_thread_no_total_counter_creator",
                "invalid counter instance parent name: " +
                    paths.parentinstancename_);
            return naming::invalid_gid;
        }

        using hpx::util::placeholders::_1;

        detail::thread_pool_base& pool = default_pool();
        if (paths.instancename_ == "total" && paths.instanceindex_ == -1)
        {
            // overall counter, not supported
            HPX_THROWS_IF(ec, bad_parameter,
                "locality_pool_thread_no_total_counter_creator",
                "invalid counter instance name: " + paths.instancename_ +
                    "'total' is not supported");
        }
        else if (paths.instancename_ == "pool")
        {
            if (paths.instanceindex_ >= 0 &&
                std::size_t(paths.instanceindex_) <
                    hpx::resource::get_num_thread_pools())
            {
                // specific for given pool counter
                detail::thread_pool_base& pool_instance =
                    hpx::resource::get_thread_pool(paths.instanceindex_);

                using performance_counters::detail::create_raw_counter;
                util::function_nonser<std::int64_t(bool)> f =
                    util::bind(pool_func, &pool_instance,
                        static_cast<std::size_t>(paths.subinstanceindex_), _1);
                return create_raw_counter(info, std::move(f), ec);
            }
        }
        else if (paths.instancename_ == "worker-thread" &&
            paths.instanceindex_ >= 0 &&
            std::size_t(paths.instanceindex_) < pool.get_os_thread_count())
        {
            // specific counter
            using performance_counters::detail::create_raw_counter;
            util::function_nonser<std::int64_t(bool)> f =
                util::bind(pool_func, &pool,
                    static_cast<std::size_t>(paths.instanceindex_), _1);
            return create_raw_counter(info, std::move(f), ec);
        }

        HPX_THROWS_IF(ec, bad_parameter,
            "locality_pool_thread_no_total_counter_creator",
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
    naming::gid_type threadmanager::thread_counts_counter_creator(
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

        using util::placeholders::_1;

        creator_data data[] = {
            // /threads{locality#%d/total}/count/stack-recycles
            {"count/stack-recycles",
                util::bind(
                    &coroutine_type::impl_type::get_stack_recycle_count, _1),
                util::function_nonser<std::uint64_t(bool)>(), "", 0},
#if !defined(HPX_WINDOWS) && !defined(HPX_HAVE_GENERIC_CONTEXT_COROUTINES)
            // /threads{locality#%d/total}/count/stack-unbinds
            {"count/stack-unbinds",
                util::bind(
                    &coroutine_type::impl_type::get_stack_unbind_count, _1),
                util::function_nonser<std::uint64_t(bool)>(), "", 0},
#endif
            // /threads{locality#%d/total}/count/objects
            // /threads{locality#%d/allocator%d}/count/objects
            {"count/objects",
                &coroutine_type::impl_type::get_allocation_count_all,
                util::bind(&coroutine_type::impl_type::get_allocation_count,
                    static_cast<std::size_t>(paths.instanceindex_), _1),
                "allocator", HPX_COROUTINE_NUM_ALL_HEAPS},
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
    void threadmanager::register_counter_types()
    {
        using util::placeholders::_1;
        using util::placeholders::_2;

        performance_counters::create_counter_func counts_creator(util::bind(
            &threadmanager::thread_counts_counter_creator, this, _1, _2));

        performance_counters::generic_counter_type_data counter_types[] = {
            // length of thread queue(s)
            {"/threadqueue/length", performance_counters::counter_raw,
                "returns the current queue length for the referenced queue",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind(&threadmanager::locality_pool_thread_counter_creator,
                    this, &threadmanager::get_queue_length,
                    &detail::thread_pool_base::get_queue_length, _1, _2),
                &performance_counters::locality_pool_thread_counter_discoverer,
                ""},
#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
            // average thread wait time for queue(s)
            {"/threads/wait-time/pending", performance_counters::counter_raw,
                "returns the average wait time of pending threads for the "
                "referenced queue",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind(&threadmanager::queue_wait_time_counter_creator,
                    this, &threadmanager::get_average_thread_wait_time,
                    &detail::thread_pool_base::get_average_thread_wait_time, _1, _2),
                &performance_counters::locality_pool_thread_counter_discoverer,
                "ns"},
            // average task wait time for queue(s)
            {"/threads/wait-time/staged", performance_counters::counter_raw,
                "returns the average wait time of staged threads (task "
                "descriptions) for the referenced queue",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind(&threadmanager::queue_wait_time_counter_creator,
                    this, &threadmanager::get_average_task_wait_time,
                    &detail::thread_pool_base::get_average_task_wait_time, _1, _2),
                &performance_counters::locality_pool_thread_counter_discoverer,
                "ns"},
#endif
#ifdef HPX_HAVE_THREAD_IDLE_RATES
            // idle rate
            {"/threads/idle-rate", performance_counters::counter_raw,
                "returns the idle rate for the referenced object",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind(&threadmanager::locality_pool_thread_counter_creator,
                    this, &threadmanager::avg_idle_rate,
                    &detail::thread_pool_base::avg_idle_rate, _1, _2),
                &performance_counters::locality_pool_thread_counter_discoverer,
                "0.01%"},
#ifdef HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES
            {"/threads/creation-idle-rate", performance_counters::counter_raw,
                "returns the % of idle-rate spent creating HPX-threads for the "
                "referenced object",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind(&threadmanager::locality_pool_thread_counter_creator,
                    this, &threadmanager::avg_creation_idle_rate,
                    &detail::thread_pool_base::avg_creation_idle_rate, _1, _2),
                &performance_counters::locality_pool_thread_counter_discoverer,
                "0.01%"},
            {"/threads/cleanup-idle-rate", performance_counters::counter_raw,
                "returns the % of time spent cleaning up terminated "
                "HPX-threads for the referenced object",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind(&threadmanager::locality_pool_thread_counter_creator,
                    this, &threadmanager::avg_cleanup_idle_rate,
                    &detail::thread_pool_base::avg_cleanup_idle_rate, _1, _2),
                &performance_counters::locality_pool_thread_counter_discoverer,
                "0.01%"},
#endif
#endif
#ifdef HPX_HAVE_THREAD_CUMULATIVE_COUNTS
            // thread counts
            {"/threads/count/cumulative", performance_counters::counter_raw,
                "returns the overall number of executed (retired) HPX-threads "
                "for the referenced locality",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind(&threadmanager::locality_pool_thread_counter_creator,
                    this, &threadmanager::get_executed_threads,
                    &detail::thread_pool_base::get_executed_threads, _1, _2),
                &performance_counters::locality_pool_thread_counter_discoverer,
                ""},
            {"/threads/count/cumulative-phases",
                performance_counters::counter_raw,
                "returns the overall number of HPX-thread phases executed for "
                "the referenced locality",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind(&threadmanager::locality_pool_thread_counter_creator,
                    this, &threadmanager::get_executed_thread_phases,
                    &detail::thread_pool_base::get_executed_thread_phases, _1, _2),
                &performance_counters::locality_pool_thread_counter_discoverer,
                ""},
#ifdef HPX_HAVE_THREAD_IDLE_RATES
            {"/threads/time/average", performance_counters::counter_raw,
                "returns the average time spent executing one HPX-thread",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind(&threadmanager::locality_pool_thread_counter_creator,
                    this, &threadmanager::get_thread_duration,
                    &detail::thread_pool_base::get_thread_duration, _1, _2),
                &performance_counters::locality_pool_thread_counter_discoverer,
                "ns"},
            {"/threads/time/average-phase", performance_counters::counter_raw,
                "returns the average time spent executing one HPX-thread phase",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind(&threadmanager::locality_pool_thread_counter_creator,
                    this, &threadmanager::get_thread_phase_duration,
                    &detail::thread_pool_base::get_thread_phase_duration, _1, _2),
                &performance_counters::locality_pool_thread_counter_discoverer,
                "ns"},
            {"/threads/time/average-overhead",
                performance_counters::counter_raw,
                "returns average overhead time executing one HPX-thread",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind(&threadmanager::locality_pool_thread_counter_creator,
                    this, &threadmanager::get_thread_overhead,
                    &detail::thread_pool_base::get_thread_overhead, _1, _2),
                &performance_counters::locality_pool_thread_counter_discoverer,
                "ns"},
            {"/threads/time/average-phase-overhead",
                performance_counters::counter_raw,
                "returns average overhead time executing one HPX-thread phase",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind(&threadmanager::locality_pool_thread_counter_creator,
                    this, &threadmanager::get_thread_phase_overhead,
                    &detail::thread_pool_base::get_thread_phase_overhead, _1, _2),
                &performance_counters::locality_pool_thread_counter_discoverer,
                "ns"},
            {"/threads/time/cumulative", performance_counters::counter_raw,
                "returns the cumulative time spent executing HPX-threads",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind(&threadmanager::locality_pool_thread_counter_creator,
                    this, &threadmanager::get_cumulative_thread_duration,
                    &detail::thread_pool_base::get_cumulative_thread_duration, _1, _2),
                &performance_counters::locality_pool_thread_counter_discoverer,
                "ns"},
            {"/threads/time/cumulative-overhead",
                performance_counters::counter_raw,
                "returns the cumulative overhead time incurred by executing "
                "HPX threads",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind(&threadmanager::locality_pool_thread_counter_creator,
                    this, &threadmanager::get_cumulative_thread_overhead,
                    &detail::thread_pool_base::get_cumulative_thread_overhead, _1, _2),
                &performance_counters::locality_pool_thread_counter_discoverer,
                "ns"},
#endif
#endif
            {"/threads/time/overall", performance_counters::counter_raw,
                "returns the overall time spent running the scheduler on a "
                "core",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind(&threadmanager::locality_pool_thread_counter_creator,
                    this, &threadmanager::get_cumulative_duration,
                    &detail::thread_pool_base::get_cumulative_duration, _1, _2),
                &performance_counters::locality_pool_thread_counter_discoverer,
                "ns"},
            {"/threads/count/instantaneous/all",
                performance_counters::counter_raw,
                "returns the overall current number of HPX-threads "
                "instantiated at the referenced locality",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind(&threadmanager::locality_pool_thread_counter_creator,
                    this, &threadmanager::get_thread_count_unknown,
                    &detail::thread_pool_base::get_thread_count_unknown, _1, _2),
                &performance_counters::locality_pool_thread_counter_discoverer,
                ""},
            {"/threads/count/instantaneous/active",
                performance_counters::counter_raw,
                "returns the current number of active HPX-threads "
                "at the referenced locality",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind(&threadmanager::locality_pool_thread_counter_creator,
                    this, &threadmanager::get_thread_count_active,
                    &detail::thread_pool_base::get_thread_count_active, _1, _2),
                &performance_counters::locality_pool_thread_counter_discoverer,
                ""},
            {"/threads/count/instantaneous/pending",
                performance_counters::counter_raw,
                "returns the current number of pending HPX-threads "
                "at the referenced locality",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind(&threadmanager::locality_pool_thread_counter_creator,
                    this, &threadmanager::get_thread_count_pending,
                    &detail::thread_pool_base::get_thread_count_pending, _1, _2),
                &performance_counters::locality_pool_thread_counter_discoverer,
                ""},
            {"/threads/count/instantaneous/suspended",
                performance_counters::counter_raw,
                "returns the current number of suspended HPX-threads "
                "at the referenced locality",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind(&threadmanager::locality_pool_thread_counter_creator,
                    this, &threadmanager::get_thread_count_suspended,
                    &detail::thread_pool_base::get_thread_count_suspended, _1, _2),
                &performance_counters::locality_pool_thread_counter_discoverer,
                ""},
            {"/threads/count/instantaneous/terminated",
                performance_counters::counter_raw,
                "returns the current number of terminated HPX-threads "
                "at the referenced locality",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind(&threadmanager::locality_pool_thread_counter_creator,
                    this, &threadmanager::get_thread_count_terminated,
                    &detail::thread_pool_base::get_thread_count_terminated, _1, _2),
                &performance_counters::locality_pool_thread_counter_discoverer,
                ""},
            {"/threads/count/instantaneous/staged",
                performance_counters::counter_raw,
                "returns the current number of staged HPX-threads (task "
                "descriptions) "
                "at the referenced locality",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind(&threadmanager::locality_pool_thread_counter_creator,
                    this, &threadmanager::get_thread_count_staged,
                    &detail::thread_pool_base::get_thread_count_staged, _1, _2),
                &performance_counters::locality_pool_thread_counter_discoverer,
                ""},
            {"/threads/count/stack-recycles", performance_counters::counter_raw,
                "returns the total number of HPX-thread recycling operations "
                "performed for the referenced locality",
                HPX_PERFORMANCE_COUNTER_V1, counts_creator,
                &performance_counters::locality_counter_discoverer, ""},
#if !defined(HPX_WINDOWS) && !defined(HPX_HAVE_GENERIC_CONTEXT_COROUTINES)
            {"/threads/count/stack-unbinds", performance_counters::counter_raw,
                "returns the total number of HPX-thread unbind (madvise) "
                "operations performed for the referenced locality",
                HPX_PERFORMANCE_COUNTER_V1, counts_creator,
                &performance_counters::locality_counter_discoverer, ""},
#endif
            {"/threads/count/objects", performance_counters::counter_raw,
                "returns the overall number of created HPX-thread objects for "
                "the referenced locality",
                HPX_PERFORMANCE_COUNTER_V1, counts_creator,
                &locality_allocator_counter_discoverer, ""},
#ifdef HPX_HAVE_THREAD_STEALING_COUNTS
            {"/threads/count/pending-misses", performance_counters::counter_raw,
                "returns the number of times that the referenced worker-thread "
                "on the referenced locality failed to find pending HPX-threads "
                "in its associated queue",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind(&threadmanager::locality_pool_thread_counter_creator,
                    this, &threadmanager::get_num_pending_misses,
                    &detail::thread_pool_base::get_num_pending_misses, _1, _2),
                &performance_counters::locality_pool_thread_counter_discoverer,
                ""},
            {"/threads/count/pending-accesses",
                performance_counters::counter_raw,
                "returns the number of times that the referenced worker-thread "
                "on the referenced locality looked for pending HPX-threads "
                "in its associated queue",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind(&threadmanager::locality_pool_thread_counter_creator,
                    this, &threadmanager::get_num_pending_accesses,
                    &detail::thread_pool_base::get_num_pending_accesses, _1, _2),
                &performance_counters::locality_pool_thread_counter_discoverer,
                ""},
            {"/threads/count/stolen-from-pending",
                performance_counters::counter_raw,
                "returns the overall number of pending HPX-threads stolen by "
                "neighboring"
                "schedulers from this scheduler for the referenced locality",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind(&threadmanager::locality_pool_thread_counter_creator,
                    this, &threadmanager::get_num_stolen_from_pending,
                    &detail::thread_pool_base::get_num_stolen_from_pending, _1, _2),
                &performance_counters::locality_pool_thread_counter_discoverer,
                ""},
            {"/threads/count/stolen-from-staged",
                performance_counters::counter_raw,
                "returns the overall number of task descriptions stolen by "
                "neighboring"
                "schedulers from this scheduler for the referenced locality",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind(&threadmanager::locality_pool_thread_counter_creator,
                    this, &threadmanager::get_num_stolen_from_staged,
                    &detail::thread_pool_base::get_num_stolen_from_staged, _1, _2),
                &performance_counters::locality_pool_thread_counter_discoverer,
                ""},
            {"/threads/count/stolen-to-pending",
                performance_counters::counter_raw,
                "returns the overall number of pending HPX-threads stolen from "
                "neighboring"
                "schedulers for the referenced locality",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind(&threadmanager::locality_pool_thread_counter_creator,
                    this, &threadmanager::get_num_stolen_to_pending,
                    &detail::thread_pool_base::get_num_stolen_to_pending, _1, _2),
                &performance_counters::locality_pool_thread_counter_discoverer,
                ""},
            {"/threads/count/stolen-to-staged",
                performance_counters::counter_raw,
                "returns the overall number of task descriptions stolen from "
                "neighboring"
                "schedulers for the referenced locality",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind(&threadmanager::locality_pool_thread_counter_creator,
                    this, &threadmanager::get_num_stolen_to_staged,
                    &detail::thread_pool_base::get_num_stolen_to_staged, _1, _2),
                &performance_counters::locality_pool_thread_counter_discoverer,
                ""},
#endif
            // scheduler utilization
            {"/scheduler/utilization/instantaneous",
                performance_counters::counter_raw,
                "returns the current scheduler utilization",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind(
                    &threadmanager::scheduler_utilization_counter_creator, this,
                    _1, _2),
                &performance_counters::locality_pool_counter_discoverer, "%"},
            // idle-loop count
            {"/threads/idle-loop-count/instantaneous",
                performance_counters::counter_raw,
                "returns the current value of the scheduler idle-loop count",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind(&threadmanager::
                               locality_pool_thread_no_total_counter_creator,
                    this, &detail::thread_pool_base::get_idle_loop_count, _1, _2),
                &performance_counters::
                    locality_pool_thread_no_total_counter_discoverer,
                ""},
            // busy-loop count
            {"/threads/busy-loop-count/instantaneous",
                performance_counters::counter_raw,
                "returns the current value of the scheduler busy-loop count",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind(&threadmanager::
                               locality_pool_thread_no_total_counter_creator,
                    this, &detail::thread_pool_base::get_busy_loop_count, _1, _2),
                &performance_counters::
                    locality_pool_thread_no_total_counter_discoverer,
                ""}
        };
        performance_counters::install_counter_types(
            counter_types, sizeof(counter_types)/sizeof(counter_types[0]));
    }

    ///////////////////////////////////////////////////////////////////////////
    bool threadmanager::run()
    {
        std::unique_lock<mutex_type> lk(mtx_);

        // startup barrier for controlling startup
        HPX_ASSERT(startup_.get() == nullptr);

        // the main thread needs to have a unique thread_num
        // worker threads are numbered 0..N-1, so we can use N for this thread
        auto& rp = hpx::get_resource_partitioner();
        init_tss(rp.get_num_threads());

#ifdef HPX_HAVE_TIMER_POOL
        LTM_(info) << "run: running timer pool";
        timer_pool_.run(false);
#endif

        for (auto& pool_iter : pools_)
        {
            std::size_t num_threads_in_pool =
                rp.get_num_threads(pool_iter->get_pool_name());
            startup_.reset(new compat::barrier(
                static_cast<unsigned>(num_threads_in_pool + 1)));

            if (pool_iter->get_os_thread_count() != 0 ||
                pool_iter->has_reached_state(state_running))
            {
                return true;    // do nothing if already running
            }

            if (!pool_iter->run(lk, std::ref(*startup_), num_threads_in_pool))
            {
#ifdef HPX_HAVE_TIMER_POOL
                timer_pool_.stop();
#endif
                return false;
            }

            // wait for all thread pools to have launched all OS threads
            startup_->wait();

            // set all states of all schedulers to "running"
            policies::scheduler_base* sched = pool_iter->get_scheduler();
            if (sched) sched->set_all_states(state_running);
        }

        LTM_(info) << "run: running";
        return true;
    }

    void threadmanager::stop(bool blocking)
    {
        LTM_(info) << "stop: blocking(" << std::boolalpha << blocking << ")";

        std::unique_lock<mutex_type> lk(mtx_);
        for (auto& pool_iter : pools_)
        {
            pool_iter->stop(lk, blocking);
        }
        deinit_tss();

#ifdef HPX_HAVE_TIMER_POOL
        LTM_(info) << "stop: stopping timer pool";
        timer_pool_.stop();    // stop timer pool as well
        if (blocking)
        {
            timer_pool_.join();
            timer_pool_.clear();
        }
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    std::int64_t get_thread_count(thread_state_enum state)
    {
        return get_thread_manager().get_thread_count(state);
    }

    std::int64_t get_thread_count(thread_priority priority,
        thread_state_enum state)
    {
        return get_thread_manager().get_thread_count(state, priority);
    }

    ///////////////////////////////////////////////////////////////////////////
    bool enumerate_threads(util::function_nonser<bool(thread_id_type)> const& f,
        thread_state_enum state)
    {
        return get_thread_manager().enumerate_threads(f, state);
    }
} // namespace threads
} // namespace hpx
