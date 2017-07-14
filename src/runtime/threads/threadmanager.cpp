//  Copyright (c) 2007-2016 Hartmut Kaiser
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
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/runtime/threads/threadmanager_impl.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/threads/thread_init_data.hpp>
#include <hpx/runtime/threads/detail/set_thread_state.hpp>
#include <hpx/runtime/threads/detail/thread_pool_impl.hpp>
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
    } // namespace detail
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
    threadmanager_impl::threadmanager_impl(
#ifdef HPX_HAVE_TIMER_POOL
            util::io_service_pool& timer_pool,
#endif
            notification_policy_type& notifier)
      : num_threads_(hpx::get_resource_partitioner().get_num_threads()),
#ifdef HPX_HAVE_TIMER_POOL
        timer_pool_(timer_pool),
#endif
        thread_logger_("threadmanager_impl::register_thread"),
        work_logger_("threadmanager_impl::register_work"),
        set_state_logger_("threadmanager_impl::set_state"),
        notifier_(notifier)
    {
        auto& rp = hpx::get_resource_partitioner();
        rp.set_threadmanager(this);
        size_t num_pools = rp.get_num_pools();
        util::command_line_handling cfg_ = rp.get_command_line_switches();
        std::string name;
        std::size_t thread_offset = 0;

        // instantiate the pools
        for (size_t i(0); i < num_pools; i++)
        {
            name = rp.get_pool_name(i);
            resource::scheduling_policy sched_type = rp.which_scheduler(name);
            std::size_t num_threads_in_pool = rp.get_num_threads(i);

            // make sure the first thread-pool that gets instantiated is the default one
            if (i == 0)
            {
                if (name != "default")
                    throw std::invalid_argument("Trying to instantiate pool " +
                        name +
                        " as first thread pool, but first thread pool must be "
                        "named default");
            }

            switch (sched_type)
            {
            case -2:    // user supplied
            {
                const auto& pool_func = rp.get_pool_creator(i);
                detail::thread_pool* pool = pool_func(notifier,
                    num_threads_in_pool, thread_offset, i, name.c_str());
                pools_.push_back(pool);
                break;
            }
            case -1:    // unspecified = -1
            {
                throw std::invalid_argument(
                    "cannot instantiate a thread-manager if the thread-pool" +
                    name + " has an unspecified scheduler type");
            }
            case 0:    // local = 0
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
                threads::policies::scheduler_base* sched =
                    new local_sched_type(init);

                // instantiate the pool
                detail::thread_pool* pool =
                    new hpx::threads::detail::thread_pool_impl<
                        local_sched_type>(static_cast<local_sched_type*>(sched),
                        notifier, i, name.c_str(),
                        policies::scheduler_mode(policies::do_background_work |
                            policies::reduce_thread_priority |
                            policies::delay_exit),
                        thread_offset);
                pools_.push_back(pool);

#else
                throw detail::command_line_error(
                    "Command line option "
                    "--hpx:queuing=local "
                    "is not configured in this build. Please rebuild with "
                    "'cmake -DHPX_WITH_THREAD_SCHEDULERS=local'.");
#endif
                break;
            }

            case 1:    // local_priority_fifo = 1
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
                threads::policies::scheduler_base* sched =
                    new local_sched_type(init);

                // instantiate the pool
                detail::thread_pool* pool =
                    new hpx::threads::detail::thread_pool_impl<
                        local_sched_type>(static_cast<local_sched_type*>(sched),
                        notifier, i, name.c_str(),
                        policies::scheduler_mode(policies::do_background_work |
                            policies::reduce_thread_priority |
                            policies::delay_exit),
                        thread_offset);
                pools_.push_back(pool);

                break;
            }

            case 2:    // local_priority_lifo = 2
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
                threads::policies::scheduler_base* sched =
                    new local_sched_type(init);

                // instantiate the pool
                detail::thread_pool* pool =
                    new hpx::threads::detail::thread_pool_impl<
                        local_sched_type>(static_cast<local_sched_type*>(sched),
                        notifier, i, name.c_str(),
                        policies::scheduler_mode(policies::do_background_work |
                            policies::reduce_thread_priority |
                            policies::delay_exit),
                        thread_offset);
                pools_.push_back(pool);

                break;
            }

            case 3:    // static_ = 3
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
                threads::policies::scheduler_base* sched =
                    new local_sched_type(init);

                // instantiate the pool
                detail::thread_pool* pool =
                    new hpx::threads::detail::thread_pool_impl<
                        local_sched_type>(static_cast<local_sched_type*>(sched),
                        notifier, i, name.c_str(),
                        policies::scheduler_mode(policies::do_background_work |
                            policies::reduce_thread_priority |
                            policies::delay_exit),
                        thread_offset);
                pools_.push_back(pool);

#else
                throw detail::command_line_error(
                    "Command line option "
                    "--hpx:queuing=static "
                    "is not configured in this build. Please rebuild with "
                    "'cmake -DHPX_WITH_THREAD_SCHEDULERS=static'.");
#endif
                break;
            }

            case 4:    // static_priority = 4
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
                threads::policies::scheduler_base* sched =
                    new local_sched_type(init);

                // instantiate the pool
                detail::thread_pool* pool =
                    new hpx::threads::detail::thread_pool_impl<
                        local_sched_type>(static_cast<local_sched_type*>(sched),
                        notifier, i, name.c_str(),
                        policies::scheduler_mode(policies::do_background_work |
                            policies::reduce_thread_priority |
                            policies::delay_exit),
                        thread_offset);
                pools_.push_back(pool);

#else
                throw detail::command_line_error(
                    "Command line option "
                    "--hpx:queuing=static-priority "
                    "is not configured in this build. Please rebuild with "
                    "'cmake -DHPX_WITH_THREAD_SCHEDULERS=static-priority'.");
#endif
                break;
            }

            case 5:    // abp_priority = 5
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
                threads::policies::scheduler_base* sched =
                    new local_sched_type(init);

                // instantiate the pool
                detail::thread_pool* pool =
                    new hpx::threads::detail::thread_pool_impl<
                        local_sched_type>(static_cast<local_sched_type*>(sched),
                        notifier, i, name.c_str(),
                        policies::scheduler_mode(policies::do_background_work |
                            policies::reduce_thread_priority |
                            policies::delay_exit),
                        thread_offset);
                pools_.push_back(pool);

#else
                throw detail::command_line_error(
                    "Command line option "
                    "--hpx:queuing=abp-priority "
                    "is not configured in this build. Please rebuild with "
                    "'cmake -DHPX_WITH_THREAD_SCHEDULERS=abp-priority'.");
#endif
                break;
            }

            case 6:    // hierarchy = 6
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
                threads::policies::scheduler_base* sched =
                    new local_sched_type(init);

                // instantiate the pool
                detail::thread_pool* pool =
                    new hpx::threads::detail::thread_pool_impl<
                        local_sched_type>(static_cast<local_sched_type*>(sched),
                        notifier, i, name.c_str(),
                        policies::scheduler_mode(policies::do_background_work |
                            policies::reduce_thread_priority |
                            policies::delay_exit),
                        thread_offset);
                pools_.push_back(pool);

#else
                throw detail::command_line_error(
                    "Command line option "
                    "--hpx:queuing=hierarchy "
                    "is not configured in this build. Please rebuild with "
                    "'cmake -DHPX_WITH_THREAD_SCHEDULERS=hierarchy'.");
#endif
                break;
            }

            case 7:    // periodic_priority = 7
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
                threads::policies::scheduler_base* sched =
                    new local_sched_type(init);

                // instantiate the pool
                detail::thread_pool* pool =
                    new hpx::threads::detail::thread_pool_impl<
                        local_sched_type>(static_cast<local_sched_type*>(sched),
                        notifier, i, name.c_str(),
                        policies::scheduler_mode(policies::do_background_work |
                            policies::reduce_thread_priority |
                            policies::delay_exit),
                        thread_offset);
                pools_.push_back(pool);

                break;
            }

            case 8:    // throttle = 8
            {
                throw hpx::detail::command_line_error(
                    "Command line option "
                    "--hpx:queuing=throttle "
                    "is not configured in this build. Please rebuild with "
                    "'cmake -DHPX_WITH_THREAD_SCHEDULERS=throttle "
                    "-DHPX_WITH_APEX'.");
                break;
            }
            }
            // update the thread_offset for the next pool
            thread_offset += num_threads_in_pool;
        }

        // fill the thread-lookup table
        for (auto& pool_iter : pools_) {
            size_t nt = rp.get_num_threads(pool_iter->get_pool_name());
            for (size_t i(0); i < nt; i++) {
                threads_lookup_.push_back(pool_iter->get_pool_id());
            }
        }

    }

    threadmanager_impl::~threadmanager_impl()
    {
        for(auto pool_iter : pools_){
            delete pool_iter;
        }
    }

    void threadmanager_impl::init()
    {
        auto& rp = hpx::get_resource_partitioner();
        std::size_t threads_offset = 0;

        // initialize all pools
        for (auto& pool_iter : pools_)
        {
            std::size_t num_threads_in_pool =
                rp.get_num_threads(pool_iter->get_pool_name());
            pool_iter->init(num_threads_in_pool, threads_offset);
            threads_offset += num_threads_in_pool;
        }
    }

    void threadmanager_impl::print_pools()
    {
        std::cout << "The threadmanager owns "
                  << pools_.size() << " pool(s) : \n";
        for(auto pool_iter : pools_){
            pool_iter->print_pool();
        }
    }

    threadmanager_impl::pool_type threadmanager_impl::default_pool() const
    {
        return pools_[0];
    }

/*
    threadmanager_impl::scheduler_type threadmanager_impl::get_scheduler(std::string pool_name) const
    {
        // if the given pool_name is default, we don't need to look for it
        if(pool_name == "default"){
            return default_scheduler();
        }

        auto pool = std::find_if(
                // don't start at begin() since the first one is the default, start one further
                ++pools_.begin(), pools_.end(),
                [&pool_name](std::pair<pool_type,scheduler_type> itp) -> bool {
                    return (itp.first->get_pool_name() == pool_name);}
        );

        if(pool != pools_.end()){
            scheduler_type ret((&(*pool))->second);
            return ret;
        }

        throw std::invalid_argument(
                "the resource partitioner does not own a thread pool named \""
                + pool_name + "\". \n");
    }
*/
    threadmanager_impl::pool_type threadmanager_impl::get_pool(
        std::string pool_name) const
    {
        // if the given pool_name is default, we don't need to look for it
        if (pool_name == "default")
        {
            return default_pool();
        }

        auto pool = std::find_if(
            // don't start at begin() since the first one is the default,
            // start one further
            ++pools_.begin(), pools_.end(),
            [&pool_name](pool_type itp) -> bool {
                return (itp->get_pool_name() == pool_name);
            });

        if (pool != pools_.end())
        {
            return *pool;
        }

        throw std::invalid_argument(
            "the resource partitioner does not own a thread pool named \"" +
            pool_name + "\". \n");
        //! FIXME Add names of available pools?
    }

    threadmanager_impl::pool_type threadmanager_impl::get_pool(
        detail::pool_id_type pool_id) const
    {
        return get_pool(pool_id.name_);
    }

    threadmanager_impl::pool_type threadmanager_impl::get_pool(
        std::size_t thread_index) const
    {
        return get_pool(threads_lookup_[thread_index]);
    }

    ///////////////////////////////////////////////////////////////////////////
    std::int64_t threadmanager_impl::get_thread_count(thread_state_enum state,
        thread_priority priority, std::size_t num_thread, bool reset) const
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
    bool threadmanager_impl::enumerate_threads(
        util::function_nonser<bool(thread_id_type)> const& f,
        thread_state_enum state) const
    {
        std::lock_guard<mutex_type> lk(mtx_);
        bool result = true;

        for(auto& pool_iter : pools_){
            result = result && pool_iter->enumerate_threads(f, state);
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Abort all threads which are in suspended state. This will set
    // the state of all suspended threads to \a pending while
    // supplying the wait_abort extended state flag
    void threadmanager_impl::abort_all_suspended_threads()
    {
        std::lock_guard<mutex_type> lk(mtx_);
        for(auto& pool_iter : pools_) {
            pool_iter->abort_all_suspended_threads();
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Clean up terminated threads. This deletes all threads which
    // have been terminated but which are still held in the queue
    // of terminated threads. Some schedulers might not do anything
    // here.
    bool threadmanager_impl::cleanup_terminated(bool delete_all)
    {
        std::lock_guard<mutex_type> lk(mtx_);
        bool result = true;

        for(auto& pool_iter : pools_) {
            result = result && pool_iter->cleanup_terminated(delete_all);
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////
    void threadmanager_impl::
        register_thread(thread_init_data& data, thread_id_type& id,
            thread_state_enum initial_state, bool run_now, error_code& ec)
    {
        util::block_profiler_wrapper<register_thread_tag> bp(thread_logger_);
        default_pool()->create_thread(data, id, initial_state, run_now, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    void threadmanager_impl::register_work(
        thread_init_data& data, thread_state_enum initial_state, error_code& ec)
    {
        util::block_profiler_wrapper<register_work_tag> bp(work_logger_);
        default_pool()->create_work(data, initial_state, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    // counter creator and discovery functions
/*
    // queue length(s) counter creation function
    naming::gid_type threadmanager_impl::queue_length_counter_creator(
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

        typedef detail::thread_pool spt;

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
            std::size_t(paths.instanceindex_) < pool_->get_os_thread_count())
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
    template <typename Scheduler>
    naming::gid_type threadmanager_impl::
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

        typedef detail::thread_pool<Scheduler> spt;

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
            std::size_t(paths.instanceindex_) < pool_->get_os_thread_count())
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
    template <typename Scheduler>
    naming::gid_type threadmanager_impl::
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

        typedef detail::thread_pool<Scheduler> spt;

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
            std::size_t(paths.instanceindex_) < pool_->get_os_thread_count())
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
    template <typename Scheduler>
    naming::gid_type threadmanager_impl::scheduler_utilization_counter_creator(
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

        typedef detail::thread_pool_impl<Scheduler> spt;

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
    template <typename Scheduler>
    naming::gid_type threadmanager_impl::idle_loop_count_counter_creator(
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

        typedef detail::thread_pool_impl<Scheduler> spt;

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
            std::size_t(paths.instanceindex_) < pool_->get_os_thread_count())
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
    template <typename Scheduler>
    naming::gid_type threadmanager_impl::busy_loop_count_counter_creator(
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

        typedef detail::thread_pool_impl<Scheduler> spt;

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
            std::size_t(paths.instanceindex_) < pool_->get_os_thread_count())
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
    naming::gid_type threadmanager_impl::
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
            std::size_t(paths.instanceindex_) < pool_->get_os_thread_count())
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
    template <typename Scheduler>
    naming::gid_type threadmanager_impl::thread_counts_counter_creator(
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

        typedef detail::thread_pool_impl<Scheduler> spt;
        typedef threadmanager_impl ti;

        using util::placeholders::_1;

        std::size_t shepherd_count = pool_->get_os_thread_count();
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
    template <typename Scheduler>
    void threadmanager_impl::
        register_counter_types()
    {
        using util::placeholders::_1;
        using util::placeholders::_2;

        typedef threadmanager_impl ti;
        performance_counters::create_counter_func counts_creator(
            util::bind(&ti::thread_counts_counter_creator<Scheduler>, this, _1, _2));

        performance_counters::generic_counter_type_data counter_types[] =
        {
            // length of thread queue(s)
            { "/threadqueue/length", performance_counters::counter_raw,
              "returns the current queue length for the referenced queue",
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&ti::queue_length_counter_creator<Scheduler>, this, _1, _2),
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
              util::bind(&ti::scheduler_utilization_counter_creator<Scheduler>, this, _1, _2),
              &performance_counters::locality_counter_discoverer,
              "%"
            },
            // idle-loop count
            { "/scheduler/idle-loop-count/instantaneous",
                    performance_counters::counter_raw,
              "returns the current value of the scheduler idle-loop count",
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&ti::idle_loop_count_counter_creator<Scheduler>, this, _1, _2),
              &performance_counters::locality_thread_counter_discoverer,
              ""
            },
            // busy-loop count
            { "/scheduler/busy-loop-count/instantaneous",
                    performance_counters::counter_raw,
              "returns the current value of the scheduler busy-loop count",
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&ti::busy_loop_count_counter_creator<Scheduler>, this, _1, _2),
              &performance_counters::locality_thread_counter_discoverer,
              ""
            }
        };
        performance_counters::install_counter_types(
            counter_types, sizeof(counter_types)/sizeof(counter_types[0]));
    }*/

    ///////////////////////////////////////////////////////////////////////////

    bool threadmanager_impl::run()
    {
        std::unique_lock<mutex_type> lk(mtx_);
        auto& rp = hpx::get_resource_partitioner();
        std::size_t num_threads(rp.get_num_threads());

        // reset the startup barrier for controlling startup
        HPX_ASSERT(startup_.get() == nullptr);
        startup_.reset(
            new compat::barrier(static_cast<unsigned>(num_threads + 1)));

        // the main thread needs to have a unique thread_num

        // the main thread needs to have a unique thread_num
        // worker threads are numbered 0..N-1, so we can use N for this thread
        init_tss(num_threads);

#ifdef HPX_HAVE_TIMER_POOL
        LTM_(info) << "run: running timer pool";
        timer_pool_.run(false);
#endif

        for (auto& pool_iter : pools_)
        {
            std::size_t num_threads_in_pool =
                rp.get_num_threads(pool_iter->get_pool_name());
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
        }

        // wait for all thread pools to have launched all OS threads
        startup_->wait();

        // set all states of all schedulers to "running"
        for (auto& pool_iter : pools_)
        {
            pool_iter->get_scheduler()->set_all_states(state_running);
        }

        LTM_(info) << "run: running";
        return true;
    }

    void threadmanager_impl::stop(bool blocking)
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

#ifdef HPX_HAVE_THREAD_CUMULATIVE_COUNTS
    std::int64_t threadmanager_impl::get_executed_threads(
        std::size_t num, bool reset)
    {
        std::int64_t result = 0;

        for (auto& pool_iter : pools_)
        {
            result += pool_iter->get_executed_threads(num, reset);
        }

        return result;
    }

    std::int64_t threadmanager_impl::get_executed_thread_phases(
        std::size_t num, bool reset)
    {
        std::int64_t result = 0;

        for (auto& pool_iter : pools_)
        {
            result += pool_iter->get_executed_thread_phases(num, reset);
        }

        return result;
    }

#ifdef HPX_HAVE_THREAD_IDLE_RATES
//     std::int64_t threadmanager_impl::get_thread_phase_duration(
//         std::size_t num, bool reset)
//     {
//         return pool_->get_thread_phase_duration(num, reset);
//     }
//
//     std::int64_t threadmanager_impl::get_thread_duration(
//         std::size_t num, bool reset)
//     {
//         return pool_->get_thread_duration(num, reset);
//     }
//
//     std::int64_t threadmanager_impl::get_thread_phase_overhead(
//         std::size_t num, bool reset)
//     {
//         return pool_->get_thread_phase_overhead(num, reset);
//     }
//
//     std::int64_t threadmanager_impl::get_thread_overhead(
//         std::size_t num, bool reset)
//     {
//         return pool_->get_thread_overhead(num, reset);
//     }
//
//     std::int64_t threadmanager_impl::get_cumulative_thread_duration(
//         std::size_t num, bool reset)
//     {
//         return pool_->get_cumulative_thread_duration(num, reset);
//     }
//
//     std::int64_t threadmanager_impl::get_cumulative_thread_overhead(
//         std::size_t num, bool reset)
//     {
//         return pool_->get_cumulative_thread_overhead(num, reset);
//     }
#endif
#endif

    std::int64_t threadmanager_impl::get_cumulative_duration(
        std::size_t num, bool reset)
    {
        std::int64_t result = 0;
        for (auto& pool_iter : pools_)
        {
            result += pool_iter->get_cumulative_duration(num, reset);
        }
        return result;
    }

#ifdef HPX_HAVE_THREAD_IDLE_RATES
//     ///////////////////////////////////////////////////////////////////////////
//     std::int64_t threadmanager_impl::avg_idle_rate(bool reset)
//     {
//         return pool_->avg_idle_rate(reset);
//     }
//
//     std::int64_t threadmanager_impl::avg_idle_rate(
//         std::size_t num_thread, bool reset)
//     {
//         return pool_->avg_idle_rate(num_thread, reset);
//     }
//
// #if defined(HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES)
//     std::int64_t threadmanager_impl::avg_creation_idle_rate(bool reset)
//     {
//         return pool_->avg_creation_idle_rate(reset);
//     }
//
//     std::int64_t threadmanager_impl::avg_cleanup_idle_rate(bool reset)
//     {
//         return pool_->avg_cleanup_idle_rate(reset);
//     }
// #endif
#endif
} // namespace threads
} // namespace hpx
