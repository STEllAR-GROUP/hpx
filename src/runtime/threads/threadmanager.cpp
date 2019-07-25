//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach, Katelyn Kufahl
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//  Copyright (c) 2015 Patricia Grubel
//  Copyright (c) 2017 Shoshana Jakobovits
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assertion.hpp>
#include <hpx/errors.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/wait_all.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/resource/detail/partitioner.hpp>
#include <hpx/runtime/thread_pool_helpers.hpp>
#include <hpx/runtime/threads/detail/scheduled_thread_pool.hpp>
#include <hpx/runtime/threads/detail/set_thread_state.hpp>
#include <hpx/runtime/threads/executors/current_executor.hpp>
#include <hpx/runtime/threads/policies/schedulers.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/threads/thread_init_data.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime/threads/thread_pool_suspension_helpers.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/util/bind_back.hpp>
#include <hpx/util/bind_front.hpp>
#include <hpx/util/detail/yield_k.hpp>
#include <hpx/hardware/timestamp.hpp>
#include <hpx/util/itt_notify.hpp>
#include <hpx/logging.hpp>
#include <hpx/util/runtime_configuration.hpp>

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
#include <vector>

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

        void ensure_queuing_option_compatibility(
            boost::program_options::variables_map const& vm)
        {
            ensure_high_priority_compatibility(vm);
            ensure_numa_sensitivity_compatibility(vm);
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
                    throw hpx::detail::command_line_error(
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
            if (cfg.affinity_domain_ != "pu")
            {
                affinity_domain = cfg.affinity_domain_;
                if (0 != std::string("pu").find(affinity_domain) &&
                    0 != std::string("core").find(affinity_domain) &&
                    0 != std::string("numa").find(affinity_domain) &&
                    0 != std::string("machine").find(affinity_domain))
                {
                    throw hpx::detail::command_line_error(
                        "Invalid command line option "
                        "--hpx:affinity, value must be one of: pu, core, numa, "
                        "or machine.");
                }
            }
            return affinity_domain;
        }

        std::size_t get_affinity_description(
            util::command_line_handling const& cfg, std::string& affinity_desc)
        {
            if (cfg.affinity_bind_.empty())
                return cfg.numa_sensitive_;

            if (!(cfg.pu_offset_ == std::size_t(-1) ||
                    cfg.pu_offset_ == std::size_t(0)) ||
                cfg.pu_step_ != 1 || cfg.affinity_domain_ != "pu")
            {
                throw hpx::detail::command_line_error(
                    "Command line option --hpx:bind "
                    "should not be used with --hpx:pu-step, --hpx:pu-offset, "
                    "or --hpx:affinity.");
            }

            affinity_desc = cfg.affinity_bind_;
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
        if (state < unknown || state > pending_boost)
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
            "high (recursive)",
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
        notification_policy_type& notifier,
        detail::network_background_callback_type network_background_callback)
      : num_threads_(hpx::resource::get_partitioner().get_num_distinct_pus())
#ifdef HPX_HAVE_TIMER_POOL
      , timer_pool_(timer_pool)
#endif
      , notifier_(notifier)
      , network_background_callback_(network_background_callback)
    {
    }

    void threadmanager::create_pools()
    {
        auto& rp = hpx::resource::get_partitioner();
        size_t num_pools = rp.get_num_pools();
        util::command_line_handling const& cfg_ = rp.get_command_line_switches();
        std::size_t thread_offset = 0;

        // instantiate the pools
        for (size_t i = 0; i != num_pools; i++)
        {
            std::string name = rp.get_pool_name(i);
            resource::scheduling_policy sched_type = rp.which_scheduler(name);
            std::size_t num_threads_in_pool = rp.get_num_threads(i);
            policies::scheduler_mode scheduler_mode = rp.get_scheduler_mode(i);

            // make sure the first thread-pool that gets instantiated is the default one
            if (i == 0)
            {
                if (name != rp.get_default_pool_name())
                {
                    throw std::invalid_argument("Trying to instantiate pool " +
                        name +
                        " as first thread pool, but first thread pool must be "
                        "named " + rp.get_default_pool_name());
                }
            }

            switch (sched_type)
            {
            case resource::user_defined:
            {
                auto pool_func = rp.get_pool_creator(i);
                std::unique_ptr<thread_pool_base> pool(
                    pool_func(notifier_, num_threads_in_pool,
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
                std::unique_ptr<thread_pool_base> pool(
                    new hpx::threads::detail::scheduled_thread_pool<
                        local_sched_type>(std::move(sched), notifier_, i,
                        name.c_str(), scheduler_mode, thread_offset,
                        network_background_callback_));
                pools_.push_back(std::move(pool));
#else
                throw hpx::detail::command_line_error(
                    "Command line option --hpx:queuing=local "
                    "is not configured in this build. Please rebuild with "
                    "'cmake -DHPX_WITH_THREAD_SCHEDULERS=local'.");
#endif
                break;
            }

            case resource::local_priority_fifo:
            {
                // set parameters for scheduler and pool instantiation and
                // perform compatibility checks
                std::size_t num_high_priority_queues =
                    hpx::detail::get_num_high_priority_queues(
                        cfg_, rp.get_num_threads(name));
                std::string affinity_desc;
                std::size_t numa_sensitive =
                    hpx::detail::get_affinity_description(cfg_, affinity_desc);

                // instantiate the scheduler
                typedef hpx::threads::policies::local_priority_queue_scheduler<
                    std::mutex, hpx::threads::policies::lockfree_fifo>
                    local_sched_type;
                local_sched_type::init_parameter_type init(num_threads_in_pool,
                    num_high_priority_queues, 1000, numa_sensitive,
                    "core-local_priority_queue_scheduler");
                std::unique_ptr<local_sched_type> sched(
                    new local_sched_type(init));

                // instantiate the pool
                std::unique_ptr<thread_pool_base> pool(
                    new hpx::threads::detail::scheduled_thread_pool<
                        local_sched_type>(std::move(sched), notifier_, i,
                        name.c_str(), scheduler_mode, thread_offset,
                        network_background_callback_));
                pools_.push_back(std::move(pool));

                break;
            }

            case resource::local_priority_lifo:
            {
#if defined(HPX_HAVE_CXX11_STD_ATOMIC_128BIT)
                // set parameters for scheduler and pool instantiation and
                // perform compatibility checks
                std::size_t num_high_priority_queues =
                    hpx::detail::get_num_high_priority_queues(
                        cfg_, rp.get_num_threads(name));
                std::string affinity_desc;
                std::size_t numa_sensitive =
                    hpx::detail::get_affinity_description(cfg_, affinity_desc);

                // instantiate the scheduler
                typedef hpx::threads::policies::local_priority_queue_scheduler<
                    std::mutex, hpx::threads::policies::lockfree_lifo>
                    local_sched_type;
                local_sched_type::init_parameter_type init(num_threads_in_pool,
                    num_high_priority_queues, 1000, numa_sensitive,
                    "core-local_priority_queue_scheduler");
                std::unique_ptr<local_sched_type> sched(
                    new local_sched_type(init));

                // instantiate the pool
                std::unique_ptr<thread_pool_base> pool(
                    new hpx::threads::detail::scheduled_thread_pool<
                        local_sched_type>(std::move(sched), notifier_, i,
                        name.c_str(), scheduler_mode, thread_offset,
                        network_background_callback_));
                pools_.push_back(std::move(pool));
#else
                throw hpx::detail::command_line_error(
                    "Command line option --hpx:queuing=local-priority-lifo "
                    "is not configured in this build. Please rebuild with "
                    "'cmake -DHPX_WITH_THREAD_SCHEDULERS=local-priority-lifo'. "
                    "Additionally, please make sure 128bit atomics are "
                    "available.");
#endif
                break;
            }

            case resource::static_:
            {
#if defined(HPX_HAVE_STATIC_SCHEDULER)
                // set parameters for scheduler and pool instantiation and
                // perform compatibility checks
                hpx::detail::ensure_high_priority_compatibility(cfg_.vm_);
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
                std::unique_ptr<thread_pool_base> pool(
                    new hpx::threads::detail::scheduled_thread_pool<
                        local_sched_type>(std::move(sched), notifier_, i,
                        name.c_str(), scheduler_mode, thread_offset,
                        network_background_callback_));
                pools_.push_back(std::move(pool));
#else
                throw hpx::detail::command_line_error(
                    "Command line option --hpx:queuing=static "
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
                std::size_t num_high_priority_queues =
                    hpx::detail::get_num_high_priority_queues(
                        cfg_, rp.get_num_threads(name));
                std::string affinity_domain =
                    hpx::detail::get_affinity_domain(cfg_);
                std::string affinity_desc;
                std::size_t numa_sensitive =
                    hpx::detail::get_affinity_description(cfg_, affinity_desc);

                // instantiate the scheduler
                using local_sched_type =
                    hpx::threads::policies::static_priority_queue_scheduler<>;
                local_sched_type::init_parameter_type init(num_threads_in_pool,
                    num_high_priority_queues, 1000, numa_sensitive,
                    "core-static_priority_queue_scheduler");
                std::unique_ptr<local_sched_type> sched(
                    new local_sched_type(init));

                // instantiate the pool
                std::unique_ptr<thread_pool_base> pool(
                    new hpx::threads::detail::scheduled_thread_pool<
                        local_sched_type>(std::move(sched), notifier_, i,
                        name.c_str(), scheduler_mode, thread_offset,
                        network_background_callback_));
                pools_.push_back(std::move(pool));
#else
                throw hpx::detail::command_line_error(
                    "Command line option --hpx:queuing=static-priority "
                    "is not configured in this build. Please rebuild with "
                    "'cmake -DHPX_WITH_THREAD_SCHEDULERS=static-priority'.");
#endif
                break;
            }

            case resource::abp_priority_fifo:
            {
#if defined(HPX_HAVE_ABP_SCHEDULER) && defined(HPX_HAVE_CXX11_STD_ATOMIC_128BIT)
                // set parameters for scheduler and pool instantiation and
                // perform compatibility checks
                std::size_t num_high_priority_queues =
                    hpx::detail::get_num_high_priority_queues(
                        cfg_, rp.get_num_threads(name));

                // instantiate the scheduler
                typedef hpx::threads::policies::local_priority_queue_scheduler<
                    std::mutex, hpx::threads::policies::lockfree_fifo>
                    local_sched_type;
                local_sched_type::init_parameter_type init(num_threads_in_pool,
                    num_high_priority_queues, 1000, cfg_.numa_sensitive_,
                    "core-abp_fifo_priority_queue_scheduler");
                std::unique_ptr<local_sched_type> sched(
                    new local_sched_type(init));

                // instantiate the pool
                std::unique_ptr<thread_pool_base> pool(
                    new hpx::threads::detail::scheduled_thread_pool<
                        local_sched_type>(std::move(sched), notifier_, i,
                        name.c_str(), scheduler_mode, thread_offset,
                        network_background_callback_));
                pools_.push_back(std::move(pool));
#else
                throw hpx::detail::command_line_error(
                    "Command line option --hpx:queuing=abp-priority-fifo "
                    "is not configured in this build. Please rebuild with "
                    "'cmake -DHPX_WITH_THREAD_SCHEDULERS=abp-priority'. "
                    "Additionally, please make sure 128bit atomics are "
                    "available.");
#endif
                break;
            }

            case resource::abp_priority_lifo:
            {
#if defined(HPX_HAVE_ABP_SCHEDULER) && defined(HPX_HAVE_CXX11_STD_ATOMIC_128BIT)
                // set parameters for scheduler and pool instantiation and
                // perform compatibility checks
                std::size_t num_high_priority_queues =
                    hpx::detail::get_num_high_priority_queues(
                        cfg_, rp.get_num_threads(name));

                // instantiate the scheduler
                typedef hpx::threads::policies::local_priority_queue_scheduler<
                    std::mutex, hpx::threads::policies::lockfree_lifo>
                    local_sched_type;
                local_sched_type::init_parameter_type init(num_threads_in_pool,
                    num_high_priority_queues, 1000, cfg_.numa_sensitive_,
                    "core-abp_fifo_priority_queue_scheduler");
                std::unique_ptr<local_sched_type> sched(
                    new local_sched_type(init));

                // instantiate the pool
                std::unique_ptr<thread_pool_base> pool(
                    new hpx::threads::detail::scheduled_thread_pool<
                        local_sched_type>(std::move(sched), notifier_, i,
                        name.c_str(), scheduler_mode, thread_offset,
                        network_background_callback_));
                pools_.push_back(std::move(pool));
#else
                throw hpx::detail::command_line_error(
                    "Command line option --hpx:queuing=abp-priority-lifo "
                    "is not configured in this build. Please rebuild with "
                    "'cmake -DHPX_WITH_THREAD_SCHEDULERS=abp-priority'.");
#endif
                break;
            }

            case resource::shared_priority:
            {
#if defined(HPX_HAVE_SHARED_PRIORITY_SCHEDULER)
                // instantiate the scheduler
                typedef hpx::threads::policies::shared_priority_queue_scheduler<>
                    local_sched_type;
                hpx::threads::policies::core_ratios ratios(4, 4, 64);
                std::unique_ptr<local_sched_type> sched(
                    new local_sched_type(num_threads_in_pool, ratios,
                        "core-shared_priority_queue_scheduler"));

                // instantiate the pool
                std::unique_ptr<thread_pool_base> pool(
                    new hpx::threads::detail::scheduled_thread_pool<
                        local_sched_type>(std::move(sched), notifier_, i,
                        name.c_str(), scheduler_mode, thread_offset,
                        network_background_callback_));
                pools_.push_back(std::move(pool));
#else
                throw hpx::detail::command_line_error(
                    "Command line option --hpx:queuing=shared-priority "
                    "is not configured in this build. Please rebuild with "
                    "'cmake -DHPX_WITH_THREAD_SCHEDULERS=shared-priority'.");
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
            std::size_t nt = rp.get_num_threads(pool_iter->get_pool_index());
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
        auto& rp = hpx::resource::get_partitioner();
        std::size_t threads_offset = 0;

        // initialize all pools
        for (auto && pool_iter : pools_)
        {
            std::size_t num_threads_in_pool =
                rp.get_num_threads(pool_iter->get_pool_index());
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

    thread_pool_base& threadmanager::default_pool() const
    {
        HPX_ASSERT(!pools_.empty());
        return *pools_[0];
    }

    thread_pool_base& threadmanager::get_pool(
        std::string const& pool_name) const
    {
        // if the given pool_name is default, we don't need to look for it
        // we must always return pool 0
        if (pool_name == "default" ||
            pool_name == resource::get_partitioner().get_default_pool_name())
        {
            return default_pool();
        }

        // now check the other pools - no need to check pool 0 again, so ++begin
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

    thread_pool_base& threadmanager::get_pool(
        pool_id_type pool_id) const
    {
        return get_pool(pool_id.name());
    }

    thread_pool_base& threadmanager::get_pool(
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

    std::int64_t threadmanager::get_background_thread_count()
    {
        std::int64_t total_count = 0;
        std::lock_guard<mutex_type> lk(mtx_);

        for (auto& pool_iter : pools_)
        {
            total_count +=
                pool_iter->get_background_thread_count();
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
            result = pool_iter->cleanup_terminated(delete_all) && result;
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////
    void threadmanager::register_thread(thread_init_data& data,
        thread_id_type& id, thread_state_enum initial_state, bool run_now,
        error_code& ec)
    {
        thread_pool_base *pool = nullptr;
        if (get_self_ptr())
        {
            auto tid = get_self_id();
            pool = tid->get_scheduler_base()->get_parent_pool();
        }
        else
        {
            pool = &default_pool();
        }
        pool->create_thread(data, id, initial_state, run_now, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    void threadmanager::register_work(
        thread_init_data& data, thread_state_enum initial_state, error_code& ec)
    {
        thread_pool_base *pool = nullptr;
        if (get_self_ptr())
        {
            auto tid = get_self_id();
            pool = tid->get_scheduler_base()->get_parent_pool();
        }
        else
        {
            pool = &default_pool();
        }
        pool->create_work(data, initial_state, ec);
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

#if defined(HPX_HAVE_BACKGROUND_THREAD_COUNTERS) && defined(HPX_HAVE_THREAD_IDLE_RATES)
    std::int64_t threadmanager::get_background_work_duration(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->get_background_work_duration(all_threads, reset);
        return result;
    }

    std::int64_t threadmanager::get_background_overhead(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->get_background_overhead(all_threads, reset);
        return result;
    }

    std::int64_t threadmanager::get_background_send_duration(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->get_background_send_duration(all_threads, reset);
        return result;
    }

    std::int64_t threadmanager::get_background_send_overhead(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->get_background_send_overhead(all_threads, reset);
        return result;
    }

    std::int64_t threadmanager::get_background_receive_duration(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result +=
                pool_iter->get_background_receive_duration(all_threads, reset);
        return result;
    }

    std::int64_t threadmanager::get_background_receive_overhead(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result +=
                pool_iter->get_background_receive_overhead(all_threads, reset);
        return result;
    }
#endif    // HPX_HAVE_BACKGROUND_THREAD_COUNTERS

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
    std::size_t threadmanager::shrink_pool(std::string const& pool_name)
    {
        return resource::get_partitioner().shrink_pool(
            pool_name,
            [this, &pool_name](std::size_t virt_core)
            {
                get_pool(pool_name).remove_processing_unit(virt_core);
            });
    }

    std::size_t threadmanager::expand_pool(std::string const& pool_name)
    {
        return resource::get_partitioner().expand_pool(
            pool_name,
            [this, &pool_name](std::size_t virt_core)
            {
                thread_pool_base& pool = get_pool(pool_name);
                pool.add_processing_unit(virt_core,
                    pool.get_thread_offset() + virt_core);
            });
    }

    ///////////////////////////////////////////////////////////////////////////
    bool threadmanager::run()
    {
        std::unique_lock<mutex_type> lk(mtx_);

        // the main thread needs to have a unique thread_num
        // worker threads are numbered 0..N-1, so we can use N for this thread
        auto& rp = hpx::resource::get_partitioner();
        init_tss(rp.get_num_threads());

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

            if (!pool_iter->run(lk, num_threads_in_pool))
            {
#ifdef HPX_HAVE_TIMER_POOL
                timer_pool_.stop();
#endif
                return false;
            }

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
    }

    void threadmanager::suspend()
    {
        if (threads::get_self_ptr())
        {
            std::vector<hpx::future<void>> fs;

            for (auto& pool_iter : pools_)
            {
                fs.push_back(suspend_pool(*pool_iter));
            }

            hpx::wait_all(fs);
        }
        else
        {
            for (auto& pool_iter : pools_)
            {
                pool_iter->suspend_direct();
            }
        }
    }

    void threadmanager::resume()
    {
        if (threads::get_self_ptr())
        {
            std::vector<hpx::future<void>> fs;

            for (auto& pool_iter : pools_)
            {
                fs.push_back(resume_pool(*pool_iter));
            }
            hpx::wait_all(fs);
        }
        else
        {
            for (auto& pool_iter : pools_)
            {
                pool_iter->resume_direct();
            }
        }
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
