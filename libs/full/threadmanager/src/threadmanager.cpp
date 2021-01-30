//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach, Katelyn Kufahl
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//  Copyright (c) 2015 Patricia Grubel
//  Copyright (c) 2017 Shoshana Jakobovits
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_combinators/wait_all.hpp>
#include <hpx/execution_base/this_thread.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/hardware/timestamp.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/modules/schedulers.hpp>
#include <hpx/modules/threadmanager.hpp>
#include <hpx/resource_partitioner/detail/partitioner.hpp>
#include <hpx/runtime/threads/thread_pool_suspension_helpers.hpp>
#include <hpx/runtime_configuration/runtime_configuration.hpp>
#include <hpx/thread_pools/scheduled_thread_pool.hpp>
#include <hpx/threading_base/set_thread_state.hpp>
#include <hpx/threading_base/thread_data.hpp>
#include <hpx/threading_base/thread_helpers.hpp>
#include <hpx/threading_base/thread_init_data.hpp>
#include <hpx/threading_base/thread_queue_init_parameters.hpp>
#include <hpx/topology/topology.hpp>
#include <hpx/type_support/unused.hpp>
#include <hpx/util/get_entry_as.hpp>

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

namespace hpx { namespace threads {
    namespace detail {
        void check_num_high_priority_queues(
            std::size_t num_threads, std::size_t num_high_priority_queues)
        {
            if (num_high_priority_queues > num_threads)
            {
                throw hpx::detail::command_line_error(
                    "Invalid command line option: "
                    "number of high priority threads ("
                    "--hpx:high-priority-threads), should not be larger "
                    "than number of threads (--hpx:threads)");
            }
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    threadmanager::threadmanager(util::runtime_configuration& rtcfg,
#ifdef HPX_HAVE_TIMER_POOL
        util::io_service_pool& timer_pool,
#endif
        notification_policy_type& notifier,
        detail::network_background_callback_type network_background_callback)
      : rtcfg_(rtcfg)
#ifdef HPX_HAVE_TIMER_POOL
      , timer_pool_(timer_pool)
#endif
      , notifier_(notifier)
      , network_background_callback_(network_background_callback)
    {
        using util::placeholders::_1;
        using util::placeholders::_3;

        // Add callbacks local to threadmanager.
        notifier.add_on_start_thread_callback(
            util::bind(&threadmanager::init_tss, this, _1));
        notifier.add_on_stop_thread_callback(
            util::bind(&threadmanager::deinit_tss, this));

        auto& rp = hpx::resource::get_partitioner();
        notifier.add_on_start_thread_callback(util::bind(
            &resource::detail::partitioner::assign_pu, std::ref(rp), _3, _1));
        notifier.add_on_stop_thread_callback(util::bind(
            &resource::detail::partitioner::unassign_pu, std::ref(rp), _3, _1));
    }

    void threadmanager::create_pools()
    {
        auto& rp = hpx::resource::get_partitioner();
        size_t num_pools = rp.get_num_pools();
        std::size_t thread_offset = 0;

        std::size_t max_background_threads =
            hpx::util::get_entry_as<std::size_t>(rtcfg_,
                "hpx.max_background_threads",
                (std::numeric_limits<std::size_t>::max)());
        std::size_t const max_idle_loop_count =
            hpx::util::get_entry_as<std::int64_t>(
                rtcfg_, "hpx.max_idle_loop_count", HPX_IDLE_LOOP_COUNT_MAX);
        std::size_t const max_busy_loop_count =
            hpx::util::get_entry_as<std::int64_t>(
                rtcfg_, "hpx.max_busy_loop_count", HPX_BUSY_LOOP_COUNT_MAX);

        std::int64_t const max_thread_count =
            hpx::util::get_entry_as<std::int64_t>(rtcfg_,
                "hpx.thread_queue.max_thread_count",
                HPX_THREAD_QUEUE_MAX_THREAD_COUNT);
        std::int64_t const min_tasks_to_steal_pending =
            hpx::util::get_entry_as<std::int64_t>(rtcfg_,
                "hpx.thread_queue.min_tasks_to_steal_pending",
                HPX_THREAD_QUEUE_MIN_TASKS_TO_STEAL_PENDING);
        std::int64_t const min_tasks_to_steal_staged =
            hpx::util::get_entry_as<std::int64_t>(rtcfg_,
                "hpx.thread_queue.min_tasks_to_steal_staged",
                HPX_THREAD_QUEUE_MIN_TASKS_TO_STEAL_STAGED);
        std::int64_t const min_add_new_count =
            hpx::util::get_entry_as<std::int64_t>(rtcfg_,
                "hpx.thread_queue.min_add_new_count",
                HPX_THREAD_QUEUE_MIN_ADD_NEW_COUNT);
        std::int64_t const max_add_new_count =
            hpx::util::get_entry_as<std::int64_t>(rtcfg_,
                "hpx.thread_queue.max_add_new_count",
                HPX_THREAD_QUEUE_MAX_ADD_NEW_COUNT);
        std::int64_t const min_delete_count =
            hpx::util::get_entry_as<std::int64_t>(rtcfg_,
                "hpx.thread_queue.min_delete_count",
                HPX_THREAD_QUEUE_MIN_DELETE_COUNT);
        std::int64_t const max_delete_count =
            hpx::util::get_entry_as<std::int64_t>(rtcfg_,
                "hpx.thread_queue.max_delete_count",
                HPX_THREAD_QUEUE_MAX_DELETE_COUNT);
        std::int64_t const max_terminated_threads =
            hpx::util::get_entry_as<std::int64_t>(rtcfg_,
                "hpx.thread_queue.max_terminated_threads",
                HPX_THREAD_QUEUE_MAX_TERMINATED_THREADS);
        double const max_idle_backoff_time = hpx::util::get_entry_as<double>(
            rtcfg_, "hpx.max_idle_backoff_time", HPX_IDLE_BACKOFF_TIME_MAX);

        std::ptrdiff_t small_stacksize =
            rtcfg_.get_stack_size(thread_stacksize::small_);
        std::ptrdiff_t medium_stacksize =
            rtcfg_.get_stack_size(thread_stacksize::medium);
        std::ptrdiff_t large_stacksize =
            rtcfg_.get_stack_size(thread_stacksize::large);
        std::ptrdiff_t huge_stacksize =
            rtcfg_.get_stack_size(thread_stacksize::huge);

        policies::thread_queue_init_parameters thread_queue_init(
            max_thread_count, min_tasks_to_steal_pending,
            min_tasks_to_steal_staged, min_add_new_count, max_add_new_count,
            min_delete_count, max_delete_count, max_terminated_threads,
            max_idle_backoff_time, small_stacksize, medium_stacksize,
            large_stacksize, huge_stacksize);

        if (!rtcfg_.enable_networking())
        {
            max_background_threads = 0;
        }

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
                        " as first thread pool, but first thread pool must "
                        "be named " +
                        rp.get_default_pool_name());
                }
            }

            thread_pool_init_parameters thread_pool_init(name, i,
                scheduler_mode, num_threads_in_pool, thread_offset, notifier_,
                rp.get_affinity_data(), network_background_callback_,
                max_background_threads, max_idle_loop_count,
                max_busy_loop_count);

            std::size_t numa_sensitive = hpx::util::get_entry_as<std::size_t>(
                rtcfg_, "hpx.numa_sensitive", 0);

            switch (sched_type)
            {
            case resource::user_defined:
            {
                auto pool_func = rp.get_pool_creator(i);
                std::unique_ptr<thread_pool_base> pool(
                    pool_func(thread_pool_init, thread_queue_init));
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
                // instantiate the scheduler
                using local_sched_type =
                    hpx::threads::policies::local_queue_scheduler<>;

                local_sched_type::init_parameter_type init(
                    thread_pool_init.num_threads_,
                    thread_pool_init.affinity_data_, thread_queue_init,
                    "core-local_queue_scheduler");

                std::unique_ptr<local_sched_type> sched(
                    new local_sched_type(init));

                // set the default scheduler flags
                sched->set_scheduler_mode(thread_pool_init.mode_);
                // conditionally set/unset this flag
                sched->update_scheduler_mode(
                    policies::enable_stealing_numa, !numa_sensitive);

                // instantiate the pool
                std::unique_ptr<thread_pool_base> pool(
                    new hpx::threads::detail::scheduled_thread_pool<
                        local_sched_type>(std::move(sched), thread_pool_init));
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
                    hpx::util::get_entry_as<std::size_t>(rtcfg_,
                        "hpx.thread_queue.high_priority_queues",
                        thread_pool_init.num_threads_);
                detail::check_num_high_priority_queues(
                    thread_pool_init.num_threads_, num_high_priority_queues);

                // instantiate the scheduler
                using local_sched_type =
                    hpx::threads::policies::local_priority_queue_scheduler<
                        std::mutex, hpx::threads::policies::lockfree_fifo>;

                local_sched_type::init_parameter_type init(
                    thread_pool_init.num_threads_,
                    thread_pool_init.affinity_data_, num_high_priority_queues,
                    thread_queue_init, "core-local_priority_queue_scheduler");

                std::unique_ptr<local_sched_type> sched(
                    new local_sched_type(init));

                // set the default scheduler flags
                sched->set_scheduler_mode(thread_pool_init.mode_);
                // conditionally set/unset this flag
                sched->update_scheduler_mode(
                    policies::enable_stealing_numa, !numa_sensitive);

                // instantiate the pool
                std::unique_ptr<thread_pool_base> pool(
                    new hpx::threads::detail::scheduled_thread_pool<
                        local_sched_type>(std::move(sched), thread_pool_init));
                pools_.push_back(std::move(pool));

                break;
            }

            case resource::local_priority_lifo:
            {
#if defined(HPX_HAVE_CXX11_STD_ATOMIC_128BIT)
                // set parameters for scheduler and pool instantiation and
                // perform compatibility checks
                std::size_t num_high_priority_queues =
                    hpx::util::get_entry_as<std::size_t>(rtcfg_,
                        "hpx.thread_queue.high_priority_queues",
                        thread_pool_init.num_threads_);
                detail::check_num_high_priority_queues(
                    thread_pool_init.num_threads_, num_high_priority_queues);

                // instantiate the scheduler
                using local_sched_type =
                    hpx::threads::policies::local_priority_queue_scheduler<
                        std::mutex, hpx::threads::policies::lockfree_lifo>;

                local_sched_type::init_parameter_type init(
                    thread_pool_init.num_threads_,
                    thread_pool_init.affinity_data_, num_high_priority_queues,
                    thread_queue_init, "core-local_priority_queue_scheduler");

                std::unique_ptr<local_sched_type> sched(
                    new local_sched_type(init));

                // set the default scheduler flags
                sched->set_scheduler_mode(thread_pool_init.mode_);
                // conditionally set/unset this flag
                sched->update_scheduler_mode(
                    policies::enable_stealing_numa, !numa_sensitive);

                // instantiate the pool
                std::unique_ptr<thread_pool_base> pool(
                    new hpx::threads::detail::scheduled_thread_pool<
                        local_sched_type>(std::move(sched), thread_pool_init));
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
                // instantiate the scheduler
                using local_sched_type =
                    hpx::threads::policies::static_queue_scheduler<>;

                local_sched_type::init_parameter_type init(
                    thread_pool_init.num_threads_,
                    thread_pool_init.affinity_data_, thread_queue_init,
                    "core-static_queue_scheduler");

                std::unique_ptr<local_sched_type> sched(
                    new local_sched_type(init));

                // set the default scheduler flags
                sched->set_scheduler_mode(thread_pool_init.mode_);
                // conditionally set/unset this flag
                sched->update_scheduler_mode(
                    policies::enable_stealing_numa, !numa_sensitive);

                // instantiate the pool
                std::unique_ptr<thread_pool_base> pool(
                    new hpx::threads::detail::scheduled_thread_pool<
                        local_sched_type>(std::move(sched), thread_pool_init));
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
                    hpx::util::get_entry_as<std::size_t>(rtcfg_,
                        "hpx.thread_queue.high_priority_queues",
                        thread_pool_init.num_threads_);
                detail::check_num_high_priority_queues(
                    thread_pool_init.num_threads_, num_high_priority_queues);

                // instantiate the scheduler
                using local_sched_type =
                    hpx::threads::policies::static_priority_queue_scheduler<>;

                local_sched_type::init_parameter_type init(
                    thread_pool_init.num_threads_,
                    thread_pool_init.affinity_data_, num_high_priority_queues,
                    thread_queue_init, "core-static_priority_queue_scheduler");

                std::unique_ptr<local_sched_type> sched(
                    new local_sched_type(init));

                // set the default scheduler flags
                sched->set_scheduler_mode(thread_pool_init.mode_);
                // conditionally set/unset this flag
                sched->update_scheduler_mode(
                    policies::enable_stealing_numa, !numa_sensitive);

                // instantiate the pool
                std::unique_ptr<thread_pool_base> pool(
                    new hpx::threads::detail::scheduled_thread_pool<
                        local_sched_type>(std::move(sched), thread_pool_init));
                pools_.push_back(std::move(pool));
#else
                throw hpx::detail::command_line_error(
                    "Command line option --hpx:queuing=static-priority "
                    "is not configured in this build. Please rebuild with "
                    "'cmake "
                    "-DHPX_WITH_THREAD_SCHEDULERS=static-priority'.");
#endif
                break;
            }

            case resource::abp_priority_fifo:
            {
#if defined(HPX_HAVE_ABP_SCHEDULER) && defined(HPX_HAVE_CXX11_STD_ATOMIC_128BIT)
                // set parameters for scheduler and pool instantiation and
                // perform compatibility checks
                std::size_t num_high_priority_queues =
                    hpx::util::get_entry_as<std::size_t>(rtcfg_,
                        "hpx.thread_queue.high_priority_queues",
                        thread_pool_init.num_threads_);
                detail::check_num_high_priority_queues(
                    thread_pool_init.num_threads_, num_high_priority_queues);

                // instantiate the scheduler
                using local_sched_type =
                    hpx::threads::policies::local_priority_queue_scheduler<
                        std::mutex, hpx::threads::policies::lockfree_fifo>;

                local_sched_type::init_parameter_type init(
                    thread_pool_init.num_threads_,
                    thread_pool_init.affinity_data_, num_high_priority_queues,
                    thread_queue_init,
                    "core-abp_fifo_priority_queue_scheduler");

                std::unique_ptr<local_sched_type> sched(
                    new local_sched_type(init));

                // set the default scheduler flags
                sched->set_scheduler_mode(thread_pool_init.mode_);
                // conditionally set/unset this flag
                sched->update_scheduler_mode(
                    policies::enable_stealing_numa, !numa_sensitive);

                // instantiate the pool
                std::unique_ptr<thread_pool_base> pool(
                    new hpx::threads::detail::scheduled_thread_pool<
                        local_sched_type>(std::move(sched), thread_pool_init));
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
                    hpx::util::get_entry_as<std::size_t>(rtcfg_,
                        "hpx.thread_queue.high_priority_queues",
                        thread_pool_init.num_threads_);
                detail::check_num_high_priority_queues(
                    thread_pool_init.num_threads_, num_high_priority_queues);

                // instantiate the scheduler
                using local_sched_type =
                    hpx::threads::policies::local_priority_queue_scheduler<
                        std::mutex, hpx::threads::policies::lockfree_lifo>;

                local_sched_type::init_parameter_type init(
                    thread_pool_init.num_threads_,
                    thread_pool_init.affinity_data_, num_high_priority_queues,
                    thread_queue_init,
                    "core-abp_fifo_priority_queue_scheduler");

                std::unique_ptr<local_sched_type> sched(
                    new local_sched_type(init));

                // set the default scheduler flags
                sched->set_scheduler_mode(thread_pool_init.mode_);
                // conditionally set/unset this flag
                sched->update_scheduler_mode(
                    policies::enable_stealing_numa, !numa_sensitive);

                // instantiate the pool
                std::unique_ptr<thread_pool_base> pool(
                    new hpx::threads::detail::scheduled_thread_pool<
                        local_sched_type>(std::move(sched), thread_pool_init));
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
                typedef hpx::threads::policies::
                    shared_priority_queue_scheduler<>
                        local_sched_type;
                local_sched_type::init_parameter_type init(
                    thread_pool_init.num_threads_, {1, 1, 1},
                    thread_pool_init.affinity_data_, thread_queue_init,
                    "core-shared_priority_queue_scheduler");

                std::unique_ptr<local_sched_type> sched(
                    new local_sched_type(init));

                // set the default scheduler flags
                sched->set_scheduler_mode(thread_pool_init.mode_);
                // conditionally set/unset this flag
                sched->update_scheduler_mode(
                    policies::enable_stealing_numa, !numa_sensitive);

                // instantiate the pool
                std::unique_ptr<thread_pool_base> pool(
                    new hpx::threads::detail::scheduled_thread_pool<
                        local_sched_type>(std::move(sched), thread_pool_init));
                pools_.push_back(std::move(pool));
#else
                throw hpx::detail::command_line_error(
                    "Command line option --hpx:queuing=shared-priority "
                    "is not configured in this build. Please rebuild with "
                    "'cmake "
                    "-DHPX_WITH_THREAD_SCHEDULERS=shared-priority'.");
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

    threadmanager::~threadmanager() {}

    void threadmanager::init()
    {
        auto& rp = hpx::resource::get_partitioner();
        std::size_t threads_offset = 0;

        // initialize all pools
        for (auto&& pool_iter : pools_)
        {
            std::size_t num_threads_in_pool =
                rp.get_num_threads(pool_iter->get_pool_index());
            pool_iter->init(num_threads_in_pool, threads_offset);
            threads_offset += num_threads_in_pool;
        }
    }

    void threadmanager::print_pools(std::ostream& os)
    {
        os << "The thread-manager owns " << pools_.size()    //  -V128
           << " pool(s) : \n";

        for (auto&& pool_iter : pools_)
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
        auto pool = std::find_if(++pools_.begin(), pools_.end(),
            [&pool_name](pool_type const& itp) -> bool {
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

    thread_pool_base& threadmanager::get_pool(pool_id_type const& pool_id) const
    {
        return get_pool(pool_id.name());
    }

    thread_pool_base& threadmanager::get_pool(std::size_t thread_index) const
    {
        return get_pool(threads_lookup_[thread_index]);
    }

    bool threadmanager::pool_exists(std::string const& pool_name) const
    {
        // if the given pool_name is default, we don't need to look for it
        // we must always return pool 0
        if (pool_name == "default" ||
            pool_name == resource::get_partitioner().get_default_pool_name())
        {
            return true;
        }

        // now check the other pools - no need to check pool 0 again, so ++begin
        auto pool = std::find_if(++pools_.begin(), pools_.end(),
            [&pool_name](pool_type const& itp) -> bool {
                return (itp->get_pool_name() == pool_name);
            });

        if (pool != pools_.end())
        {
            return true;
        }

        return false;
    }

    bool threadmanager::pool_exists(std::size_t pool_index) const
    {
        return pool_index < pools_.size();
    }

    ///////////////////////////////////////////////////////////////////////////
    std::int64_t threadmanager::get_thread_count(thread_schedule_state state,
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

    std::int64_t threadmanager::get_idle_core_count()
    {
        std::int64_t total_count = 0;
        std::lock_guard<mutex_type> lk(mtx_);

        for (auto& pool_iter : pools_)
        {
            total_count += pool_iter->get_idle_core_count();
        }

        return total_count;
    }

    mask_type threadmanager::get_idle_core_mask()
    {
        mask_type mask = mask_type();
        resize(mask, hardware_concurrency());

        std::lock_guard<mutex_type> lk(mtx_);

        for (auto& pool_iter : pools_)
        {
            pool_iter->get_idle_core_mask(mask);
        }

        return mask;
    }

    std::int64_t threadmanager::get_background_thread_count()
    {
        std::int64_t total_count = 0;
        std::lock_guard<mutex_type> lk(mtx_);

        for (auto& pool_iter : pools_)
        {
            total_count += pool_iter->get_background_thread_count();
        }

        return total_count;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Enumerate all matching threads
    bool threadmanager::enumerate_threads(
        util::function_nonser<bool(thread_id_type)> const& f,
        thread_schedule_state state) const
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
    void threadmanager::register_thread(
        thread_init_data& data, thread_id_type& id, error_code& ec)
    {
        thread_pool_base* pool = nullptr;
        auto thrd_data = get_self_id_data();
        if (thrd_data)
        {
            pool = thrd_data->get_scheduler_base()->get_parent_pool();
        }
        else
        {
            pool = &default_pool();
        }
        pool->create_thread(data, id, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    void threadmanager::register_work(thread_init_data& data, error_code& ec)
    {
        thread_pool_base* pool = nullptr;
        auto thrd_data = get_self_id_data();
        if (thrd_data)
        {
            pool = thrd_data->get_scheduler_base()->get_parent_pool();
        }
        else
        {
            pool = &default_pool();
        }
        pool->create_work(data, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    constexpr std::size_t all_threads = std::size_t(-1);

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

#if defined(HPX_HAVE_BACKGROUND_THREAD_COUNTERS) &&                            \
    defined(HPX_HAVE_THREAD_IDLE_RATES)
    std::int64_t threadmanager::get_background_work_duration(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result +=
                pool_iter->get_background_work_duration(all_threads, reset);
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
            result +=
                pool_iter->get_background_send_duration(all_threads, reset);
        return result;
    }

    std::int64_t threadmanager::get_background_send_overhead(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result +=
                pool_iter->get_background_send_overhead(all_threads, reset);
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
            result +=
                pool_iter->get_num_stolen_from_pending(all_threads, reset);
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
#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY)
        return resource::get_partitioner().shrink_pool(
            pool_name, [this, &pool_name](std::size_t virt_core) {
                get_pool(pool_name).remove_processing_unit(virt_core);
            });
#else
        HPX_UNUSED(pool_name);
        HPX_THROW_EXCEPTION(no_success, "threadmanager::shrink_pool",
            "shrink_pool is not available because "
            "HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY=OFF");
#endif
    }

    std::size_t threadmanager::expand_pool(std::string const& pool_name)
    {
#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY)
        return resource::get_partitioner().expand_pool(
            pool_name, [this, &pool_name](std::size_t virt_core) {
                thread_pool_base& pool = get_pool(pool_name);
                pool.add_processing_unit(
                    virt_core, pool.get_thread_offset() + virt_core);
            });
#else
        HPX_UNUSED(pool_name);
        HPX_THROW_EXCEPTION(no_success, "threadmanager::shrink_pool",
            "shrink_pool is not available because "
            "HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY=OFF");
#endif
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
            if (sched)
                sched->set_all_states(state_running);
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
}}    // namespace hpx::threads
