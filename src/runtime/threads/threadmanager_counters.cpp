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
#include <hpx/modules/errors.hpp>
#include <hpx/functional/bind_back.hpp>
#include <hpx/functional/bind_front.hpp>
#include <hpx/performance_counters/counter_creators.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/manage_counter_type.hpp>
#include <hpx/runtime_local/thread_pool_helpers.hpp>
#include <hpx/modules/threadmanager.hpp>
#include <hpx/runtime/threads/threadmanager_counters.hpp>
#include <hpx/schedulers/maintain_queue_wait_times.hpp>

#include <cstddef>
#include <cstdint>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads {
    namespace detail {
#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
        naming::gid_type queue_wait_time_counter_creator(threadmanager* tm,
            threadmanager_counter_func total_func,
            threadpool_counter_func pool_func,
            performance_counters::counter_info const& info, error_code& ec)
        {
            naming::gid_type gid = locality_pool_thread_counter_creator(
                tm, total_func, pool_func, info, ec);

            if (!ec)
                policies::set_maintain_queue_wait_times_enabled(true);

            return gid;
        }
#endif

        naming::gid_type locality_pool_thread_counter_creator(threadmanager* tm,
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

            thread_pool_base& pool = tm->default_pool();
            if (paths.instancename_ == "total" && paths.instanceindex_ == -1)
            {
                // overall counter
                using performance_counters::detail::create_raw_counter;
                util::function_nonser<std::int64_t(bool)> f =
                    util::bind_front(total_func, tm);
                return create_raw_counter(info, std::move(f), ec);
            }
            else if (paths.instancename_ == "pool")
            {
                if (paths.instanceindex_ >= 0 &&
                    std::size_t(paths.instanceindex_) <
                        hpx::resource::get_num_thread_pools())
                {
                    // specific for given pool counter
                    thread_pool_base& pool_instance =
                        hpx::resource::get_thread_pool(paths.instanceindex_);

                    using performance_counters::detail::create_raw_counter;
                    util::function_nonser<std::int64_t(bool)> f =
                        util::bind_front(pool_func, &pool_instance,
                            static_cast<std::size_t>(paths.subinstanceindex_));
                    return create_raw_counter(info, std::move(f), ec);
                }
            }
            else if (paths.instancename_ == "worker-thread" &&
                paths.instanceindex_ >= 0 &&
                std::size_t(paths.instanceindex_) < pool.get_os_thread_count())
            {
                // specific counter from default
                using performance_counters::detail::create_raw_counter;
                util::function_nonser<std::int64_t(bool)> f =
                    util::bind_front(pool_func, &pool,
                        static_cast<std::size_t>(paths.instanceindex_));
                return create_raw_counter(info, std::move(f), ec);
            }

            HPX_THROWS_IF(ec, bad_parameter,
                "locality_pool_thread_counter_creator",
                "invalid counter instance name: " + paths.instancename_);
            return naming::invalid_gid;
        }

        // scheduler utilization counter creation function
        naming::gid_type scheduler_utilization_counter_creator(
            threadmanager* tm, performance_counters::counter_info const& info,
            error_code& ec)
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
                HPX_THROWS_IF(ec, bad_parameter,
                    "scheduler_utilization_creator",
                    "invalid counter instance parent name: " +
                        paths.parentinstancename_);
                return naming::invalid_gid;
            }

            using performance_counters::detail::create_raw_counter;

            thread_pool_base& pool = tm->default_pool();
            if (paths.instancename_ == "total" && paths.instanceindex_ == -1)
            {
                // counter for default pool
                util::function_nonser<std::int64_t()> f = util::bind_back(
                    &thread_pool_base::get_scheduler_utilization, &pool);
                return create_raw_counter(info, std::move(f), ec);
            }
            else if (paths.instancename_ == "pool")
            {
                if (paths.instanceindex_ < 0)
                {
                    // counter for default pool
                    util::function_nonser<std::int64_t()> f = util::bind_back(
                        &thread_pool_base::get_scheduler_utilization, &pool);
                    return create_raw_counter(info, std::move(f), ec);
                }
                else if (std::size_t(paths.instanceindex_) <
                    hpx::resource::get_num_thread_pools())
                {
                    // counter specific for given pool
                    thread_pool_base& pool_instance =
                        hpx::resource::get_thread_pool(paths.instanceindex_);

                    util::function_nonser<std::int64_t()> f = util::bind_back(
                        &thread_pool_base::get_scheduler_utilization,
                        &pool_instance);
                    return create_raw_counter(info, std::move(f), ec);
                }
            }

            HPX_THROWS_IF(ec, bad_parameter, "scheduler_utilization_creator",
                "invalid counter instance name: " + paths.instancename_);
            return naming::invalid_gid;
        }

        ///////////////////////////////////////////////////////////////////////////
        // locality/pool/worker-thread counter creation function with no total
        // /threads{locality#%d/worker-thread#%d}/idle-loop-count/instantaneous
        // /threads{locality#%d/pool#%s/worker-thread#%d}/idle-loop-count/instantaneous
        naming::gid_type locality_pool_thread_no_total_counter_creator(
            threadmanager* tm, threadpool_counter_func pool_func,
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

            thread_pool_base& pool = tm->default_pool();
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
                    thread_pool_base& pool_instance =
                        hpx::resource::get_thread_pool(paths.instanceindex_);

                    using performance_counters::detail::create_raw_counter;
                    util::function_nonser<std::int64_t(bool)> f =
                        util::bind_front(pool_func, &pool_instance,
                            static_cast<std::size_t>(paths.subinstanceindex_));
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
                    util::bind_front(pool_func, &pool,
                        static_cast<std::size_t>(paths.instanceindex_));
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
            if (!status_is_valid(status))
                return false;

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

                if (mode == performance_counters::discover_counters_full)
                {
                    for (std::size_t t = 0; t != HPX_COROUTINE_NUM_ALL_HEAPS;
                         ++t)
                    {
                        p.instancename_ = "allocator";
                        p.instanceindex_ = static_cast<std::int32_t>(t);
                        status = get_counter_name(p, i.fullname_, ec);
                        if (!status_is_valid(status) || !f(i, ec) || ec)
                            return false;
                    }
                }
                else
                {
                    status = get_counter_name(p, i.fullname_, ec);
                    if (!status_is_valid(status) || !f(i, ec) || ec)
                        return false;
                }
            }
            else if (p.instancename_ == "total" && p.instanceindex_ == -1)
            {
                // overall counter
                status = get_counter_name(p, i.fullname_, ec);
                if (!status_is_valid(status) || !f(i, ec) || ec)
                    return false;
            }
            else if (p.instancename_ == "allocator#*")
            {
                for (std::size_t t = 0; t != HPX_COROUTINE_NUM_ALL_HEAPS; ++t)
                {
                    p.instancename_ = "allocator";
                    p.instanceindex_ = static_cast<std::int32_t>(t);
                    status = get_counter_name(p, i.fullname_, ec);
                    if (!status_is_valid(status) || !f(i, ec) || ec)
                        return false;
                }
            }
            else if (!f(i, ec) || ec)
            {
                return false;
            }

            if (&ec != &throws)
                ec = make_success_code();

            return true;
        }

        ///////////////////////////////////////////////////////////////////////////
        naming::gid_type counter_creator(
            performance_counters::counter_info const& info,
            performance_counters::counter_path_elements const& paths,
            util::function_nonser<std::int64_t(bool)> const& total_creator,
            util::function_nonser<std::int64_t(bool)> const& individual_creator,
            char const* individual_name, std::size_t individual_count,
            error_code& ec)
        {
            if (paths.parentinstance_is_basename_)
            {
                HPX_THROWS_IF(ec, bad_parameter, "counter_creator",
                    "invalid counter instance parent name: " +
                        paths.parentinstancename_);
                return naming::invalid_gid;
            }

            if (!total_creator.empty() && paths.instancename_ == "total" &&
                paths.instanceindex_ == -1)
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
#if defined(HPX_HAVE_COROUTINE_COUNTERS)
        naming::gid_type thread_counts_counter_creator(
            performance_counters::counter_info const& info, error_code& ec)
        {
            // verify the validity of the counter instance name
            performance_counters::counter_path_elements paths;
            performance_counters::get_counter_path_elements(
                info.fullname_, paths, ec);
            if (ec)
                return naming::invalid_gid;

            struct creator_data
            {
                char const* const countername;
                util::function_nonser<std::int64_t(bool)> total_func;
                util::function_nonser<std::int64_t(bool)> individual_func;
                char const* const individual_name;
                std::size_t individual_count;
            };

            creator_data data[] = {
                // /threads{locality#%d/total}/count/stack-recycles
                {"count/stack-recycles",
                    util::bind_front(
                        &coroutine_type::impl_type::get_stack_recycle_count),
                    util::function_nonser<std::uint64_t(bool)>(), "", 0},
#if !defined(HPX_WINDOWS) && !defined(HPX_HAVE_GENERIC_CONTEXT_COROUTINES)
                // /threads{locality#%d/total}/count/stack-unbinds
                {"count/stack-unbinds",
                    util::bind_front(
                        &coroutine_type::impl_type::get_stack_unbind_count),
                    util::function_nonser<std::uint64_t(bool)>(), "", 0},
#endif
            };
            std::size_t const data_size = sizeof(data) / sizeof(data[0]);

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
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    void register_counter_types(threadmanager& tm)
    {
#if defined(HPX_HAVE_COROUTINE_COUNTERS)
        performance_counters::create_counter_func counts_creator(
            util::bind_front(&detail::thread_counts_counter_creator));
#endif

        performance_counters::generic_counter_type_data counter_types[] = {
            // length of thread queue(s)
            {"/threadqueue/length", performance_counters::counter_raw,
                "returns the current queue length for the referenced queue",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind_front(&detail::locality_pool_thread_counter_creator,
                    &tm, &threadmanager::get_queue_length,
                    &thread_pool_base::get_queue_length),
                &performance_counters::locality_pool_thread_counter_discoverer,
                ""},
#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
            // average thread wait time for queue(s)
            {"/threads/wait-time/pending",
                performance_counters::counter_average_timer,
                "returns the average wait time of pending threads for the "
                "referenced queue",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind_front(&detail::queue_wait_time_counter_creator, &tm,
                    &threadmanager::get_average_thread_wait_time,
                    &thread_pool_base::get_average_thread_wait_time),
                &performance_counters::locality_pool_thread_counter_discoverer,
                "ns"},
            // average task wait time for queue(s)
            {"/threads/wait-time/staged",
                performance_counters::counter_average_timer,
                "returns the average wait time of staged threads (task "
                "descriptions) for the referenced queue",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind_front(&detail::queue_wait_time_counter_creator, &tm,
                    &threadmanager::get_average_task_wait_time,
                    &thread_pool_base::get_average_task_wait_time),
                &performance_counters::locality_pool_thread_counter_discoverer,
                "ns"},
#endif
#ifdef HPX_HAVE_THREAD_IDLE_RATES
            // idle rate
            {"/threads/idle-rate", performance_counters::counter_average_count,
                "returns the idle rate for the referenced object",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind_front(&detail::locality_pool_thread_counter_creator,
                    &tm, &threadmanager::avg_idle_rate,
                    &thread_pool_base::avg_idle_rate),
                &performance_counters::locality_pool_thread_counter_discoverer,
                "0.01%"},
#ifdef HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES
            {"/threads/creation-idle-rate",
                performance_counters::counter_average_count,
                "returns the % of idle-rate spent creating HPX-threads for the "
                "referenced object",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind_front(&detail::locality_pool_thread_counter_creator,
                    &tm, &threadmanager::avg_creation_idle_rate,
                    &thread_pool_base::avg_creation_idle_rate),
                &performance_counters::locality_pool_thread_counter_discoverer,
                "0.01%"},
            {"/threads/cleanup-idle-rate",
                performance_counters::counter_average_count,
                "returns the % of time spent cleaning up terminated "
                "HPX-threads for the referenced object",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind_front(&detail::locality_pool_thread_counter_creator,
                    &tm, &threadmanager::avg_cleanup_idle_rate,
                    &thread_pool_base::avg_cleanup_idle_rate),
                &performance_counters::locality_pool_thread_counter_discoverer,
                "0.01%"},
#endif
#endif
#ifdef HPX_HAVE_THREAD_CUMULATIVE_COUNTS
            // thread counts
            {"/threads/count/cumulative",
                performance_counters::counter_monotonically_increasing,
                "returns the overall number of executed (retired) HPX-threads "
                "for the referenced locality",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind_front(&detail::locality_pool_thread_counter_creator,
                    &tm, &threadmanager::get_executed_threads,
                    &thread_pool_base::get_executed_threads),
                &performance_counters::locality_pool_thread_counter_discoverer,
                ""},
            {"/threads/count/cumulative-phases",
                performance_counters::counter_monotonically_increasing,
                "returns the overall number of HPX-thread phases executed for "
                "the referenced locality",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind_front(&detail::locality_pool_thread_counter_creator,
                    &tm, &threadmanager::get_executed_thread_phases,
                    &thread_pool_base::get_executed_thread_phases),
                &performance_counters::locality_pool_thread_counter_discoverer,
                ""},
#ifdef HPX_HAVE_THREAD_IDLE_RATES
            {"/threads/time/average",
                performance_counters::counter_average_timer,
                "returns the average time spent executing one HPX-thread",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind_front(&detail::locality_pool_thread_counter_creator,
                    &tm, &threadmanager::get_thread_duration,
                    &thread_pool_base::get_thread_duration),
                &performance_counters::locality_pool_thread_counter_discoverer,
                "ns"},
            {"/threads/time/average-phase",
                performance_counters::counter_average_timer,
                "returns the average time spent executing one HPX-thread phase",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind_front(&detail::locality_pool_thread_counter_creator,
                    &tm, &threadmanager::get_thread_phase_duration,
                    &thread_pool_base::get_thread_phase_duration),
                &performance_counters::locality_pool_thread_counter_discoverer,
                "ns"},
            {"/threads/time/average-overhead",
                performance_counters::counter_average_timer,
                "returns average overhead time executing one HPX-thread",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind_front(&detail::locality_pool_thread_counter_creator,
                    &tm, &threadmanager::get_thread_overhead,
                    &thread_pool_base::get_thread_overhead),
                &performance_counters::locality_pool_thread_counter_discoverer,
                "ns"},
            {"/threads/time/average-phase-overhead",
                performance_counters::counter_average_timer,
                "returns average overhead time executing one HPX-thread phase",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind_front(&detail::locality_pool_thread_counter_creator,
                    &tm, &threadmanager::get_thread_phase_overhead,
                    &thread_pool_base::get_thread_phase_overhead),
                &performance_counters::locality_pool_thread_counter_discoverer,
                "ns"},
            {"/threads/time/cumulative",
                performance_counters::counter_elapsed_time,
                "returns the cumulative time spent executing HPX-threads",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind_front(&detail::locality_pool_thread_counter_creator,
                    &tm, &threadmanager::get_cumulative_thread_duration,
                    &thread_pool_base::get_cumulative_thread_duration),
                &performance_counters::locality_pool_thread_counter_discoverer,
                "ns"},
            {"/threads/time/cumulative-overhead",
                performance_counters::counter_elapsed_time,
                "returns the cumulative overhead time incurred by executing "
                "HPX threads",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind_front(&detail::locality_pool_thread_counter_creator,
                    &tm, &threadmanager::get_cumulative_thread_overhead,
                    &thread_pool_base::get_cumulative_thread_overhead),
                &performance_counters::locality_pool_thread_counter_discoverer,
                "ns"},
#endif
#endif

#if defined(HPX_HAVE_BACKGROUND_THREAD_COUNTERS) &&                            \
    defined(HPX_HAVE_THREAD_IDLE_RATES)
            {"/threads/time/background-work-duration",
                performance_counters::counter_elapsed_time,
                "returns the overall time spent running background work",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind_front(&detail::locality_pool_thread_counter_creator,
                    &tm, &threadmanager::get_background_work_duration,
                    &thread_pool_base::get_background_work_duration),
                &performance_counters::locality_pool_thread_counter_discoverer,
                "ns"},
            {"/threads/background-overhead",
                performance_counters::counter_aggregating,
                "returns the overall background overhead",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind_front(&detail::locality_pool_thread_counter_creator,
                    &tm, &threadmanager::get_background_overhead,
                    &thread_pool_base::get_background_overhead),
                &performance_counters::locality_pool_thread_counter_discoverer,
                "0.1%"},
            {"/threads/time/background-send-duration",
                performance_counters::counter_elapsed_time,
                "returns the overall time spent running background work "
                "related to sending parcels",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind_front(&detail::locality_pool_thread_counter_creator,
                    &tm, &threadmanager::get_background_send_duration,
                    &thread_pool_base::get_background_send_duration),
                &performance_counters::locality_pool_thread_counter_discoverer,
                "ns"},
            {"/threads/background-send-overhead",
                performance_counters::counter_aggregating,
                "returns the overall background overhead "
                "related to sending parcels",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind_front(&detail::locality_pool_thread_counter_creator,
                    &tm, &threadmanager::get_background_send_overhead,
                    &thread_pool_base::get_background_send_overhead),
                &performance_counters::locality_pool_thread_counter_discoverer,
                "0.1%"},
            {"/threads/time/background-receive-duration",
                performance_counters::counter_elapsed_time,
                "returns the overall time spent running background work "
                "related to receiving parcels",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind_front(&detail::locality_pool_thread_counter_creator,
                    &tm, &threadmanager::get_background_receive_duration,
                    &thread_pool_base::get_background_receive_duration),
                &performance_counters::locality_pool_thread_counter_discoverer,
                "ns"},
            {"/threads/background-receive-overhead",
                performance_counters::counter_aggregating,
                "returns the overall background overhead "
                "related to receiving parcels",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind_front(&detail::locality_pool_thread_counter_creator,
                    &tm, &threadmanager::get_background_receive_overhead,
                    &thread_pool_base::get_background_receive_overhead),
                &performance_counters::locality_pool_thread_counter_discoverer,
                "0.1%"},
#endif    // HPX_HAVE_BACKGROUND_THREAD_COUNTERS

            {   "/threads/time/overall",
                performance_counters::counter_elapsed_time,
                "returns the overall time spent running the scheduler on a "
                "core",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind_front(&detail::locality_pool_thread_counter_creator,
                    &tm, &threadmanager::get_cumulative_duration,
                    &thread_pool_base::get_cumulative_duration),
                &performance_counters::locality_pool_thread_counter_discoverer,
                "ns"},
            {   "/threads/count/instantaneous/all",
                performance_counters::counter_raw,
                "returns the overall current number of HPX-threads "
                "instantiated at the referenced locality",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind_front(&detail::locality_pool_thread_counter_creator,
                    &tm, &threadmanager::get_thread_count_unknown,
                    &thread_pool_base::get_thread_count_unknown),
                &performance_counters::locality_pool_thread_counter_discoverer,
                ""},
            {   "/threads/count/instantaneous/active",
                performance_counters::counter_raw,
                "returns the current number of active HPX-threads "
                "at the referenced locality",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind_front(&detail::locality_pool_thread_counter_creator,
                    &tm, &threadmanager::get_thread_count_active,
                    &thread_pool_base::get_thread_count_active),
                &performance_counters::locality_pool_thread_counter_discoverer,
                ""},
            {   "/threads/count/instantaneous/pending",
                performance_counters::counter_raw,
                "returns the current number of pending HPX-threads "
                "at the referenced locality",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind_front(&detail::locality_pool_thread_counter_creator,
                    &tm, &threadmanager::get_thread_count_pending,
                    &thread_pool_base::get_thread_count_pending),
                &performance_counters::locality_pool_thread_counter_discoverer,
                ""},
            {   "/threads/count/instantaneous/suspended",
                performance_counters::counter_raw,
                "returns the current number of suspended HPX-threads "
                "at the referenced locality",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind_front(&detail::locality_pool_thread_counter_creator,
                    &tm, &threadmanager::get_thread_count_suspended,
                    &thread_pool_base::get_thread_count_suspended),
                &performance_counters::locality_pool_thread_counter_discoverer,
                ""},
            {   "/threads/count/instantaneous/terminated",
                performance_counters::counter_raw,
                "returns the current number of terminated HPX-threads "
                "at the referenced locality",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind_front(&detail::locality_pool_thread_counter_creator,
                    &tm, &threadmanager::get_thread_count_terminated,
                    &thread_pool_base::get_thread_count_terminated),
                &performance_counters::locality_pool_thread_counter_discoverer,
                ""},
            {   "/threads/count/instantaneous/staged",
                performance_counters::counter_raw,
                "returns the current number of staged HPX-threads (task "
                "descriptions) "
                "at the referenced locality",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind_front(&detail::locality_pool_thread_counter_creator,
                    &tm, &threadmanager::get_thread_count_staged,
                    &thread_pool_base::get_thread_count_staged),
                &performance_counters::locality_pool_thread_counter_discoverer,
                ""},
#if defined(HPX_HAVE_COROUTINE_COUNTERS)
            {   "/threads/count/stack-recycles",
                performance_counters::counter_monotonically_increasing,
                "returns the total number of HPX-thread recycling operations "
                "performed for the referenced locality",
                HPX_PERFORMANCE_COUNTER_V1, counts_creator,
                &performance_counters::locality_counter_discoverer, ""},
#if !defined(HPX_WINDOWS) && !defined(HPX_HAVE_GENERIC_CONTEXT_COROUTINES)
            {   "/threads/count/stack-unbinds",
                performance_counters::counter_monotonically_increasing,
                "returns the total number of HPX-thread unbind (madvise) "
                "operations performed for the referenced locality",
                HPX_PERFORMANCE_COUNTER_V1, counts_creator,
                &performance_counters::locality_counter_discoverer, ""},
#endif
            {   "/threads/count/objects",
                performance_counters::counter_monotonically_increasing,
                "returns the overall number of created HPX-thread objects for "
                "the referenced locality",
                HPX_PERFORMANCE_COUNTER_V1, counts_creator,
                &detail::locality_allocator_counter_discoverer, ""},
#endif
#ifdef HPX_HAVE_THREAD_STEALING_COUNTS
            {   "/threads/count/pending-misses",
                performance_counters::counter_monotonically_increasing,
                "returns the number of times that the referenced worker-thread "
                "on the referenced locality failed to find pending HPX-threads "
                "in its associated queue",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind_front(&detail::locality_pool_thread_counter_creator,
                    &tm, &threadmanager::get_num_pending_misses,
                    &thread_pool_base::get_num_pending_misses),
                &performance_counters::locality_pool_thread_counter_discoverer,
                ""},
            {   "/threads/count/pending-accesses",
                performance_counters::counter_monotonically_increasing,
                "returns the number of times that the referenced worker-thread "
                "on the referenced locality looked for pending HPX-threads "
                "in its associated queue",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind_front(&detail::locality_pool_thread_counter_creator,
                    &tm, &threadmanager::get_num_pending_accesses,
                    &thread_pool_base::get_num_pending_accesses),
                &performance_counters::locality_pool_thread_counter_discoverer,
                ""},
            {   "/threads/count/stolen-from-pending",
                performance_counters::counter_monotonically_increasing,
                "returns the overall number of pending HPX-threads stolen by "
                "neighboring"
                "schedulers from &tm scheduler for the referenced locality",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind_front(&detail::locality_pool_thread_counter_creator,
                    &tm, &threadmanager::get_num_stolen_from_pending,
                    &thread_pool_base::get_num_stolen_from_pending),
                &performance_counters::locality_pool_thread_counter_discoverer,
                ""},
            {   "/threads/count/stolen-from-staged",
                performance_counters::counter_monotonically_increasing,
                "returns the overall number of task descriptions stolen by "
                "neighboring"
                "schedulers from tm scheduler for the referenced locality",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind_front(&detail::locality_pool_thread_counter_creator,
                    &tm, &threadmanager::get_num_stolen_from_staged,
                    &thread_pool_base::get_num_stolen_from_staged),
                &performance_counters::locality_pool_thread_counter_discoverer,
                ""},
            {   "/threads/count/stolen-to-pending",
                performance_counters::counter_monotonically_increasing,
                "returns the overall number of pending HPX-threads stolen from "
                "neighboring"
                "schedulers for the referenced locality",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind_front(&detail::locality_pool_thread_counter_creator,
                    &tm, &threadmanager::get_num_stolen_to_pending,
                    &thread_pool_base::get_num_stolen_to_pending),
                &performance_counters::locality_pool_thread_counter_discoverer,
                ""},
            {   "/threads/count/stolen-to-staged",
                performance_counters::counter_monotonically_increasing,
                "returns the overall number of task descriptions stolen from "
                "neighboring"
                "schedulers for the referenced locality",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind_front(&detail::locality_pool_thread_counter_creator,
                    &tm, &threadmanager::get_num_stolen_to_staged,
                    &thread_pool_base::get_num_stolen_to_staged),
                &performance_counters::locality_pool_thread_counter_discoverer,
                ""},
#endif
            // scheduler utilization
            {   "/scheduler/utilization/instantaneous",
                performance_counters::counter_raw,
                "returns the current scheduler utilization",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind_front(
                    &detail::scheduler_utilization_counter_creator, &tm),
                &performance_counters::locality_pool_counter_discoverer, "%"},
            // idle-loop count
            {   "/threads/idle-loop-count/instantaneous",
                performance_counters::counter_raw,
                "returns the current value of the scheduler idle-loop count",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind_front(
                    &detail::locality_pool_thread_no_total_counter_creator, &tm,
                    &thread_pool_base::get_idle_loop_count),
                &performance_counters::
                    locality_pool_thread_no_total_counter_discoverer,
                ""},
            // busy-loop count
            {   "/threads/busy-loop-count/instantaneous",
                performance_counters::counter_raw,
                "returns the current value of the scheduler busy-loop count",
                HPX_PERFORMANCE_COUNTER_V1,
                util::bind_front(
                    &detail::locality_pool_thread_no_total_counter_creator, &tm,
                    &thread_pool_base::get_busy_loop_count),
                &performance_counters::
                    locality_pool_thread_no_total_counter_discoverer,
                ""}
        };
        performance_counters::install_counter_types(
            counter_types, sizeof(counter_types) / sizeof(counter_types[0]));
    }
}    // namespace threads
}    // namespace hpx
