//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/naming.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

char const* const locality_pool_thread_counter_names[] =
{
    "/threadqueue/length",
#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
    "/threads/wait-time/pending",
    "/threads/wait-time/staged",
#endif
#ifdef HPX_HAVE_THREAD_IDLE_RATES
    "/threads/idle-rate",
#ifdef HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES
    "/threads/creation-idle-rate",
    "/threads/cleanup-idle-rate",
#endif
#endif
    "/threads/time/overall",
    "/threads/count/instantaneous/all",
    "/threads/count/instantaneous/active",
    "/threads/count/instantaneous/pending",
    "/threads/count/instantaneous/suspended",
    "/threads/count/instantaneous/terminated",
    "/threads/count/instantaneous/staged",
#ifdef HPX_HAVE_THREAD_CUMULATIVE_COUNTS
    "/threads/count/cumulative",
    "/threads/count/cumulative-phases",
#ifdef HPX_HAVE_THREAD_IDLE_RATES
    "/threads/time/average",
    "/threads/time/average-phase",
    "/threads/time/average-overhead",
    "/threads/time/average-phase-overhead",
    "/threads/time/cumulative",
    "/threads/time/cumulative-overhead",
#endif
#endif
#ifdef HPX_HAVE_THREAD_STEALING_COUNTS
    "/threads/count/pending-misses",
    "/threads/count/pending-accesses",
    "/threads/count/stolen-from-pending",
    "/threads/count/stolen-from-staged",
    "/threads/count/stolen-to-pending",
    "/threads/count/stolen-to-staged",
#endif
    nullptr
};

char const* const locality_pool_thread_no_total_counter_names[] =
{
    "/threads/idle-loop-count/instantaneous",
    "/threads/busy-loop-count/instantaneous",
    nullptr
};

///////////////////////////////////////////////////////////////////////////////
// char const* const locality_thread_counter_names[] =
// {
//     nullptr
// };

char const* const locality_counter_names[] =
{
    "/threads/count/stack-recycles",
#if !defined(HPX_WINDOWS) && !defined(HPX_HAVE_GENERIC_CONTEXT_COROUTINES)
    "/threads/count/stack-unbinds",
#endif
    "/scheduler/utilization/instantaneous",
    nullptr
};

///////////////////////////////////////////////////////////////////////////////
void test_all_locality_thread_counters(char const* const* counter_names,
    std::size_t locality_id, std::size_t pool, std::size_t core)
{
    for (char const* const*p = counter_names; *p != nullptr; ++p)
    {
        // split counter type into counter path elements
        hpx::performance_counters::counter_path_elements path;
        HPX_TEST_EQ(
            hpx::performance_counters::get_counter_path_elements(*p, path),
            hpx::performance_counters::status_valid_data);

        // augment the counter path elements
        path.parentinstancename_ = "locality";
        path.parentinstanceindex_ = locality_id;
        if (pool != std::size_t(-1))
        {
            path.instancename_ = "pool";
            path.instanceindex_ = pool;

            if (core != std::size_t(-1))
            {
                path.subinstancename_ = "worker-thread";
                path.subinstanceindex_ = core;
            }
            else
            {
                path.subinstancename_ = "total";
                path.subinstanceindex_ = std::size_t(-1);
            }
        }
        else if (core != std::size_t(-1))
        {
            path.instancename_ = "worker-thread";
            path.instanceindex_ = core;
        }
        else
        {
            path.instancename_ = "total";
            path.instanceindex_ = std::size_t(-1);
        }

        std::string name;
        HPX_TEST_EQ(
            hpx::performance_counters::get_counter_name(path, name),
            hpx::performance_counters::status_valid_data);

        std::cout << name << '\n';

        try
        {
            hpx::performance_counters::performance_counter counter(name);
            HPX_TEST_EQ(counter.get_name(hpx::launch::sync), name);
            counter.get_value<std::size_t>(hpx::launch::sync);
        }
        catch (...)
        {
            HPX_TEST(false);        // should never happen
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
void test_all_locality_pool_thread_counters(
    std::size_t locality_id, std::size_t pool, std::size_t core)
{
    test_all_locality_thread_counters(
        locality_pool_thread_counter_names, locality_id, pool, core);
}

void test_all_locality_pool_thread_no_total_counters(
    std::size_t locality_id, std::size_t pool, std::size_t core)
{
    test_all_locality_thread_counters(
        locality_pool_thread_no_total_counter_names, locality_id, pool, core);
}

// void test_all_locality_thread_counters(
//     std::size_t locality_id, std::size_t core)
// {
//     test_all_locality_thread_counters(
//         locality_thread_counter_names, locality_id, std::size_t(-1), core);
// }

void test_all_locality_counters(std::size_t locality_id)
{
    test_all_locality_thread_counters(
        locality_counter_names, locality_id, std::size_t(-1), std::size_t(-1));
}

///////////////////////////////////////////////////////////////////////////////
void test_all_counters_locality(std::size_t locality_id)
{
    std::size_t cores = hpx::get_num_worker_threads();

    // locality/pool/thread
    test_all_locality_pool_thread_counters(
        locality_id, std::size_t(-1), std::size_t(-1));

    std::size_t pools = hpx::resource::get_num_thread_pools();
    for (std::size_t pool = 0; pool != pools; ++pool)
    {
        test_all_locality_pool_thread_counters(
            locality_id, pool, std::size_t(-1));

        for (std::size_t core = 0; core != cores; ++core)
        {
            test_all_locality_pool_thread_counters(locality_id, pool, core);
            test_all_locality_pool_thread_no_total_counters(
                locality_id, pool, core);
        }
    }

    for (std::size_t core = 0; core != cores; ++core)
    {
        test_all_locality_pool_thread_counters(
            locality_id, std::size_t(-1), core);
    }

//     // locality/thread (same as locality/pool#default/threads)
//     test_all_locality_thread_counters(locality_id, std::size_t(-1));
//
//     for (std::size_t core = 0; core != cores; ++core)
//     {
//         test_all_locality_thread_counters(locality_id, core);
//     }

    // locality/total
    test_all_locality_counters(locality_id);
}

int main(int argc, char* argv[])
{
    for (auto const& id : hpx::find_all_localities())
        test_all_counters_locality(hpx::naming::get_locality_id_from_id(id));

    return hpx::util::report_errors();
}
