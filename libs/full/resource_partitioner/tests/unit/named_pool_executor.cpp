//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple test verifying basic resource partitioner
// pool and executor

#include <hpx/hpx_init.hpp>

#include <hpx/async_combinators/when_all.hpp>
#include <hpx/include/parallel_execution.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/resource_partitioner.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/modules/async_local.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

int const max_threads = 4;

// dummy function we will call using async
void dummy_task(std::size_t n, std::string const& text)
{
    for (std::size_t i(0); i < n; ++i)
    {
        std::cout << text << " iteration " << i << "\n";
    }
}

int hpx_main()
{
    HPX_TEST_EQ(std::size_t(4), hpx::resource::get_num_threads());
    HPX_TEST_EQ(std::size_t(4), hpx::resource::get_num_thread_pools());
    HPX_TEST_EQ(std::size_t(0), hpx::resource::get_pool_index("default"));
    HPX_TEST_EQ(std::size_t(0), hpx::resource::get_pool_index("pool-0"));
    HPX_TEST(hpx::resource::pool_exists("default"));
    HPX_TEST(hpx::resource::pool_exists("pool-0"));
    HPX_TEST(!hpx::resource::pool_exists("nonexistent"));
    for (int pool_index = 0; pool_index < max_threads; ++pool_index)
    {
        HPX_TEST(hpx::resource::pool_exists(pool_index));
    }
    HPX_TEST(!hpx::resource::pool_exists(max_threads));

    for (int i = 0; i < max_threads; ++i)
    {
        std::string pool_name = "pool-" + std::to_string(i);
        HPX_TEST_EQ(pool_name, hpx::resource::get_pool_name(i));
        HPX_TEST_EQ(std::size_t(1), hpx::resource::get_num_threads(i));
    }

    // Make sure default construction works
    hpx::execution::parallel_executor exec_default;
    HPX_UNUSED(exec_default);

    // setup executors for different task priorities on the pools
    // segfaults or exceptions in any of the following will cause
    // the test to fail
    hpx::execution::parallel_executor exec_0_hp(
        &hpx::resource::get_thread_pool("default"),
        hpx::threads::thread_priority::high);

    hpx::execution::parallel_executor exec_0(
        &hpx::resource::get_thread_pool("default"),
        hpx::threads::thread_priority::default_);

    std::vector<hpx::future<void>> lotsa_futures;

    // use executors to schedule work on pools
    lotsa_futures.push_back(
        hpx::async(exec_0_hp, &dummy_task, 3, "HP default"));

    lotsa_futures.push_back(
        hpx::async(exec_0, &dummy_task, 3, "Normal default"));

    std::vector<hpx::execution::parallel_executor> execs;
    std::vector<hpx::execution::parallel_executor> execs_hp;
    //
    for (int i = 0; i < max_threads; ++i)
    {
        std::string pool_name = "pool-" + std::to_string(i);
        execs.push_back(hpx::execution::parallel_executor(
            &hpx::resource::get_thread_pool(pool_name),
            hpx::threads::thread_priority::default_));
        execs_hp.push_back(hpx::execution::parallel_executor(
            &hpx::resource::get_thread_pool(pool_name),
            hpx::threads::thread_priority::high));
    }

    for (int i = 0; i < max_threads; ++i)
    {
        std::string pool_name = "pool-" + std::to_string(i);
        lotsa_futures.push_back(
            hpx::async(execs[i], &dummy_task, 3, pool_name + " normal"));
        lotsa_futures.push_back(
            hpx::async(execs_hp[i], &dummy_task, 3, pool_name + " HP"));
    }

    // check that the default executor still works
    hpx::parallel::execution::default_executor large_stack_executor(
        hpx::threads::thread_stacksize::large);

    lotsa_futures.push_back(hpx::async(
        large_stack_executor, &dummy_task, 3, "true default + large stack"));

    // just wait until everything is done
    when_all(lotsa_futures).get();

    return hpx::finalize();
}

void init_resource_partitioner_handler(
    hpx::resource::partitioner& rp, hpx::program_options::variables_map const&)
{
    // before adding pools - set the default pool name to "pool-0"
    rp.set_default_pool_name("pool-0");

    // create N pools
    for (int i = 0; i < max_threads; i++)
    {
        std::string pool_name = "pool-" + std::to_string(i);
        rp.create_thread_pool(
            pool_name, hpx::resource::scheduling_policy::local_priority_fifo);
    }

    // add one PU to each pool
    int thread_count = 0;
    for (hpx::resource::numa_domain const& d : rp.numa_domains())
    {
        for (hpx::resource::core const& c : d.cores())
        {
            for (hpx::resource::pu const& p : c.pus())
            {
                if (thread_count < max_threads)
                {
                    std::string pool_name =
                        "pool-" + std::to_string(thread_count);
                    std::cout << "Added pu " << thread_count << " to "
                              << pool_name << "\n";
                    rp.add_resource(p, pool_name);
                    thread_count++;
                }
            }
        }
    }
}

// this test must be run with 4 threads
int main(int argc, char* argv[])
{
    hpx::init_params init_args;
    init_args.cfg = {"hpx.os_threads=" + std::to_string(max_threads)};
    // Set the callback to init the thread_pools
    init_args.rp_callback = &init_resource_partitioner_handler;

    // now run the test
    HPX_TEST_EQ(hpx::init(argc, argv, init_args), 0);
    return hpx::util::report_errors();
}
