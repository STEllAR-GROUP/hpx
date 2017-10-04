//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple test verifying basic resource partitioner
// pool and executor

#include <hpx/hpx_init.hpp>
#include <hpx/include/resource_partitioner.hpp>
#include <hpx/include/parallel_execution.hpp>
#include <hpx/runtime/threads/executors/pool_executor.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

const int max_threads = 4;

// dummy function we will call using async
void dummy_task(std::size_t n, std::string text)
{
    for (std::size_t i(0); i < n; ++i)
    {
        std::cout << text << " iteration " << i << "\n";
    }
}

int hpx_main(int argc, char* argv[])
{
    HPX_TEST_EQ(std::size_t(4), hpx::resource::get_num_threads());
    HPX_TEST_EQ(std::size_t(4), hpx::resource::get_num_thread_pools());
    HPX_TEST_EQ(std::size_t(0), hpx::resource::get_pool_index("default"));
    HPX_TEST_EQ(std::size_t(0), hpx::resource::get_pool_index("pool-0"));

    for (int i=0; i<max_threads; ++i) {
        std::string pool_name = "pool-"+std::to_string(i);
        HPX_TEST_EQ(pool_name , hpx::resource::get_pool_name(i));
        HPX_TEST_EQ(std::size_t(1), hpx::resource::get_num_threads(i));
    }

    // setup executors for different task priorities on the pools
    // segfaults or exceptions in any of the following will cause
    // the test to fail
    hpx::threads::scheduled_executor exec_0_hp =
        hpx::threads::executors::pool_executor("default",
        hpx::threads::thread_priority_high);

    hpx::threads::scheduled_executor exec_0 =
        hpx::threads::executors::pool_executor("default",
        hpx::threads::thread_priority_default);

    std::vector<hpx::future<void>> lotsa_futures;

    // use executors to schedule work on pools
    lotsa_futures.push_back(
        hpx::async(exec_0_hp, &dummy_task, 3, "HP default")
    );

    lotsa_futures.push_back(
        hpx::async(exec_0, &dummy_task, 3, "Normal default")
    );

    std::vector<hpx::threads::scheduled_executor> execs;
    std::vector<hpx::threads::scheduled_executor> execs_hp;
    //
    for (int i=0; i<max_threads; ++i) {
        std::string pool_name = "pool-"+std::to_string(i);
        execs.push_back(
            hpx::threads::executors::pool_executor(pool_name,
            hpx::threads::thread_priority_default)
        );
        execs_hp.push_back(
            hpx::threads::executors::pool_executor(pool_name,
            hpx::threads::thread_priority_high)
        );
    }

    for (int i=0; i<max_threads; ++i) {
        std::string pool_name = "pool-"+std::to_string(i);
        lotsa_futures.push_back(
            hpx::async(execs[i], &dummy_task, 3, pool_name + " normal")
        );
        lotsa_futures.push_back(
            hpx::async(execs_hp[i], &dummy_task, 3, pool_name + " HP")
        );
    }

    // check that the default executor still works
    hpx::parallel::execution::default_executor large_stack_executor(
        hpx::threads::thread_stacksize_large);

    lotsa_futures.push_back(
        hpx::async(large_stack_executor, &dummy_task, 3, "true default + large stack")
    );

    // just wait until everything is done
    when_all(lotsa_futures).get();

    return hpx::finalize();
}

// this test must be run with 4 threads
int main(int argc, char* argv[])
{
    std::vector<std::string> cfg = {
        "hpx.os_threads=" + std::to_string(max_threads)
    };

    // create the resource partitioner
    hpx::resource::partitioner rp(argc, argv, std::move(cfg));

    // before adding pools - set the default pool name to "pool-0"
    rp.set_default_pool_name("pool-0");

    // create N pools
    for (int i=0; i<max_threads; i++) {
        std::string pool_name = "pool-"+std::to_string(i);
        rp.create_thread_pool(pool_name,
            hpx::resource::scheduling_policy::local_priority_fifo);
    }

    // add one PU to each pool
    int thread_count = 0;
    for (const hpx::resource::numa_domain& d : rp.numa_domains())
    {
        for (const hpx::resource::core& c : d.cores())
        {
            for (const hpx::resource::pu& p : c.pus())
            {
                if (thread_count < max_threads)
                {
                    std::string pool_name = "pool-" + std::to_string(thread_count);
                    std::cout << "Added pu " << thread_count
                              << " to " << pool_name << "\n";
                    rp.add_resource(p, pool_name);
                    thread_count++;
                }
            }
        }
    }

    // now run the test
    HPX_TEST_EQ(hpx::init(), 0);
    return hpx::util::report_errors();
}
