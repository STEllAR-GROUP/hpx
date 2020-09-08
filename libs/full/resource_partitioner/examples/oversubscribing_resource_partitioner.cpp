//  Copyright (c) 2017 Shoshana Jakobovits
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/parallel_for_loop.hpp>
#include <hpx/include/resource_partitioner.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/iostream.hpp>
//
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <utility>
//
#include "system_characteristics.hpp"

namespace resource { namespace pools {
    enum ids
    {
        DEFAULT = 0,
        MPI = 1,
        GPU = 2,
        MATRIX = 3,
    };
}}    // namespace resource::pools

static bool use_pools = false;
static bool use_scheduler = false;
static int pool_threads = 1;

// this is our custom scheduler type
using high_priority_sched =
    hpx::threads::policies::shared_priority_queue_scheduler<>;
using namespace hpx::threads::policies;
using hpx::threads::policies::scheduler_mode;

// dummy function we will call using async
void do_stuff(std::size_t n, bool printout)
{
    if (printout)
        hpx::cout << "[do stuff] " << n << "\n";
    for (std::size_t i(0); i < n; ++i)
    {
        double f = std::sin(2 * M_PI * i / n);
        if (printout)
            hpx::cout << "sin(" << i << ") = " << f << ", ";
    }
    if (printout)
        hpx::cout << "\n";
}

// this is called on an hpx thread after the runtime starts up
int hpx_main(hpx::program_options::variables_map& vm)
{
    if (vm.count("use-pools"))
        use_pools = true;
    if (vm.count("use-scheduler"))
        use_scheduler = true;
    //
    std::cout << "[hpx_main] starting ..."
              << "use_pools " << use_pools << " "
              << "use_scheduler " << use_scheduler << "\n";

    std::size_t num_threads = hpx::get_num_worker_threads();
    hpx::cout << "HPX using threads = " << num_threads << std::endl;

    if (num_threads == 1)
    {
        HPX_THROW_EXCEPTION(hpx::commandline_option_error, "hpx_main",
            "the oversubscribing_resource_partitioner example requires at "
            "least 2 worker threads (1 given)");
    }

    std::size_t loop_count = num_threads * 1;
    std::size_t async_count = num_threads * 1;

    // create an executor with high priority for important tasks
    hpx::execution::parallel_executor high_priority_executor(
        hpx::this_thread::get_pool(), hpx::threads::thread_priority_critical);
    hpx::execution::parallel_executor normal_priority_executor;

    hpx::execution::parallel_executor mpi_executor;
    // create an executor on the mpi pool
    if (use_pools)
    {
        // get executors
        mpi_executor = hpx::execution::parallel_executor(
            &hpx::resource::get_thread_pool("mpi"));
        hpx::cout << "\n[hpx_main] got mpi executor " << std::endl;
    }
    else
    {
        mpi_executor = high_priority_executor;
    }

    // print partition characteristics
    std::cout << "\n\n[hpx_main] print resource_partitioner characteristics : "
              << "\n";
    hpx::resource::get_partitioner().print_init_pool_data(std::cout);

    // print partition characteristics
    std::cout << "\n\n[hpx_main] print thread-manager pools : "
              << "\n";
    hpx::threads::get_thread_manager().print_pools(std::cout);

    // print system characteristics
    print_system_characteristics();

    // use executor to schedule work on custom pool
    hpx::future<void> future_1 = hpx::async(mpi_executor, &do_stuff, 5, true);

    hpx::future<void> future_2 = future_1.then(
        mpi_executor, [](hpx::future<void>&& f) { do_stuff(5, true); });

    hpx::future<void> future_3 = future_2.then(mpi_executor,
        [mpi_executor, high_priority_executor, async_count](
            hpx::future<void>&& f) mutable {
            hpx::future<void> future_4, future_5;
            for (std::size_t i = 0; i < async_count; i++)
            {
                if (i % 2 == 0)
                {
                    future_4 =
                        hpx::async(mpi_executor, &do_stuff, async_count, false);
                }
                else
                {
                    future_5 = hpx::async(
                        high_priority_executor, &do_stuff, async_count, false);
                }
            }
            // the last futures we made are stored in here
            future_4.get();
            future_5.get();
        });

    future_3.get();

    hpx::lcos::local::mutex m;
    std::set<std::thread::id> thread_set;

    // test a parallel algorithm on custom pool with high priority
    hpx::execution::static_chunk_size fixed(1);
    hpx::for_loop_strided(
        hpx::execution::par.with(fixed).on(high_priority_executor), 0,
        loop_count, 1, [&](std::size_t i) {
            std::lock_guard<hpx::lcos::local::mutex> lock(m);
            if (thread_set.insert(std::this_thread::get_id()).second)
            {
                hpx::cout << std::hex << hpx::this_thread::get_id() << " "
                          << std::hex << std::this_thread::get_id()
                          << " high priority i " << std::dec << i << std::endl;
            }
        });
    hpx::cout << "thread set contains " << std::dec << thread_set.size()
              << std::endl;
    thread_set.clear();

    // test a parallel algorithm on custom pool with normal priority
    hpx::for_loop_strided(
        hpx::execution::par.with(fixed).on(normal_priority_executor), 0,
        loop_count, 1, [&](std::size_t i) {
            std::lock_guard<hpx::lcos::local::mutex> lock(m);
            if (thread_set.insert(std::this_thread::get_id()).second)
            {
                hpx::cout << std::hex << hpx::this_thread::get_id() << " "
                          << std::hex << std::this_thread::get_id()
                          << " normal priority i " << std::dec << i
                          << std::endl;
            }
        });
    hpx::cout << "thread set contains " << std::dec << thread_set.size()
              << std::endl;
    thread_set.clear();

    // test a parallel algorithm on mpi_executor
    hpx::for_loop_strided(hpx::execution::par.with(fixed).on(mpi_executor), 0,
        loop_count, 1, [&](std::size_t i) {
            std::lock_guard<hpx::lcos::local::mutex> lock(m);
            if (thread_set.insert(std::this_thread::get_id()).second)
            {
                hpx::cout << std::hex << hpx::this_thread::get_id() << " "
                          << std::hex << std::this_thread::get_id()
                          << " mpi pool i " << std::dec << i << std::endl;
            }
        });
    hpx::cout << "thread set contains " << std::dec << thread_set.size()
              << std::endl;
    thread_set.clear();

    //     auto high_priority_async_policy =
    //         hpx::launch::async_policy(hpx::threads::thread_priority_critical);
    //     auto normal_priority_async_policy = hpx::launch::async_policy();

    // test a parallel algorithm on custom pool with high priority
    hpx::for_loop_strided(
        hpx::execution::par.with(fixed /*, high_priority_async_policy*/)
            .on(mpi_executor),
        0, loop_count, 1, [&](std::size_t i) {
            std::lock_guard<hpx::lcos::local::mutex> lock(m);
            if (thread_set.insert(std::this_thread::get_id()).second)
            {
                hpx::cout << std::hex << hpx::this_thread::get_id() << " "
                          << std::hex << std::this_thread::get_id()
                          << " high priority mpi i " << std::dec << i
                          << std::endl;
            }
        });
    hpx::cout << "thread set contains " << std::dec << thread_set.size()
              << std::endl;
    thread_set.clear();

    return hpx::finalize();
}

// the normal int main function that is called at startup and runs on an OS thread
// the user must call hpx::init to start the hpx runtime which will execute hpx_main
// on an hpx thread
int main(int argc, char* argv[])
{
    hpx::program_options::options_description desc_cmdline("Test options");
    desc_cmdline.add_options()(
        "use-pools,u", "Enable advanced HPX thread pools and executors")(
        "use-scheduler,s", "Enable custom priority scheduler")("pool-threads,m",
        hpx::program_options::value<int>()->default_value(1),
        "Number of threads to assign to custom pool");

    // HPX uses a boost program options variable map, but we need it before
    // hpx-main, so we will create another one here and throw it away after use
    hpx::program_options::variables_map vm;
    hpx::program_options::store(
        hpx::program_options::command_line_parser(argc, argv)
            .allow_unregistered()
            .options(desc_cmdline)
            .run(),
        vm);

    if (vm.count("use-pools"))
    {
        use_pools = true;
    }
    if (vm.count("use-scheduler"))
    {
        use_scheduler = true;
    }

    pool_threads = vm["pool-threads"].as<int>();

    hpx::init_params iparams;

    iparams.desc_cmdline = desc_cmdline;
    iparams.rp_mode = hpx::resource::mode_allow_oversubscription;
    iparams.rp_callback = [](auto& rp) {
        //    auto &topo = rp.get_topology();
        std::cout << "[main] obtained reference to the resource_partitioner\n";

        // create a thread pool and supply a lambda that returns a new pool with
        // the a user supplied scheduler attached
        rp.create_thread_pool("default",
            [](hpx::threads::thread_pool_init_parameters init,
                hpx::threads::policies::thread_queue_init_parameters
                    thread_queue_init)
                -> std::unique_ptr<hpx::threads::thread_pool_base> {
                std::cout << "User defined scheduler creation callback "
                          << std::endl;

                high_priority_sched::init_parameter_type scheduler_init(
                    init.num_threads_, {1, 1, 64}, init.affinity_data_,
                    thread_queue_init, "shared-priority-scheduler");
                std::unique_ptr<high_priority_sched> scheduler(
                    new high_priority_sched(scheduler_init));

                init.mode_ = scheduler_mode(scheduler_mode::do_background_work |
                    scheduler_mode::delay_exit);

                std::unique_ptr<hpx::threads::thread_pool_base> pool(
                    new hpx::threads::detail::scheduled_thread_pool<
                        high_priority_sched>(std::move(scheduler), init));
                return pool;
            });

        rp.add_resource(rp.numa_domains(), "default");

        if (use_pools)
        {
            // Create a thread pool using the default scheduler provided by HPX
            //        rp.create_thread_pool("mpi",
            //            hpx::resource::scheduling_policy::local_priority_fifo);
            //std::cout << "[main] " << "thread_pools created \n";

            // create a thread pool and supply a lambda that returns a new pool with
            // the a user supplied scheduler attached
            rp.create_thread_pool("mpi",
                [](hpx::threads::thread_pool_init_parameters init,
                    hpx::threads::policies::thread_queue_init_parameters
                        thread_queue_init)
                    -> std::unique_ptr<hpx::threads::thread_pool_base> {
                    std::cout << "User defined scheduler creation callback "
                              << std::endl;

                    high_priority_sched::init_parameter_type scheduler_init(
                        init.num_threads_, {1, 1, 64}, init.affinity_data_,
                        thread_queue_init, "shared-priority-scheduler");
                    std::unique_ptr<high_priority_sched> scheduler(
                        new high_priority_sched(scheduler_init));

                    init.mode_ = scheduler_mode(scheduler_mode::delay_exit);

                    std::unique_ptr<hpx::threads::thread_pool_base> pool(
                        new hpx::threads::detail::scheduled_thread_pool<
                            high_priority_sched>(std::move(scheduler), init));
                    return pool;
                });

            // rp.add_resource(rp.numa_domains()[0].cores()[0].pus(), "mpi");
            // add N cores to mpi pool
            int count = 0;
            for (const hpx::resource::numa_domain& d : rp.numa_domains())
            {
                for (const hpx::resource::core& c : d.cores())
                {
                    for (const hpx::resource::pu& p : c.pus())
                    {
                        if (count < pool_threads)
                        {
                            std::cout << "Added pu " << count++
                                      << " to mpi pool\n";
                            rp.add_resource(p, "mpi");
                        }
                    }
                }
            }

            std::cout << "[main] resources added to thread_pools \n";
        }
    };

    return hpx::init(argc, argv, iparams);
}
