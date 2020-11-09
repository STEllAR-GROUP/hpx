//  Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/format.hpp>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/iostream.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "worker_timed.hpp"

using hpx::program_options::variables_map;
using hpx::program_options::options_description;
using hpx::program_options::value;

using hpx::get_os_thread_count;

using hpx::this_thread::suspend;
using hpx::threads::get_thread_count;

using hpx::chrono::high_resolution_timer;

using hpx::cout;
using hpx::flush;

///////////////////////////////////////////////////////////////////////////////
// Command-line variables.
std::uint64_t tasks = 500000;
std::uint64_t delay = 0;
bool header = true;

///////////////////////////////////////////////////////////////////////////////
void print_results(
    std::uint64_t cores
  , double walltime
    )
{
    if (header)
        cout << "OS-threads,Tasks,Delay (iterations),"
                "Total Walltime (seconds),Walltime per Task (seconds)\n"
             << flush;

    std::string const cores_str = hpx::util::format("{},", cores);
    std::string const tasks_str = hpx::util::format("{},", tasks);
    std::string const delay_str = hpx::util::format("{},", delay);

    hpx::util::format_to(cout,
        "{:-21} {:-21} {:-21} {:10.12}, {:10.12}\n",
        cores_str, tasks_str, delay_str,
        walltime, walltime / tasks) << flush;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(
    variables_map& vm
    )
{
    if (vm.count("no-header"))
        header = false;

    std::size_t num_os_threads = hpx::get_os_thread_count();

    int num_executors = vm["executors"].as<int>();
    if (num_executors <= 0)
        throw std::invalid_argument("number of executors to use must be larger than 0");

    if (std::size_t(num_executors) > num_os_threads)
        throw std::invalid_argument("number of executors to use must be \
                                        smaller than number of OS threads");

    std::size_t num_cores_per_executor = vm["cores"].as<int>();

    if ((num_executors - 1) * num_cores_per_executor > num_os_threads)
        throw std::invalid_argument("number of cores per executor should not \
                                     cause oversubscription");

    if (0 == tasks)
        throw std::invalid_argument("count of 0 tasks specified\n");

    // Start the clock.
    high_resolution_timer t;

    // create the executor instances
    using hpx::threads::executors::local_priority_queue_executor;

    {
        std::vector<local_priority_queue_executor> executors;
        for (std::size_t i = 0; i != std::size_t(num_executors); ++i)
        {
            // make sure we don't oversubscribe the cores, the last executor will
            // be bound to the remaining number of cores
            if ((i + 1) * num_cores_per_executor > num_os_threads)
            {
                HPX_TEST_EQ(i, std::size_t(num_executors) - 1);
                num_cores_per_executor = num_os_threads - i * num_cores_per_executor;
            }
            executors.push_back(local_priority_queue_executor(num_cores_per_executor));
        }

        t.restart();

        // schedule normal threads
        for (std::uint64_t i = 0; i < tasks; ++i)
            executors[i % num_executors].add(
                hpx::util::bind(&worker_timed, delay * 1000));
    // destructors of executors will wait for all tasks to finish executing
    }

    print_results(num_os_threads, t.elapsed());

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(
    int argc
  , char* argv[]
    )
{
    // Configure application-specific options.
    options_description cmdline("usage: " HPX_APPLICATION_STRING " [options]");

    cmdline.add_options()
        ( "tasks"
        , value<std::uint64_t>(&tasks)->default_value(500000)
        , "number of tasks to invoke")

        ( "delay"
        , value<std::uint64_t>(&delay)->default_value(0)
        , "number of iterations in the delay loop")

        ( "executors,e"
        , value<int>()->default_value(1)
        , "number of executor instances to use")

        ( "cores"
        , value<int>()->default_value(1)
        , "number of cores to bind to each of the executor instances")

        ( "no-header"
        , "do not print out the csv header row")
        ;

    // Initialize and run HPX.
    hpx::init_params init_args;
    init_args.desc_cmdline = cmdline;

    return hpx::init(argc, argv, init_args);
}

