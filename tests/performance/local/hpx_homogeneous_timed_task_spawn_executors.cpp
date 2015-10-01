//  Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/thread_executors.hpp>

#include "worker_timed.hpp"

#include <stdexcept>

#include <boost/format.hpp>
#include <boost/cstdint.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::init;
using hpx::finalize;
using hpx::get_os_thread_count;

using hpx::applier::register_work;

using hpx::this_thread::suspend;
using hpx::threads::get_thread_count;

using hpx::util::high_resolution_timer;

using hpx::cout;
using hpx::flush;

///////////////////////////////////////////////////////////////////////////////
// Command-line variables.
boost::uint64_t tasks = 500000;
boost::uint64_t delay = 0;
bool header = true;

///////////////////////////////////////////////////////////////////////////////
void print_results(
    boost::uint64_t cores
  , double walltime
    )
{
    if (header)
        cout << "OS-threads,Tasks,Delay (iterations),"
                "Total Walltime (seconds),Walltime per Task (seconds)\n"
             << flush;

    std::string const cores_str = boost::str(boost::format("%lu,") % cores);
    std::string const tasks_str = boost::str(boost::format("%lu,") % tasks);
    std::string const delay_str = boost::str(boost::format("%lu,") % delay);

    cout << ( boost::format("%-21s %-21s %-21s %10.12s, %10.12s\n")
            % cores_str % tasks_str % delay_str
            % walltime % (walltime / tasks)) << flush;
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
                HPX_ASSERT(i == std::size_t(num_executors) - 1);
                num_cores_per_executor = num_os_threads - i * num_cores_per_executor;
            }
            executors.push_back(local_priority_queue_executor(num_cores_per_executor));
        }

        t.restart();

        // schedule normal threads
        for (boost::uint64_t i = 0; i < tasks; ++i)
            executors[i % num_executors].add(
                hpx::util::bind(&worker_timed, delay * 1000));
    // destructors of executors will wait for all tasks to finish executing
    }

    print_results(num_os_threads, t.elapsed());

    return finalize();
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
        , value<boost::uint64_t>(&tasks)->default_value(500000)
        , "number of tasks to invoke")

        ( "delay"
        , value<boost::uint64_t>(&delay)->default_value(0)
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
    return init(cmdline, argc, argv);
}

