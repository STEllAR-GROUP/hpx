//  Copyright (c) 2018=2019 Mikael Simberg
//  Copyright (c) 2018-2019 John Biddiscombe
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/util/format.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/apply.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/parallel_for_loop.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/lcos/wait_each.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
//#include <hpx/testing.hpp>
//#include <hpx/timing.hpp>
#include <hpx/util/annotated_function.hpp>
#include <hpx/util/yield_while.hpp>

#include <hpx/include/parallel_executor_parameters.hpp>
#include <hpx/include/parallel_execution.hpp>
#include <hpx/runtime/threads/executors/limiting_executor.hpp>
#include <hpx/runtime/threads/executors/pool_executor.hpp>
//#include <hpx/synchronization.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using boost::program_options::options_description;
using boost::program_options::value;
using boost::program_options::variables_map;

using hpx::finalize;
using hpx::init;

using hpx::find_here;
using hpx::naming::id_type;

using hpx::apply;
using hpx::async;
using hpx::future;
using hpx::lcos::wait_each;

using hpx::util::high_resolution_timer;

using hpx::cout;
using hpx::flush;

// global vars we stick here to make printouts easy for plotting
static std::string queuing = "default";
static std::size_t numa_sensitive = 0;
static std::uint64_t num_threads = 1;
static std::string info_string = "";

///////////////////////////////////////////////////////////////////////////////
void print_stats(const char* title, const char* wait, const char* exec,
    std::int64_t count, double duration, bool csv)
{
    std::ostringstream temp;
    double us = 1e6 * duration / count;
    if (csv)
    {
        hpx::util::format_to(temp,
            "{1}, {:27}, {:15}, {:18}, {:8}, {:8}, {:20}, {:4}, {:4}, "
            "{:20}\n",
            count, title, wait, exec, duration, us, queuing, numa_sensitive,
            num_threads, info_string);
    }
    else
    {
        hpx::util::format_to(temp,
            "num_iterations {:1}, {:27} {:15} {:18} in {:8} seconds : {:8} "
            "us/future, queue {:20}, numa {:4}, threads {:4}, info {:20}\n",
            count, title, wait, exec, duration, us, queuing, numa_sensitive,
            num_threads, info_string);
    }
    std::cout << temp.str() << std::endl;
    // CDash graph plotting
    //hpx::util::print_cdash_timing(title, duration);
}

const char* ExecName(const hpx::parallel::execution::parallel_executor& exec)
{
    return "parallel_executor";
}
const char* ExecName(const hpx::parallel::execution::default_executor& exec)
{
    return "default_executor";
}

///////////////////////////////////////////////////////////////////////////////
// we use globals here to prevent the delay from being optimized away
double global_scratch = 0;
std::uint64_t num_num_iterations = 0;

///////////////////////////////////////////////////////////////////////////////
void measure_function_futures_for_loop(std::uint64_t count, bool csv, std::uint64_t chunk_size, std::uint64_t iter_length)
{
    // start the clock
    high_resolution_timer walltime;
    hpx::parallel::for_loop(hpx::parallel::execution::par.with(
                                hpx::parallel::execution::dynamic_chunk_size( chunk_size )),
        0, count, [&](std::uint64_t) { hpx::this_thread::sleep_for(std::chrono::microseconds(iter_length)); });

    // stop the clock
    const double duration = walltime.elapsed();
    print_stats("for_loop", "par", "parallel_executor", count, duration, csv);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    {
        if (vm.count("hpx:queuing"))
            queuing = vm["hpx:queuing"].as<std::string>();

        if (vm.count("hpx:numa-sensitive"))
            numa_sensitive = 1;
        else
            numa_sensitive = 0;

        const int repetitions = vm["repetitions"].as<int>();

        num_threads = hpx::get_num_worker_threads();

        const std::uint64_t chunk_size = vm["chunk_size"].as<std::uint64_t>();
        const std::uint64_t iter_length = vm["iter_length"].as<std::uint64_t>();

        const std::uint64_t count = vm["num_iterations"].as<std::uint64_t>();
        bool csv = vm.count("csv") != 0;
        if (HPX_UNLIKELY(0 == count))
            throw std::logic_error("error: count of 0 futures specified\n");

        //hpx::parallel::execution::default_executor def;
        //hpx::parallel::execution::parallel_executor par;

        for (int i = 0; i < repetitions; i++)
        {
            measure_function_futures_for_loop(count, csv, chunk_size, iter_length);
        }
    }

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description cmdline("usage: " HPX_APPLICATION_STRING " [options]");

    // clang-format off
    cmdline.add_options()("num_iterations",
        value<std::uint64_t>()->default_value(500000),
        "number of iterations to invoke")

        ("csv", "output results as csv (format: count,duration)")
        ("repetitions", value<int>()->default_value(1),
         "number of repetitions of the full benchmark")

        ("iter_length",value<std::uint64_t>()->default_value(1), "length of each iteration")
    	("chunk_size",value<std::uint64_t>()->default_value(1), "chunk size");
    // clang-format on

    // Initialize and run HPX.
    return init(cmdline, argc, argv);
}
