//  Copyright (c) 2018-2019 Mikael Simberg
//  Copyright (c) 2018-2019 John Biddiscombe
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/parallel_algorithm.hpp>
#include <hpx/include/parallel_execution.hpp>
#include <hpx/include/parallel_executor_parameters.hpp>
//#include <hpx/timing.hpp>

//#include <hpx/execution/executors/parallel_executor_aggregated.hpp>

#include "worker_timed.hpp"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
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
    double us = 1e6 * duration;
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
            "num_iterations {:1}, {:27} {:15} {:18} in {:8} microseconds "
            ", queue {:20}, numa {:4}, threads {:4}\n",
            count, title, wait, exec, us, queuing, numa_sensitive, num_threads);
    }
    std::cout << temp.str() << std::endl;
    // CDash graph plotting
    //hpx::util::print_cdash_timing(title, duration);
}

const char* ExecName(const hpx::parallel::execution::parallel_executor& exec)
{
    return "parallel_executor";
}

///////////////////////////////////////////////////////////////////////////////
// we use globals here to prevent the delay from being optimized away
double global_scratch = 0;
std::uint64_t num_num_iterations = 0;

///////////////////////////////////////////////////////////////////////////////
void measure_function_futures_for_loop(std::uint64_t count, bool csv,
    std::uint64_t chunk_size, std::uint64_t iter_length)
{
    // start the clock
    high_resolution_timer walltime;
    hpx::parallel::for_loop(
        hpx::parallel::execution::par.with(
            hpx::parallel::execution::dynamic_chunk_size(chunk_size)),
        0, count, [&](std::uint64_t) { worker_timed(iter_length * 1000); });

    // stop the clock
    const double duration = walltime.elapsed();
    print_stats("for_loop", "par", "parallel_executor", count, duration, csv);
}

void measure_function_futures_for_loopctr(std::uint64_t count, bool csv,
    std::uint64_t chunk_size, std::uint64_t iter_length)
{
    // start the clock
    high_resolution_timer walltime;
    hpx::evaluate_active_counters(true, "Initialization");

    hpx::parallel::for_loop(
        hpx::parallel::execution::par.with(
            hpx::parallel::execution::dynamic_chunk_size(chunk_size)),
        0, count, [&](std::uint64_t) { worker_timed(iter_length * 1000); });
    hpx::evaluate_active_counters(false, "Done");

    // stop the clock
    const double duration = walltime.elapsed();
    print_stats("for_loop", "par", "parallel_executor", count, duration, csv);
}

void measure_function_futures_for_loop_sptctr(std::uint64_t count, bool csv,
    std::uint64_t iter_length,
    hpx::parallel::execution::splittable_mode split_type)
{
    // start the clock
    high_resolution_timer walltime;
    hpx::evaluate_active_counters(true, "Initialization");

    hpx::parallel::for_loop(
        hpx::parallel::execution::par.on(
            hpx::parallel::execution::splittable_executor(split_type)),
        0, count, [&](std::uint64_t) { worker_timed(iter_length * 1000); });

    hpx::evaluate_active_counters(false, "Done");
    // stop the clock
    const double duration = walltime.elapsed();
    std::cout << "split type:" << get_splittable_mode_name(split_type)
              << std::endl;
    print_stats("for_loop", "par", "splittable_executor", count, duration, csv);
}

void measure_function_futures_for_loop_spt(std::uint64_t count, bool csv,
    std::uint64_t iter_length,
    hpx::parallel::execution::splittable_mode split_type)
{
    // start the clock
    high_resolution_timer walltime;
    hpx::parallel::for_loop(
        hpx::parallel::execution::par.on(
            hpx::parallel::execution::splittable_executor(split_type)),
        0, count, [&](std::uint64_t) { worker_timed(iter_length * 1000); });

    // stop the clock
    const double duration = walltime.elapsed();
    std::cout << "split type:" << get_splittable_mode_name(split_type)
              << std::endl;

    print_stats("for_loop", "par", "splittable_executor", count, duration, csv);
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

        int const repetitions = vm["repetitions"].as<int>();

        num_threads = hpx::get_num_worker_threads();

        std::uint64_t const chunk_size = vm["chunk_size"].as<std::uint64_t>();
        std::uint64_t const iter_length = vm["iter_length"].as<std::uint64_t>();

        std::uint64_t const count = vm["num_iterations"].as<std::uint64_t>();
        if (HPX_UNLIKELY(0 == count))
        {
            throw std::logic_error("error: count of 0 futures specified\n");
        }

        bool csv = vm.count("csv") != 0;
        bool spt = vm.count("spt") != 0;
        bool ctr = vm.count("counter") != 0;

        if (spt)
        {
            std::string const split_type = vm["split_type"].as<std::string>();
            hpx::parallel::execution::splittable_mode split_mode =
                hpx::parallel::execution::splittable_mode::all;
            if (split_type == "idle")
            {
                split_mode = hpx::parallel::execution::splittable_mode::idle;
            }
            else if (split_type == "idle_mask")
            {
                split_mode =
                    hpx::parallel::execution::splittable_mode::idle_mask;
            }

            if (ctr)
            {
                for (int i = 0; i < repetitions; i++)
                {
                    measure_function_futures_for_loop_sptctr(
                        count, csv, iter_length, split_mode);
                }
            }
            else
            {
                for (int i = 0; i < repetitions; i++)
                {
                    measure_function_futures_for_loop_spt(
                        count, csv, iter_length, split_mode);
                }
            }
        }

        else
        {
            if (ctr)
            {
                for (int i = 0; i < repetitions; i++)
                {
                    measure_function_futures_for_loopctr(
                        count, csv, chunk_size, iter_length);
                }
            }
            else
            {
                for (int i = 0; i < repetitions; i++)
                {
                    measure_function_futures_for_loop(
                        count, csv, chunk_size, iter_length);
                }
            }
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
    cmdline.add_options()
        ("num_iterations", value<std::uint64_t>()->default_value(500000),
            "number of iterations to invoke")
        ("csv", "output results as csv (format: count,duration)")
        ("repetitions", value<int>()->default_value(1),
            "number of repetitions of the full benchmark")
        ("spt","run using splittable executor")
        ("split_type", value<std::string>()->default_value("all"),
            "split tasks based on idle cores  or all cores")
        ("iter_length", value<std::uint64_t>()->default_value(1),
            "length of each iteration")
        ("chunk_size", value<std::uint64_t>()->default_value(1), "chunk size")
        ("counter", "print data collected from performance counters");
    // clang-format on

    // Initialize and run HPX.
    return init(cmdline, argc, argv);
}
