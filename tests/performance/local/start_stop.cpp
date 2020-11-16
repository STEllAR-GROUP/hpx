//  Copyright (c) 2018 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This example benchmarks the time it takes to start and stop the HPX runtime.
// This is meant to be compared to resume_suspend and openmp_parallel_region.

#include <hpx/hpx.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/execution_base/this_thread.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/modules/timing.hpp>

#include <hpx/modules/program_options.hpp>

#include <cstddef>
#include <iostream>

int hpx_main()
{
    return hpx::finalize();
}

int main(int argc, char ** argv)
{
    hpx::program_options::options_description desc_commandline;
    desc_commandline.add_options()
        ("repetitions",
         hpx::program_options::value<std::uint64_t>()->default_value(100),
         "Number of repetitions");

    hpx::program_options::variables_map vm;
    hpx::program_options::store(
        hpx::program_options::command_line_parser(argc, argv)
        .allow_unregistered()
        .options(desc_commandline)
        .run(),
        vm);

    std::uint64_t repetitions = vm["repetitions"].as<std::uint64_t>();

    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;

    hpx::start(argc, argv, init_args);
    std::uint64_t threads = hpx::resource::get_num_threads("default");
    hpx::stop();

    std::cout
        << "threads, resume [s], apply [s], suspend [s]"
        << std::endl;

    double start_time = 0;
    double stop_time  = 0;
    hpx::chrono::high_resolution_timer timer;

    for (std::size_t i = 0; i < repetitions; ++i)
    {
        timer.restart();

        hpx::init_params init_args;
        init_args.desc_cmdline = desc_commandline;

        hpx::start(argc, argv, init_args);
        auto t_start = timer.elapsed();
        start_time += t_start;

        for (std::size_t thread = 0; thread < threads; ++thread)
        {
            hpx::apply([](){});
        }

        auto t_apply = timer.elapsed();

        hpx::stop();
        auto t_stop = timer.elapsed();
        stop_time += t_stop;

        std::cout
            << threads << ", "
            << t_start << ", "
            << t_apply << ", "
            << t_stop
            << std::endl;
    }
    hpx::util::print_cdash_timing("StartTime", start_time);
    hpx::util::print_cdash_timing("StopTime",  stop_time);
}

