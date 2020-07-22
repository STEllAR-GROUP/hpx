//  Copyright (c) 2018 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This example benchmarks the time it takes to enter and exit an OpenMP
// parallel region. This is meant to be compared to resume_suspend and
// start_stop.

#include <hpx/hpx.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/modules/timing.hpp>
#include <hpx/execution_base/this_thread.hpp>

#include <hpx/modules/program_options.hpp>
#include <omp.h>

#include <cstddef>
#include <iostream>

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

    // Do one warmup iteration and get the number of threads
    int x = 0;
#   pragma omp parallel
    {
        x += 1;
    }
    (void) x;

    std::size_t threads = omp_get_max_threads();

    std::cout
        << "threads, parallel region [s]"
        << std::endl;

    hpx::util::high_resolution_timer timer;

    for (std::size_t i = 0; i < repetitions; ++i)
    {
        timer.restart();

        // TODO: Is there a more minimal way of starting all OpenMP threads?
#       pragma omp parallel
        {
            x += 1;
        }

        auto t_parallel = timer.elapsed();

        std::cout
            << threads << ", "
            << t_parallel
            << std::endl;
    }
}
