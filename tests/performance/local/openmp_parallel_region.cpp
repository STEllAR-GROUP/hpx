//  Copyright (c) 2018 Mikael Simberg
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This example benchmarks the time it takes to enter and exit an OpenMP
// parallel region. This is meant to be compared to resume_suspend and
// start_stop.

#include <hpx/hpx.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/util/high_resolution_clock.hpp>
#include <hpx/util/yield_while.hpp>

#include <boost/program_options.hpp>
#include <omp.h>

#include <cstddef>
#include <iostream>

int main(int argc, char ** argv)
{
    boost::program_options::options_description desc_commandline;
    desc_commandline.add_options()
        ("repetitions",
         boost::program_options::value<std::uint64_t>()->default_value(100),
         "Number of repetitions");

    boost::program_options::variables_map vm;
    boost::program_options::store(
        boost::program_options::command_line_parser(argc, argv)
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
