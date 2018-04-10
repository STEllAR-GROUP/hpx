//  Copyright (c) 2018 Mikael Simberg
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This example benchmarks the time it takes to start and stop the HPX runtime.
// This is meant to be compared to resume_suspend and openmp_parallel_region.

#include <hpx/hpx.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/util/high_resolution_clock.hpp>
#include <hpx/util/yield_while.hpp>

#include <boost/program_options.hpp>

#include <cstddef>
#include <iostream>

int hpx_main()
{
    return hpx::finalize();
}

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

    hpx::start(desc_commandline, argc, argv);
    std::uint64_t threads = hpx::resource::get_num_threads("default");
    hpx::stop();

    std::cout
        << "threads, resume [s], async [s], suspend [s]"
        << std::endl;

    hpx::util::high_resolution_timer timer;

    for (std::size_t i = 0; i < repetitions; ++i)
    {
        timer.restart();

        hpx::start(desc_commandline, argc, argv);
        auto t_start = timer.elapsed();

        for (std::size_t thread = 0; thread < threads; ++thread)
        {
            hpx::async([](){});
        }

        auto t_async = timer.elapsed();

        hpx::stop();
        auto t_stop = timer.elapsed();

        std::cout
            << threads << ", "
            << t_start << ", "
            << t_async << ", "
            << t_stop
            << std::endl;
    }
}

