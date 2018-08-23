//  Copyright (c) 2018 Mikael Simberg
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This example benchmarks the time it takes to resume and suspend the HPX
// runtime. This is meant to be compared to start_stop and
// openmp_parallel_region.

#include <hpx/hpx.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/util/high_resolution_clock.hpp>
#include <hpx/util/yield_while.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/program_options.hpp>

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

    hpx::start(nullptr, desc_commandline, argc, argv);
    hpx::runtime* rt = hpx::get_runtime_ptr();
    hpx::util::yield_while([rt]()
        {
            return rt->get_state() < hpx::state_running;
        });
    hpx::suspend();

    std::uint64_t threads = hpx::resource::get_num_threads("default");

    std::cout
        << "threads, resume [s], async [s], suspend [s]"
        << std::endl;

    double suspend_time = 0;
    double resume_time  = 0;
    hpx::util::high_resolution_timer timer;

    for (std::size_t i = 0; i < repetitions; ++i)
    {
        timer.restart();

        hpx::resume();
        auto t_resume = timer.elapsed();
        resume_time += t_resume;

        for (std::size_t thread = 0; thread < threads; ++thread)
        {
            hpx::async([](){});
        }

        auto t_async = timer.elapsed();

        hpx::suspend();
        auto t_suspend = timer.elapsed();
        suspend_time += t_suspend;

        std::cout
            << threads << ", "
            << t_resume << ", "
            << t_async << ", "
            << t_suspend
            << std::endl;
    }

    hpx::util::print_cdash_timing("ResumeTime",  resume_time);
    hpx::util::print_cdash_timing("SuspendTime", suspend_time);

    hpx::resume();
    hpx::async([]() { hpx::finalize(); });
    hpx::stop();
}


