//  Copyright (c) 2018 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This example benchmarks the time it takes to resume and suspend the HPX
// runtime. This is meant to be compared to start_stop and
// openmp_parallel_region.

#include <hpx/hpx.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/hpx_suspend.hpp>
#include <hpx/execution_base/this_thread.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/modules/timing.hpp>

#include <hpx/modules/program_options.hpp>

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

    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;

    hpx::start(nullptr, argc, argv, init_args);
    hpx::suspend();

    std::uint64_t threads = hpx::resource::get_num_threads("default");

    std::cout
        << "threads, resume [s], apply [s], suspend [s]"
        << std::endl;

    double suspend_time = 0;
    double resume_time  = 0;
    hpx::chrono::high_resolution_timer timer;

    for (std::size_t i = 0; i < repetitions; ++i)
    {
        timer.restart();

        hpx::resume();
        auto t_resume = timer.elapsed();
        resume_time += t_resume;

        for (std::size_t thread = 0; thread < threads; ++thread)
        {
            hpx::apply([](){});
        }

        auto t_apply = timer.elapsed();

        hpx::suspend();
        auto t_suspend = timer.elapsed();
        suspend_time += t_suspend;

        std::cout
            << threads << ", "
            << t_resume << ", "
            << t_apply << ", "
            << t_suspend
            << std::endl;
    }

    hpx::util::print_cdash_timing("ResumeTime",  resume_time);
    hpx::util::print_cdash_timing("SuspendTime", suspend_time);

    hpx::resume();
    hpx::apply([]() { hpx::finalize(); });
    hpx::stop();
}
