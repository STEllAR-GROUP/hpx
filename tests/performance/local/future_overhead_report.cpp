//  Copyright (c) 2018-2020 Mikael Simberg
//  Copyright (c) 2018-2019 John Biddiscombe
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/async_combinators/wait_each.hpp>
#include <hpx/execution_base/this_thread.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/apply.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/parallel_execution.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/modules/synchronization.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/modules/timing.hpp>
#include <hpx/threading_base/annotated_function.hpp>

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

using hpx::program_options::options_description;
using hpx::program_options::value;
using hpx::program_options::variables_map;

using hpx::apply;
using hpx::async;
using hpx::future;
using hpx::lcos::wait_each;

using hpx::chrono::high_resolution_timer;

// global vars we stick here to make printouts easy for plotting
static std::string queuing = "default";
static std::size_t numa_sensitive = 0;
static std::uint64_t num_threads = 1;
static std::string info_string = "";

const char* exec_name(hpx::execution::parallel_executor const&)
{
    return "parallel_executor";
}

const char* exec_name(
    hpx::parallel::execution::parallel_executor_aggregated const&)
{
    return "parallel_executor_aggregated";
}

///////////////////////////////////////////////////////////////////////////////
// we use globals here to prevent the delay from being optimized away
double global_scratch = 0;
std::uint64_t num_iterations = 0;

///////////////////////////////////////////////////////////////////////////////
double null_function() noexcept
{
    if (num_iterations > 0)
    {
        const int array_size = 4096;
        std::array<double, array_size> dummy;
        for (std::uint64_t i = 0; i < num_iterations; ++i)
        {
            for (std::uint64_t j = 0; j < array_size; ++j)
            {
                dummy[j] = 1.0 / (2.0 * i * j + 1.0);
            }
        }
        return dummy[0];
    }
    return 0.0;
}

struct scratcher
{
    void operator()(future<double> r) const
    {
        global_scratch += r.get();
    }
};

void measure_function_futures_create_thread_hierarchical_placement(
    std::uint64_t count, const int repetitions)
{
    auto sched = hpx::threads::get_self_id_data()->get_scheduler_base();

    if (std::string("core-shared_priority_queue_scheduler") ==
        sched->get_description())
    {
        sched->add_remove_scheduler_mode(
            hpx::threads::policies::scheduler_mode(
                hpx::threads::policies::assign_work_thread_parent),
            hpx::threads::policies::scheduler_mode(
                hpx::threads::policies::enable_stealing |
                hpx::threads::policies::enable_stealing_numa |
                hpx::threads::policies::assign_work_round_robin |
                hpx::threads::policies::steal_after_local |
                hpx::threads::policies::steal_high_priority_first));
    }
    auto const desc = hpx::util::thread_description();
    auto prio = hpx::threads::thread_priority::normal;
    auto const stack_size = hpx::threads::thread_stacksize::small_;
    auto const num_threads = hpx::get_num_worker_threads();
    hpx::error_code ec;

    hpx::util::perf_test_report(
        "future overhead - create_thread_hierarchical - latch", "no-executor",
        repetitions, [&]() -> void {
            hpx::lcos::local::latch l(count);

            auto const func = [&l]() {
                null_function();
                l.count_down(1);
            };
            auto const thread_func =
                hpx::threads::detail::thread_function_nullary<decltype(func)>{func};
            for (std::size_t t = 0; t < num_threads; ++t)
            {
                auto const hint = hpx::threads::thread_schedule_hint(
                    static_cast<std::int16_t>(t));
                auto spawn_func = [&thread_func, sched, hint, t, count,
                                      num_threads, desc, prio]() {
                    std::uint64_t const count_start = t * count / num_threads;
                    std::uint64_t const count_end =
                        (t + 1) * count / num_threads;
                    hpx::error_code ec;
                    for (std::uint64_t i = count_start; i < count_end; ++i)
                    {
                        hpx::threads::thread_init_data init(
                            hpx::threads::thread_function_type(thread_func),
                            desc, prio, hint, stack_size,
                            hpx::threads::thread_schedule_state::pending, false,
                            sched);
                        sched->create_thread(init, nullptr, ec);
                    }
                };
                auto const thread_spawn_func =
                    hpx::threads::detail::thread_function_nullary<decltype(
                        spawn_func)>{spawn_func};

                hpx::threads::thread_init_data init(
                    hpx::threads::thread_function_type(thread_spawn_func), desc,
                    prio, hint, stack_size,
                    hpx::threads::thread_schedule_state::pending, false, sched);
                sched->create_thread(init, nullptr, ec);
            }
            l.wait();
        });
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

        bool test_all = (vm.count("test-all") > 0);
        const int repetitions = vm["repetitions"].as<int>();

        if (vm.count("info"))
            info_string = vm["info"].as<std::string>();

        num_threads = hpx::get_num_worker_threads();

        num_iterations = vm["delay-iterations"].as<std::uint64_t>();

        const std::uint64_t count = vm["futures"].as<std::uint64_t>();
        if (HPX_UNLIKELY(0 == count))
            throw std::logic_error("error: count of 0 futures specified\n");

        if (test_all)
        {
            measure_function_futures_create_thread_hierarchical_placement(
                count, repetitions);
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
    cmdline.add_options()("futures",
        value<std::uint64_t>()->default_value(500000),
        "number of futures to invoke")

        ("delay-iterations", value<std::uint64_t>()->default_value(0),
         "number of iterations in the delay loop")

        ("test-all", "run all benchmarks")
        ("repetitions", value<int>()->default_value(1),
         "number of repetitions of the full benchmark")

        ("info", value<std::string>()->default_value("no-info"),
         "extra info for plot output (e.g. branch name)");
    // clang-format on

    // Initialize and run HPX.
    hpx::init_params init_args;
    init_args.desc_cmdline = cmdline;

    return hpx::init(argc, argv, init_args);
}
