//  Copyright (c) 2018 Mikael Simberg
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/format.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/apply.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/parallel_for_loop.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/lcos/wait_each.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/testing.hpp>
#include <hpx/timing.hpp>
#include <hpx/util/annotated_function.hpp>
#include <hpx/util/yield_while.hpp>

#include <hpx/include/parallel_execution.hpp>
#include <hpx/lcos/local/sliding_semaphore.hpp>
#include <hpx/runtime/threads/executors/limiting_executor.hpp>
#include <hpx/runtime/threads/executors/pool_executor.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using hpx::program_options::options_description;
using hpx::program_options::value;
using hpx::program_options::variables_map;

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
    double us = 1e6 * duration / count;
    if (csv)
    {
        hpx::util::format_to(cout,
            "{1}, {:27}, {:15}, {:18}, {:8}, {:8}, {:20}, {:4}, {:4}, "
            "{:20}\n",
            count, title, wait, exec, duration, us, queuing, numa_sensitive,
            num_threads, info_string)
            << flush;
    }
    else
    {
        hpx::util::format_to(cout,
            "invoked {:1}, futures {:27} {:15} {:18} in {:8} seconds : {:8} "
            "us/future, queue {:20}, numa {:4}, threads {:4}, info {:20}\n",
            count, title, wait, exec, duration, us, queuing, numa_sensitive,
            num_threads, info_string)
            << flush;
    }
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
std::uint64_t num_iterations = 0;

///////////////////////////////////////////////////////////////////////////////
double null_function()
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

HPX_PLAIN_ACTION(null_function, null_action)

struct scratcher
{
    void operator()(future<double> r) const
    {
        global_scratch += r.get();
    }
};

// Time async action execution using wait each on futures vector
void measure_action_futures_wait_each(std::uint64_t count, bool csv)
{
    const id_type here = find_here();
    std::vector<future<double>> futures;
    futures.reserve(count);

    // start the clock
    high_resolution_timer walltime;
    for (std::uint64_t i = 0; i < count; ++i)
        futures.push_back(async<null_action>(here));
    wait_each(scratcher(), futures);

    // stop the clock
    const double duration = walltime.elapsed();
    print_stats("action", "WaitEach", "no-executor", count, duration, csv);
}

// Time async action execution using wait each on futures vector
void measure_action_futures_wait_all(std::uint64_t count, bool csv)
{
    const id_type here = find_here();
    std::vector<future<double>> futures;
    futures.reserve(count);

    // start the clock
    high_resolution_timer walltime;
    for (std::uint64_t i = 0; i < count; ++i)
        futures.push_back(async<null_action>(here));
    wait_all(futures);

    // stop the clock
    const double duration = walltime.elapsed();
    print_stats("action", "WaitAll", "no-executor", count, duration, csv);
}

// Time async execution using wait each on futures vector
template <typename Executor>
void measure_function_futures_wait_each(
    std::uint64_t count, bool csv, Executor& exec)
{
    std::vector<future<double>> futures;
    futures.reserve(count);

    // start the clock
    high_resolution_timer walltime;
    for (std::uint64_t i = 0; i < count; ++i)
        futures.push_back(async(exec, &null_function));
    wait_each(scratcher(), futures);

    // stop the clock
    const double duration = walltime.elapsed();
    print_stats("async", "WaitEach", ExecName(exec), count, duration, csv);
}

template <typename Executor>
void measure_function_futures_wait_all(
    std::uint64_t count, bool csv, Executor& exec)
{
    std::vector<future<double>> futures;
    futures.reserve(count);

    // start the clock
    high_resolution_timer walltime;
    for (std::uint64_t i = 0; i < count; ++i)
        futures.push_back(async(exec, &null_function));
    wait_all(futures);

    const double duration = walltime.elapsed();
    print_stats("async", "WaitAll", ExecName(exec), count, duration, csv);
}

template <typename Executor>
void measure_function_futures_thread_count(
    std::uint64_t count, bool csv, Executor& exec)
{
    std::atomic<std::uint64_t> sanity_check(count);
    auto this_pool = hpx::this_thread::get_pool();

    // start the clock
    high_resolution_timer walltime;
    for (std::uint64_t i = 0; i < count; ++i)
    {
        hpx::apply(exec, [&sanity_check]() {
            null_function();
            sanity_check--;
        });
    }

    // Yield until there is only this and background threads left.
    hpx::util::yield_while([this_pool]() {
        auto u = this_pool->get_thread_count_unknown(std::size_t(-1), false);
        auto b = this_pool->get_background_thread_count() + 1;
        return u > b;
    });

    // stop the clock
    const double duration = walltime.elapsed();

    if (sanity_check != 0)
    {
        int count = this_pool->get_thread_count_unknown(std::size_t(-1), false);
        throw std::runtime_error(
            "This test is faulty " + std::to_string(count));
    }

    print_stats("apply", "ThreadCount", ExecName(exec), count, duration, csv);
}

template <typename Executor>
void measure_function_futures_limiting_executor(
    std::uint64_t count, bool csv, Executor exec)
{
    using namespace hpx::parallel::execution;
    std::uint64_t const num_threads = hpx::get_num_worker_threads();
    std::uint64_t const tasks = num_threads * 2000;
    std::atomic<std::uint64_t> sanity_check(count);

    // start the clock
    high_resolution_timer walltime;
    {
        hpx::threads::executors::limiting_executor<Executor> signal_exec(
            exec, tasks, tasks + 1000);
        hpx::parallel::for_loop(
            hpx::parallel::execution::par, 0, count, [&](int) {
                hpx::apply(signal_exec, [&]() {
                    null_function();
                    sanity_check--;
                });
            });
    }

    if (sanity_check != 0)
    {
        throw std::runtime_error(
            "This test is faulty " + std::to_string(sanity_check));
    }

    // stop the clock
    const double duration = walltime.elapsed();
    print_stats("apply", "limiting-Exec", ExecName(exec), count, duration, csv);
}

template <typename Executor>
void measure_function_futures_sliding_semaphore(
    std::uint64_t count, bool csv, Executor& exec)
{
    // start the clock
    high_resolution_timer walltime;
    const int sem_count = 5000;
    hpx::lcos::local::sliding_semaphore sem(sem_count);
    for (std::uint64_t i = 0; i < count; ++i)
    {
        hpx::async(exec, [i, &sem]() {
            null_function();
            sem.signal(i);
        });
        sem.wait(i);
    }
    sem.wait(count + sem_count - 1);

    // stop the clock
    const double duration = walltime.elapsed();
    print_stats("apply", "Sliding-Sem", ExecName(exec), count, duration, csv);
}

struct unlimited_number_of_chunks
{
    template <typename Executor>
    std::size_t maximal_number_of_chunks(
        Executor&& executor, std::size_t cores, std::size_t num_tasks)
    {
        return num_tasks;
    }
};

namespace hpx { namespace parallel { namespace execution {
    template <>
    struct is_executor_parameters<unlimited_number_of_chunks> : std::true_type
    {
    };
}}}    // namespace hpx::parallel::execution

void measure_function_futures_for_loop(std::uint64_t count, bool csv)
{
    // start the clock
    high_resolution_timer walltime;
    hpx::parallel::for_loop(hpx::parallel::execution::par.with(
                                hpx::parallel::execution::static_chunk_size(1),
                                unlimited_number_of_chunks()),
        0, count, [](std::uint64_t) {
            null_function();
        });

    // stop the clock
    const double duration = walltime.elapsed();
    print_stats("for_loop", "par", "parallel_executor", count, duration, csv);
}

void measure_function_futures_register_work(std::uint64_t count, bool csv)
{
    hpx::lcos::local::latch l(count);

    // start the clock
    high_resolution_timer walltime;
    for (std::uint64_t i = 0; i < count; ++i)
    {
        hpx::applier::register_work_nullary([&l]() {
            null_function();
            l.count_down(1);
        });
    }
    l.wait();

    // stop the clock
    const double duration = walltime.elapsed();
    print_stats("register_work", "latch", "none", count, duration, csv);
}

void measure_function_futures_create_thread(std::uint64_t count, bool csv)
{
    hpx::lcos::local::latch l(count);

    auto const sched = hpx::threads::get_self_id()->get_scheduler_base();
    auto func = [&l]() {
        null_function();
        l.count_down(1);
    };
    auto const thread_func =
        hpx::applier::detail::thread_function_nullary<decltype(func)>{func};
    auto const desc = hpx::util::thread_description();
    auto const prio = hpx::threads::thread_priority_normal;
    auto const hint = hpx::threads::thread_schedule_hint();
    auto const stack_size =
        hpx::threads::get_stack_size(hpx::threads::thread_stacksize_small);
    hpx::error_code ec;

    // start the clock
    high_resolution_timer walltime;
    for (std::uint64_t i = 0; i < count; ++i)
    {
        auto init = hpx::threads::thread_init_data(
            hpx::threads::thread_function_type(thread_func), desc, prio, hint,
            stack_size, sched);
        sched->create_thread(init, nullptr, hpx::threads::pending, false, ec);
    }
    l.wait();

    // stop the clock
    const double duration = walltime.elapsed();
    print_stats("create_thread", "latch", "none", count, duration, csv);
}

void measure_function_futures_create_thread_hierarchical_placement(
    std::uint64_t count, bool csv)
{
    hpx::lcos::local::latch l(count);

    auto sched = hpx::threads::get_self_id()->get_scheduler_base();
    auto const func = [&l]() {
        null_function();
        l.count_down(1);
    };
    auto const thread_func =
        hpx::applier::detail::thread_function_nullary<decltype(func)>{func};
    auto const desc = hpx::util::thread_description();
    auto const prio = hpx::threads::thread_priority_normal;
    auto const stack_size =
        hpx::threads::get_stack_size(hpx::threads::thread_stacksize_small);
    auto const num_threads = hpx::get_num_worker_threads();
    hpx::error_code ec;

    // start the clock
    high_resolution_timer walltime;
    for (std::size_t t = 0; t < num_threads; ++t)
    {
        auto const hint = hpx::threads::thread_schedule_hint(t);
        auto spawn_func = [&thread_func, sched, hint, t, count, num_threads,
                              desc, stack_size]() {
            std::uint64_t const count_start = t * count / num_threads;
            std::uint64_t const count_end = (t + 1) * count / num_threads;
            hpx::error_code ec;
            for (std::uint64_t i = count_start; i < count_end; ++i)
            {
                hpx::threads::thread_init_data init(
                    hpx::threads::thread_function_type(thread_func), desc, prio,
                    hint, stack_size, sched);
                sched->create_thread(
                    init, nullptr, hpx::threads::pending, false, ec);
            }
        };
        auto const thread_spawn_func =
            hpx::applier::detail::thread_function_nullary<decltype(spawn_func)>{
                spawn_func};

        hpx::threads::thread_init_data init(
            hpx::threads::thread_function_type(thread_spawn_func), desc, prio,
            hint, stack_size, sched);
        sched->create_thread(init, nullptr, hpx::threads::pending, false, ec);
    }
    l.wait();

    // stop the clock
    const double duration = walltime.elapsed();
    print_stats(
        "create_thread_hierarchical", "latch", "none", count, duration, csv);
}

void measure_function_futures_apply_hierarchical_placement(
    std::uint64_t count, bool csv)
{
    hpx::lcos::local::latch l(count);

    auto const func = [&l]() {
        null_function();
        l.count_down(1);
    };
    auto const num_threads = hpx::get_num_worker_threads();

    // start the clock
    high_resolution_timer walltime;
    for (std::size_t t = 0; t < num_threads; ++t)
    {
        auto const hint = hpx::threads::thread_schedule_hint(t);
        auto spawn_func = [&func, hint, t, count, num_threads]() {
            auto exec = hpx::threads::executors::default_executor(hint);
            std::uint64_t const count_start = t * count / num_threads;
            std::uint64_t const count_end = (t + 1) * count / num_threads;

            for (std::uint64_t i = count_start; i < count_end; ++i)
            {
                hpx::apply(exec, func);
            }
        };

        auto exec = hpx::threads::executors::default_executor(hint);
        hpx::apply(exec, spawn_func);
    }
    l.wait();

    // stop the clock
    const double duration = walltime.elapsed();
    print_stats("apply_hierarchical", "latch", "default_executor", count,
        duration, csv);
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
        bool csv = vm.count("csv") != 0;
        if (HPX_UNLIKELY(0 == count))
            throw std::logic_error("error: count of 0 futures specified\n");

        hpx::parallel::execution::default_executor def;
        hpx::parallel::execution::parallel_executor par;

        for (int i = 0; i < repetitions; i++)
        {
            measure_function_futures_limiting_executor(count, csv, def);
            measure_function_futures_limiting_executor(count, csv, par);
            if (test_all)
            {
                measure_action_futures_wait_each(count, csv);
                measure_action_futures_wait_all(count, csv);
                measure_function_futures_wait_each(count, csv, def);
                measure_function_futures_wait_each(count, csv, par);
                measure_function_futures_wait_all(count, csv, def);
                measure_function_futures_wait_all(count, csv, par);
                measure_function_futures_thread_count(count, csv, def);
                measure_function_futures_thread_count(count, csv, par);
                measure_function_futures_sliding_semaphore(count, csv, def);
                measure_function_futures_sliding_semaphore(count, csv, par);
                measure_function_futures_for_loop(count, csv);
                measure_function_futures_register_work(count, csv);
                measure_function_futures_create_thread(count, csv);
                measure_function_futures_create_thread_hierarchical_placement(
                    count, csv);
                measure_function_futures_apply_hierarchical_placement(
                    count, csv);
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
    cmdline.add_options()("futures",
        value<std::uint64_t>()->default_value(500000),
        "number of futures to invoke")

        ("delay-iterations", value<std::uint64_t>()->default_value(0),
         "number of iterations in the delay loop")

        ("csv", "output results as csv (format: count,duration)")
        ("test-all", "run all benchmarks")
        ("repetitions", value<int>()->default_value(1),
         "number of repetitions of the full benchmark")

        ("info", value<std::string>()->default_value("none"),
         "extra info for plot output (e.g. branch name)");
    // clang-format on

    // Initialize and run HPX.
    return init(cmdline, argc, argv);
}
