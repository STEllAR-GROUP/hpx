//  Copyright (c) 2014 Grant Mercer
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/algorithm.hpp>
#include <hpx/chrono.hpp>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_executor_parameters.hpp>
#include <hpx/iostream.hpp>

#include <hpx/executors/parallel_executor_aggregated.hpp>

#include "worker_timed.hpp"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
int delay = 1000;
int test_count = 100;
int chunk_size = 0;
int num_overlapping_loops = 0;
bool disable_stealing = false;
bool fast_idle_mode = false;
unsigned int seed = std::random_device{}();
std::mt19937 gen(seed);

struct disable_stealing_parameter
{
    template <typename Executor>
    void mark_begin_execution(Executor&&)
    {
        hpx::threads::add_remove_scheduler_mode(
            hpx::threads::policies::enable_stealing,
            hpx::threads::policies::enable_idle_backoff);
    }

    template <typename Executor>
    void mark_end_of_scheduling(Executor&&)
    {
        hpx::threads::remove_scheduler_mode(
            hpx::threads::policies::enable_stealing);
    }

    template <typename Executor>
    void mark_end_execution(Executor&&)
    {
        hpx::threads::add_remove_scheduler_mode(
            hpx::threads::policies::enable_idle_backoff,
            hpx::threads::policies::enable_stealing);
    }
};

namespace hpx { namespace parallel { namespace execution {
    template <>
    struct is_executor_parameters<disable_stealing_parameter> : std::true_type
    {
    };
}}}    // namespace hpx::parallel::execution

///////////////////////////////////////////////////////////////////////////////
void measure_sequential_foreach(
    std::vector<std::size_t> const& data_representation)
{
    if (disable_stealing)
    {
        // disable stealing from inside the algorithm
        disable_stealing_parameter dsp;

        // invoke sequential for_each
        hpx::ranges::for_each(hpx::parallel::execution::seq.with(dsp),
            data_representation, [](std::size_t) { worker_timed(delay); });
    }
    else
    {
        // invoke sequential for_each
        hpx::ranges::for_each(hpx::parallel::execution::seq,
            data_representation, [](std::size_t) { worker_timed(delay); });
    }
}

template <typename Executor>
void measure_parallel_foreach(
    std::vector<std::size_t> const& data_representation, Executor&& exec)
{
    // create executor parameters object
    hpx::parallel::execution::static_chunk_size cs(chunk_size);

    if (disable_stealing)
    {
        // disable stealing from inside the algorithm
        disable_stealing_parameter dsp;

        // invoke parallel for_each
        hpx::ranges::for_each(
            hpx::parallel::execution::par.with(cs, dsp).on(exec),
            data_representation, [](std::size_t) { worker_timed(delay); });
    }
    else
    {
        // invoke parallel for_each
        hpx::ranges::for_each(hpx::parallel::execution::par.with(cs).on(exec),
            data_representation, [](std::size_t) { worker_timed(delay); });
    }
}

template <typename Executor>
hpx::future<void> measure_task_foreach(
    std::shared_ptr<std::vector<std::size_t>> data_representation,
    Executor&& exec)
{
    // create executor parameters object
    hpx::parallel::execution::static_chunk_size cs(chunk_size);

    if (disable_stealing)
    {
        // disable stealing from inside the algorithm
        disable_stealing_parameter dsp;

        // invoke parallel for_each
        return hpx::ranges::for_each(
            hpx::parallel::execution::par(hpx::parallel::execution::task)
                .with(cs, dsp)
                .on(exec),
            *data_representation, [](std::size_t) { worker_timed(delay); })
            .then([data_representation](hpx::future<void>) {});
    }
    else
    {
        // invoke parallel for_each
        return hpx::ranges::for_each(
            hpx::parallel::execution::par(hpx::parallel::execution::task)
                .with(cs)
                .on(exec),
            *data_representation, [](std::size_t) { worker_timed(delay); })
            .then([data_representation](hpx::future<void>) {});
    }
}

///////////////////////////////////////////////////////////////////////////////
void measure_sequential_forloop(
    std::vector<std::size_t> const& data_representation)
{
    using iterator = typename std::vector<std::size_t>::const_iterator;

    if (disable_stealing)
    {
        // disable stealing from inside the algorithm
        disable_stealing_parameter dsp;

        // invoke sequential for_loop
        hpx::for_loop(hpx::parallel::execution::seq.with(dsp),
            std::begin(data_representation), std::end(data_representation),
            [](iterator) { worker_timed(delay); });
    }
    else
    {
        // invoke sequential for_loop
        hpx::for_loop(hpx::parallel::execution::seq,
            std::begin(data_representation), std::end(data_representation),
            [](iterator) { worker_timed(delay); });
    }
}

template <typename Executor>
void measure_parallel_forloop(
    std::vector<std::size_t> const& data_representation, Executor&& exec)
{
    using iterator = typename std::vector<std::size_t>::const_iterator;

    // create executor parameters object
    hpx::parallel::execution::static_chunk_size cs(chunk_size);

    if (disable_stealing)
    {
        // disable stealing from inside the algorithm
        disable_stealing_parameter dsp;

        // invoke parallel for_loop
        hpx::for_loop(hpx::parallel::execution::par.with(cs, dsp).on(exec),
            std::begin(data_representation), std::end(data_representation),
            [](iterator) { worker_timed(delay); });
    }
    else
    {
        // invoke parallel for_loop
        hpx::for_loop(hpx::parallel::execution::par.with(cs).on(exec),
            std::begin(data_representation), std::end(data_representation),
            [](iterator) { worker_timed(delay); });
    }
}

template <typename Executor>
hpx::future<void> measure_task_forloop(
    std::shared_ptr<std::vector<std::size_t>> data_representation,
    Executor&& exec)
{
    using iterator = typename std::vector<std::size_t>::const_iterator;

    // create executor parameters object
    hpx::parallel::execution::static_chunk_size cs(chunk_size);

    if (disable_stealing)
    {
        // disable stealing from inside the algorithm
        disable_stealing_parameter dsp;

        // invoke parallel for_looph
        return hpx::for_loop(
            hpx::parallel::execution::par(hpx::parallel::execution::task)
                .with(cs, dsp)
                .on(exec),
            std::begin(*data_representation), std::end(*data_representation),
            [](iterator) { worker_timed(delay); })
            .then([data_representation](hpx::future<void>) {});
    }
    else
    {
        // invoke parallel for_looph
        return hpx::for_loop(
            hpx::parallel::execution::par(hpx::parallel::execution::task)
                .with(cs)
                .on(exec),
            std::begin(*data_representation), std::end(*data_representation),
            [](iterator) { worker_timed(delay); })
            .then([data_representation](hpx::future<void>) {});
    }
}

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
std::uint64_t averageout_parallel_foreach(
    std::size_t vector_size, Executor&& exec)
{
    std::vector<std::size_t> data_representation(vector_size);
    std::iota(
        std::begin(data_representation), std::end(data_representation), gen());

    std::uint64_t start = hpx::util::high_resolution_clock::now();

    // average out 100 executions to avoid varying results
    for (auto i = 0; i < test_count; i++)
        measure_parallel_foreach(data_representation, exec);

    return (hpx::util::high_resolution_clock::now() - start) / test_count;
}

template <typename Executor>
std::uint64_t averageout_task_foreach(std::size_t vector_size, Executor&& exec)
{
    std::shared_ptr<std::vector<std::size_t>> data_representation(
        std::make_shared<std::vector<std::size_t>>(vector_size));

    std::iota(std::begin(*data_representation), std::end(*data_representation),
        gen());

    if (num_overlapping_loops <= 0)
    {
        std::uint64_t start = hpx::util::high_resolution_clock::now();

        for (auto i = 0; i < test_count; i++)
            measure_task_foreach(data_representation, exec).wait();

        return (hpx::util::high_resolution_clock::now() - start) / test_count;
    }

    std::vector<hpx::shared_future<void>> tests;
    tests.resize(num_overlapping_loops);

    std::uint64_t start = hpx::util::high_resolution_clock::now();

    for (auto i = 0; i < test_count; i++)
    {
        hpx::future<void> curr =
            measure_task_foreach(data_representation, exec);
        if (i >= num_overlapping_loops)
            tests[(i - num_overlapping_loops) % tests.size()].wait();
        tests[i % tests.size()] = curr.share();
    }

    hpx::wait_all(tests);
    return (hpx::util::high_resolution_clock::now() - start) / test_count;
}

std::uint64_t averageout_sequential_foreach(std::size_t vector_size)
{
    std::vector<std::size_t> data_representation(vector_size);
    std::iota(
        std::begin(data_representation), std::end(data_representation), gen());

    std::uint64_t start = hpx::util::high_resolution_clock::now();

    // average out 100 executions to avoid varying results
    for (auto i = 0; i < test_count; i++)
        measure_sequential_foreach(data_representation);

    return (hpx::util::high_resolution_clock::now() - start) / test_count;
}

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
std::uint64_t averageout_parallel_forloop(
    std::size_t vector_size, Executor&& exec)
{
    std::vector<std::size_t> data_representation(vector_size);
    std::iota(
        std::begin(data_representation), std::end(data_representation), gen());

    std::uint64_t start = hpx::util::high_resolution_clock::now();

    // average out 100 executions to avoid varying results
    for (auto i = 0; i < test_count; i++)
        measure_parallel_forloop(data_representation, exec);

    return (hpx::util::high_resolution_clock::now() - start) / test_count;
}

template <typename Executor>
std::uint64_t averageout_task_forloop(std::size_t vector_size, Executor&& exec)
{
    std::shared_ptr<std::vector<std::size_t>> data_representation(
        std::make_shared<std::vector<std::size_t>>(vector_size));

    std::iota(std::begin(*data_representation), std::end(*data_representation),
        gen());

    if (num_overlapping_loops <= 0)
    {
        std::uint64_t start = hpx::util::high_resolution_clock::now();

        for (auto i = 0; i < test_count; i++)
            measure_task_forloop(data_representation, exec).wait();

        return (hpx::util::high_resolution_clock::now() - start) / test_count;
    }

    std::vector<hpx::shared_future<void>> tests;
    tests.resize(num_overlapping_loops);

    std::uint64_t start = hpx::util::high_resolution_clock::now();

    for (auto i = 0; i < test_count; i++)
    {
        hpx::future<void> curr =
            measure_task_forloop(data_representation, exec);
        if (i >= num_overlapping_loops)
            tests[(i - num_overlapping_loops) % tests.size()].wait();
        tests[i % tests.size()] = curr.share();
    }

    hpx::wait_all(tests);
    return (hpx::util::high_resolution_clock::now() - start) / test_count;
}

std::uint64_t averageout_sequential_forloop(std::size_t vector_size)
{
    std::vector<std::size_t> data_representation(vector_size);
    std::iota(
        std::begin(data_representation), std::end(data_representation), gen());

    std::uint64_t start = hpx::util::high_resolution_clock::now();

    // average out 100 executions to avoid varying results
    for (auto i = 0; i < test_count; i++)
        measure_sequential_forloop(data_representation);

    return (hpx::util::high_resolution_clock::now() - start) / test_count;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map& vm)
{
    // pull values from cmd
    std::size_t vector_size = vm["vector_size"].as<std::size_t>();
    bool csvoutput = vm["csv_output"].as<int>() ? true : false;
    delay = vm["work_delay"].as<int>();
    test_count = vm["test_count"].as<int>();
    chunk_size = vm["chunk_size"].as<int>();
    num_overlapping_loops = vm["overlapping_loops"].as<int>();
    disable_stealing = vm.count("disable_stealing");
    fast_idle_mode = vm.count("fast_idle_mode");

    // verify that input is within domain of program
    if (test_count == 0 || test_count < 0)
    {
        hpx::cout << "test_count cannot be zero or negative...\n" << hpx::flush;
    }
    else if (delay < 0)
    {
        hpx::cout << "delay cannot be a negative number...\n" << hpx::flush;
    }
    else
    {
        if (disable_stealing)
        {
            hpx::threads::remove_scheduler_mode(
                hpx::threads::policies::enable_stealing);
        }
        if (fast_idle_mode)
        {
            hpx::threads::add_scheduler_mode(
                hpx::threads::policies::fast_idle_mode);
        }

        // results
        std::uint64_t par_time_foreach;
        std::uint64_t task_time_foreach;
        std::uint64_t seq_time_foreach;

        std::uint64_t par_time_forloop;
        std::uint64_t task_time_forloop;
        std::uint64_t seq_time_forloop;

        if (vm.count("aggregated") != 0)
        {
            hpx::parallel::execution::parallel_executor_aggregated par;

            par_time_foreach = averageout_parallel_foreach(vector_size, par);
            task_time_foreach = averageout_task_foreach(vector_size, par);
            seq_time_foreach = averageout_sequential_foreach(vector_size);

            par_time_forloop = averageout_parallel_forloop(vector_size, par);
            task_time_forloop = averageout_task_forloop(vector_size, par);
            seq_time_forloop = averageout_sequential_forloop(vector_size);
        }
        else
        {
            hpx::parallel::execution::parallel_executor par;

            par_time_foreach = averageout_parallel_foreach(vector_size, par);
            task_time_foreach = averageout_task_foreach(vector_size, par);
            seq_time_foreach = averageout_sequential_foreach(vector_size);

            par_time_forloop = averageout_parallel_forloop(vector_size, par);
            task_time_forloop = averageout_task_forloop(vector_size, par);
            seq_time_forloop = averageout_sequential_forloop(vector_size);
        }

        if (disable_stealing)
        {
            hpx::threads::add_scheduler_mode(
                hpx::threads::policies::enable_stealing);
        }
        if (fast_idle_mode)
        {
            hpx::threads::remove_scheduler_mode(
                hpx::threads::policies::fast_idle_mode);
        }

        if (csvoutput)
        {
            hpx::cout << "," << seq_time_foreach / 1e9 << ","
                      << par_time_foreach / 1e9 << ","
                      << task_time_foreach / 1e9 << "\n"
                      << hpx::flush;
        }
        else
        {
            // print results(Formatted). setw(x) assures that all output is
            // right justified
            hpx::cout << std::left
                      << "----------------Parameters---------------------\n"
                      << std::left << "Vector size: " << std::right
                      << std::setw(30) << vector_size << "\n"
                      << std::left << "Number of tests" << std::right
                      << std::setw(28) << test_count << "\n"
                      << std::left << "Delay per iteration(nanoseconds)"
                      << std::right << std::setw(11) << delay << "\n"
                      << std::left << "Display time in: " << std::right
                      << std::setw(27) << "Seconds\n"
                      << hpx::flush;

            hpx::cout << "-------------Average-(for_each)----------------\n"
                      << std::left
                      << "Average parallel execution time  : " << std::right
                      << std::setw(8) << par_time_foreach / 1e9 << "\n"
                      << std::left
                      << "Average task execution time      : " << std::right
                      << std::setw(8) << task_time_foreach / 1e9 << "\n"
                      << std::left
                      << "Average sequential execution time: " << std::right
                      << std::setw(8) << seq_time_foreach / 1e9 << "\n"
                      << hpx::flush;

            hpx::cout << "-----Execution Time Difference-(for_each)------\n"
                      << std::left << "Parallel Scale: " << std::right
                      << std::setw(27)
                      << (double(seq_time_foreach) / par_time_foreach) << "\n"
                      << std::left << "Task Scale    : " << std::right
                      << std::setw(27)
                      << (double(seq_time_foreach) / task_time_foreach) << "\n"
                      << hpx::flush;

            hpx::cout << "-------------Average-(for_loop)----------------\n"
                      << std::left
                      << "Average parallel execution time  : " << std::right
                      << std::setw(8) << par_time_forloop / 1e9 << "\n"
                      << std::left
                      << "Average task execution time      : " << std::right
                      << std::setw(8) << task_time_forloop / 1e9 << "\n"
                      << std::left
                      << "Average sequential execution time: " << std::right
                      << std::setw(8) << seq_time_forloop / 1e9 << "\n"
                      << hpx::flush;

            hpx::cout << "-----Execution Time Difference-(for_loop)------\n"
                      << std::left << "Parallel Scale: " << std::right
                      << std::setw(27)
                      << (double(seq_time_forloop) / par_time_forloop) << "\n"
                      << std::left << "Task Scale    : " << std::right
                      << std::setw(27)
                      << (double(seq_time_forloop) / task_time_forloop) << "\n"
                      << hpx::flush;
        }
    }

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    //initialize program
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    hpx::program_options::options_description cmdline(
        "usage: " HPX_APPLICATION_STRING " [options]");

    // clang-format off
    cmdline.add_options()
        ( "vector_size"
        , hpx::program_options::value<std::size_t>()->default_value(1000)
        , "size of vector")

        ("work_delay"
        , hpx::program_options::value<int>()->default_value(1)
        , "loop delay per element in nanoseconds")

        ("test_count"
        , hpx::program_options::value<int>()->default_value(100)
        , "number of tests to be averaged")

        ("chunk_size"
        , hpx::program_options::value<int>()->default_value(0)
        , "number of iterations to combine while parallelization")

        ("overlapping_loops"
        , hpx::program_options::value<int>()->default_value(0)
        , "number of overlapping task loops")

        ("csv_output"
        , hpx::program_options::value<int>()->default_value(0)
        ,"print results in csv format")

        ("aggregated"
        ,"use aggregated executor")

        ("disable_stealing"
        ,"disable thread stealing")

        ("fast_idle_mode"
        ,"enable fast idle mode")
        ;
    // clang-format on

    return hpx::init(cmdline, argc, argv, cfg);
}
