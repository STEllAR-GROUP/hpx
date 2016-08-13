//  Copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/util/high_resolution_clock.hpp>
#include <hpx/include/parallel_algorithm.hpp>
#include <hpx/include/parallel_executor_parameters.hpp>
#include <hpx/include/iostreams.hpp>
#include "worker_timed.hpp"

#include <boost/format.hpp>
#include <boost/range/functions.hpp>

#include <cstdint>
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

///////////////////////////////////////////////////////////////////////////////
void measure_sequential_foreach(std::size_t size)
{
    std::vector<std::size_t> data_representation(size);
    std::iota(boost::begin(data_representation),
        boost::end(data_representation),
        std::rand());

    // invoke sequential for_each
    hpx::parallel::for_each(hpx::parallel::seq,
        boost::begin(data_representation),
        boost::end(data_representation),
        [](std::size_t) {
            worker_timed(delay);
        });
}

void measure_parallel_foreach(std::size_t size)
{
    std::vector<std::size_t> data_representation(size);
    std::iota(boost::begin(data_representation),
        boost::end(data_representation),
        std::rand());

    // create executor parameters object
    hpx::parallel::static_chunk_size cs(chunk_size);

    // invoke parallel for_each
    hpx::parallel::for_each(hpx::parallel::par.with(cs),
        boost::begin(data_representation),
        boost::end(data_representation),
        [](std::size_t) {
            worker_timed(delay);
        });
}

hpx::future<void> measure_task_foreach(std::size_t size)
{
    std::shared_ptr<std::vector<std::size_t> > data_representation(
        std::make_shared<std::vector<std::size_t> >(size));
    std::iota(boost::begin(*data_representation),
        boost::end(*data_representation),
        std::rand());

    // create executor parameters object
    hpx::parallel::static_chunk_size cs(chunk_size);

    // invoke parallel for_each
    return
        hpx::parallel::for_each(
            hpx::parallel::par(hpx::parallel::task).with(cs),
            boost::begin(*data_representation),
            boost::end(*data_representation),
            [](std::size_t) {
                worker_timed(delay);
            }
        ).then(
            [data_representation](hpx::future<void>) {}
        );
}

std::uint64_t average_out_parallel(std::size_t vector_size)
{
    std::uint64_t start = hpx::util::high_resolution_clock::now();

    // average out 100 executions to avoid varying results
    for(auto i = 0; i < test_count; i++)
        measure_parallel_foreach(vector_size);

    return (hpx::util::high_resolution_clock::now() - start) / test_count;
}

std::uint64_t average_out_task(std::size_t vector_size)
{
    if (num_overlapping_loops <= 0)
    {
        std::uint64_t start = hpx::util::high_resolution_clock::now();

        for(auto i = 0; i < test_count; i++)
            measure_task_foreach(vector_size).wait();

        return (hpx::util::high_resolution_clock::now() - start) / test_count;
    }

    std::vector<hpx::shared_future<void> > tests;
    tests.resize(num_overlapping_loops);

    std::uint64_t start = hpx::util::high_resolution_clock::now();

    for(auto i = 0; i < test_count; i++)
    {
        hpx::future<void> curr = measure_task_foreach(vector_size);
        if (i >= num_overlapping_loops)
            tests[(i-num_overlapping_loops) % tests.size()].wait();
        tests[i % tests.size()] = curr.share();
    }

    hpx::wait_all(tests);
    return (hpx::util::high_resolution_clock::now() - start) / test_count;
}

std::uint64_t average_out_sequential(std::size_t vector_size)
{
    std::uint64_t start = hpx::util::high_resolution_clock::now();

    // average out 100 executions to avoid varying results
    for(auto i = 0; i < test_count; i++)
        measure_sequential_foreach(vector_size);

    return (hpx::util::high_resolution_clock::now() - start) / test_count;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    //pull values from cmd
    std::size_t vector_size = vm["vector_size"].as<std::size_t>();
    bool csvoutput = vm["csv_output"].as<int>() ?true : false;
    delay = vm["work_delay"].as<int>();
    test_count = vm["test_count"].as<int>();
    chunk_size = vm["chunk_size"].as<int>();
    num_overlapping_loops = vm["overlapping_loops"].as<int>();

    //verify that input is within domain of program
    if(test_count == 0 || test_count < 0) {
        hpx::cout << "test_count cannot be zero or negative...\n" << hpx::flush;
    } else if (delay < 0) {
        hpx::cout << "delay cannot be a negative number...\n" << hpx::flush;
    } else {

        //results
        std::uint64_t par_time = average_out_parallel(vector_size);
        std::uint64_t task_time = average_out_task(vector_size);
        std::uint64_t seq_time = average_out_sequential(vector_size);

        if(csvoutput) {
            hpx::cout << "," << seq_time/1e9
                      << "," << par_time/1e9
                      << "," << task_time/1e9 << "\n" << hpx::flush;
        }
        else {
        // print results(Formatted). Setw(x) assures that all output is right justified
            hpx::cout << std::left << "----------------Parameters-----------------\n"
                << std::left << "Vector size: " << std::right
                             << std::setw(30) << vector_size << "\n"
                << std::left << "Number of tests" << std::right
                             << std::setw(28) << test_count << "\n"
                << std::left << "Delay per iteration(nanoseconds)"
                             << std::right << std::setw(11) << delay << "\n"
                << std::left << "Display time in: "
                << std::right << std::setw(27) << "Seconds\n" << hpx::flush;

            hpx::cout << "------------------Average------------------\n"
                << std::left << "Average parallel execution time  : "
                             << std::right << std::setw(8) << par_time/1e9 << "\n"
                << std::left << "Average task execution time      : "
                             << std::right << std::setw(8) << task_time/1e9 << "\n"
                << std::left << "Average sequential execution time: "
                             << std::right << std::setw(8) << seq_time/1e9 << "\n"
                             << hpx::flush;

            hpx::cout << "---------Execution Time Difference---------\n"
                << std::left << "Parallel Scale: " << std::right  << std::setw(27)
                             << (double(seq_time) / par_time) << "\n"
                << std::left << "Task Scale    : " << std::right  << std::setw(27)
                             << (double(seq_time) / task_time) << "\n" << hpx::flush;
        }
    }

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    //initialize program
    std::vector<std::string> const cfg = {
        "hpx.os_threads=all"
    };

    boost::program_options::options_description cmdline(
        "usage: " HPX_APPLICATION_STRING " [options]");

    cmdline.add_options()
        ( "vector_size"
        , boost::program_options::value<std::size_t>()->default_value(1000)
        , "size of vector")

        ("work_delay"
        , boost::program_options::value<int>()->default_value(1)
        , "loop delay per element in nanoseconds")

        ("test_count"
        , boost::program_options::value<int>()->default_value(100)
        , "number of tests to be averaged")

        ("chunk_size"
        , boost::program_options::value<int>()->default_value(0)
        , "number of iterations to combine while parallelization")

        ("overlapping_loops"
        , boost::program_options::value<int>()->default_value(0)
        , "number of overlapping task loops")

        ("csv_output"
        , boost::program_options::value<int>()->default_value(0)
        ,"print results in csv format")
        ;

    return hpx::init(cmdline, argc, argv, cfg);
}

