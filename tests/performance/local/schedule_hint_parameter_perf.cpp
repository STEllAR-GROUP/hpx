//  Copyright (c) 2019 Mikael Simberg
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/parallel_algorithm.hpp>
#include <hpx/include/parallel_executor_parameters.hpp>
#include <hpx/util/high_resolution_clock.hpp>

#include <memory>
#include <string>
#include <tuple>

void print_header()
{
    hpx::cout << "schedule,chunk_size,num_elements,threads,min,avg,max"
              << std::endl;
}

void print_result(std::string const& name, int chunk_size, int num_elements,
    std::tuple<double, double, double> timings)
{
    hpx::cout << name << "," << chunk_size << "," << num_elements << ","
              << hpx::get_num_worker_threads() << "," << std::get<0>(timings)
              << "," << std::get<1>(timings) << "," << std::get<2>(timings)
              << std::endl;
}

template <typename Policy>
std::tuple<double, double, double> test_schedule(
    Policy&& p, int num_warmups, int num_iterations, int num_elements)
{
    hpx::util::high_resolution_timer timer;

    double t_min = (std::numeric_limits<double>::max)();
    double t_max = (std::numeric_limits<double>::min)();
    double t_avg = 0.0;

    for (int r = 0; r < num_warmups + num_iterations; ++r)
    {
        double* a = new double[num_elements];

        hpx::parallel::for_loop(
            p, 0, num_elements, [&a](int i) { a[i] = double(i); });

        timer.restart();
        hpx::parallel::for_loop(
            p, 0, num_elements, [&a](int i) { a[i] = a[i] * a[i]; });
        double t = timer.elapsed();

        delete[] a;

        if (r >= num_warmups)
        {
            t_min = (std::min)(t, t_min);
            t_max = (std::max)(t, t_max);
            t_avg += t;
        }
    }

    t_avg /= num_iterations;

    return std::make_tuple(t_min, t_avg, t_max);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    int const num_warmups = vm["num_warmups"].as<int>();
    int const num_iterations = vm["num_iterations"].as<int>();
    int const num_elements = vm["num_elements"].as<int>();
    int const chunk_size = vm["chunk_size"].as<int>();

    print_header();

    using namespace hpx::parallel::execution;

    auto chunked_result = test_schedule(
        par.with(chunked_placement(), static_chunk_size(chunk_size)),
        num_warmups, num_iterations, num_elements);
    print_result("chunked", chunk_size, num_elements, chunked_result);

    auto round_robin_result = test_schedule(
        par.with(round_robin_placement(), static_chunk_size(chunk_size)),
        num_warmups, num_iterations, num_elements);
    print_result("round_robin", chunk_size, num_elements, round_robin_result);

    auto scheduler_result = test_schedule(
        par.with(scheduler_placement(), static_chunk_size(chunk_size)),
        num_warmups, num_iterations, num_elements);
    print_result(
        "scheduler_default", chunk_size, num_elements, scheduler_result);

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    boost::program_options::options_description cmdline(
        "usage: " HPX_APPLICATION_STRING " [options]");

    using boost::program_options::value;

    cmdline.add_options()("num_warmups", value<int>()->default_value(10),
        "number of warmup iterations")("num_iterations",
        value<int>()->default_value(100),
        "number of tests to be averaged")("num_elements",
        value<int>()->default_value(1000000), "number of elements in vector")(
        "chunk_size", value<int>()->default_value(0),
        "number of elements to combine into a single task");

    std::vector<std::string> const cfg = {"hpx.os_threads=all"};
    hpx::init(cmdline, argc, argv, cfg);
}
