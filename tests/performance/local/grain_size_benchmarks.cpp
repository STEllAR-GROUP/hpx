//  Copyright (c) 2020 Shahrzad Shirzad
//  Copyright (c) 2018-2019 Mikael Simberg
//  Copyright (c) 2018-2019 John Biddiscombe
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/parallel_algorithm.hpp>
#include <hpx/include/parallel_execution.hpp>
#include <hpx/include/parallel_executor_parameters.hpp>
//#include <hpx/timing.hpp>

//#include <hpx/execution/executors/parallel_executor_aggregated.hpp>

#include "worker_timed.hpp"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using boost::program_options::options_description;
using boost::program_options::value;
using boost::program_options::variables_map;

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
    std::int64_t count, std::uint64_t chunk_size, double duration)
{
    std::ostringstream temp;
    double us = 1e6 * duration;

    hpx::util::format_to(temp,
        "array_size {:1}, {:27} {:15} {:18} in {:8} microseconds "
        ",  chunk size {:4}, threads {:4}\n" ,
        count, title, wait, exec, us, chunk_size, num_threads);

    std::cout << temp.str() << std::endl;
}

const char* ExecName(const hpx::parallel::execution::parallel_executor& exec)
{
    return "parallel_executor";
}

///////////////////////////////////////////////////////////////////////////////
// we use globals here to prevent the delay from being optimized away
double global_scratch = 0;
std::uint64_t num_num_iterations = 0;

void initialize(std::vector<double>& x, std::vector<double>& y,
    std::vector<double>& z, std::uint64_t n)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        x[i] = (double) i * (double) i;
        y[i] = (double) (i + 1) * (double) (i - 1);
        z[i] = y[i];
    }
}
//bool daxpy(std::vector<double>& x, std::vector<double>& y, std::vector<double>& z, const double a)
//{
//    for (std::size_t i = 0; i < x.size(); ++i)
//    {
//        if (z[i] != a * x[i] + y[i])
//            return false;
//    }
//    return true;
//}

struct benchmark
{
    virtual void operator()(double& x, double& y) const = 0;
    virtual bool validate(std::vector<double>& x, std::vector<double>& y,
        std::vector<double>& z) const = 0;
};

struct daxpy : benchmark
{
    const double a = 3.0;
    void operator()(double& x, double& y) const override
    {
        y += a * x;
    }
    bool validate(std::vector<double>& x, std::vector<double>& y,
        std::vector<double>& z) const override
    {
        for (std::size_t i = 0; i < x.size(); ++i)
        {
            if (y[i] != a * x[i] + z[i])
                return false;
        }
        return true;
    }
};

///////////////////////////////////////////////////////////////////////////////
void measure_function_futures_for_loop(
    std::uint64_t array_size, std::uint64_t chunk_size)
{
    const double a = 3.0;

    std::vector<double> x(array_size, 0.0), y(array_size, 0.0),
        z(array_size, 0.0);
    initialize(x, y, z, array_size);

    daxpy f;
    // start the clock
    high_resolution_timer walltime;
    hpx::parallel::for_loop(
        hpx::parallel::execution::par.with(
            hpx::parallel::execution::dynamic_chunk_size(chunk_size)),
        0, array_size, [&](std::uint64_t i) { worker_timed(100 * 1000); });

//    hpx::parallel::for_loop(
//        hpx::parallel::execution::par.with(
//            hpx::parallel::execution::dynamic_chunk_size(chunk_size)),
//        0, array_size, [&](std::uint64_t i) { f(x[i], y[i]); });

    // stop the clock
    const double duration = walltime.elapsed();
//    if (f.validate(x, y, z))
        print_stats(
            "for_loop", "par", "parallel_executor", array_size, chunk_size, duration);
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

        int const repetitions = vm["repetitions"].as<int>();

        num_threads = hpx::get_num_worker_threads();

        std::uint64_t const chunk_size = vm["chunk_size"].as<std::uint64_t>();
        std::uint64_t const array_size = vm["array_size"].as<std::uint64_t>();

        if (HPX_UNLIKELY(0 == array_size))
        {
            throw std::logic_error("error: count of 0 futures specified\n");
        }

        for (int i = 0; i < repetitions; i++)
        {
            measure_function_futures_for_loop(array_size, chunk_size);
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
    cmdline.add_options()
        ("array_size", value<std::uint64_t>()->default_value(1000000), "input array size")
        ("repetitions", value<int>()->default_value(1),
         "number of repetitions of the full benchmark")
        ("chunk_size", value<std::uint64_t>()->default_value(1), "chunk size");
    // clang-format on


    return init(cmdline, argc, argv);

}
