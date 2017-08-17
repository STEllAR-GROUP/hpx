//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/util/high_resolution_timer.hpp>

#include <boost/format.hpp>
#include <boost/program_options.hpp>

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include "worker_timed.hpp"

///////////////////////////////////////////////////////////////////////////////
std::size_t iterations = 10000;
std::uint64_t delay = 0;

void just_wait()
{
    worker_timed(delay * 1000);
}

///////////////////////////////////////////////////////////////////////////////
template <typename Policy>
double measure_one(Policy policy)
{
    std::vector<hpx::future<void> > threads;
    threads.reserve(iterations);

    std::uint64_t start = hpx::util::high_resolution_clock::now();

    for (std::size_t i = 0; i != iterations; ++i)
    {
        threads.push_back(hpx::async(policy, &just_wait));
    }

    hpx::wait_all(threads);

    std::uint64_t stop = hpx::util::high_resolution_clock::now();
    return (stop - start) / 1e9;
}

template <typename Policy>
double measure(Policy policy)
{
    std::size_t num_cores = hpx::get_os_thread_count();
    std::vector<hpx::future<double> > cores;
    cores.reserve(num_cores);

    for (std::size_t i = 0; i != num_cores; ++i)
    {
        cores.push_back(hpx::async(&measure_one<Policy>, policy));
    }

    std::vector<double> times = hpx::util::unwrap(cores);
    return std::accumulate(times.begin(), times.end(), 0.0);
}

int hpx_main(boost::program_options::variables_map& vm)
{
    bool print_header = vm.count("no-header") == 0;
    bool do_child = vm.count("no-child") == 0;      // fork only
    bool do_parent = vm.count("no-parent") == 0;    // async only
    std::size_t num_cores = hpx::get_os_thread_count();
    if (vm.count("num_cores") != 0)
        num_cores = vm["num_cores"].as<std::size_t>();

    // first collect child stealing times
    double child_stealing_time = 0;
    if (do_parent)
        child_stealing_time = measure(hpx::launch::async);

    // now collect parent stealing times
    double parent_stealing_time = 0;
    if (do_child)
        parent_stealing_time = measure(hpx::launch::fork);

    if (print_header)
    {
        hpx::cout
            << "num_cores,num_threads,child_stealing_time[s],parent_stealing_time[s]"
            << hpx::endl;
    }

    hpx::cout
        << (boost::format("%d,%d,%f,%f") %
                num_cores %
                iterations %
                child_stealing_time %
                parent_stealing_time)
        << hpx::endl;

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // Configure application-specific options.
    namespace po = boost::program_options;
    po::options_description cmdline(
        "usage: " HPX_APPLICATION_STRING " [options]");

    cmdline.add_options()
        ("delay",
            po::value<std::uint64_t>(&delay)->default_value(0),
            "time to busy wait in delay loop [microseconds] "
            "(default: no busy waiting)")
        ("num_threads",
            po::value<std::size_t>(&iterations)->default_value(10000),
            "number of threads to create while measuring execution "
            "(default: 10000)")
        ("num_cores",
            po::value<std::size_t>(),
            "number of spawning tasks to execute (default: number of cores)")
        ("no-header", "do not print out the csv header row")
        ("no-child", "do not test child-stealing (launch::fork only)")
        ("no-parent", "do not test child-stealing (launch::async only)")
        ;

    return hpx::init(cmdline, argc, argv);
}

