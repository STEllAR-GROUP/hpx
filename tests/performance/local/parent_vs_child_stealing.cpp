//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/util/high_resolution_timer.hpp>

#include "worker_timed.hpp"

#include <boost/format.hpp>
#include <boost/program_options.hpp>

///////////////////////////////////////////////////////////////////////////////
std::size_t iterations = 10000;
boost::uint64_t delay = 0;

void just_wait()
{
    worker_timed(delay * 1000);
}

///////////////////////////////////////////////////////////////////////////////
template <typename Policy>
double measure(Policy policy)
{
    std::vector<hpx::future<void> > threads;
    threads.reserve(iterations);

    boost::uint64_t start = hpx::util::high_resolution_clock::now();

    for (std::size_t i = 0; i != iterations; ++i)
    {
        threads.push_back(hpx::async(policy, &just_wait));
    }

    hpx::wait_all(threads);

    boost::uint64_t stop = hpx::util::high_resolution_clock::now();
    return (stop - start) / 1e9;
}

int hpx_main(boost::program_options::variables_map& vm)
{
    bool print_header = vm.count("no-header") == 0;

    // first collect child stealing times
    double child_stealing_time = measure(hpx::launch::async);
    double parent_stealing_time = measure(hpx::launch::fork);

    if (print_header)
    {
        hpx::cout
            << "testcount,child_stealing_time[s],parent_stealing_time[s]"
            << hpx::endl;
    }

    hpx::cout
        << (boost::format("%d,%f,%f") %
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
            po::value<boost::uint64_t>(&delay)->default_value(0),
            "time to busy wait in delay loop [microseconds] "
            "(default: no busy waiting)")
        ("iterations",
            po::value<std::size_t>(&iterations)->default_value(10000),
            "number of threads to create while measuring execution "
            "(default: 10000)")
        ("no-header", "do not print out the csv header row")
        ;

    return hpx::init(cmdline, argc, argv);
}

