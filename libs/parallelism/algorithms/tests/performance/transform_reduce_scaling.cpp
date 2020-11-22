//  Copyright (c) 2014 Grant Mercer
//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <hpx/chrono.hpp>
#include <hpx/iostream.hpp>
#include <hpx/numeric.hpp>
#include "worker_timed.hpp"

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
int test_count = 100;
unsigned int seed = std::random_device{}();
std::mt19937 gen(seed);

struct Point
{
    double x, y;
};

///////////////////////////////////////////////////////////////////////////////
void measure_transform_reduce(std::size_t size)
{
    std::vector<Point> data_representation(
        size, Point{double(gen()), double(gen())});

    // invoke transform_reduce
    double result = hpx::transform_reduce(hpx::execution::par,
        std::begin(data_representation), std::end(data_representation), 0.0,
        std::plus<double>(), [](Point r) { return r.x * r.y; });
    HPX_UNUSED(result);
}

void measure_transform_reduce_old(std::size_t size)
{
    std::vector<Point> data_representation(
        size, Point{double(gen()), double(gen())});

    //invoke old reduce
    Point result = hpx::ranges::reduce(hpx::execution::par,
        std::begin(data_representation), std::end(data_representation),
        Point{0.0, 0.0}, [](Point res, Point curr) {
            return Point{res.x * res.y + curr.x * curr.y, 1.0};
        });
    HPX_UNUSED(result);
}

std::uint64_t average_out_transform_reduce(std::size_t vector_size)
{
    measure_transform_reduce(vector_size);
    return std::uint64_t(1);
}

std::uint64_t average_out_transform_reduce_old(std::size_t vector_size)
{
    measure_transform_reduce_old(vector_size);
    return std::uint64_t(1);
}

int hpx_main(hpx::program_options::variables_map& vm)
{
    std::size_t vector_size = vm["vector_size"].as<std::size_t>();
    bool csvoutput = vm["csv_output"].as<int>() ? true : false;
    test_count = vm["test_count"].as<int>();
    if (test_count < 0 || test_count == 0)
    {
        hpx::cout << "test_count cannot be less than zero...\n" << hpx::flush;
    }
    else
    {
        std::uint64_t tr_time = average_out_transform_reduce(vector_size);
        std::uint64_t tr_old_time =
            average_out_transform_reduce_old(vector_size);

        if (csvoutput)
        {
            hpx::cout << "," << tr_time / 1e9 << "," << tr_old_time / 1e9
                      << "\n"
                      << hpx::flush;
        }
        else
        {
            hpx::cout << "transform_reduce: " << std::right << std::setw(30)
                      << tr_time / 1e9 << "\n"
                      << hpx::flush;
            hpx::cout << "old_transform_reduce" << std::right << std::setw(30)
                      << tr_old_time / 1e9 << "\n"
                      << hpx::flush;
        }
    }
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    hpx::program_options::options_description cmdline(
        "usage: " HPX_APPLICATION_STRING " [options]");

    cmdline.add_options()("vector_size",
        hpx::program_options::value<std::size_t>()->default_value(1000),
        "size of vector")

        ("csv_output", hpx::program_options::value<int>()->default_value(0),
            "print results in csv format")

            ("test_count",
                hpx::program_options::value<int>()->default_value(100),
                "number of tests to take average from");

    hpx::init_params init_args;
    init_args.desc_cmdline = cmdline;
    init_args.cfg = cfg;

    return hpx::init(argc, argv, init_args);
}
