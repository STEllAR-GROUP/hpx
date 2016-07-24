//  Copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/util/high_resolution_clock.hpp>
#include <hpx/include/parallel_numeric.hpp>
#include <hpx/include/iostreams.hpp>
#include "worker_timed.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#include <boost/cstdint.hpp>
#include <boost/format.hpp>
#include <boost/range/functions.hpp>

///////////////////////////////////////////////////////////////////////////////
int test_count = 100;

struct Point
{
    double x, y;
};

///////////////////////////////////////////////////////////////////////////////
void measure_transform_reduce(std::size_t size)
{
    std::vector<Point> data_representation(size,
        Point{double(std::rand()), double(std::rand())});

    // invode transform_reduce
    double result =
        hpx::parallel::transform_reduce(hpx::parallel::par,
        boost::begin(data_representation),
        boost::end(data_representation),
        [](Point r)
        {
            return r.x * r.y;
        },
        0.0,
        std::plus<double>()
    );
    HPX_UNUSED(result);
}

void measure_transform_reduce_old(std::size_t size)
{
    std::vector<Point> data_representation(size,
        Point{double(std::rand()), double(std::rand())});

    //invode old reduce
    Point result =
        hpx::parallel::reduce(hpx::parallel::par,
        boost::begin(data_representation),
        boost::end(data_representation),
        Point{0.0, 0.0},
        [](Point res, Point curr)
        {
            return Point{
                res.x * res.y + curr.x * curr.y, 1.0};
        }
    );
    HPX_UNUSED(result);
}

boost::uint64_t average_out_transform_reduce(std::size_t vector_size)
{
    measure_transform_reduce(vector_size);
    return boost::uint64_t(1);
}

boost::uint64_t average_out_transform_reduce_old(std::size_t vector_size)
{
    measure_transform_reduce_old(vector_size);
    return boost::uint64_t(1);
}

int hpx_main(boost::program_options::variables_map& vm)
{
    std::size_t vector_size = vm["vector_size"].as<std::size_t>();
    bool csvoutput = vm["csv_output"].as<int>() ?true : false;
    test_count = vm["test_count"].as<int>();
    if(test_count < 0 || test_count == 0) {
        hpx::cout << "test_count cannot be less than zero...\n" << hpx::flush;
    } else {
        boost::uint64_t tr_time = average_out_transform_reduce(vector_size);
        boost::uint64_t tr_old_time = average_out_transform_reduce_old(
            vector_size);

        if(csvoutput) {
            hpx::cout << "," << tr_time/1e9
                      << "," << tr_old_time/1e9 << "\n" << hpx::flush;
        } else {
            hpx::cout << "transform_reduce: " << std::right
                << std::setw(30) << tr_time/1e9 << "\n" << hpx::flush;
            hpx::cout << "old_transform_reduce" << std::right
                << std::setw(30) << tr_old_time/1e9 << "\n" << hpx::flush;
        }
    }
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> const cfg = {
        "hpx.os_threads=" +
            std::to_string(hpx::threads::hardware_concurrency())
    };

    boost::program_options::options_description cmdline(
        "usage: " HPX_APPLICATION_STRING " [options]");

    cmdline.add_options()
        ("vector_size"
        , boost::program_options::value<std::size_t>()->default_value(1000)
        , "size of vector")

        ("csv_output"
        , boost::program_options::value<int>()->default_value(0)
        , "print results in csv format")

        ("test_count"
        , boost::program_options::value<int>()->default_value(100)
        , "number of tests to take average from")
        ;

    return hpx::init(cmdline, argc, argv, cfg);

}

