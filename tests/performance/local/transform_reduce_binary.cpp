//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>

#include <hpx/util/high_resolution_clock.hpp>
#include <hpx/include/parallel_transform_reduce.hpp>
#include <hpx/include/iostreams.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/range/functions.hpp>

struct plus
{
    template <typename T1, typename T2>
    auto operator()(T1 && t1, T2 && t2) const -> decltype(t1 + t2)
    {
        return t1 + t2;
    }
};

struct multiplies
{
    template <typename T1, typename T2>
    auto operator()(T1 && t1, T2 && t2) const -> decltype(t1 * t2)
    {
        return t1 * t2;
    }
};

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy>
float measure_inner_product(ExPolicy && policy,
    std::vector<float> const& data1, std::vector<float> const& data2)
{
    return
        hpx::parallel::transform_reduce(
            policy,
            std::begin(data1),
            std::end(data1),
            std::begin(data2),
            0.0f,
            ::multiplies(),
            ::plus()
        );
}

template <typename ExPolicy>
std::int64_t measure_inner_product(int count, ExPolicy && policy,
    std::vector<float> const& data1, std::vector<float> const& data2)
{
    std::int64_t start = hpx::util::high_resolution_clock::now();

    for (int i = 0; i != count; ++i)
        measure_inner_product(policy, data1, data2);

    return (hpx::util::high_resolution_clock::now() - start) / count;
}

int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    hpx::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    std::size_t size = vm["vector_size"].as<std::size_t>();
    bool csvoutput = vm["csv_output"].as<int>() ?true : false;
    int test_count = vm["test_count"].as<int>();

    std::vector<float> data1(size);
    std::vector<float> data2(size);

    std::iota(std::begin(data1), std::end(data1), float(std::rand()));
    std::iota(std::begin(data2), std::end(data2), float(std::rand()));

    if (test_count <= 0)
    {
        hpx::cout << "test_count cannot be less than zero...\n" << hpx::flush;
    }
    else
    {
        // warm up caches
        measure_inner_product(hpx::parallel::par, data1, data2);

        // do measurements
        std::uint64_t tr_time_datapar = measure_inner_product(
            test_count, hpx::parallel::datapar_execution, data1, data2);
        std::uint64_t tr_time_par = measure_inner_product(
            test_count, hpx::parallel::par, data1, data2);

        if (csvoutput)
        {
            hpx::cout
                << "," << tr_time_par / 1e9
                << "," << tr_time_datapar / 1e9
                << "\n" << hpx::flush;
        }
        else
        {
            hpx::cout
                << "transform_reduce(execution::par): " << std::right
                    << std::setw(15) << tr_time_par / 1e9 << "\n"
                << "transform_reduce(datapar): " << std::right
                    << std::setw(15) << tr_time_datapar / 1e9 << "\n"
                << hpx::flush;
        }
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> const cfg = {
        "hpx.os_threads=all"
    };

    boost::program_options::options_description cmdline(
        "usage: " HPX_APPLICATION_STRING " [options]");

    cmdline.add_options()
        ("vector_size"
        , boost::program_options::value<std::size_t>()->default_value(1024)
        , "size of vector")

        ("csv_output"
        , boost::program_options::value<int>()->default_value(0)
        , "print results in csv format")

        ("test_count"
        , boost::program_options::value<int>()->default_value(10)
        , "number of tests to take average from")

        ("seed,s"
        , boost::program_options::value<unsigned int>()
        , "the random number generator seed to use for this run")
        ;

    return hpx::init(cmdline, argc, argv, cfg);

}

