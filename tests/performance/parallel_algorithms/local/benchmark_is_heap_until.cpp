///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2017 Taeguk Kwon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_is_heap.hpp>
#include <hpx/include/parallel_generate.hpp>
#include <hpx/util/high_resolution_clock.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <boost/random.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <limits>
#include <iostream>
#include <string>

///////////////////////////////////////////////////////////////////////////////
struct random_fill
{
    random_fill()
        : gen(std::rand()),
        dist(0, RAND_MAX)
    {}

    int operator()()
    {
        return dist(gen);
    }

    boost::random::mt19937 gen;
    boost::random::uniform_int_distribution<> dist;

    template <typename Archive>
    void serialize(Archive& ar, unsigned)
    {}
};

///////////////////////////////////////////////////////////////////////////////
double run_is_heap_until_benchmark_std(int test_count,
    std::vector<int> const& v)
{
    std::cout << "--- run_is_heap_until_benchmark_std ---" << std::endl;
    decltype(std::begin(v)) result;
    std::uint64_t time = hpx::util::high_resolution_clock::now();

    for (int i = 0; i < test_count; ++i)
    {
        result = std::is_heap_until(std::begin(v), std::end(v));
    }

    time = hpx::util::high_resolution_clock::now() - time;

    auto heap_range = std::distance(std::begin(v), result);
    std::cout << "Heap Range : " << heap_range << std::endl;

    return (time * 1e-9) / test_count;
}

///////////////////////////////////////////////////////////////////////////////
double run_is_heap_until_benchmark_seq(int test_count,
    std::vector<int> const& v)
{
    std::cout << "--- run_is_heap_until_benchmark_par ---" << std::endl;
    decltype(std::begin(v)) result;
    std::uint64_t time = hpx::util::high_resolution_clock::now();

    for (int i = 0; i < test_count; ++i)
    {
        using namespace hpx::parallel;
        result = is_heap_until(execution::seq, std::begin(v), std::end(v));
    }

    time = hpx::util::high_resolution_clock::now() - time;

    auto heap_range = std::distance(std::begin(v), result);
    std::cout << "Heap Range : " << heap_range << std::endl;

    return (time * 1e-9) / test_count;
}

///////////////////////////////////////////////////////////////////////////////
double run_is_heap_until_benchmark_par(int test_count,
    std::vector<int> const& v)
{
    std::cout << "--- run_is_heap_until_benchmark_par ---" << std::endl;
    decltype(std::begin(v)) result;
    std::uint64_t time = hpx::util::high_resolution_clock::now();

    for (int i = 0; i < test_count; ++i)
    {
        using namespace hpx::parallel;
        result = is_heap_until(execution::par, std::begin(v), std::end(v));
    }

    time = hpx::util::high_resolution_clock::now() - time;

    auto heap_range = std::distance(std::begin(v), result);
    std::cout << "Heap Range : " << heap_range << std::endl;

    return (time * 1e-9) / test_count;
}

///////////////////////////////////////////////////////////////////////////////
double run_is_heap_until_benchmark_par_unseq(int test_count,
    std::vector<int> const& v)
{
    std::cout << "--- run_is_heap_until_benchmark_par_unseq ---" << std::endl;
    decltype(std::begin(v)) result;
    std::uint64_t time = hpx::util::high_resolution_clock::now();

    for (int i = 0; i < test_count; ++i)
    {
        using namespace hpx::parallel;
        result = is_heap_until(execution::par_unseq, std::begin(v), std::end(v));
    }

    time = hpx::util::high_resolution_clock::now() - time;

    auto heap_range = std::distance(std::begin(v), result);
    std::cout << "Heap Range : " << heap_range << std::endl;

    return (time * 1e-9) / test_count;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::srand(seed);

    // pull values from cmd
    std::size_t vector_size = vm["vector_size"].as<std::size_t>();
    std::size_t break_pos = vm["break_pos"].as<std::size_t>();
    int test_count = vm["test_count"].as<int>();

    std::size_t const os_threads = hpx::get_os_thread_count();

    if (break_pos > vector_size)
        break_pos = vector_size;

    std::cout << "-------------- Benchmark Config --------------" << std::endl;
    std::cout << "seed        : " << seed << std::endl;
    std::cout << "vector_size : " << vector_size << std::endl;
    std::cout << "break_pos   : " << break_pos << std::endl;
    std::cout << "test_count  : " << test_count << std::endl;
    std::cout << "os threads  : " << os_threads << std::endl;
    std::cout << "----------------------------------------------\n" << std::endl;

    std::cout << "* Preparing Benchmark..." << std::endl;
    std::vector<int> v(vector_size);

    // initialize data
    using namespace hpx::parallel;
    generate(execution::par, std::begin(v), std::end(v), random_fill());
    std::make_heap(std::begin(v), std::next(std::begin(v), break_pos));
    if (break_pos < vector_size)
        v[break_pos] = static_cast<int>((std::numeric_limits<std::size_t>::max)());

    std::cout << "* Running Benchmark..." << std::endl;
    double time_std =
        run_is_heap_until_benchmark_std(test_count, v);
    double time_seq =
        run_is_heap_until_benchmark_seq(test_count, v);
    double time_par =
        run_is_heap_until_benchmark_par(test_count, v);
    double time_par_unseq =
        run_is_heap_until_benchmark_par_unseq(test_count, v);

    std::cout << "\n-------------- Benchmark Result --------------" << std::endl;
    auto fmt = "is_heap_until (%1%) : %2%(sec)";
    std::cout << (boost::format(fmt) % "std" % time_std) << std::endl;
    std::cout << (boost::format(fmt) % "seq" % time_seq) << std::endl;
    std::cout << (boost::format(fmt) % "par" % time_par) << std::endl;
    std::cout << (boost::format(fmt) % "par_unseq" % time_par_unseq) << std::endl;
    std::cout << "----------------------------------------------" << std::endl;

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using namespace boost::program_options;
    options_description desc_commandline(
        "usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("vector_size",
            boost::program_options::value<std::size_t>()->default_value(1000000),
            "size of vector (default: 1000000)")
        ("break_pos",
            boost::program_options::value<std::size_t>()->
                default_value((std::numeric_limits<std::size_t>::max)()),
            "a position which breaks max heap (default: vector_size)")
        ("test_count",
            boost::program_options::value<int>()->default_value(10),
            "number of tests to be averaged (default: 10)")
        ("seed,s", boost::program_options::value<unsigned int>(),
            "the random number generator seed to use for this run")
        ;

    // initialize program
    std::vector<std::string> const cfg = {
        "hpx.os_threads=all"
    };

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
