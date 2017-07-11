///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2017 Taeguk Kwon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_unique.hpp>
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
#include <iostream>
#include <string>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
struct random_fill
{
    random_fill(std::size_t random_range)
        : gen(std::rand()),
        dist(0, random_range - 1)
    {}

    int operator()()
    {
        return dist(gen);
    }

    boost::random::mt19937 gen;
    boost::random::uniform_int_distribution<> dist;
};

///////////////////////////////////////////////////////////////////////////////
template <typename InIter, typename OutIter>
double run_unique_copy_benchmark_std(int test_count,
    InIter first, InIter last, OutIter dest)
{
    std::uint64_t time = hpx::util::high_resolution_clock::now();

    for (int i = 0; i < test_count; ++i)
    {
        std::unique_copy(first, last, dest);
    }

    time = hpx::util::high_resolution_clock::now() - time;

    return (time * 1e-9) / test_count;
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename FwdIter1, typename FwdIter2>
double run_unique_copy_benchmark_hpx(int test_count, ExPolicy policy,
    FwdIter1 first, FwdIter1 last, FwdIter2 dest)
{
    std::uint64_t time = hpx::util::high_resolution_clock::now();

    for (int i = 0; i < test_count; ++i)
    {
        hpx::parallel::unique_copy(policy, first, last, dest);
    }

    time = hpx::util::high_resolution_clock::now() - time;

    return (time * 1e-9) / test_count;
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void run_benchmark(std::size_t vector_size, std::size_t random_range,
    int test_count, IteratorTag)
{
    std::cout << "* Preparing Benchmark..." << std::endl;

    typedef typename std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> v(vector_size);
    std::vector<int> result(v.size());
    iterator first = iterator(std::begin(v));
    iterator last = iterator(std::end(v));
    iterator dest = iterator(std::begin(result));

    // initialize data
    using namespace hpx::parallel;
    generate(execution::par, std::begin(v), std::end(v),
        random_fill(random_range));

    auto dest_dist = std::distance(std::begin(result),
        std::unique_copy(first, last, dest).base());

    std::cout << "*** Destination iterator distance : "
        << dest_dist << std::endl << std::endl;

    std::cout << "* Running Benchmark..." << std::endl;
    std::cout << "--- run_unique_copy_benchmark_std ---" << std::endl;
    double time_std =
        run_unique_copy_benchmark_std(test_count,
            first, last, dest);

    std::cout << "--- run_unique_copy_benchmark_seq ---" << std::endl;
    double time_seq =
        run_unique_copy_benchmark_hpx(test_count, execution::seq,
            first, last, dest);

    std::cout << "--- run_unique_copy_benchmark_par ---" << std::endl;
    double time_par =
        run_unique_copy_benchmark_hpx(test_count, execution::par,
            first, last, dest);

    std::cout << "--- run_unique_copy_benchmark_par_unseq ---" << std::endl;
    double time_par_unseq =
        run_unique_copy_benchmark_hpx(test_count, execution::par_unseq,
            first, last, dest);

    std::cout << "\n-------------- Benchmark Result --------------" << std::endl;
    auto fmt = "unique_copy (%1%) : %2%(sec)";
    std::cout << (boost::format(fmt) % "std" % time_std) << std::endl;
    std::cout << (boost::format(fmt) % "seq" % time_seq) << std::endl;
    std::cout << (boost::format(fmt) % "par" % time_par) << std::endl;
    std::cout << (boost::format(fmt) % "par_unseq" % time_par_unseq) << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
std::string correct_iterator_tag_str(std::string iterator_tag)
{
    if (iterator_tag != "random" &&
        iterator_tag != "bidirectional" &&
        iterator_tag != "forward")
        return "random";
    else
        return iterator_tag;
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
    std::size_t random_range = vm["random_range"].as<std::size_t>();
    int test_count = vm["test_count"].as<int>();
    std::string iterator_tag_str = correct_iterator_tag_str(
        vm["iterator_tag"].as<std::string>());

    std::size_t const os_threads = hpx::get_os_thread_count();

    if (random_range < 1)
        random_range = 1;

    std::cout << "-------------- Benchmark Config --------------" << std::endl;
    std::cout << "seed         : " << seed << std::endl;
    std::cout << "vector_size  : " << vector_size << std::endl;
    std::cout << "random_range : " << random_range << std::endl;
    std::cout << "iterator_tag : " << iterator_tag_str << std::endl;
    std::cout << "test_count   : " << test_count << std::endl;
    std::cout << "os threads   : " << os_threads << std::endl;
    std::cout << "----------------------------------------------\n" << std::endl;

    if (iterator_tag_str == "random")
        run_benchmark(vector_size, test_count, random_range,
            std::random_access_iterator_tag());
    else if (iterator_tag_str == "bidirectional")
        run_benchmark(vector_size, test_count, random_range,
            std::bidirectional_iterator_tag());
    else // forward
        run_benchmark(vector_size, test_count, random_range,
            std::forward_iterator_tag());

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
        ("random_range",
            boost::program_options::value<std::size_t>()->default_value(6),
            "range of random numbers [0, x) (default: 6)")
        ("iterator_tag",
            boost::program_options::value<std::string>()->default_value("random"),
            "the kind of iterator tag (random/bidirectional/forward)")
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
