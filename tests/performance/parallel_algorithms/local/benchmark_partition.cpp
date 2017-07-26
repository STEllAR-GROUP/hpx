///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2017 Taeguk Kwon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_partition.hpp>
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

#include "test_utils.hpp"

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
template <typename InIter, typename Pred>
double run_partition_benchmark_std(int test_count,
    InIter first, InIter last, Pred pred)
{
    std::uint64_t time = hpx::util::high_resolution_clock::now();

    for (int i = 0; i < test_count; ++i)
    {
        std::partition(first, last, pred);
    }

    time = hpx::util::high_resolution_clock::now() - time;

    return (time * 1e-9) / test_count;
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename FwdIter, typename Pred>
double run_partition_benchmark_hpx(int test_count, ExPolicy policy,
    FwdIter first, FwdIter last, Pred pred)
{
    std::uint64_t time = hpx::util::high_resolution_clock::now();

    for (int i = 0; i < test_count; ++i)
    {
        using namespace hpx::parallel;
        partition(policy, first, last, pred);
    }

    time = hpx::util::high_resolution_clock::now() - time;

    return (time * 1e-9) / test_count;
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void run_benchmark(std::size_t vector_size, int test_count, int base_num,
    IteratorTag)
{
    std::cout << "* Preparing Benchmark..." << std::endl;

    typedef typename std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> v(vector_size);
    iterator first = iterator(std::begin(v));
    iterator last = iterator(std::end(v));

    // initialize data
    using namespace hpx::parallel;
    generate(execution::par, std::begin(v), std::end(v), random_fill());

    std::cout << "* Running Benchmark..." << std::endl;

    auto pred = [base_num](auto const& t) {
        return t < base_num;
    };

    std::cout << "--- run_partition_benchmark_std ---" << std::endl;
    double time_std =
        run_partition_benchmark_std(test_count, first, last, pred);

    std::cout << "--- run_partition_benchmark_seq ---" << std::endl;
    double time_seq =
        run_partition_benchmark_hpx(test_count, execution::seq,
            first, last, pred);

    std::cout << "--- run_partition_benchmark_par ---" << std::endl;
    double time_par =
        run_partition_benchmark_hpx(test_count, execution::par,
            first, last, pred);

    std::cout << "--- run_partition_benchmark_par_unseq ---" << std::endl;
    double time_par_unseq =
        run_partition_benchmark_hpx(test_count, execution::par_unseq,
            first, last, pred);

    std::cout << "\n-------------- Benchmark Result --------------" << std::endl;
    auto fmt = "partition (%1%) : %2%(sec)";
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
// Robert Jenkins' 32 bit integer hash function.
std::uint32_t hash(std::uint32_t n)
{
    n = (n + 0x7ed55d16) + (n << 12);
    n = (n ^ 0xc761c23c) ^ (n >> 19);
    n = (n + 0x165667b1) + (n << 5);
    n = (n + 0xd3a2646c) ^ (n << 9);
    n = (n + 0xfd7046c5) + (n << 3);
    n = (n ^ 0xb55a4f09) ^ (n >> 16);
    return n;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    // If simply using current time as seed, random numbers are closer
    //     when time is closer.
    std::uint32_t seed = hash(std::uint32_t(std::time(nullptr)));
    if (vm.count("seed"))
        seed = vm["seed"].as<std::uint32_t>();

    std::srand(static_cast<unsigned int>(seed));

    // pull values from cmd
    std::size_t vector_size = vm["vector_size"].as<std::size_t>();
    int base_num = vm["base_num"].as<int>();
    if (base_num == -1)
        base_num = std::rand();
    int test_count = vm["test_count"].as<int>();
    std::string iterator_tag_str = correct_iterator_tag_str(
        vm["iterator_tag"].as<std::string>());

    std::size_t const os_threads = hpx::get_os_thread_count();

    std::cout << "-------------- Benchmark Config --------------" << std::endl;
    std::cout << "seed         : " << seed << std::endl;
    std::cout << "vector_size  : " << vector_size << std::endl;
    std::cout << "base_num     : " << base_num << std::endl;
    std::cout << "iterator_tag : " << iterator_tag_str << std::endl;
    std::cout << "test_count   : " << test_count << std::endl;
    std::cout << "os threads   : " << os_threads << std::endl;
    std::cout << "----------------------------------------------\n" << std::endl;

    if (iterator_tag_str == "random")
        run_benchmark(vector_size, test_count, base_num,
            std::random_access_iterator_tag());
    else if (iterator_tag_str == "bidirectional")
        run_benchmark(vector_size, test_count, base_num,
            std::bidirectional_iterator_tag());
    else // forward
        run_benchmark(vector_size, test_count, base_num,
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
        ("iterator_tag",
            boost::program_options::value<std::string>()->default_value("random"),
            "the kind of iterator tag (random/bidirectional/forward)")
        ("base_num",
            boost::program_options::value<int>()->default_value(-1),
            (boost::format(
                "the base number for partitioning. The range is [0, %1%]"
                " (default: random number)") % RAND_MAX).str().c_str())
        ("test_count",
            boost::program_options::value<int>()->default_value(10),
            "number of tests to be averaged (default: 10)")
        ("seed,s", boost::program_options::value<std::uint32_t>(),
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
