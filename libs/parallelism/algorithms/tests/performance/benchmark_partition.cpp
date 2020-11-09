///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2017 Taeguk Kwon
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_copy.hpp>
#include <hpx/include/parallel_generate.hpp>
#include <hpx/include/parallel_partition.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/modules/timing.hpp>

#include <hpx/modules/program_options.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "utils.hpp"

///////////////////////////////////////////////////////////////////////////////
const int random_fill_range = (std::min)(100000, RAND_MAX);
unsigned int seed = std::random_device{}();

///////////////////////////////////////////////////////////////////////////////
struct random_fill
{
    random_fill()
      : gen(seed)
      , dist(0, random_fill_range)
    {
    }

    int operator()()
    {
        return dist(gen);
    }

    std::mt19937 gen;
    std::uniform_int_distribution<> dist;
};

///////////////////////////////////////////////////////////////////////////////
template <typename OrgIter, typename FwdIter, typename Pred>
double run_partition_benchmark_std(int test_count, OrgIter org_first,
    OrgIter org_last, FwdIter first, FwdIter last, Pred pred)
{
    std::uint64_t time = std::uint64_t(0);

    for (int i = 0; i < test_count; ++i)
    {
        // Restore [first, last) with original data.
        hpx::copy(hpx::execution::par, org_first, org_last, first);

        std::uint64_t elapsed = hpx::chrono::high_resolution_clock::now();
        std::partition(first, last, pred);
        time += hpx::chrono::high_resolution_clock::now() - elapsed;
    }

    return (time * 1e-9) / test_count;
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename OrgIter, typename FwdIter, typename Pred>
double run_partition_benchmark_hpx(int test_count, ExPolicy policy,
    OrgIter org_first, OrgIter org_last, FwdIter first, FwdIter last, Pred pred)
{
    std::uint64_t time = std::uint64_t(0);

    for (int i = 0; i < test_count; ++i)
    {
        // Restore [first, last) with original data.
        hpx::copy(hpx::execution::par, org_first, org_last, first);

        std::uint64_t elapsed = hpx::chrono::high_resolution_clock::now();
        hpx::parallel::partition(policy, first, last, pred);
        time += hpx::chrono::high_resolution_clock::now() - elapsed;
    }

    return (time * 1e-9) / test_count;
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void run_benchmark(
    std::size_t vector_size, int test_count, int base_num, IteratorTag)
{
    std::cout << "* Preparing Benchmark..." << std::endl;

    typedef test_container<IteratorTag> test_container;
    typedef typename test_container::type container;

    container v = test_container::get_container(vector_size);
    container org_v;

    auto first = std::begin(v);
    auto last = std::end(v);

    // initialize data
    using namespace hpx::execution;
    hpx::generate(par, std::begin(v), std::end(v), random_fill());
    org_v = v;

    auto org_first = std::begin(org_v);
    auto org_last = std::end(org_v);

    std::cout << "* Running Benchmark..." << std::endl;

    auto pred = [base_num](int t) { return t < base_num; };

    std::cout << "--- run_partition_benchmark_std ---" << std::endl;
    double time_std = run_partition_benchmark_std(
        test_count, org_first, org_last, first, last, pred);

    std::cout << "--- run_partition_benchmark_seq ---" << std::endl;
    double time_seq = run_partition_benchmark_hpx(
        test_count, seq, org_first, org_last, first, last, pred);

    std::cout << "--- run_partition_benchmark_par ---" << std::endl;
    double time_par = run_partition_benchmark_hpx(
        test_count, par, org_first, org_last, first, last, pred);

    std::cout << "--- run_partition_benchmark_par_unseq ---" << std::endl;
    double time_par_unseq = run_partition_benchmark_hpx(
        test_count, par_unseq, org_first, org_last, first, last, pred);

    std::cout << "\n-------------- Benchmark Result --------------"
              << std::endl;
    auto fmt = "partition ({1}) : {2}(sec)";
    hpx::util::format_to(std::cout, fmt, "std", time_std) << std::endl;
    hpx::util::format_to(std::cout, fmt, "seq", time_seq) << std::endl;
    hpx::util::format_to(std::cout, fmt, "par", time_par) << std::endl;
    hpx::util::format_to(std::cout, fmt, "par_unseq", time_par_unseq)
        << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
std::string correct_iterator_tag_str(std::string iterator_tag)
{
    if (iterator_tag != "random" && iterator_tag != "bidirectional" &&
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
int hpx_main(hpx::program_options::variables_map& vm)
{
    // If simply using current time as seed, random numbers are closer
    //     when time is closer.
    if (vm.count("seed"))
        seed = vm["seed"].as<std::uint32_t>();

    std::mt19937 gen(static_cast<unsigned int>(seed));
    std::uniform_int_distribution<> dis(0, random_fill_range - 1);

    // pull values from cmd
    std::size_t vector_size = vm["vector_size"].as<std::size_t>();
    int base_num = dis(gen);
    if (vm.count("base_num"))
        base_num = vm["base_num"].as<int>();
    int test_count = vm["test_count"].as<int>();
    std::string iterator_tag_str =
        correct_iterator_tag_str(vm["iterator_tag"].as<std::string>());

    std::size_t const os_threads = hpx::get_os_thread_count();

    std::cout << "-------------- Benchmark Config --------------" << std::endl;
    std::cout << "seed            : " << seed << std::endl;
    std::cout << "vector_size     : " << vector_size << std::endl;
    std::cout << "rand_fill range : " << random_fill_range << std::endl;
    std::cout << "base_num        : " << base_num << std::endl;
    std::cout << "iterator_tag    : " << iterator_tag_str << std::endl;
    std::cout << "test_count      : " << test_count << std::endl;
    std::cout << "os threads      : " << os_threads << std::endl;
    std::cout << "----------------------------------------------\n"
              << std::endl;

    if (iterator_tag_str == "random")
        run_benchmark(vector_size, test_count, base_num,
            std::random_access_iterator_tag());
    else if (iterator_tag_str == "bidirectional")
        run_benchmark(vector_size, test_count, base_num,
            std::bidirectional_iterator_tag());
    else    // forward
        run_benchmark(
            vector_size, test_count, base_num, std::forward_iterator_tag());

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using namespace hpx::program_options;
    options_description desc_commandline(
        "usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()("vector_size",
        hpx::program_options::value<std::size_t>()->default_value(1000000),
        "size of vector (default: 1000000)")("iterator_tag",
        hpx::program_options::value<std::string>()->default_value("random"),
        "the kind of iterator tag (random/bidirectional/forward)")("base_num",
        hpx::program_options::value<int>(),
        hpx::util::format("the base number for partitioning."
                          " The range of random_fill is [0, {1}]"
                          " (default: random number in the range [0, {2}]",
            random_fill_range, random_fill_range)
            .c_str())("test_count",
        hpx::program_options::value<int>()->default_value(10),
        "number of tests to be averaged (default: 10)")("seed,s",
        hpx::program_options::value<std::uint32_t>(),
        "the random number generator seed to use for this run");

    // initialize program
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::init(argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
