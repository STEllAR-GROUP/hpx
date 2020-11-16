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
#include <hpx/include/parallel_merge.hpp>
#include <hpx/include/parallel_sort.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/modules/timing.hpp>

#include <hpx/modules/program_options.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "utils.hpp"

///////////////////////////////////////////////////////////////////////////////
unsigned int seed = (unsigned int) std::random_device{}();
std::mt19937 _rand(seed);
///////////////////////////////////////////////////////////////////////////////

struct random_fill
{
    random_fill(std::size_t random_range)
      : gen(_rand())
      , dist(0, random_range - 1)
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
template <typename OrgIter, typename BidirIter>
double run_inplace_merge_benchmark_std(int test_count, OrgIter org_first,
    OrgIter org_last, BidirIter first, BidirIter middle, BidirIter last)
{
    std::uint64_t time = std::uint64_t(0);

    for (int i = 0; i < test_count; ++i)
    {
        // Restore [first, last) with original data.
        hpx::copy(hpx::execution::par, org_first, org_last, first);

        std::uint64_t elapsed = hpx::chrono::high_resolution_clock::now();
        std::inplace_merge(first, middle, last);
        time += hpx::chrono::high_resolution_clock::now() - elapsed;
    }

    return (time * 1e-9) / test_count;
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename OrgIter, typename BidirIter>
double run_inplace_merge_benchmark_hpx(int test_count, ExPolicy policy,
    OrgIter org_first, OrgIter org_last, BidirIter first, BidirIter middle,
    BidirIter last)
{
    std::uint64_t time = std::uint64_t(0);

    for (int i = 0; i < test_count; ++i)
    {
        // Restore [first, last) with original data.
        hpx::copy(hpx::execution::par, org_first, org_last, first);

        std::uint64_t elapsed = hpx::chrono::high_resolution_clock::now();
        hpx::inplace_merge(policy, first, middle, last);
        time += hpx::chrono::high_resolution_clock::now() - elapsed;
    }

    return (time * 1e-9) / test_count;
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void run_benchmark(std::size_t vector_left_size, std::size_t vector_right_size,
    int test_count, std::size_t random_range, IteratorTag)
{
    std::cout << "* Preparing Benchmark..." << std::endl;

    typedef test_container<IteratorTag> test_container;
    typedef typename test_container::type container;

    container c =
        test_container::get_container(vector_left_size + vector_right_size);
    container org_c;

    auto first = std::begin(c);
    auto middle = first + vector_left_size;
    auto last = std::end(c);

    // initialize data
    using namespace hpx::execution;
    hpx::generate(par, first, middle, random_fill(random_range));
    hpx::generate(par, middle, last, random_fill(random_range));
    hpx::parallel::sort(par, first, middle);
    hpx::parallel::sort(par, middle, last);
    org_c = c;

    auto org_first = std::begin(org_c);
    auto org_last = std::end(org_c);

    std::cout << "* Running Benchmark..." << std::endl;
    std::cout << "--- run_inplace_merge_benchmark_std ---" << std::endl;
    double time_std = run_inplace_merge_benchmark_std(
        test_count, org_first, org_last, first, middle, last);

    std::cout << "--- run_inplace_merge_benchmark_seq ---" << std::endl;
    double time_seq = run_inplace_merge_benchmark_hpx(
        test_count, seq, org_first, org_last, first, middle, last);

    std::cout << "--- run_inplace_merge_benchmark_par ---" << std::endl;
    double time_par = run_inplace_merge_benchmark_hpx(
        test_count, par, org_first, org_last, first, middle, last);

    std::cout << "--- run_inplace_merge_benchmark_par_unseq ---" << std::endl;
    double time_par_unseq = run_inplace_merge_benchmark_hpx(
        test_count, par_unseq, org_first, org_last, first, middle, last);

    std::cout << "\n-------------- Benchmark Result --------------"
              << std::endl;
    auto fmt = "inplace_merge ({1}) : {2}(sec)";
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
    if (iterator_tag != "random"/* &&
        iterator_tag != "bidirectional"*/)
        return "random";
    else
        return iterator_tag;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map& vm)
{
    if (vm.count("seed"))
    {
        seed = vm["seed"].as<unsigned int>();
        _rand.seed(seed);
    }

    // pull values from cmd
    std::size_t vector_size = vm["vector_size"].as<std::size_t>();
    double vector_ratio = vm["vector_ratio"].as<double>();
    std::size_t random_range = vm["random_range"].as<std::size_t>();
    int test_count = vm["test_count"].as<int>();
    std::string iterator_tag_str =
        correct_iterator_tag_str(vm["iterator_tag"].as<std::string>());

    std::size_t const os_threads = hpx::get_os_thread_count();

    if (random_range < 1)
        random_range = 1;

    std::size_t vector_left_size = std::size_t(vector_size * vector_ratio);
    std::size_t vector_right_size = vector_size - vector_left_size;

    std::cout << "-------------- Benchmark Config --------------" << std::endl;
    std::cout << "seed              : " << seed << std::endl;
    std::cout << "vector_left_size  : " << vector_left_size << std::endl;
    std::cout << "vector_right_size : " << vector_right_size << std::endl;
    std::cout << "random_range      : " << random_range << std::endl;
    std::cout << "iterator_tag      : " << iterator_tag_str << std::endl;
    std::cout << "test_count        : " << test_count << std::endl;
    std::cout << "os threads        : " << os_threads << std::endl;
    std::cout << "----------------------------------------------\n"
              << std::endl;

    if (iterator_tag_str == "random")
        run_benchmark(vector_left_size, vector_right_size, test_count,
            random_range, std::random_access_iterator_tag());
    //else // bidirectional
    //    run_benchmark(vector_left_size, vector_right_size,
    //        test_count, random_range,
    //        std::bidirectional_iterator_tag());

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using namespace hpx::program_options;
    options_description desc_commandline(
        "usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()("vector_size",
        hpx::program_options::value<std::size_t>()->default_value(1000000),
        "sum of sizes of two vectors (default: 1000000)")("vector_ratio",
        hpx::program_options::value<double>()->default_value(0.7),
        "ratio of two vector sizes (default: 0.7)")("random_range",
        hpx::program_options::value<std::size_t>()->default_value(6),
        "range of random numbers [0, x) (default: 6)")("iterator_tag",
        hpx::program_options::value<std::string>()->default_value("random"),
        "the kind of iterator tag (random/bidirectional/forward)")("test_count",
        hpx::program_options::value<int>()->default_value(10),
        "number of tests to be averaged (default: 10)")("seed,s",
        hpx::program_options::value<unsigned int>(),
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
