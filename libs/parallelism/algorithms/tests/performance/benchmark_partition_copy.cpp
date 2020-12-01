///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2017 Taeguk Kwon
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
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
unsigned int seed = (unsigned int) std::random_device{}();
std::mt19937 _rand(seed);

///////////////////////////////////////////////////////////////////////////////

struct random_fill
{
    random_fill()
      : gen(_rand())
      , dist(0, RAND_MAX)
    {
    }

    int operator()()
    {
        return dist(gen);
    }

    std::mt19937 gen;
    std::uniform_int_distribution<> dist;

    template <typename Archive>
    void serialize(Archive&, unsigned)
    {
    }
};

///////////////////////////////////////////////////////////////////////////////
template <typename InIter, typename OutIter1, typename OutIter2, typename Pred>
double run_partition_copy_benchmark_std(int test_count, InIter first,
    InIter last, OutIter1 dest_true, OutIter2 dest_false, Pred pred)
{
    std::uint64_t time = hpx::chrono::high_resolution_clock::now();

    for (int i = 0; i < test_count; ++i)
    {
        std::partition_copy(first, last, dest_true, dest_false, pred);
    }

    time = hpx::chrono::high_resolution_clock::now() - time;

    return (time * 1e-9) / test_count;
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
    typename FwdIter3, typename Pred>
double run_partition_copy_benchmark_hpx(int test_count, ExPolicy policy,
    FwdIter1 first, FwdIter1 last, FwdIter2 dest_true, FwdIter3 dest_false,
    Pred pred)
{
    std::uint64_t time = hpx::chrono::high_resolution_clock::now();

    for (int i = 0; i < test_count; ++i)
    {
        using namespace hpx::execution;
        hpx::parallel::partition_copy(
            policy, first, last, dest_true, dest_false, pred);
    }

    time = hpx::chrono::high_resolution_clock::now() - time;

    return (time * 1e-9) / test_count;
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void run_benchmark(std::size_t vector_size, int test_count, IteratorTag)
{
    std::cout << "* Preparing Benchmark..." << std::endl;

    typedef test_container<IteratorTag> test_container;
    typedef typename test_container::type container;

    container v = test_container::get_container(vector_size);
    container result_true = test_container::get_container(vector_size);
    container result_false = test_container::get_container(vector_size);

    auto first = std::begin(v);
    auto last = std::end(v);
    auto dest_true = std::begin(result_true);
    auto dest_false = std::begin(result_false);

    // initialize data
    using namespace hpx::execution;
    hpx::generate(par, std::begin(v), std::end(v), random_fill());

    std::cout << "* Running Benchmark..." << std::endl;

    int rand_base = _rand();

    auto pred = [rand_base](int t) { return t < rand_base; };

    std::cout << "--- run_partition_copy_benchmark_std ---" << std::endl;
    double time_std = run_partition_copy_benchmark_std(
        test_count, first, last, dest_true, dest_false, pred);

    std::cout << "--- run_partition_copy_benchmark_seq ---" << std::endl;
    double time_seq = run_partition_copy_benchmark_hpx(
        test_count, seq, first, last, dest_true, dest_false, pred);

    std::cout << "--- run_partition_copy_benchmark_par ---" << std::endl;
    double time_par = run_partition_copy_benchmark_hpx(
        test_count, par, first, last, dest_true, dest_false, pred);

    std::cout << "--- run_partition_copy_benchmark_par_unseq ---" << std::endl;
    double time_par_unseq = run_partition_copy_benchmark_hpx(
        test_count, par_unseq, first, last, dest_true, dest_false, pred);

    std::cout << "\n-------------- Benchmark Result --------------"
              << std::endl;
    auto fmt = "partition_copy ({1}) : {2}(sec)";
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
int hpx_main(hpx::program_options::variables_map& vm)
{
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    _rand.seed(seed);

    // pull values from cmd
    std::size_t vector_size = vm["vector_size"].as<std::size_t>();
    int test_count = vm["test_count"].as<int>();
    std::string iterator_tag_str =
        correct_iterator_tag_str(vm["iterator_tag"].as<std::string>());

    std::size_t const os_threads = hpx::get_os_thread_count();

    std::cout << "-------------- Benchmark Config --------------" << std::endl;
    std::cout << "seed         : " << seed << std::endl;
    std::cout << "vector_size  : " << vector_size << std::endl;
    std::cout << "iterator_tag : " << iterator_tag_str << std::endl;
    std::cout << "test_count   : " << test_count << std::endl;
    std::cout << "os threads   : " << os_threads << std::endl;
    std::cout << "----------------------------------------------\n"
              << std::endl;

    if (iterator_tag_str == "random")
        run_benchmark(
            vector_size, test_count, std::random_access_iterator_tag());
    else if (iterator_tag_str == "bidirectional")
        run_benchmark(
            vector_size, test_count, std::bidirectional_iterator_tag());
    else    // forward
        run_benchmark(vector_size, test_count, std::forward_iterator_tag());

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
