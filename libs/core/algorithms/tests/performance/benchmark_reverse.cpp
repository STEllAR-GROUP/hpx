///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2026 Arpit Khandelwal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <hpx/algorithm.hpp>
#include <hpx/chrono.hpp>
#include <hpx/format.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/program_options.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "utils.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename OrgIter, typename BidirIter>
double run_reverse_benchmark_std(int test_count, OrgIter org_first,
    OrgIter org_last, BidirIter first, BidirIter last)
{
    std::uint64_t time = std::uint64_t(0);

    for (int i = 0; i < test_count; ++i)
    {
        // Restore [first, last) with original data.
        hpx::copy(hpx::execution::par, org_first, org_last, first);

        std::uint64_t elapsed = hpx::chrono::high_resolution_clock::now();
        std::reverse(first, last);
        time += hpx::chrono::high_resolution_clock::now() - elapsed;
    }

    return (static_cast<double>(time) * 1e-9) / test_count;
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename OrgIter, typename BidirIter>
double run_reverse_benchmark_hpx(int test_count, ExPolicy policy,
    OrgIter org_first, OrgIter org_last, BidirIter first, BidirIter last)
{
    std::uint64_t time = std::uint64_t(0);

    for (int i = 0; i < test_count; ++i)
    {
        // Restore [first, last) with original data.
        hpx::copy(hpx::execution::par, org_first, org_last, first);

        std::uint64_t elapsed = hpx::chrono::high_resolution_clock::now();
        hpx::reverse(policy, first, last);
        time += hpx::chrono::high_resolution_clock::now() - elapsed;
    }

    return (static_cast<double>(time) * 1e-9) / test_count;
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void run_benchmark(std::size_t vector_size, int test_count, IteratorTag)
{
    std::cout << "* Preparing Benchmark..." << std::endl;

    typedef test_container<IteratorTag> test_container;
    typedef typename test_container::type container;

    container c = test_container::get_container(vector_size);
    container org_c;

    auto first = std::begin(c);
    auto last = std::end(c);

    // initialize data
    std::iota(first, last, 0);
    org_c = c;

    auto org_first = std::begin(org_c);
    auto org_last = std::end(org_c);

    std::cout << "* Running Benchmark..." << std::endl;
    std::cout << "--- run_reverse_benchmark_std ---" << std::endl;
    double time_std =
        run_reverse_benchmark_std(test_count, org_first, org_last, first, last);

    std::cout << "--- run_reverse_benchmark_seq ---" << std::endl;
    double time_seq = run_reverse_benchmark_hpx(
        test_count, hpx::execution::seq, org_first, org_last, first, last);

    std::cout << "--- run_reverse_benchmark_par ---" << std::endl;
    double time_par = run_reverse_benchmark_hpx(
        test_count, hpx::execution::par, org_first, org_last, first, last);

    std::cout << "--- run_reverse_benchmark_par_unseq ---" << std::endl;
    double time_par_unseq = run_reverse_benchmark_hpx(test_count,
        hpx::execution::par_unseq, org_first, org_last, first, last);

    std::cout << "\n-------------- Benchmark Result --------------"
              << std::endl;
    auto fmt = "reverse ({1}) : {2}(sec)";
    hpx::util::format_to(std::cout, fmt, "std", time_std) << std::endl;
    hpx::util::format_to(std::cout, fmt, "seq", time_seq) << std::endl;
    hpx::util::format_to(std::cout, fmt, "par", time_par) << std::endl;
    hpx::util::format_to(std::cout, fmt, "par_unseq", time_par_unseq)
        << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map& vm)
{
    // pull values from cmd
    std::size_t vector_size = vm["vector_size"].as<std::size_t>();
    int test_count = vm["test_count"].as<int>();
    std::string iterator_tag_str = vm["iterator_tag"].as<std::string>();

    std::size_t const os_threads = hpx::get_os_thread_count();

    std::cout << "-------------- Benchmark Config --------------" << std::endl;
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
    else
    {
        std::cerr << "unsupported iterator tag: " << iterator_tag_str
                  << std::endl;
        return hpx::local::finalize();
    }

    return hpx::local::finalize();
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
        "the kind of iterator tag (random/bidirectional)")("test_count",
        hpx::program_options::value<int>()->default_value(10),
        "number of tests to be averaged (default: 10)");

    // initialize program
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    // Initialize and run HPX
    hpx::local::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
