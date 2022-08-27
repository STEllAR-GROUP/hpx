//  Copyright (c) 2020 Francisco Jose Tapia (fjtapia@gmail.com )
//  Copyright (c) 2021 Akhil J Nair
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/local/execution.hpp>
#include <hpx/local/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/parallel/algorithms/partial_sort.hpp>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
unsigned int seed = std::random_device{}();
std::mt19937 gen(seed);

#define SIZE 1007

template <typename IteratorTag>
void test_partial_sort(IteratorTag)
{
    using compare_t = std::less<std::uint64_t>;

    std::vector<std::uint64_t> A, B;
    A.reserve(SIZE);
    B.reserve(SIZE);

    for (std::uint64_t i = 0; i < SIZE; ++i)
    {
        A.emplace_back(i);
    }
    std::shuffle(A.begin(), A.end(), gen);

    for (std::uint64_t i = 1; i < SIZE; ++i)
    {
        B = A;
        hpx::partial_sort(B.begin(), B.begin() + i, B.end(), compare_t());

        for (std::uint64_t j = 0; j < i; ++j)
        {
            HPX_TEST(B[j] == j);
        }
    }
}

template <typename ExPolicy, typename IteratorTag>
void test_partial_sort(ExPolicy policy, IteratorTag)
{
    using compare_t = std::less<std::uint64_t>;

    std::vector<std::uint64_t> A, B;
    A.reserve(SIZE);
    B.reserve(SIZE);

    for (std::uint64_t i = 0; i < SIZE; ++i)
    {
        A.emplace_back(i);
    }
    std::shuffle(A.begin(), A.end(), gen);

    for (std::uint64_t i = 1; i < SIZE; ++i)
    {
        B = A;
        hpx::partial_sort(
            policy, B.begin(), B.begin() + i, B.end(), compare_t());

        for (std::uint64_t j = 0; j < i; ++j)
        {
            HPX_TEST(B[j] == j);
        }
    }
}

template <typename ExPolicy, typename IteratorTag>
void test_partial_sort_async(ExPolicy p, IteratorTag)
{
    using compare_t = std::less<std::uint64_t>;

    std::vector<std::uint64_t> A, B;
    A.reserve(SIZE);
    B.reserve(SIZE);

    for (std::uint64_t i = 0; i < SIZE; ++i)
    {
        A.emplace_back(i);
    }
    std::shuffle(A.begin(), A.end(), gen);

    for (std::uint64_t i = 1; i < SIZE; ++i)
    {
        B = A;
        auto result = hpx::partial_sort(
            p, B.begin(), B.begin() + i, B.end(), compare_t());
        result.wait();

        for (std::uint64_t j = 0; j < i; ++j)
        {
            HPX_TEST(B[j] == j);
        }
    }
}

template <typename IteratorTag>
void test_partial_sort()
{
    using namespace hpx::execution;
    test_partial_sort(IteratorTag());
    test_partial_sort(seq, IteratorTag());
    test_partial_sort(par, IteratorTag());
    test_partial_sort(par_unseq, IteratorTag());

    test_partial_sort_async(seq(task), IteratorTag());
    test_partial_sort_async(par(task), IteratorTag());
}

void partial_sort_test()
{
    test_partial_sort<std::random_access_iterator_tag>();
    test_partial_sort<std::forward_iterator_tag>();
}

int hpx_main(hpx::program_options::variables_map& vm)
{
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    gen.seed(seed);

    partial_sort_test();

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace hpx::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()("seed,s", value<unsigned int>(),
        "the random number generator seed to use for this run");

    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    // Initialize and run HPX
    hpx::local::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
