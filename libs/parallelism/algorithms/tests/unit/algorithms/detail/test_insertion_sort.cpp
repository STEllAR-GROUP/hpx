//  Copyright (c) 2015-2017 Francisco Jose Tapia
//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/parallel/algorithms/detail/insertion_sort.hpp>

#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>

using hpx::parallel::v1::detail::insertion_sort;

void test01()
{
    unsigned A[] = {7, 4, 23, 15, 17, 2, 24, 13, 8, 3, 11, 16, 6, 14, 21, 5, 1,
        12, 19, 22, 25, 8};

    insertion_sort(&A[0], &A[22]);
    for (unsigned i = 0; i < 21; i++)
    {
        HPX_TEST(A[i] <= A[i + 1]);
    }

    unsigned B[] = {1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19,
        20, 21, 23, 24, 25};

    insertion_sort(&B[0], &B[22]);
    for (unsigned i = 0; i < 21; i++)
    {
        HPX_TEST(B[i] <= B[i + 1]);
    }

    unsigned C[] = {27, 26, 25, 23, 22, 21, 19, 18, 17, 16, 15, 14, 13, 11, 10,
        9, 8, 7, 6, 5, 3, 2};

    insertion_sort(&C[0], &C[22]);
    for (unsigned i = 0; i < 21; i++)
    {
        HPX_TEST(C[i] <= C[i + 1]);
    }

    unsigned D[] = {
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};

    insertion_sort(&D[0], &D[22]);
    for (unsigned i = 0; i < 21; i++)
    {
        HPX_TEST(D[i] <= D[i + 1]);
    }

    unsigned F[100];
    for (unsigned i = 0; i < 100; i++)
        F[i] = std::rand() % 1000;
    insertion_sort(&F[0], &F[100]);
    for (unsigned i = 0; i < 99; i++)
    {
        HPX_TEST(F[i] <= F[i + 1]);
    }

    constexpr unsigned NG = 10000;

    unsigned G[NG];
    for (unsigned i = 0; i < NG; i++)
        G[i] = std::rand() % 1000;
    insertion_sort(&G[0], &G[NG]);
    for (unsigned i = 0; i < NG - 1; i++)
    {
        HPX_TEST(G[i] <= G[i + 1]);
    }
}

void test02()
{
    typedef typename std::vector<std::uint64_t>::iterator iter_t;
#if defined(HPX_DEBUG)
    constexpr std::uint32_t NELEM = 667;
#else
    constexpr std::uint32_t NELEM = 6667;
#endif
    std::vector<std::uint64_t> A;
    A.reserve(NELEM + 2000);

    for (std::uint32_t i = 0; i < 1000; ++i)
        A.push_back(0);
    for (std::uint32_t i = 0; i < NELEM; ++i)
        A.push_back(NELEM - i);
    for (std::uint32_t i = 0; i < 1000; ++i)
        A.push_back(0);

    insertion_sort(A.begin() + 1000, A.begin() + (1000 + NELEM));

    for (iter_t it = A.begin() + 1000; it != A.begin() + (1000 + NELEM); ++it)
    {
        HPX_TEST((*(it - 1)) <= (*it));
    }
    HPX_TEST(A[998] == 0 && A[999] == 0 && A[1000 + NELEM] == 0 &&
        A[1001 + NELEM] == 0);

    //------------------------------------------------------------------------
    A.clear();
    A.reserve(NELEM + 2000);

    for (std::uint32_t i = 0; i < 1000; ++i)
        A.push_back(999999999);
    for (std::uint32_t i = 0; i < NELEM; ++i)
        A.push_back(NELEM - i);
    for (std::uint32_t i = 0; i < 1000; ++i)
        A.push_back(999999999);

    insertion_sort(A.begin() + 1000, A.begin() + (1000 + NELEM));

    for (iter_t it = A.begin() + 1001; it != A.begin() + (1000 + NELEM); ++it)
    {
        HPX_TEST((*(it - 1)) <= (*it));
    }
    HPX_TEST(A[998] == 999999999 && A[999] == 999999999 &&
        A[1000 + NELEM] == 999999999 && A[1001 + NELEM] == 999999999);
}

int hpx_main(hpx::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    test01();
    test02();

    return hpx::finalize();
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
    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::init(argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
