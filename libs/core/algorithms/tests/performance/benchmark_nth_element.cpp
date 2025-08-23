//----------------------------------------------------------------------------
/// \file benchmark_nth_element.cpp
/// \brief Benchmark program of the nth_element function
///
//  Copyright (c) 2020 Francisco Jose Tapia (fjtapia@gmail.com )
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//-----------------------------------------------------------------------------
#include <hpx/algorithm.hpp>
#include <hpx/assert.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/program_options.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <vector>
#include <version>

//---------------------------------------------------------------------------
// Compare the speed with the implementation of the compiler, in a vector
// with 10000 elements, checking all the positions
//----------------------------------------------------------------------------
void function01(void)
{
    std::cout << "BENCHMARK 1\n";
    std::cout
        << "Accumulate the times of all the positions with 10000 elements\n";
    std::cout
        << "-----------------------------------------------------------\n";
    typedef std::less<uint64_t> compare_t;
    std::mt19937 my_rand(0);
    std::vector<uint64_t> A, B;
    const uint32_t NELEM = 10000;
    A.reserve(NELEM);
    B.reserve(NELEM);

    for (uint64_t i = 0; i < NELEM; ++i)
        A.emplace_back(i);
    std::shuffle(A.begin(), A.end(), my_rand);

    auto start = std::chrono::high_resolution_clock::now();
    for (uint64_t i = 0; i < NELEM; ++i)
    {
        B = A;
        hpx::nth_element(B.begin(), B.begin() + static_cast<std::ptrdiff_t>(i),
            B.end(), compare_t());
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<long unsigned, std::nano> nanotime1 = end - start;
    std::cout << "hpx::parallel::nth_element :" << (nanotime1.count() / 1000000)
              << " msec\n";

    start = std::chrono::high_resolution_clock::now();
    for (uint64_t i = 0; i < NELEM; ++i)
    {
        B = A;
        std::nth_element(B.begin(), B.begin() + static_cast<std::ptrdiff_t>(i),
            B.end(), compare_t());
    }
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<long unsigned, std::nano> nanotime2 = end - start;
    std::cout << "std::nth_element           :" << (nanotime2.count() / 1000000)
              << " msec\n";
}
//----------------------------------------------------------------------------
/// nth_element in a vector with 10000000 elements, with different number
//  of positions to sort.
/// Accumulate all the times and compare the two algorithms with 1 thread
//----------------------------------------------------------------------------
void function02(void)
{
    std::cout << "\nBENCHMARK 2\n";
    std::cout << "Times with different positionss with 10000000 elements\n";
    std::cout << "-------------------------------------------------\n";
    typedef std::less<uint64_t> compare_t;
    std::mt19937 my_rand(0);
    std::vector<uint64_t> A, B;
    const uint32_t NELEM = 10000000;
    A.reserve(NELEM);
    B.reserve(NELEM);

    for (uint64_t i = 0; i < NELEM; ++i)
        A.emplace_back(i);
    std::shuffle(A.begin(), A.end(), my_rand);

    uint64_t ac1 = 0, ac2 = 0;
    const uint32_t STEP = NELEM / 20;
    for (uint64_t i = 0; i < NELEM; i += STEP)
    {
        std::cout << "Searched position [" << i << "]   \t";

        B = A;
        auto start = std::chrono::high_resolution_clock::now();
        hpx::nth_element(::hpx::execution::seq, B.begin(),
            B.begin() + static_cast<std::ptrdiff_t>(i), B.end(), compare_t());
        //hpx::nth_element (B.begin(), B.begin() + i, B.end(),compare_t());
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<long unsigned, std::nano> nanotime1 = end - start;
        ac1 += nanotime1.count();
        std::cout << "hpx::parallel::nth_element :"
                  << (nanotime1.count() / 1000000) << " msec";
        HPX_ASSERT(B[i] == i);

        B = A;
        start = std::chrono::high_resolution_clock::now();
        std::nth_element(B.begin(), B.begin() + static_cast<std::ptrdiff_t>(i),
            B.end(), compare_t());
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<long unsigned, std::nano> nanotime2 = end - start;
        ac2 += nanotime2.count();
        std::cout << "  \tstd::nth_element       :"
                  << (nanotime2.count() / 1000000) << " msec\n";
        HPX_ASSERT(B[i] == i);
    }
    std::cout << "\n\n";
    std::cout << "Accumulated (msec) hpx::parallel::nth_element "
              << ac1 / 1000000 << " msec\n";
    std::cout << "Accumulated (msec) std::nth_element           "
              << ac2 / 1000000 << " msec\n";
}

int test_main(void)
{
    std::cout << "\n\n";
    std::cout
        << "**********************************************************\n\n";
    std::cout << "           BENCHMARK_NTH_ELEMENT (1 Thread)\n";
    std::cout << "(Times are expressed in millisecods (msecs)";
    std::cout << " Less is better)\n\n";
    std::cout
        << "**********************************************************\n\n";
    function01();
    function02();
    std::cout
        << "------------------------ end -------------------------------\n";
    return 0;
}

int hpx_main(hpx::program_options::variables_map&)
{
    test_main();

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=all");
    hpx::local::init_params init_args;
    init_args.cfg = cfg;

    // Initialize and run HPX.
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
