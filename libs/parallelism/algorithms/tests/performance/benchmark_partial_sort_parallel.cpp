//  Copyright (c) 2020 Francisco Jose Tapia (fjtapia@gmail.com )
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/execution.hpp>
#include <hpx/hpx.hpp>
#include <hpx/init.hpp>
#include <hpx/parallel/algorithms/partial_sort.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

////////////////////////////////////////////////////////////////////////////
unsigned int seed = std::random_device{}();
std::mt19937 gen(seed);

// Compare the speed with the implementation of the compiler
void function01()
{
    typedef std::less<std::uint64_t> compare_t;
#if defined(HPX_DEBUG)
    constexpr std::uint32_t NELEM = 100;
#else
    constexpr std::uint32_t NELEM = 10000;
#endif

    std::vector<std::uint64_t> A, B;
    A.reserve(NELEM);
    B.reserve(NELEM);

    for (std::uint64_t i = 0; i < NELEM; ++i)
    {
        A.emplace_back(i);
    }
    std::shuffle(A.begin(), A.end(), gen);

    auto start = std::chrono::high_resolution_clock::now();
    for (std::uint64_t i = 0; i <= NELEM; ++i)
    {
        B = A;
        hpx::partial_sort(::hpx::execution::par, B.begin(), B.begin() + i,
            B.end(), compare_t());
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<long unsigned, std::nano> nanotime1 = end - start;
    std::cout << "hpx::parallel::partial_sort :"
              << (nanotime1.count() / 1000000) << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for (std::uint64_t i = 0; i <= NELEM; ++i)
    {
        B = A;
        std::partial_sort(B.begin(), B.begin() + i, B.end(), compare_t());
    }
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<long unsigned, std::nano> nanotime2 = end - start;
    std::cout << "std::partial_sort           :"
              << (nanotime2.count() / 1000000) << std::endl;
}

void function02()
{
    std::less<std::uint64_t> comp;
#if defined(HPX_DEBUG)
    constexpr std::uint32_t NELEM = 100000;
#else
    constexpr std::uint32_t NELEM = 10000000;
#endif

    std::vector<std::uint64_t> A, B;
    A.reserve(NELEM);
    B.reserve(NELEM);

    for (uint64_t i = 0; i < NELEM; ++i)
        A.emplace_back(i);
    std::shuffle(A.begin(), A.end(), gen);
    std::cout << "\n";

    // std::sort
    B = A;
    auto start = std::chrono::high_resolution_clock::now();
    std::sort(B.begin(), B.end(), comp);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<long unsigned, std::nano> nanotime1 = end - start;
    std::cout << "std::sort                    :"
              << (nanotime1.count() / 1000000) << std::endl;

    // heap sort
    B = A;
    start = std::chrono::high_resolution_clock::now();
    std::make_heap(B.begin(), B.end(), comp);
    std::sort_heap(B.begin(), B.end(), comp);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<long unsigned, std::nano> nanotime2 = end - start;
    std::cout << "std::heap_sort               :"
              << (nanotime2.count() / 1000000) << std::endl;

    // hpx::parallel::partial sort
    B = A;
    start = std::chrono::high_resolution_clock::now();
    hpx::partial_sort(hpx::execution::par, B.begin(), B.end(), B.end(), comp);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<long unsigned, std::nano> nanotime4 = end - start;
    std::cout << "hpx::parallel::partial_sort  :"
              << (nanotime4.count() / 1000000) << std::endl;

    // std::partial_sort
    B = A;
    start = std::chrono::high_resolution_clock::now();
    std::partial_sort(B.begin(), B.end(), B.end(), comp);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<long unsigned, std::nano> nanotime5 = end - start;
    std::cout << "std::partial_sort            :"
              << (nanotime5.count() / 1000000) << std::endl;

    std::cout << "\n";
}

// partial_sort with different intervals of data to sort.
// accumulate all the times and compare the two algorithms with 1 thread
void function03()
{
    typedef std::less<std::uint64_t> compare_t;
#if defined(HPX_DEBUG)
    constexpr std::uint32_t NELEM = 10000;
#else
    constexpr std::uint32_t NELEM = 1000000;
#endif

    std::vector<std::uint64_t> A, B;
    A.reserve(NELEM);
    B.reserve(NELEM);

    for (std::uint64_t i = 0; i < NELEM; ++i)
    {
        A.emplace_back(i);
    }
    std::shuffle(A.begin(), A.end(), gen);

    std::uint64_t ac1 = 0, ac2 = 0;
    constexpr uint32_t STEP = NELEM / 100;

    for (std::uint64_t i = 0; i <= NELEM; i += STEP)
    {
        std::cout << "[" << i << "] \t";

        B = A;
        auto start = std::chrono::high_resolution_clock::now();
        hpx::partial_sort(hpx::execution::par, B.begin(), B.begin() + i,
            B.end(), compare_t());
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<long unsigned, std::nano> nanotime1 = end - start;
        ac1 += nanotime1.count();
        std::cout << "hpx::partial_sort :" << (nanotime1.count() / 1000000);

        B = A;
        start = std::chrono::high_resolution_clock::now();
        std::partial_sort(B.begin(), B.begin() + i, B.end(), compare_t());
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<long unsigned, std::nano> nanotime2 = end - start;
        ac2 += nanotime2.count();
        std::cout << "\t\tstd::partial_sort :" << (nanotime2.count() / 1000000)
                  << std::endl;
    }

    std::cout << "\n\n";
    std::cout << "Accumulated (msec) hpx::partial_sort " << ac1 / 1000000
              << std::endl;
    std::cout << "Accumulated (msec) std::partial_sort " << ac2 / 1000000
              << std::endl;
}

int test_main()
{
    std::cout
        << "------------ test_partial_sort_copy_ parallel --------------\n";
    function01();
    function02();
    function03();
    std::cout
        << "------------------------ end -------------------------------\n";
    return 0;
}

int hpx_main(hpx::program_options::variables_map& vm)
{
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    gen.seed(seed);

    test_main();

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

    return hpx::init(argc, argv, init_args);
}
