//  Copyright (c) 2020 Francisco Jose Tapia (fjtapia@gmail.com )
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/execution.hpp>
#include <hpx/hpx.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/parallel/algorithms/partial_sort.hpp>

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

////////////////////////////////////////////////////////////////////////////
unsigned int seed = std::random_device{}();
std::mt19937 gen(seed);

void function01()
{
    typedef std::less<std::uint64_t> compare_t;
    constexpr std::uint32_t NELEM = 1000;

    std::vector<std::uint64_t> A, B;
    A.reserve(NELEM);
    B.reserve(NELEM);

    for (std::uint64_t i = 0; i < NELEM; ++i)
    {
        A.emplace_back(i);
    }
    std::shuffle(A.begin(), A.end(), gen);

    for (std::uint64_t i = 1; i <= NELEM; ++i)
    {
        B = A;
        hpx::partial_sort(hpx::execution::par, B.begin(), B.begin() + i,
            B.end(), compare_t());

        for (std::uint64_t j = 0; j < i; ++j)
        {
            HPX_TEST(B[j] == j);
        }
    }
}

// partial_sort with different intervals of data to sort.
// accumulate all the times and compare the two algorithms with 1 thread
void function02()
{
    typedef std::less<std::uint64_t> compare_t;
    constexpr std::uint32_t NELEM = 100000;

    std::vector<std::uint64_t> A, B;
    A.reserve(NELEM);
    B.reserve(NELEM);

    for (std::uint64_t i = 0; i < NELEM; ++i)
    {
        A.emplace_back(i);
    }
    std::shuffle(A.begin(), A.end(), gen);

    constexpr std::uint32_t STEP = NELEM / 100;
    for (std::uint64_t i = 0; i < NELEM; i += STEP)
    {
        B = A;
        hpx::partial_sort(hpx::execution::par, B.begin(), B.begin() + i,
            B.end(), compare_t());
        for (std::uint64_t k = 0; k < i; ++k)
        {
            HPX_TEST(B[k] == k);
        }
    }
}

int hpx_main(hpx::program_options::variables_map& vm)
{
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    gen.seed(seed);

    function01();
    function02();

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
