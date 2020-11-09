//  Copyright (c) 2015 Daniel Bourgeois
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_adjacent_difference.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy>
void test_adjacent_difference(ExPolicy policy)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    std::vector<std::size_t> c = test::random_iota(10007);
    std::vector<std::size_t> d(10007);
    std::vector<std::size_t> d_ans(10007);

    auto it = hpx::parallel::adjacent_difference(
        policy, std::begin(c), std::end(c), std::begin(d));
    std::adjacent_difference(std::begin(c), std::end(c), std::begin(d_ans));

    HPX_TEST(std::equal(std::begin(d), std::end(d), std::begin(d_ans),
        [](std::size_t lhs, std::size_t rhs) -> bool { return lhs == rhs; }));

    HPX_TEST(std::end(d) == it);
}

template <typename ExPolicy>
void test_adjacent_difference_async(ExPolicy p)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    std::vector<std::size_t> c = test::random_iota(10007);
    std::vector<std::size_t> d(10007);
    std::vector<std::size_t> d_ans(10007);

    auto f_it = hpx::parallel::adjacent_difference(
        p, std::begin(c), std::end(c), std::begin(d));
    std::adjacent_difference(std::begin(c), std::end(c), std::begin(d_ans));

    f_it.wait();
    HPX_TEST(std::equal(std::begin(d), std::end(d), std::begin(d_ans),
        [](std::size_t lhs, std::size_t rhs) -> bool { return lhs == rhs; }));

    HPX_TEST(std::end(d) == f_it.get());
}

void adjacent_difference_test()
{
    using namespace hpx::execution;
    test_adjacent_difference(seq);
    test_adjacent_difference(par);
    test_adjacent_difference(par_unseq);

    test_adjacent_difference_async(seq(task));
    test_adjacent_difference_async(par(task));
}

int hpx_main(hpx::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    adjacent_difference_test();
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
