//  Copyright (c) 2014 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/local/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/parallel/algorithms/shift_left.hpp>

#include <atomic>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_shift_left(IteratorTag)
{
    using base_iterator = std::vector<std::size_t>::iterator;
    using iterator = test::test_iterator<base_iterator, IteratorTag>;

    std::vector<std::size_t> c(10);
    std::iota(std::begin(c), std::end(c), std::rand());
    std::vector<std::size_t> d = c;

    auto n = std::rand() % 10;
    hpx::shift_left(iterator(std::begin(c)), iterator(std::end(c)), n);

    std::move(std::begin(d) + n, std::end(d), std::begin(d));

    // verify values
    HPX_TEST(std::equal(std::begin(c), std::begin(c) + n, std::begin(d)));
}

template <typename ExPolicy, typename IteratorTag>
void test_shift_left(ExPolicy policy, IteratorTag)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    using base_iterator = std::vector<std::size_t>::iterator;
    using iterator = test::test_iterator<base_iterator, IteratorTag>;

    std::vector<std::size_t> c(10);
    std::iota(std::begin(c), std::end(c), std::rand());
    std::vector<std::size_t> d = c;

    auto n = std::rand() % 10;
    hpx::shift_left(policy, iterator(std::begin(c)), iterator(std::end(c)), n);

    std::move(std::begin(d) + n, std::end(d), std::begin(d));

    // verify values
    HPX_TEST(std::equal(std::begin(c), std::begin(c) + n, std::begin(d)));
}
/*
template <typename ExPolicy, typename IteratorTag>
void test_shift_left_async(ExPolicy p, IteratorTag)
{
    using base_iterator = std::vector<std::size_t>::iterator;
    using iterator = test::test_iterator<base_iterator, IteratorTag>;

    std::vector<std::size_t> c(10);
    std::iota(std::begin(c), std::end(c), std::rand());
    std::vector<std::size_t> d = c;

    auto n = std::rand() % 10;
    auto f =
        hpx::shift_left(p, iterator(std::begin(c)), iterator(std::end(c)), n);
    f.wait();

    std::move(std::begin(d) + n, std::end(d), std::begin(d));

    // verify values
    HPX_TEST(std::equal(std::begin(c), std::begin(c) + n, std::begin(d)));
}
*/
template <typename IteratorTag>
void test_shift_left()
{
    using namespace hpx::execution;
    test_shift_left(IteratorTag());
    test_shift_left(seq, IteratorTag());
    test_shift_left(par, IteratorTag());
    //test_shift_left(par_unseq, IteratorTag());

    //test_shift_left_async(seq(task), IteratorTag());
    //test_shift_left_async(par(task), IteratorTag());
}

void shift_left_test()
{
    test_shift_left<std::random_access_iterator_tag>();
    test_shift_left<std::forward_iterator_tag>();
}

int hpx_main(hpx::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    shift_left_test();
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
