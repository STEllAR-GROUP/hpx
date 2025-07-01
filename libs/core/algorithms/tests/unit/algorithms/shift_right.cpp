//  Copyright (c) 2014 Hartmut Kaiser
//  Copyright (c) 2021 Akhil J Nair
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/algorithm.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>
#include <cstddef>
#include <forward_list>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

#define ARR_SIZE 100007

////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_shift_right_nonbidir(IteratorTag)
{
    std::forward_list<std::size_t> c(ARR_SIZE);
    std::iota(std::begin(c), std::end(c), std::rand());
    std::vector<std::size_t> d;

    for (auto elem : c)
    {
        d.push_back(elem);
    }

    // shift by zero should have no effect
    hpx::shift_right(std::begin(c), std::end(c), 0);
    HPX_TEST(std::equal(std::begin(c), std::end(c), std::begin(d)));

    // shift by a negative number should have no effect
    hpx::shift_right(std::begin(c), std::end(c), -4);
    HPX_TEST(std::equal(std::begin(c), std::end(c), std::begin(d)));

    std::size_t n = (std::rand() % (std::size_t) ARR_SIZE) + 1;
    hpx::shift_right(std::begin(c), std::end(c), n);

    std::move_backward(std::begin(d),
        std::end(d) - static_cast<std::ptrdiff_t>(n), std::end(d));

    // verify values
    HPX_TEST(std::equal(std::next(std::begin(c), n), std::end(c),
        std::begin(d) + static_cast<std::ptrdiff_t>(n)));

    // ensure shift by more than n does not crash
    hpx::shift_right(std::begin(c), std::end(c), (std::size_t) (ARR_SIZE + 1));
}

template <typename IteratorTag>
void test_shift_right(IteratorTag)
{
    std::vector<std::size_t> c(ARR_SIZE);
    std::iota(std::begin(c), std::end(c), std::rand());
    std::vector<std::size_t> d = c;

    // shift by zero should have no effect
    hpx::shift_right(std::begin(c), std::end(c), 0);
    HPX_TEST(std::equal(std::begin(c), std::end(c), std::begin(d)));

    // shift by a negative number should have no effect
    hpx::shift_right(std::begin(c), std::end(c), -4);
    HPX_TEST(std::equal(std::begin(c), std::end(c), std::begin(d)));

    std::size_t n = (std::rand() % (std::size_t) ARR_SIZE) + 1;
    hpx::shift_right(std::begin(c), std::end(c), n);

    std::move_backward(std::begin(d),
        std::end(d) - static_cast<std::ptrdiff_t>(n), std::end(d));

    // verify values
    HPX_TEST(std::equal(std::begin(c) + static_cast<std::ptrdiff_t>(n),
        std::end(c), std::begin(d) + static_cast<std::ptrdiff_t>(n)));

    // ensure shift by more than n does not crash
    hpx::shift_right(std::begin(c), std::end(c), (std::size_t) (ARR_SIZE + 1));
}

template <typename ExPolicy, typename IteratorTag>
void test_shift_right(ExPolicy policy, IteratorTag)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    std::vector<std::size_t> c(ARR_SIZE);
    std::iota(std::begin(c), std::end(c), std::rand());
    std::vector<std::size_t> d = c;

    // shift by zero should have no effect
    hpx::shift_right(policy, std::begin(c), std::end(c), 0);
    HPX_TEST(std::equal(std::begin(c), std::end(c), std::begin(d)));

    // shift by a negative number should have no effect
    hpx::shift_right(policy, std::begin(c), std::end(c), -4);
    HPX_TEST(std::equal(std::begin(c), std::end(c), std::begin(d)));

    std::size_t n = (std::rand() % (std::size_t) ARR_SIZE) + 1;
    hpx::shift_right(policy, std::begin(c), std::end(c), n);

    std::move_backward(std::begin(d),
        std::end(d) - static_cast<std::ptrdiff_t>(n), std::end(d));

    // verify values
    HPX_TEST(std::equal(std::begin(c) + static_cast<std::ptrdiff_t>(n),
        std::end(c), std::begin(d) + static_cast<std::ptrdiff_t>(n)));

    // ensure shift by more than n does not crash
    hpx::shift_right(
        policy, std::begin(c), std::end(c), (std::size_t) (ARR_SIZE + 1));
}

template <typename ExPolicy, typename IteratorTag>
void test_shift_right_async(ExPolicy p, IteratorTag)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    std::vector<std::size_t> c(ARR_SIZE);
    std::iota(std::begin(c), std::end(c), std::rand());
    std::vector<std::size_t> d = c;

    // shift by zero should have no effect
    auto f = hpx::shift_right(p, std::begin(c), std::end(c), 0);
    f.wait();
    HPX_TEST(std::equal(std::begin(c), std::end(c), std::begin(d)));

    // shift by a negative number should have no effect
    auto f1 = hpx::shift_right(p, std::begin(c), std::end(c), -4);
    f1.wait();
    HPX_TEST(std::equal(std::begin(c), std::end(c), std::begin(d)));

    std::size_t n = (std::rand() % (std::size_t) ARR_SIZE) + 1;
    auto f2 = hpx::shift_right(p, std::begin(c), std::end(c), n);
    f2.wait();

    std::move_backward(std::begin(d),
        std::end(d) - static_cast<std::ptrdiff_t>(n), std::end(d));

    // verify values
    HPX_TEST(std::equal(std::begin(c) + static_cast<std::ptrdiff_t>(n),
        std::end(c), std::begin(d) + static_cast<std::ptrdiff_t>(n)));

    // ensure shift by more than n does not crash
    auto f3 = hpx::shift_right(
        p, std::begin(c), std::end(c), (std::size_t) (ARR_SIZE + 1));
    f3.wait();
}

template <typename IteratorTag>
void test_shift_right()
{
    using namespace hpx::execution;
    test_shift_right_nonbidir(IteratorTag());
    test_shift_right(IteratorTag());
    test_shift_right(seq, IteratorTag());
    test_shift_right(par, IteratorTag());
    test_shift_right(par_unseq, IteratorTag());

    test_shift_right_async(seq(task), IteratorTag());
    test_shift_right_async(par(task), IteratorTag());
}

void shift_right_test()
{
    test_shift_right<std::random_access_iterator_tag>();
    test_shift_right<std::forward_iterator_tag>();
}

int hpx_main(hpx::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    shift_right_test();
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
