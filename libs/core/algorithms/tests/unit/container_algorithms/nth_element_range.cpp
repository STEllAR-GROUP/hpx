//  Copyright (c) 2018 Christopher Ogle
//  Copyright (c) 2020 Hartmut Kaiser
//  Copyright (c) 2021 Akhil J Nair
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/iterator_support/tests/iter_sent.hpp>
#include <hpx/local/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/parallel/container_algorithms/nth_element.hpp>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
#define SIZE 10007

template <typename IteratorTag>
void test_nth_element_sent(IteratorTag)
{
    using base_iterator = std::vector<std::size_t>::iterator;
    using iterator = test::test_iterator<base_iterator, IteratorTag>;
    using sentinel = test::sentinel_from_iterator<iterator>;

    // generate vector of unique random integers
    std::unordered_set<std::size_t> uset;
    while (uset.size() != SIZE)
    {
        uset.insert(std::rand());
    }

    std::vector<std::size_t> c(SIZE);
    c.insert(c.end(), uset.begin(), uset.end());
    std::vector<std::size_t> d = c;

    auto rand_index = std::rand() % SIZE;

    hpx::ranges::nth_element(iterator(std::begin(c)),
        iterator(std::begin(c) + rand_index),
        sentinel{std::end(c) - 1});

    std::nth_element(std::begin(d), std::begin(d) + rand_index, std::end(d) - 1);

    HPX_TEST(*(std::begin(c) + rand_index) == *(std::begin(d) + rand_index));
}

template <typename ExPolicy, typename IteratorTag>
void test_nth_element_sent(ExPolicy policy, IteratorTag)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    // ensure all characters are unique
    std::unordered_set<unsigned char> uset;

    std::vector<char> c1(7);
    std::generate(std::begin(c1), std::end(c1), [&uset]() {
        unsigned char c = 'a' + dis(gen);
        while (uset.find(c) != uset.end())
        {
            c = 'a' + dis(gen);
        }
        uset.insert(c);
        return c;
    });

    uset.clear();
    std::vector<char> c2(7);
    std::generate(std::begin(c2), std::end(c2), [&uset]() {
        unsigned char c = 'a' + dis(gen);
        while (uset.find(c) != uset.end())
        {
            c = 'a' + dis(gen);
        }
        uset.insert(c);
        return c;
    });

    bool actual_result1 = std::nth_element(
        std::begin(c1), std::begin(c1) + 5, std::begin(c2), std::begin(c2) + 5);
    bool result1 = hpx::ranges::nth_element(policy, std::begin(c1),
        sentinel<char>{*(std::begin(c1) + 5)}, std::begin(c2),
        sentinel<char>{*(std::begin(c2) + 5)});

    bool actual_result2 = std::nth_element(
        std::begin(c2), std::begin(c2) + 5, std::begin(c1), std::begin(c1) + 5);
    bool result2 = hpx::ranges::nth_element(policy, std::begin(c2),
        sentinel<char>{*(std::begin(c2) + 5)}, std::begin(c1),
        sentinel<char>{*(std::begin(c1) + 5)});

    bool actual_result3 = std::nth_element(
        std::begin(c1), std::begin(c1) + 5, std::begin(c1), std::begin(c1) + 5);
    bool result3 = hpx::ranges::nth_element(policy, std::begin(c1),
        sentinel<char>{*(std::begin(c1) + 5)}, std::begin(c1),
        sentinel<char>{*(std::begin(c1) + 5)});

    HPX_TEST_EQ(actual_result1, result1);
    HPX_TEST_EQ(actual_result2, result2);
    HPX_TEST_EQ(actual_result3, result3);

    // check corner cases
    std::vector<char> c3 = {1, 1, 1, 1, 3, 2, 2, 8};
    std::vector<char> c4 = {1, 1, 1, 1, 3, 5, 5, 8};
    auto result4 = hpx::ranges::nth_element(policy, std::begin(c3),
        sentinel<char>{3}, std::begin(c4), sentinel<char>{3});
    auto result5 = hpx::ranges::nth_element(policy, std::begin(c3),
        sentinel<char>{8}, std::begin(c4), sentinel<char>{8});

    HPX_TEST_EQ(false, result4);
    HPX_TEST_EQ(true, result5);
}

template <typename IteratorTag>
void test_nth_element(IteratorTag)
{
    std::vector<char> c1(10);
    std::generate(
        std::begin(c1), std::end(c1), []() { return 'a' + dis(gen); });

    std::vector<char> c2(10);
    std::generate(
        std::begin(c2), std::end(c2), []() { return 'a' + dis(gen); });

    bool actual_result1 = std::nth_element(
        std::begin(c1), std::end(c1), std::begin(c2), std::end(c2));
    bool result1 = hpx::ranges::nth_element(c1, c2);

    bool actual_result2 = std::nth_element(
        std::begin(c1), std::end(c1), std::begin(c2), std::end(c2));
    bool result2 = hpx::ranges::nth_element(c1, c2);

    bool actual_result3 = std::nth_element(
        std::begin(c1), std::end(c1), std::begin(c2), std::end(c2));
    bool result3 = hpx::ranges::nth_element(c1, c2);

    HPX_TEST_EQ(actual_result1, result1);
    HPX_TEST_EQ(actual_result2, result2);
    HPX_TEST_EQ(actual_result3, result3);
}

template <typename ExPolicy, typename IteratorTag>
void test_nth_element(ExPolicy policy, IteratorTag)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    std::vector<char> c1(10);
    std::generate(
        std::begin(c1), std::end(c1), []() { return 'a' + dis(gen); });

    std::vector<char> c2(10);
    std::generate(
        std::begin(c2), std::end(c2), []() { return 'a' + dis(gen); });

    bool actual_result1 = std::nth_element(
        std::begin(c1), std::end(c1), std::begin(c2), std::end(c2));
    bool result1 = hpx::ranges::nth_element(policy, c1, c2);

    bool actual_result2 = std::nth_element(
        std::begin(c1), std::end(c1), std::begin(c2), std::end(c2));
    bool result2 = hpx::ranges::nth_element(policy, c1, c2);

    bool actual_result3 = std::nth_element(
        std::begin(c1), std::end(c1), std::begin(c2), std::end(c2));
    bool result3 = hpx::ranges::nth_element(policy, c1, c2);

    HPX_TEST_EQ(actual_result1, result1);
    HPX_TEST_EQ(actual_result2, result2);
    HPX_TEST_EQ(actual_result3, result3);
}

template <typename ExPolicy, typename IteratorTag>
void test_nth_element_async(ExPolicy policy, IteratorTag)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    std::vector<char> c1(10);
    std::generate(
        std::begin(c1), std::end(c1), []() { return 'a' + dis(gen); });

    std::vector<char> c2(10);
    std::generate(
        std::begin(c2), std::end(c2), []() { return 'a' + dis(gen); });

    bool actual_result1 = std::nth_element(
        std::begin(c1), std::end(c1), std::begin(c2), std::end(c2));
    hpx::future<bool> result1 =
        hpx::ranges::nth_element(policy, c1, c2);

    bool actual_result2 = std::nth_element(
        std::begin(c1), std::end(c1), std::begin(c2), std::end(c2));
    hpx::future<bool> result2 =
        hpx::ranges::nth_element(policy, c1, c2);

    bool actual_result3 = std::nth_element(
        std::begin(c1), std::end(c1), std::begin(c2), std::end(c2));
    hpx::future<bool> result3 =
        hpx::ranges::nth_element(policy, c1, c2);

    result1.wait();
    result2.wait();
    result3.wait();

    HPX_TEST_EQ(actual_result1, result1.get());
    HPX_TEST_EQ(actual_result2, result2.get());
    HPX_TEST_EQ(actual_result3, result3.get());
}

template <typename IteratorTag>
void test_nth_element()
{
    using namespace hpx::execution;

    /*test_nth_element(IteratorTag());
    test_nth_element(seq, IteratorTag());
    test_nth_element(par, IteratorTag());
    test_nth_element(par_unseq, IteratorTag());

    test_nth_element_async(seq(task), IteratorTag());
    test_nth_element_async(par(task), IteratorTag());*/

    test_nth_element_sent(IteratorTag());
    /*test_nth_element_sent(seq);
    test_nth_element_sent(par);
    test_nth_element_sent(par_unseq);*/
}

void nth_element_test()
{
    test_nth_element<std::random_access_iterator_tag>();
    test_nth_element<std::forward_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    nth_element_test();
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
