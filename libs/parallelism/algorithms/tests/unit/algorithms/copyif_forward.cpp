//  Copyright (c) 2014 Grant Mercer
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_copy.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
unsigned int seed = std::random_device{}();
std::mt19937 gen(seed);
std::uniform_int_distribution<> dis(0, (std::numeric_limits<int>::max)());

void test_copy_if_seq()
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, std::forward_iterator_tag>
        iterator;

    std::vector<int> c(10007);
    std::vector<int> d(c.size());
    auto middle = std::begin(c) + c.size() / 2;
    std::iota(std::begin(c), middle, dis(gen));
    std::fill(middle, std::end(c), -1);

    hpx::copy_if(iterator(std::begin(c)), iterator(std::end(c)), std::begin(d),
        [](int i) { return !(i < 0); });

    std::size_t count = 0;
    HPX_TEST(std::equal(
        std::begin(c), middle, std::begin(d), [&count](int v1, int v2) -> bool {
            HPX_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));

    HPX_TEST(std::equal(middle, std::end(c), std::begin(d) + d.size() / 2,
        [&count](int v1, int v2) -> bool {
            HPX_TEST_NEQ(v1, v2);
            ++count;
            return v1 != v2;
        }));

    HPX_TEST_EQ(count, d.size());
}

template <typename ExPolicy>
void test_copy_if(ExPolicy&& policy)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, std::forward_iterator_tag>
        iterator;

    std::vector<int> c(10007);
    std::vector<int> d(c.size());
    auto middle = std::begin(c) + c.size() / 2;
    std::iota(std::begin(c), middle, dis(gen));
    std::fill(middle, std::end(c), -1);

    hpx::copy_if(policy, iterator(std::begin(c)), iterator(std::end(c)),
        std::begin(d), [](int i) { return !(i < 0); });

    std::size_t count = 0;
    HPX_TEST(std::equal(
        std::begin(c), middle, std::begin(d), [&count](int v1, int v2) -> bool {
            HPX_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));

    HPX_TEST(std::equal(middle, std::end(c), std::begin(d) + d.size() / 2,
        [&count](int v1, int v2) -> bool {
            HPX_TEST_NEQ(v1, v2);
            ++count;
            return v1 != v2;
        }));

    HPX_TEST_EQ(count, d.size());
}

template <typename ExPolicy>
void test_copy_if_async(ExPolicy&& p)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, std::forward_iterator_tag>
        iterator;

    std::vector<int> c(10007);
    std::vector<int> d(c.size());
    auto middle = std::begin(c) + c.size() / 2;
    std::iota(std::begin(c), middle, dis(gen));
    std::fill(middle, std::end(c), -1);

    auto f = hpx::copy_if(p, iterator(std::begin(c)), iterator(std::end(c)),
        std::begin(d), [](int i) { return !(i < 0); });
    f.wait();

    std::size_t count = 0;
    HPX_TEST(std::equal(
        std::begin(c), middle, std::begin(d), [&count](int v1, int v2) -> bool {
            HPX_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));

    HPX_TEST(std::equal(middle, std::end(c), std::begin(d) + d.size() / 2,
        [&count](int v1, int v2) -> bool {
            HPX_TEST_NEQ(v1, v2);
            ++count;
            return v1 != v2;
        }));

    HPX_TEST_EQ(count, d.size());
}

void test_copy_if()
{
    using namespace hpx::parallel;

    test_copy_if_seq();

    test_copy_if(execution::seq);
    test_copy_if(execution::par);
    test_copy_if(execution::par_unseq);

    test_copy_if_async(execution::seq(execution::task));
    test_copy_if_async(execution::par(execution::task));
}

int hpx_main(hpx::program_options::variables_map& vm)
{
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    gen.seed(seed);

    test_copy_if();
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
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
