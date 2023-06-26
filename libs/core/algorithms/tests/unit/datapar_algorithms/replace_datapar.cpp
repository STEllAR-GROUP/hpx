//  Copyright (c) 2022 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/algorithm.hpp>
#include <hpx/datapar.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "../algorithms/test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_replace(ExPolicy policy, IteratorTag)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    std::vector<int> d(c.size());
    std::iota(std::begin(c), std::end(c), std::rand());
    std::copy(std::begin(c), std::end(c), std::begin(d));

    std::size_t idx = std::rand() % c.size();    //-V104

    hpx::replace(policy, iterator(std::begin(c)), iterator(std::end(c)), c[idx],
        c[idx] + 1);

    std::replace(std::begin(d), std::end(d), d[idx], d[idx] + 1);

    std::size_t count = 0;
    HPX_TEST(std::equal(std::begin(c), std::end(c), std::begin(d),
        [&count](auto v1, auto v2) -> bool {
            HPX_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    HPX_TEST_EQ(count, d.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_replace_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    std::vector<int> d(c.size());
    std::iota(std::begin(c), std::end(c), std::rand());
    std::copy(std::begin(c), std::end(c), std::begin(d));

    std::size_t idx = std::rand() % c.size();    //-V104

    hpx::future<void> f = hpx::replace(
        p, iterator(std::begin(c)), iterator(std::end(c)), c[idx], c[idx] + 1);
    f.wait();

    std::replace(std::begin(d), std::end(d), d[idx], d[idx] + 1);

    std::size_t count = 0;
    HPX_TEST(std::equal(std::begin(c), std::end(c), std::begin(d),
        [&count](auto v1, auto v2) -> bool {
            HPX_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    HPX_TEST_EQ(count, d.size());
}

template <typename IteratorTag>
void test_replace()
{
    using namespace hpx::execution;
    test_replace(simd, IteratorTag());
    test_replace(par_simd, IteratorTag());

    test_replace_async(simd(task), IteratorTag());
    test_replace_async(par_simd(task), IteratorTag());
}

void replace_test()
{
    test_replace<std::random_access_iterator_tag>();
    test_replace<std::forward_iterator_tag>();
}

int hpx_main(hpx::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    replace_test();
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
