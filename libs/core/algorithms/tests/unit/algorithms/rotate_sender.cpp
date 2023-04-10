//  Copyright (c) 2007-2022 Hartmut Kaiser
//  Copyright (c) 2021 Chuanqiu He
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

// Clang V11 ICE's on this test
#if !defined(HPX_CLANG_VERSION) || (HPX_CLANG_VERSION / 10000) != 11

#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/async_base.hpp>
#include <hpx/modules/iterator_support.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/thread.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename Policy, typename ExPolicy, typename IteratorTag>
void test_rotate_direct(Policy l, ExPolicy&& policy, IteratorTag)
{
    static_assert(hpx::is_execution_policy_v<ExPolicy>,
        "hpx::is_execution_policy<ExPolicy>::value");

    using base_iterator = std::vector<std::size_t>::iterator;
    using iterator = test::test_iterator<base_iterator, IteratorTag>;

    namespace ex = hpx::execution::experimental;
    namespace tt = hpx::this_thread::experimental;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d1;

    std::iota(std::begin(c), std::end(c), std::rand());
    std::copy(std::begin(c), std::end(c), std::back_inserter(d1));

    std::size_t mid_pos = std::rand() % c.size();    //-V104
    base_iterator mid = std::begin(c);
    std::advance(mid, mid_pos);

    using scheduler_t = ex::thread_pool_policy_scheduler<Policy>;

    auto exec = ex::explicit_scheduler_executor(scheduler_t(l));
    hpx::rotate(policy.on(exec), iterator(std::begin(c)), iterator(mid),
        iterator(std::end(c)));

    base_iterator mid1 = std::begin(d1);
    std::advance(mid1, mid_pos);
    std::rotate(std::begin(d1), mid1, std::end(d1));

    std::size_t count = 0;
    HPX_TEST(std::equal(std::begin(c), std::end(c), std::begin(d1),
        [&count](std::size_t v1, std::size_t v2) -> bool {
            HPX_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    HPX_TEST_EQ(count, d1.size());
}

template <typename Policy, typename ExPolicy, typename IteratorTag>
void test_rotate(Policy l, ExPolicy&& policy, IteratorTag)
{
    static_assert(hpx::is_execution_policy_v<ExPolicy>,
        "hpx::is_execution_policy<ExPolicy>::value");

    using base_iterator = std::vector<std::size_t>::iterator;
    using iterator = test::test_iterator<base_iterator, IteratorTag>;

    namespace ex = hpx::execution::experimental;
    namespace tt = hpx::this_thread::experimental;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d1;

    std::iota(std::begin(c), std::end(c), std::rand());
    std::copy(std::begin(c), std::end(c), std::back_inserter(d1));

    std::size_t mid_pos = std::rand() % c.size();    //-V104
    base_iterator mid = std::begin(c);
    std::advance(mid, mid_pos);

    using scheduler_t = ex::thread_pool_policy_scheduler<Policy>;

    auto exec = ex::explicit_scheduler_executor(scheduler_t(l));
    ex::just(iterator(std::begin(c)), iterator(mid), iterator(std::end(c))) |
        hpx::rotate(policy.on(exec)) | tt::sync_wait();

    base_iterator mid1 = std::begin(d1);
    std::advance(mid1, mid_pos);
    std::rotate(std::begin(d1), mid1, std::end(d1));

    std::size_t count = 0;
    HPX_TEST(std::equal(std::begin(c), std::end(c), std::begin(d1),
        [&count](std::size_t v1, std::size_t v2) -> bool {
            HPX_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    HPX_TEST_EQ(count, d1.size());
}

template <typename Policy, typename ExPolicy, typename IteratorTag>
void test_rotate_async_direct(Policy l, ExPolicy&& p, IteratorTag)
{
    using base_iterator = std::vector<std::size_t>::iterator;
    using iterator = test::test_iterator<base_iterator, IteratorTag>;

    namespace ex = hpx::execution::experimental;
    namespace tt = hpx::this_thread::experimental;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d1;

    std::iota(std::begin(c), std::end(c), std::rand());
    std::copy(std::begin(c), std::end(c), std::back_inserter(d1));

    std::size_t mid_pos = std::rand() % c.size();    //-V104

    base_iterator mid = std::begin(c);
    std::advance(mid, mid_pos);

    using scheduler_t = ex::thread_pool_policy_scheduler<Policy>;

    auto exec = ex::explicit_scheduler_executor(scheduler_t(l));
    hpx::rotate(p.on(exec), iterator(std::begin(c)), iterator(mid),
        iterator(std::end(c))) |
        tt::sync_wait();

    base_iterator mid1 = std::begin(d1);
    std::advance(mid1, mid_pos);
    std::rotate(std::begin(d1), mid1, std::end(d1));

    std::size_t count = 0;
    HPX_TEST(std::equal(std::begin(c), std::end(c), std::begin(d1),
        [&count](std::size_t v1, std::size_t v2) -> bool {
            HPX_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    HPX_TEST_EQ(count, d1.size());
}

template <typename IteratorTag>
void test_rotate_direct()
{
    using namespace hpx::execution;

    test_rotate_direct(hpx::launch::sync, seq, IteratorTag());
    test_rotate_direct(hpx::launch::sync, unseq, IteratorTag());

    test_rotate_direct(hpx::launch::async, par, IteratorTag());
    test_rotate_direct(hpx::launch::async, par_unseq, IteratorTag());

    test_rotate_async_direct(hpx::launch::sync, seq(task), IteratorTag());
    test_rotate_async_direct(hpx::launch::async, par(task), IteratorTag());
}

template <typename IteratorTag>
void test_rotate()
{
    using namespace hpx::execution;

    test_rotate(hpx::launch::sync, seq(task), IteratorTag());
    test_rotate(hpx::launch::sync, unseq(task), IteratorTag());

    test_rotate(hpx::launch::async, par(task), IteratorTag());
    test_rotate(hpx::launch::async, par_unseq(task), IteratorTag());
}

void rotate_test()
{
    test_rotate_direct<std::random_access_iterator_tag>();
    test_rotate_direct<std::forward_iterator_tag>();

    test_rotate<std::random_access_iterator_tag>();
    test_rotate<std::forward_iterator_tag>();
}

int hpx_main(hpx::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    rotate_test();

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

#else

int main(int, char*[])
{
    return 0;
}

#endif
