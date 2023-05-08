//  Copyright (c) 2007-2022 Hartmut Kaiser
//  Copyright (c)      2021 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/async_base.hpp>
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
void test_reverse_direct(Policy l, ExPolicy&& policy, IteratorTag)
{
    static_assert(hpx::is_execution_policy_v<ExPolicy>,
        "hpx::is_execution_policy_v<ExPolicy>");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    namespace ex = hpx::execution::experimental;
    namespace tt = hpx::this_thread::experimental;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d1;

    std::iota(std::begin(c), std::end(c), std::rand());
    std::copy(std::begin(c), std::end(c), std::back_inserter(d1));

    using scheduler_t = ex::thread_pool_policy_scheduler<Policy>;

    auto exec = ex::explicit_scheduler_executor(scheduler_t(l));
    hpx::reverse(
        policy.on(exec), iterator(std::begin(c)), iterator(std::end(c)));

    std::reverse(std::begin(d1), std::end(d1));

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
void test_reverse(Policy l, ExPolicy&& policy, IteratorTag)
{
    static_assert(hpx::is_execution_policy_v<ExPolicy>,
        "hpx::is_execution_policy_v<ExPolicy>");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    namespace ex = hpx::execution::experimental;
    namespace tt = hpx::this_thread::experimental;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d1;

    std::iota(std::begin(c), std::end(c), std::rand());
    std::copy(std::begin(c), std::end(c), std::back_inserter(d1));

    using scheduler_t = ex::thread_pool_policy_scheduler<Policy>;

    auto exec = ex::explicit_scheduler_executor(scheduler_t(l));
    ex::just(iterator(std::begin(c)), iterator(std::end(c))) |
        hpx::reverse(policy.on(exec)) | tt::sync_wait();

    std::reverse(std::begin(d1), std::end(d1));

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
void test_reverse_async_direct(Policy l, ExPolicy&& p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    namespace ex = hpx::execution::experimental;
    namespace tt = hpx::this_thread::experimental;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d1;

    std::iota(std::begin(c), std::end(c), std::rand());
    std::copy(std::begin(c), std::end(c), std::back_inserter(d1));

    using scheduler_t = ex::thread_pool_policy_scheduler<Policy>;

    auto exec = ex::explicit_scheduler_executor(scheduler_t(l));
    hpx::reverse(p.on(exec), iterator(std::begin(c)), iterator(std::end(c))) |
        tt::sync_wait();

    std::reverse(std::begin(d1), std::end(d1));

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
void test_reverse_direct()
{
    using namespace hpx::execution;

    test_reverse_direct(hpx::launch::sync, seq, IteratorTag());
    test_reverse_direct(hpx::launch::sync, unseq, IteratorTag());

    test_reverse_direct(hpx::launch::async, par, IteratorTag());
    test_reverse_direct(hpx::launch::async, par_unseq, IteratorTag());

    test_reverse_async_direct(hpx::launch::sync, seq(task), IteratorTag());
    test_reverse_async_direct(hpx::launch::async, par(task), IteratorTag());
}

template <typename IteratorTag>
void test_reverse()
{
    using namespace hpx::execution;

    test_reverse(hpx::launch::sync, seq(task), IteratorTag());
    test_reverse(hpx::launch::sync, unseq(task), IteratorTag());

    test_reverse(hpx::launch::async, par(task), IteratorTag());
    test_reverse(hpx::launch::async, par_unseq(task), IteratorTag());
}

void reverse_test()
{
    test_reverse_direct<std::random_access_iterator_tag>();
    test_reverse_direct<std::bidirectional_iterator_tag>();

    test_reverse<std::random_access_iterator_tag>();
    test_reverse<std::bidirectional_iterator_tag>();
}

int hpx_main(hpx::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    reverse_test();

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
