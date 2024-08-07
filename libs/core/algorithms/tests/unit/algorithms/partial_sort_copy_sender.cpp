//  Copyright (c) 2024 Tobias Wukovitsch
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/algorithm.hpp>
#include <hpx/assert.hpp>
#include <hpx/execution.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "test_utils.hpp"

#if defined(HPX_DEBUG)
constexpr std::uint64_t NELEM{111};
#else
constexpr std::uint64_t NELEM{1007};
#endif

////////////////////////////////////////////////////////////////////////////
unsigned int seed = std::random_device{}();
std::mt19937 gen(seed);

template <typename LnPolicy, typename ExPolicy, typename IteratorTag>
void test_partial_sort_copy_sender(
    LnPolicy ln_policy, ExPolicy&& ex_policy, IteratorTag)
{
    static_assert(hpx::is_async_execution_policy_v<ExPolicy>,
        "hpx::is_async_execution_policy_v<ExPolicy>");

    using base_iterator = std::vector<std::uint64_t>::iterator;
    using iterator = test::test_iterator<base_iterator, IteratorTag>;
    using compare_t = std::less<std::uint64_t>;

    namespace ex = hpx::execution::experimental;
    namespace tt = hpx::this_thread::experimental;
    using scheduler_t = ex::thread_pool_policy_scheduler<LnPolicy>;

    std::vector<std::uint64_t> l{9, 7, 6, 8, 5, 4, 1, 2, 3};
    std::uint64_t v1[20], v2[20];

    //------------------------------------------------------------------------
    // Output size is smaller than input size
    //------------------------------------------------------------------------
    for (int i = 0; i < 20; ++i)
        v1[i] = v2[i] = 999;

    auto exec = ex::explicit_scheduler_executor(scheduler_t(ln_policy));

    tt::sync_wait(ex::just(iterator(std::begin(l)), iterator(std::end(l)),
                      &v1[0], &v1[4]) |
        hpx::partial_sort_copy(ex_policy.on(exec)));

    std::partial_sort_copy(l.begin(), l.end(), &v2[0], &v2[4]);

    for (int i = 0; i < 4; ++i)
    {
        HPX_TEST(v1[i] == v2[i]);
    };
    for (int i = 4; i < 20; ++i)
    {
        HPX_TEST(v1[i] == v2[i]);
    };

    //------------------------------------------------------------------------
    // Output size is equal than input size
    //------------------------------------------------------------------------
    for (int i = 0; i < 20; ++i)
        v1[i] = v2[i] = 999;

    tt::sync_wait(ex::just(iterator(std::begin(l)), iterator(std::end(l)),
                      &v1[0], &v1[9]) |
        hpx::partial_sort_copy(ex_policy.on(exec)));

    std::partial_sort_copy(l.begin(), l.end(), &v2[0], &v2[9]);

    for (int i = 0; i < 9; ++i)
    {
        HPX_TEST(v1[i] == v2[i]);
    };
    for (int i = 9; i < 20; ++i)
    {
        HPX_TEST(v1[i] == v2[i]);
    };

    //------------------------------------------------------------------------
    // Output size is greater than input size
    //------------------------------------------------------------------------
    for (int i = 0; i < 20; ++i)
        v1[i] = v2[i] = 999;

    tt::sync_wait(ex::just(iterator(std::begin(l)), iterator(std::end(l)),
                      &v1[0], &v1[20]) |
        hpx::partial_sort_copy(ex_policy.on(exec)));

    std::partial_sort_copy(l.begin(), l.end(), &v2[0], &v2[20]);

    for (int i = 0; i < 9; ++i)
    {
        HPX_TEST(v1[i] == v2[i]);
    };
    for (int i = 9; i < 20; ++i)
    {
        HPX_TEST(v1[i] == v2[i]);
    };
}

template <typename IteratorTag>
void partial_sort_copy_sender_test()
{
    using namespace hpx::execution;
    test_partial_sort_copy_sender(hpx::launch::sync, seq(task), IteratorTag());
    test_partial_sort_copy_sender(
        hpx::launch::sync, unseq(task), IteratorTag());

    test_partial_sort_copy_sender(hpx::launch::async, par(task), IteratorTag());
    test_partial_sort_copy_sender(
        hpx::launch::async, par_unseq(task), IteratorTag());
}

int hpx_main(hpx::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    partial_sort_copy_sender_test<std::forward_iterator_tag>();
    partial_sort_copy_sender_test<std::random_access_iterator_tag>();

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
