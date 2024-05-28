//  Copyright (c) 2024 Tobias Wukovitsch
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/algorithm.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/parallel/algorithms/sort_by_key.hpp>

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>
//
#if defined(HPX_DEBUG)
#define HPX_SORT_BY_KEY_TEST_SIZE (1 << 8)
#else
#define HPX_SORT_BY_KEY_TEST_SIZE (1 << 18)
#endif
//
#include "sort_tests.hpp"
//
#define EXTRA_DEBUG
//
namespace debug {
    template <typename T>
    void output(const std::string& name, const std::vector<T>& v)
    {
#ifdef EXTRA_DEBUG7
        std::cout << name.c_str() << "\t : {" << v.size() << "} : ";
        std::copy(std::begin(v), std::end(v),
            std::ostream_iterator<T>(std::cout, ", "));
        std::cout << "\n";
#else
        HPX_UNUSED(name);
        HPX_UNUSED(v);
#endif
    }

    template <typename Iter>
    void output(const std::string& name, Iter begin, Iter end)
    {
#ifdef EXTRA_DEBUG
        std::cout << name.c_str() << "\t : {" << std::distance(begin, end)
                  << "} : ";
        std::copy(begin, end,
            std::ostream_iterator<
                typename std::iterator_traits<Iter>::value_type>(
                std::cout, ", "));
        std::cout << "\n";
#else
        HPX_UNUSED(name);
        HPX_UNUSED(begin);
        HPX_UNUSED(end);
#endif
    }

#if defined(EXTRA_DEBUG)
#define debug_msg(a) std::cout << a
#else
#define debug_msg(a)
#endif
}    // namespace debug

#undef msg
#define msg(a, b, d)                                      \
    std::cout << std::setw(60) << a << std::setw(12) << b \
              << std::setw(8) << #d << "\t";

////////////////////////////////////////////////////////////////////////////////

template <typename LnPolicy, typename ExPolicy>
void test_sort_by_key_sender(LnPolicy ln_policy, ExPolicy&& ex_policy)
{
    using Tkey = std::size_t;
    using Tval = std::size_t;

    static_assert(hpx::is_async_execution_policy_v<ExPolicy>,
        "hpx::is_async_execution_policy_v<ExPolicy>");
    msg(typeid(ExPolicy).name(), typeid(Tval).name(), sync);
    std::cout << "\n";

    namespace ex = hpx::execution::experimental;
    namespace tt = hpx::this_thread::experimental;
    using scheduler_t = ex::thread_pool_policy_scheduler<LnPolicy>;

    // vector of values, and keys
    std::vector<Tval> values, o_values;
    std::vector<Tkey> keys, o_keys;
    //
    values.assign(HPX_SORT_BY_KEY_TEST_SIZE, 0);
    keys.assign(HPX_SORT_BY_KEY_TEST_SIZE, 0);

    // generate a sequence as the values
    std::iota(values.begin(), values.end(), 0);
    // generate a sequence as the keys
    std::iota(keys.begin(), keys.end(), 0);

    // shuffle the keys up,
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(keys.begin(), keys.end(), g);

    // make copies of initial states
    o_keys = keys;
    o_values = values;

    auto exec = ex::explicit_scheduler_executor(scheduler_t(ln_policy));

    tt::sync_wait(
        ex::just(keys.begin(), keys.end(), values.begin())
        | hpx::experimental::sort_by_key(ex_policy.on(exec))
    );

    // after sorting by key, the values should be equal to the original keys
    bool is_equal = std::equal(keys.begin(), keys.end(), o_values.begin());
    if (is_equal)
    {
        //std::cout << "Test Passed\n";
    }
    else
    {
        debug::output("keys     ", o_keys);
        debug::output("values   ", o_values);
        debug::output("key range", keys);
        debug::output("val range", values);
        throw std::string("Problem");
    }
    HPX_TEST(is_equal);
}

void sort_by_key_sender_test()
{
    using namespace hpx::execution;
    test_sort_by_key_sender(hpx::launch::sync, seq(task));
    test_sort_by_key_sender(hpx::launch::sync, unseq(task));

    test_sort_by_key_sender(hpx::launch::async, par(task));
    test_sort_by_key_sender(hpx::launch::async, par_unseq(task));
}

int hpx_main(hpx::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    sort_by_key_sender_test();

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
