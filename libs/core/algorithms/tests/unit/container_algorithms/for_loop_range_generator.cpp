//  Copyright (c) 2016-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

// clang up to V12 (Apple clang up to v15) and gcc above V13 refuse to compile
// the code below
#if defined(HPX_HAVE_CXX20_COROUTINES) &&                                      \
    (!defined(HPX_CLANG_VERSION) || HPX_CLANG_VERSION >= 130000) &&            \
    (!defined(HPX_GCC_VERSION) || HPX_GCC_VERSION < 140000) &&                 \
    (!defined(HPX_APPLE_CLANG_VERSION) || HPX_APPLE_CLANG_VERSION >= 160000)

#include <hpx/algorithm.hpp>
#include <hpx/generator.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
unsigned int seed = std::random_device{}();
std::mt19937 gen(seed);

///////////////////////////////////////////////////////////////////////////////
namespace test {

    struct index_pair
    {
        int first;
        int last;
        int stride = 1;

        [[nodiscard]] constexpr bool operator==(
            index_pair const& rhs) const noexcept
        {
            return first == rhs.first && last == rhs.last &&
                stride == rhs.stride;
        }
    };

    hpx::generator<int&, int> iterate(index_pair const p) noexcept
    {
        auto const lo = p.first;
        auto const hi = p.last;

        if (auto const stride = p.stride; stride > 0)
        {
            for (auto i = lo; i < hi; i += stride)
                co_yield i;
        }
        else if (stride < 0)
        {
            for (auto i = hi - 1; i >= lo; i += stride)
                co_yield i;
        }
    }

    bool empty(index_pair const& p)
    {
        return p.first == p.last;
    }

    auto size(index_pair const& p)
    {
        return p.last - p.first;
    }

    auto distance(index_pair const& p1, index_pair const& p2)
    {
        return p1.first - p2.first;
    }

    index_pair subrange(
        index_pair const& p, std::ptrdiff_t const first, std::size_t const size)
    {
        return {static_cast<int>(p.first + first),
            static_cast<int>(p.first + first + size), p.stride};
    }

    static_assert(hpx::traits::is_range_generator_v<index_pair>);
}    // namespace test

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy>
void test_for_loop(ExPolicy&& policy)
{
    static_assert(hpx::is_execution_policy_v<ExPolicy>,
        "hpx::is_execution_policy_v<ExPolicy>");

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), gen());

    hpx::ranges::experimental::for_loop(std::forward<ExPolicy>(policy),
        test::index_pair{0, static_cast<int>(c.size())},
        [&](int const i) { c[i] = 42; });

    // verify values
    std::size_t count = 0;
    std::for_each(
        std::begin(c), std::end(c), [&count](std::size_t const v) -> void {
            HPX_TEST_EQ(v, static_cast<std::size_t>(42));
            ++count;
        });
    HPX_TEST_EQ(count, c.size());
}

template <typename ExPolicy>
void test_for_loop_async(ExPolicy&& p)
{
    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), gen());

    auto f = hpx::ranges::experimental::for_loop(std::forward<ExPolicy>(p),
        test::index_pair{0, static_cast<int>(c.size())},
        [&](int const i) { c[i] = 42; });
    f.wait();

    // verify values
    std::size_t count = 0;
    std::for_each(
        std::begin(c), std::end(c), [&count](std::size_t const v) -> void {
            HPX_TEST_EQ(v, static_cast<std::size_t>(42));
            ++count;
        });
    HPX_TEST_EQ(count, c.size());
}

void test_for_loop()
{
    test_for_loop(hpx::execution::seq);
    test_for_loop(hpx::execution::par);
    test_for_loop(hpx::execution::par_unseq);

    test_for_loop_async(hpx::execution::seq(hpx::execution::task));
    test_for_loop_async(hpx::execution::par(hpx::execution::task));
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map& vm)
{
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    gen.seed(seed);

    test_for_loop();

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

    // By default, this test should run on all available cores
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
