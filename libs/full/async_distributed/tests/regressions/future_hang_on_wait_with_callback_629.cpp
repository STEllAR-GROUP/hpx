////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2012 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/async_combinators/wait_each.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/plain_actions.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/pack_traversal/unwrap.hpp>

#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

using hpx::program_options::options_description;
using hpx::program_options::value;
using hpx::program_options::variables_map;

using hpx::finalize;
using hpx::init;

///////////////////////////////////////////////////////////////////////////////
// x to the p power
template <typename T>
inline T sign(T a)
{
    if (a > 0)
        return 1;
    else if (a < 0)
        return -1;
    else
        return 0;
}

template <typename T>
T ipow(T x, T p)
{
    T i = 1;
    for (T j = 1; j < p; j++)
        i *= x;
    return i;
}

template <typename T>
inline bool compare_real(T x, T y, T epsilon)
{
    if ((x + epsilon >= y) && (x - epsilon <= y))
        return true;
    else
        return false;
}

double null_function(std::uint64_t seed, std::uint64_t delay_iterations)
{
    std::mt19937_64 prng(seed);

    std::uniform_real_distribution<double> v(0, 1.);
    std::uniform_int_distribution<short> s(0, 1);

    double d = 0.;

    for (std::uint64_t i = 0; i < delay_iterations; ++i)
    {
        double v0 = v(prng);
        double v1 = v(prng);

        if (compare_real(v1, 0.0, 1e-10))
            v1 = 1e-10;

        d += (s(prng) ? 1.0 : -1.0) * (v0 / v1);
    }

    return d;
}

///////////////////////////////////////////////////////////////////////////////
double null_tree(std::uint64_t seed, std::uint64_t children,
    std::uint64_t depth, std::uint64_t max_depth,
    std::uint64_t delay_iterations);

HPX_PLAIN_ACTION(null_tree, null_tree_action);

double null_tree(std::uint64_t seed, std::uint64_t children,
    std::uint64_t depth, std::uint64_t max_depth,
    std::uint64_t delay_iterations)
{
    if (depth == max_depth)
        return null_function(seed, delay_iterations);

    hpx::id_type const here = hpx::find_here();

    double d = 0.;

    std::vector<hpx::future<double>> futures;
    futures.reserve(children);

    std::uint64_t p = seed + ipow(depth, children);

    for (std::uint64_t j = 0; j < children; ++j)
    {
        futures.push_back(hpx::async<null_tree_action>(
            here, j + p, children, depth + 1, max_depth, delay_iterations));
    }

    null_function(seed, delay_iterations);

    hpx::lcos::wait_each(
        hpx::util::unwrapping([&](double r) { d += r; }), futures);

    return d;
}

int hpx_main(variables_map& vm)
{
    {
        std::uint64_t test_runs = vm["test-runs"].as<std::uint64_t>();

        std::uint64_t children = vm["children"].as<std::uint64_t>();

        std::uint64_t max_depth = vm["depth"].as<std::uint64_t>() + 1;

        std::uint64_t delay_iterations =
            vm["delay-iterations"].as<std::uint64_t>();

        bool verbose = vm.count("verbose") != 0;

        hpx::id_type const here = hpx::find_here();

        double d = 0.;

        null_tree_action null_act;
        for (std::uint64_t i = 0; (test_runs == 0) || (i < test_runs); ++i)
        {
            d += null_act(here, 0, children, 1, max_depth, delay_iterations);

            if (verbose)
                hpx::util::format_to(std::cout, "{:016u} : {}\n", i, d)
                    << std::flush;
        }
    }

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description cmdline("usage: " HPX_APPLICATION_STRING " [options]");

    // clang-format off
    cmdline.add_options()
        ("test-runs"
        , value<std::uint64_t>()->default_value(1000)
        , "number of times to repeat the test (0 == infinite)")

        ("verbose"
        , "print state every iteration")

        ("children"
        , value<std::uint64_t>()->default_value(8)
        , "number of children each node has")

        ("depth"
        , value<std::uint64_t>()->default_value(2)
        , "depth of the tree structure")

        ("delay-iterations"
        , value<std::uint64_t>()->default_value(1000)
        , "number of iterations in the delay loop");
    // clang-format on

    // Initialize and run HPX.
    hpx::init_params init_args;
    init_args.desc_cmdline = cmdline;

    return hpx::init(argc, argv, init_args);
}
#endif
