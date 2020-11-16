//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
void barrier_test(
    std::size_t num, std::size_t rank, std::atomic<std::size_t>& c)
{
    hpx::lcos::barrier b(
        hpx::util::format("local_barrier_test_{}", hpx::get_locality_id()), num,
        rank);
    ++c;

    // wait for all threads to enter the barrier
    b.wait();
}

///////////////////////////////////////////////////////////////////////////////
void local_tests(hpx::program_options::variables_map& vm)
{
    std::size_t pxthreads = 0;
    if (vm.count("pxthreads"))
        pxthreads = vm["pxthreads"].as<std::size_t>();

    std::size_t iterations = 0;
    if (vm.count("iterations"))
        iterations = vm["iterations"].as<std::size_t>();

    hpx::id_type here = hpx::find_here();
    for (std::size_t i = 0; i < iterations; ++i)
    {
        std::atomic<std::size_t> c(0);
        for (std::size_t j = 1; j <= pxthreads; ++j)
        {
            hpx::async(&barrier_test, pxthreads + 1, j, std::ref(c));
        }

        hpx::lcos::barrier b(
            hpx::util::format("local_barrier_test_{}", hpx::get_locality_id()),
            pxthreads + 1, 0);
        b.wait();    // wait for all threads to enter the barrier

        HPX_TEST_EQ(pxthreads, c.load());
    }
}

///////////////////////////////////////////////////////////////////////////////
void remote_test_multiple(hpx::program_options::variables_map& vm)
{
    std::size_t iterations = 0;
    if (vm.count("iterations"))
        iterations = vm["iterations"].as<std::size_t>();

    char const* const barrier_test_name = "/test/barrier/multiple";

    hpx::lcos::barrier b(barrier_test_name);
    for (std::size_t i = 0; i != iterations; ++i)
        b.wait();
}

void remote_test_single(hpx::program_options::variables_map& vm)
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    if (localities.size() == 1)
        return;    // nothing to be done here

    std::size_t iterations = 0;
    if (vm.count("iterations"))
        iterations = vm["iterations"].as<std::size_t>();

    char const* const barrier_test_name_outer = "/test/barrier/single_outer";
    hpx::lcos::barrier outer(barrier_test_name_outer);

    char const* const barrier_test_name = "/test/barrier/single";
    for (std::size_t i = 0; i != iterations; ++i)
    {
        hpx::lcos::barrier b(barrier_test_name);
        b.wait();

        outer.wait();
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map& vm)
{
    local_tests(vm);

    remote_test_multiple(vm);
    remote_test_multiple(vm);

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    using namespace hpx::program_options;

    // Configure application-specific options
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    // clang-format off
    desc_commandline.add_options()
        ("pxthreads,T", value<std::size_t>()->default_value(64),
            "the number of PX threads to invoke")
        ("iterations", value<std::size_t>()->default_value(64),
            "the number of times to repeat the test")
        ;
    // clang-format on

    // We force this test to use several threads by default.
    std::vector<std::string> const cfg = {
        "hpx.os_threads=all", "hpx.run_hpx_main!=1"};

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::init(argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
#endif
