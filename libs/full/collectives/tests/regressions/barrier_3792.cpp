// Copyright (c) 2019 Weile Wei
// Copyright (c) 2019 Maxwell Reeser
// Copyright (c) 2019-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
void run_barrier_test(
    std::size_t root_locality, std::vector<size_t> const& locs)
{
    auto const this_locality = hpx::get_locality_id();
    auto const it = std::find(locs.begin(), locs.end(), this_locality);
    if (it == locs.end())
        return;

    auto const root_it = std::find(locs.begin(), locs.end(), root_locality);
    if (root_it == locs.end())
        return;

    // this assumes that the first locality in the array has the smallest ordinal
    // number
    std::size_t const offset = it - locs.begin();
    std::string const barrier_name =
        "/loc_list/barrier" + std::to_string(locs[0]) + std::to_string(locs[1]);

    hpx::distributed::barrier b(barrier_name,
        hpx::collectives::num_sites_arg(locs.size()),
        hpx::collectives::this_site_arg(offset),
        hpx::collectives::generation_arg(),
        hpx::collectives::root_site_arg(root_it - locs.begin()));
    b.wait();
}

int hpx_main()
{
    std::cout << "Hello world from locality " << hpx::get_locality_id()
              << std::endl;

    run_barrier_test(0, {0, 1});
    run_barrier_test(1, {0, 1});

    run_barrier_test(0, {0, 2});
    run_barrier_test(2, {0, 2});

    run_barrier_test(1, {1, 2});
    run_barrier_test(2, {1, 2});

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // We force hpx_main to run on all processes
    std::vector<std::string> const cfg = {"hpx.run_hpx_main!=1"};
    hpx::init_params init_args;
    init_args.cfg = cfg;

    return hpx::init(argc, argv, init_args);
}
#endif
