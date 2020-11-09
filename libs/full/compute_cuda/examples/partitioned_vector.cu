//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/algorithm.hpp>
#include <hpx/include/compute.hpp>
#include <hpx/include/partitioned_vector.hpp>

#include <hpx/hpx_init.hpp>

#include <iostream>
#include <numeric>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
// Define the partitioned vector types to be used.
typedef hpx::cuda::experimental::allocator<int> target_allocator;
typedef hpx::compute::vector<int, target_allocator> target_vector;

HPX_REGISTER_PARTITIONED_VECTOR(int, target_vector);

///////////////////////////////////////////////////////////////////////////////
struct pfo
{
    template <typename T>
    HPX_HOST_DEVICE void operator()(T& val) const
    {
        int v = val;
        val = ++v;
    }
};

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map& vm)
{
    hpx::cuda::experimental::target_distribution_policy policy =
        hpx::cuda::experimental::target_layout(
            hpx::cuda::experimental::get_local_targets());

    {
        hpx::partitioned_vector<int, target_vector> v(1000, policy);
        hpx::ranges::for_each(hpx::execution::seq, v, pfo());
        hpx::ranges::for_each(hpx::execution::par, v, pfo());
        hpx::ranges::for_each(
            hpx::execution::seq(hpx::execution::task), v, pfo())
            .get();
        hpx::ranges::for_each(
            hpx::execution::par(hpx::execution::task), v, pfo())
            .get();
    }

    // TODO: add more

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace hpx::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;

    return hpx::init(argc, argv, init_args);
}
