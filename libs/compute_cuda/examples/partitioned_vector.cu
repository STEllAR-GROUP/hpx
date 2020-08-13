//  Copyright (c) 2016 Hartmut Kaiser
//
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
        using namespace hpx::parallel;
        hpx::partitioned_vector<int, target_vector> v(1000, policy);
        hpx::ranges::for_each(execution::seq, v, pfo());
        hpx::ranges::for_each(execution::par, v, pfo());
        hpx::ranges::for_each(execution::seq(execution::task), v, pfo()).get();
        hpx::ranges::for_each(execution::par(execution::task), v, pfo()).get();
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
    return hpx::init(desc_commandline, argc, argv);
}
