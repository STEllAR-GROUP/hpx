//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/include/compute.hpp>
#include <hpx/include/partitioned_vector.hpp>
#include <hpx/include/parallel_for_each.hpp>

#include <hpx/hpx_init.hpp>

#include <iostream>
#include <numeric>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
// Define the partitioned vector types to be used.
typedef hpx::compute::cuda::allocator<int> target_allocator;
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
    hpx::compute::cuda::target_distribution_policy policy =
        hpx::compute::cuda::target_layout(hpx::compute::cuda::get_local_targets());

    {
        using namespace hpx::parallel;
        hpx::partitioned_vector<int, target_vector> v(1000, policy);
        hpx::parallel::for_each(execution::seq, v.begin(), v.end(), pfo());
        hpx::parallel::for_each(execution::par, v.begin(), v.end(), pfo());
        hpx::parallel::for_each(execution::seq(execution::task),
            v.begin(), v.end(), pfo()).get();
        hpx::parallel::for_each(execution::par(execution::task),
            v.begin(), v.end(), pfo()).get();
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
