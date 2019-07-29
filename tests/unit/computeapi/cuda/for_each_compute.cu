///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <hpx/include/compute.hpp>
#include <hpx/include/parallel_for_each.hpp>
#include <hpx/include/parallel_copy.hpp>

#include <hpx/testing.hpp>

#include <hpx/hpx_init.hpp>

#include <numeric>
#include <iostream>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

typedef hpx::compute::cuda::default_executor executor_type;
typedef hpx::compute::cuda::allocator<int> target_allocator;
typedef hpx::compute::vector<int, target_allocator> target_vector;

void test_for_each(executor_type& exec, target_vector& d_A)
{
    std::vector<int> h_C(d_A.size());
    hpx::parallel::copy(
        hpx::parallel::execution::par,
        d_A.begin(), d_A.end(), h_C.begin());

    // FIXME : Lambda function given to for_each() is momentarily defined as
    //         HPX_HOST_DEVICE in place of HPX_DEVICE to allow the host_side
    //         result_of<> (used inside for_each()) to get the return
    //         type

    hpx::parallel::for_each(
        hpx::parallel::execution::par.on(exec),
        d_A.begin(), d_A.end(),
        [] HPX_HOST_DEVICE (int & i)
        {
             i += 5;
        });

    for(std::size_t i = 0; i != h_C.size(); ++i)
    {
        HPX_TEST_EQ(h_C[i] + 5, d_A[i]);
    }
}

int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::random_device{}();
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(2, 101);

    int const N = 100;
    std::vector<int> h_A(N);

    std::iota(h_A.begin(), h_A.end(), dis(gen));

    // define execution target (here device 0)
    hpx::compute::cuda::target target;

    // allocate data on the device
    target_allocator alloc(target);
    target_vector d_A(N, alloc);

    // copy data to device
    hpx::parallel::copy(
        hpx::parallel::execution::par,
        h_A.begin(), h_A.end(), d_A.begin());

    // create executor
    executor_type exec(target);

    // Run tests:
    test_for_each(exec, d_A);

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace boost::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("seed,s", value<unsigned int>(),
        "the random number generator seed to use for this run")
        ;

    // Initialize and run HPX
    hpx::init(desc_commandline, argc, argv);

    return hpx::util::report_errors();
}
