///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <hpx/algorithm.hpp>
#include <hpx/async_cuda/target.hpp>
#include <hpx/include/compute.hpp>

#include <hpx/modules/testing.hpp>

#include <hpx/hpx_init.hpp>

#include <cstddef>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <hpx/program_options.hpp>

typedef hpx::cuda::experimental::default_executor executor_type;
typedef hpx::cuda::experimental::allocator<int> target_allocator;
typedef hpx::compute::vector<int, target_allocator> target_vector;

void test_for_loop(executor_type& exec, target_vector& d_A, target_vector& d_B,
    target_vector& d_C, std::vector<int> const& ref)
{
    // FIXME : Lambda function given to for_loop_n() is momentarily defined as
    //         HPX_HOST_DEVICE in place of HPX_DEVICE to allow the host_side
    //         result_of<> (used inside for_each()) to get the return
    //         type

    hpx::for_loop_n(hpx::execution::par.on(exec), d_A.data(), d_A.size(),
        hpx::parallel::induction(d_B.data()),
        hpx::parallel::induction(d_C.data()),
        [] HPX_HOST_DEVICE(int* A, int* B, int* C) { *C = *A + 3.0 * *B; });

    std::vector<int> h_C(d_C.size());
    hpx::copy(hpx::execution::par, d_C.begin(), d_C.end(), h_C.begin());

    HPX_TEST_EQ(h_C.size(), ref.size());
    HPX_TEST_EQ(d_C.size(), ref.size());
    for (std::size_t i = 0; i != ref.size(); ++i)
    {
        HPX_TEST_EQ(h_C[i], ref[i]);
        HPX_TEST_EQ(d_C[i], ref[i]);
    }
}

int hpx_main(hpx::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::random_device{}();
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(2, 101);

    int const N = 100;
    std::vector<int> h_A(N);
    std::vector<int> h_B(N);
    std::vector<int> h_C_ref(N);

    std::iota(h_A.begin(), h_A.end(), dis(gen));
    std::iota(h_B.begin(), h_B.end(), dis(gen));

    std::transform(h_A.begin(), h_A.end(), h_B.begin(), h_C_ref.begin(),
        [](int a, int b) { return a + 3.0 * b; });

    // define execution targets (here device 0), allows overlapping operations
    hpx::cuda::experimental::target targetA, targetB;

    // allocate data on the device
    target_allocator allocA(targetA);
    target_allocator allocB(targetB);

    target_vector d_A(N, allocA);
    target_vector d_B(N, allocB);
    target_vector d_C(N, allocA);

    // copy data to device
    hpx::future<void> f = hpx::copy(hpx::execution::par(hpx::execution::task),
        h_A.begin(), h_A.end(), d_A.begin());

    hpx::copy(hpx::execution::par, h_B.begin(), h_B.end(), d_B.begin());

    // synchronize with copy operation to A
    f.get();

    // create executor
    executor_type exec(targetB);

    // Run tests:
    test_for_loop(exec, d_A, d_B, d_C, h_C_ref);

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace hpx::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()("seed,s", value<unsigned int>(),
        "the random number generator seed to use for this run");

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;

    hpx::init(argc, argv, init_args);

    return hpx::util::report_errors();
}
