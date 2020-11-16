//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/async_cuda/target.hpp>
#include <hpx/include/compute.hpp>
#include <hpx/include/parallel_copy.hpp>

#include <hpx/hpx_init.hpp>

#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

int hpx_main(hpx::program_options::variables_map& vm)
{
    auto seed = std::random_device{}();
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(2, 101);

    // create data vector on host
    int const N = 100;
    std::vector<int> h_A(N);
    std::vector<int> h_B(N);
    std::iota(h_A.begin(), h_A.end(), dis(gen));

    hpx::cuda::experimental::target target;

    // create data vector on device
    typedef hpx::cuda::experimental::allocator<int> allocator_type;
    allocator_type alloc(target);

    hpx::compute::vector<int, allocator_type> d_A(N, alloc);

    hpx::future<void> f = target.get_future_with_callback();

    f.get();

    // copy data from host to device
    hpx::copy(hpx::execution::par, h_A.begin(), h_A.end(), d_A.begin());

    // copy data from device to host
    hpx::copy(hpx::execution::par, d_A.begin(), d_A.end(), h_B.begin());

    if (std::equal(h_A.begin(), h_A.end(), h_B.begin()))
        std::cout << "Copy succeeded!\n";
    else
        std::cout << "Copy not successful :(\n";

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

    return hpx::init(argc, argv, init_args);
}
