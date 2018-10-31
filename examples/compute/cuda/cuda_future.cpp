//  Copyright (c) 2018 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_NO_CXX11_ALLOCATOR
//
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_copy.hpp>
#include <hpx/include/parallel_for_each.hpp>
#include <hpx/include/parallel_for_loop.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/parallel_executor_parameters.hpp>
//
#include <hpx/include/compute.hpp>
#include <hpx/compute/cuda/target.hpp>
// CUDA runtime
#include <cuda_runtime.h>
//
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>
//
#include "cuda_future_helper.h"

// -------------------------------------------------------------------------
// This example uses the normal C++ compiler to compile all the HPX stuff
// but the cuda functions go in their own .cu file and are compiled with
// nvcc, we don't mix them.
// Declare functions we are importing - note that template instantiations
// must be present in the .cu file and compiled so that we can link to them
template <typename T>
extern void cuda_trivial_kernel(T, cudaStream_t stream);

// -------------------------------------------------------------------------
int hpx_main(boost::program_options::variables_map& vm)
{
    std::size_t device     = vm["device"].as<std::size_t>();
    //
    unsigned int seed = (unsigned int)std::time(nullptr);
     if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    hpx::compute::cuda::target target(device);
    //
    hpx::compute::util::cuda_future_helper helper(device);
    helper.print_local_targets();
    //
    float testf = 2.345;
    cuda_trivial_kernel(testf, helper.get_stream());

    double testd = 5.678;
    cuda_trivial_kernel(testd, helper.get_stream());

    auto fn = &cuda_trivial_kernel<double>;
    double d = 3.1415;
    auto f = helper.async(fn, d);
    f.then([](hpx::future<void> &&f) {
        std::cout << "trivial kernel completed \n";
    });

    return hpx::finalize();
}

// -------------------------------------------------------------------------
int main(int argc, char **argv)
{
    printf("[HPX Cuda future] - Starting...\n");

    using namespace boost::program_options;
    options_description cmdline("usage: " HPX_APPLICATION_STRING " [options]");
    cmdline.add_options()
        (   "device",
            boost::program_options::value<std::size_t>()->default_value(0),
            "Device to use")
        (   "iterations",
            boost::program_options::value<std::size_t>()->default_value(30),
            "iterations")
        ("seed,s",
            boost::program_options::value<unsigned int>(),
            "the random number generator seed to use for this run")
        ;

    return hpx::init(cmdline, argc, argv);
}
