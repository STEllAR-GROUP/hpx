//  Copyright (c) 2018 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_copy.hpp>
#include <hpx/include/parallel_executor_parameters.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/parallel_for_each.hpp>
#include <hpx/include/parallel_for_loop.hpp>
#include <hpx/modules/testing.hpp>
//
#include <hpx/async_cuda/cuda_executor.hpp>
#include <hpx/async_cuda/target.hpp>

// CUDA runtime
#include <hpx/async_cuda/custom_gpu_api.hpp>
//
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>

// -------------------------------------------------------------------------
// This example uses the normal C++ compiler to compile all the HPX stuff
// but the cuda functions go in their own .cu file and are compiled with
// nvcc, we don't mix them.
// Declare functions we are importing - note that template instantiations
// must be present in the .cu file and compiled so that we can link to them
template <typename T>
extern void cuda_trivial_kernel(T, cudaStream_t stream);

extern __global__ void saxpy(int n, float a, float* x, float* y);
// -------------------------------------------------------------------------
int test_saxpy(hpx::cuda::experimental::cuda_executor& cudaexec)
{
    int N = 1 << 20;

    // host arrays
    std::vector<float> h_A(N);
    std::vector<float> h_B(N);

    float *d_A, *d_B;
    hpx::cuda::experimental::check_cuda_error(
        cudaMalloc((void**) &d_A, N * sizeof(float)));

    hpx::cuda::experimental::check_cuda_error(
        cudaMalloc((void**) &d_B, N * sizeof(float)));

    // init host data
    for (int idx = 0; idx < N; idx++)
    {
        h_A[idx] = 1.0f;
        h_B[idx] = 2.0f;
    }

    // copy both arrays from cpu to gpu, putting both copies onto the stream
    // no need to get a future back yet
    hpx::apply(cudaexec, cudaMemcpyAsync, d_A, h_A.data(), N * sizeof(float),
        cudaMemcpyHostToDevice);
    hpx::apply(cudaexec, cudaMemcpyAsync, d_B, h_B.data(), N * sizeof(float),
        cudaMemcpyHostToDevice);

    unsigned int threads = 256;
    unsigned int blocks = (N + 255) / threads;
    float ratio = 2.0f;

    // now launch a kernel on the stream
    void* args[] = {&N, &ratio, &d_A, &d_B};
#ifdef HPX_HAVE_HIP
    hpx::apply(cudaexec, cudaLaunchKernel,
#else
    hpx::apply(cudaexec, cudaLaunchKernel<void>,
#endif
        reinterpret_cast<const void*>(&saxpy), dim3(blocks), dim3(threads),
        args, std::size_t(0));

    // finally, perform a copy from the gpu back to the cpu all on the same stream
    // grab a future to when this completes
    auto cuda_future = hpx::async(cudaexec, cudaMemcpyAsync, h_B.data(), d_B,
        N * sizeof(float), cudaMemcpyDeviceToHost);

    // we can add a continuation to the memcpy future, so that when the
    // memory copy completes, we can do new things ...
    cuda_future
        .then([&](hpx::future<void>&&) {
            std::cout
                << "saxpy completed on GPU, checking results in continuation"
                << std::endl;
            float max_error = 0.0f;
            for (int jdx = 0; jdx < N; jdx++)
            {
                max_error = (std::max)(max_error, abs(h_B[jdx] - 4.0f));
            }
            std::cout << "Max Error: " << max_error << std::endl;
        })
        .get();

    // the .get() is important in the line above because without it, this function
    // returns and the task above goes out of scope and the refs it holds
    // are invalidated.

    return 1;
}

// -------------------------------------------------------------------------
int hpx_main(hpx::program_options::variables_map& vm)
{
    // install cuda future polling handler
    hpx::cuda::experimental::enable_user_polling poll("default");
    //
    std::size_t device = vm["device"].as<std::size_t>();
    //
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    // create a cuda target using device number 0,1,2...
    hpx::cuda::experimental::target target(device);

    // for debug purposes, print out available targets
    hpx::cuda::experimental::print_local_targets();

    // create a stream helper object
    hpx::cuda::experimental::cuda_executor cudaexec(
        device, hpx::cuda::experimental::event_mode{});

    // --------------------
    // test kernel launch<float> using apply and async
    float testf = 1.2345;
    std::cout << "apply : cuda kernel <float>  : " << testf << std::endl;
    hpx::apply(cudaexec, cuda_trivial_kernel<float>, testf);

    std::cout << "async : cuda kernel <float>  : " << testf + 1 << std::endl;
    auto f1 = hpx::async(cudaexec, cuda_trivial_kernel<float>, testf + 1);
    f1.get();

    // --------------------
    // test kernel launch<double> using apply and async
    double testd = 2.3456;
    std::cout << "apply : cuda kernel <double> : " << testd << std::endl;
    hpx::apply(cudaexec, cuda_trivial_kernel<double>, testd);

    std::cout << "async : cuda kernel <double> : " << testd + 1 << std::endl;
    auto f2 = hpx::async(cudaexec, cuda_trivial_kernel<double>, testd + 1);
    f2.get();

    // --------------------
    // test adding a continuation to a cuda call
    double testd2 = 3.1415;
    std::cout << "future/continuation : " << testd2 << std::endl;
    auto f3 = hpx::async(cudaexec, cuda_trivial_kernel<double>, testd2);
    f3.then([](hpx::future<void>&&) {
          std::cout << "continuation triggered\n";
      }).get();

    // --------------------
    // test using a copy of a cuda executor
    // and adding a continuation with a copy of a copy
    std::cout << "Copying executor : " << testd2 + 1 << std::endl;
    auto exec_copy = cudaexec;
    auto f4 = hpx::async(exec_copy, cuda_trivial_kernel<double>, testd2 + 1);
    f4.then([exec_copy](hpx::future<void>&&) {
          std::cout << "copy continuation triggered\n";
      }).get();

    // --------------------
    // test a full kernel example
    HPX_TEST(test_saxpy(cudaexec));

    return hpx::finalize();
}

// -------------------------------------------------------------------------
int main(int argc, char** argv)
{
    printf("[HPX Cuda future] - Starting...\n");

    using namespace hpx::program_options;
    options_description cmdline("usage: " HPX_APPLICATION_STRING " [options]");
    cmdline.add_options()("device",
        hpx::program_options::value<std::size_t>()->default_value(0),
        "Device to use")("iterations",
        hpx::program_options::value<std::size_t>()->default_value(30),
        "iterations")("seed,s", hpx::program_options::value<unsigned int>(),
        "the random number generator seed to use for this run");

    hpx::init_params init_args;
    init_args.desc_cmdline = cmdline;

    auto result = hpx::init(argc, argv, init_args);
    return result || hpx::util::report_errors();
}
