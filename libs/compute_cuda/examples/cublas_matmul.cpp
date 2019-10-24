//  Copyright (c) 2017-2018 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// For compliance with the NVIDIA EULA:
// "This software contains source code provided by NVIDIA Corporation."

// This is a conversion of the NVIDIA cublas example matrixMulCUBLAS to use
// HPX style data structures, executors and futures and demonstrate a simple use
// of computing a number of iteration of a matrix multiply on a stream and returning
// a future when it completes. This can be used to chain/schedule other task
// in a manner consistent with the future based API of HPX.
//
// Example usage: bin/cublas_matmul --sizemult=10 --iterations=25 --hpx:threads=8
// NB. The hpx::threads param only controls how many parallel tasks to use for the CPU
// comparison/checks and makes no difference to the GPU execution.
//
// Note: The hpx::compute::cuda::allocator makes use of device code and if used
// this example must be compiled with nvcc instead of c++ which requires the following
// cmake setting
// set_source_files_properties(cublas_matmul.cpp
//     PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
// Currently, nvcc does not handle lambda functions properly and it is simpler to use
// cudaMalloc/cudaMemcpy etc, so we do not #define HPX_CUBLAS_DEMO_WITH_ALLOCATOR

#define BOOST_NO_CXX11_ALLOCATOR
//
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_copy.hpp>
#include <hpx/include/parallel_executor_parameters.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/parallel_for_each.hpp>
#include <hpx/include/parallel_for_loop.hpp>
//
#include <hpx/compute/cuda/target.hpp>
#include <hpx/include/compute.hpp>
#include <hpx/timing.hpp>
#ifdef HPX_CUBLAS_DEMO_WITH_ALLOCATOR
#include <hpx/compute/cuda/allocator.hpp>
#endif
// CUDA runtime
#include <cublas_v2.h>
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
//
std::mt19937 gen;

// -------------------------------------------------------------------------
// a simple cublas wrapper helper object that can be used to synchronize
// cublas calls with an hpx future.
// -------------------------------------------------------------------------
template <typename T>
struct cublas_helper : hpx::compute::util::cuda_future_helper
{
#ifdef HPX_CUBLAS_DEMO_WITH_ALLOCATOR
    using allocator_type = typename hpx::compute::cuda::allocator<T>;
    using vector_type = typename hpx::compute::vector<T, allocator_type>;
#endif

    // construct a cublas stream
    cublas_helper(std::size_t device = 0)
      : hpx::compute::util::cuda_future_helper(device)
    {
        handle_ = 0;
        hpx::compute::util::cublas_error(cublasCreate(&handle_));
    }

    cublas_helper(cublas_helper& other) = delete;
    cublas_helper(const cublas_helper& other) = delete;
    cublas_helper operator=(const cublas_helper& other) = delete;

    ~cublas_helper()
    {
        hpx::compute::util::cublas_error(cublasDestroy(handle_));
    }

    // -------------------------------------------------------------------------
    // launch a cuBlas function and return a future that will become ready
    // when the task completes, this allows integregration of GPU kernels with
    // hpx::futuresa and the tasking DAG.
    template <typename R, typename... Params, typename... Args>
    hpx::future<void> async(R (*cublas_function)(Params...), Args&&... args)
    {
        // make sue we run on the correct device
        hpx::compute::util::cuda_error(
            cudaSetDevice(target_.native_handle().get_device()));
        // make sure this operation takes place on our stream
        hpx::compute::util::cublas_error(cublasSetStream(handle_, stream_));
        // insert the cublas handle in the arg list and call the cublas function
        hpx::compute::util::detail::async_helper<R, Params...> helper;
        helper(cublas_function, handle_, std::forward<Args>(args)...);
        return get_future();
    }

    // This is a simple wrapper for any cublas call, pass in the same arguments
    // that you would use for a cublas call except the cublas handle which is omitted
    // as the wrapper will supply that for you
    template <typename R, typename... Params, typename... Args>
    R apply(R (*cublas_function)(Params...), Args&&... args)
    {
        // make sue we run on the correct device
        hpx::compute::util::cuda_error(
            cudaSetDevice(target_.native_handle().get_device()));
        // make sure this operation takes place on our stream
        hpx::compute::util::cublas_error(cublasSetStream(handle_, stream_));
        // insert the cublas handle in the arg list and call the cublas function
        hpx::compute::util::detail::async_helper<R, Params...> helper;
        return helper(cublas_function, handle_, std::forward<Args>(args)...);
    }

    // return a copy of the cublas handle
    cublasHandle_t get_handle()
    {
        return handle_;
    }

private:
    cublasHandle_t handle_;
};

// -------------------------------------------------------------------------
// Optional Command-line multiplier for matrix sizes
struct sMatrixSize
{
    unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
};

// -------------------------------------------------------------------------
// Compute reference data set matrix multiply on CPU
// C = A * B
// @param C          reference data, computed but preallocated
// @param A          matrix A as provided to device
// @param B          matrix B as provided to device
// @param hA         height of matrix A
// @param wB         width of matrix B
// -------------------------------------------------------------------------
template <typename T>
void matrixMulCPU(T* C, const T* A, const T* B, unsigned int hA,
    unsigned int wA, unsigned int wB)
{
    hpx::parallel::for_loop(hpx::parallel::execution::par, 0, hA, [&](int i) {
        for (unsigned int j = 0; j < wB; ++j)
        {
            T sum = 0;
            for (unsigned int k = 0; k < wA; ++k)
            {
                T a = A[i * wA + k];
                T b = B[k * wB + j];
                sum += a * b;
            }
            C[i * wB + j] = (T) sum;
        }
    });
}

// -------------------------------------------------------------------------
// Compute the L2 norm difference between two arrays
inline bool compare_L2_err(const float* reference, const float* data,
    const unsigned int len, const float epsilon)
{
    HPX_ASSERT(epsilon >= 0);

    float error = 0;
    float ref = 0;

    hpx::parallel::for_loop(hpx::parallel::execution::par, 0, len, [&](int i) {
        float diff = reference[i] - data[i];
        error += diff * diff;
        ref += reference[i] * reference[i];
    });

    float normRef = sqrtf(ref);
    if (std::fabs(ref) < 1e-7f)
    {
        return false;
    }

    float normError = sqrtf(error);
    error = normError / normRef;
    bool result = error < epsilon;
    return result;
}

// -------------------------------------------------------------------------
// Run a simple test matrix multiply using CUBLAS
// -------------------------------------------------------------------------
template <typename T>
void matrixMultiply(
    sMatrixSize& matrix_size, std::size_t device, std::size_t iterations)
{
    using hpx::parallel::execution::par;

    // Allocate host memory for matrices A and B
    unsigned int size_A = matrix_size.uiWA * matrix_size.uiHA;
    unsigned int size_B = matrix_size.uiWB * matrix_size.uiHB;
    unsigned int size_C = matrix_size.uiWC * matrix_size.uiHC;

    std::vector<T> h_A(size_A);
    std::vector<T> h_B(size_B);
    std::vector<T> h_C(size_C);
    std::vector<T> h_CUBLAS(size_C);

    // Fill A and B with random numbers
    auto randfunc = [](T& x) { x = gen() / (T) RAND_MAX; };
    hpx::parallel::for_each(par, h_A.begin(), h_A.end(), randfunc);
    hpx::parallel::for_each(par, h_B.begin(), h_B.end(), randfunc);

    // create a cublas helper object we'll use to futurize the cuda events
    cublas_helper<T> cublas(device);
    using cublas_future = typename cublas_helper<T>::future_type;

#ifdef HPX_CUBLAS_DEMO_WITH_ALLOCATOR
    // for convenience
    using device_allocator = typename cublas_helper<T>::allocator_type;
    using device_vector = typename cublas_helper<T>::vector_type;
    // The policy used in the parallel algorithms
    auto policy = hpx::parallel::execution::par;

    // Create a cuda allocator
    device_allocator alloc(cublas.target());

    // Allocate device memory
    device_vector d_vA(size_A, alloc);
    device_vector d_vB(size_B, alloc);
    device_vector d_vC(size_C, alloc);

    // copy host memory to device
    hpx::parallel::copy(policy, h_A.begin(), h_A.end(), d_vA.begin());
    hpx::parallel::copy(policy, h_B.begin(), h_B.end(), d_vB.begin());

    // just to make the rest of code the same for both cases
    T* d_A = d_vA.device_data();
    T* d_B = d_vB.device_data();
    T* d_C = d_vC.device_data();

#else
    T *d_A, *d_B, *d_C;
    hpx::compute::util::cuda_error(
        cudaMalloc((void**) &d_A, size_A * sizeof(T)));

    hpx::compute::util::cuda_error(
        cudaMalloc((void**) &d_B, size_B * sizeof(T)));

    hpx::compute::util::cuda_error(
        cudaMalloc((void**) &d_C, size_C * sizeof(T)));

    // adding async copy operations into the stream before cublas calls puts
    // the copies in the queue before the matrix operations.
    cublas.memcpy_apply(
        d_A, h_A.data(), size_A * sizeof(T), cudaMemcpyHostToDevice);

    auto copy_future = cublas.memcpy_async(
        d_B, h_B.data(), size_B * sizeof(T), cudaMemcpyHostToDevice);

    // we can call get_future multiple times on the cublas helper.
    // Each one returns a new future that will be set ready when the stream event
    // for this point is triggered
    copy_future.then([](cublas_future&& f) {
        std::cout << "The async host->device copy operation completed"
                  << std::endl;
    });

#endif

    std::cout << "Computing result using CUBLAS...\n";
    const T alpha = 1.0f;
    const T beta = 0.0f;

    // Perform warmup operation with cublas
    // note cublas is column major ordering : transpose the order
    hpx::util::high_resolution_timer t1;
    //
    std::cout << "calling CUBLAS...\n";
    auto fut = cublas.async(&cublasSgemm, CUBLAS_OP_N, CUBLAS_OP_N,
        matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B,
        matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWA);

    // wait until the operation completes
    fut.get();

    double us1 = t1.elapsed_microseconds();
    std::cout << "warmup: elapsed_microseconds " << us1 << std::endl;

    // once the future has been retrieved, the next call to
    // get_future will create a new event and a new future so
    // we can reuse the same cublas wrapper object and stream if we want

    hpx::util::high_resolution_timer t2;
    for (std::size_t j = 0; j < iterations; j++)
    {
        cublas.apply(&cublasSgemm, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB,
            matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB,
            d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWA);
    }
    // get a future for when the stream reaches this point (matrix operations complete)
    auto matrix_finished = cublas.get_future();

#ifndef HPX_CUBLAS_DEMO_WITH_ALLOCATOR
    // when the matrix operations complete, copy the result to the host
    auto copy_finished = cublas.memcpy_async(
        h_CUBLAS.data(), d_C, size_C * sizeof(T), cudaMemcpyDeviceToHost);

#endif

    // attach a continuation to the cublas future
    auto new_future = matrix_finished.then([&](cublas_future&& f) {
        double us2 = t2.elapsed_microseconds();
        std::cout << "actual: elapsed_microseconds " << us2 << " iterations "
                  << iterations << std::endl;

        // Compute and print the performance
        double usecPerMatrixMul = us2 / iterations;
        double flopsPerMatrixMul = 2.0 * (double) matrix_size.uiWA *
            (double) matrix_size.uiHA * (double) matrix_size.uiWB;
        double gigaFlops =
            (flopsPerMatrixMul * 1.0e-9) / (usecPerMatrixMul / 1e6);
        printf("Performance = %.2f GFlop/s, Time = %.3f msec/iter, Size = %.0f "
               "Ops\n",
            gigaFlops, 1e-3 * usecPerMatrixMul, flopsPerMatrixMul);
    });

    // wait for the timing to complete, and then do a CPU comparison
    auto finished = new_future.then([&](cublas_future&& f) {
        // compute reference solution on the CPU
        std::cout << "\nComputing result using host CPU...\n";
#ifdef HPX_CUBLAS_DEMO_WITH_ALLOCATOR
        // copy result from device to host
        hpx::parallel::copy(policy, d_C.begin(), d_C.end(), h_CUBLAS.begin());
#else
        // just wait for the device->host copy to complete if it hasn't already
        copy_finished.get();
#endif

        // compute reference solution on the CPU
        // allocate storage for the CPU result
        std::vector<T> reference(size_C);

        hpx::util::high_resolution_timer t3;
        matrixMulCPU<T>(reference.data(), h_A.data(), h_B.data(),
            matrix_size.uiHA, matrix_size.uiWA, matrix_size.uiWB);
        double us3 = t3.elapsed_microseconds();
        std::cout << "CPU elapsed_microseconds (1 iteration) " << us3
                  << std::endl;

        // check result (CUBLAS)
        bool resCUBLAS =
            compare_L2_err(reference.data(), h_CUBLAS.data(), size_C, 1e-6);
        if (resCUBLAS != true)
        {
            throw std::runtime_error("matrix CPU/GPU comparison error");
        }
        // if the result was incorrect, we throw an exception, so here it's ok
        std::cout
            << "\nComparing CUBLAS Matrix Multiply with CPU results: OK \n";
    });

    finished.get();
#ifndef HPX_CUBLAS_DEMO_WITH_ALLOCATOR
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
#endif
}

// -------------------------------------------------------------------------
int hpx_main(hpx::program_options::variables_map& vm)
{
    std::size_t device = vm["device"].as<std::size_t>();
    std::size_t sizeMult = vm["sizemult"].as<std::size_t>();
    std::size_t iterations = vm["iterations"].as<std::size_t>();
    //
    unsigned int seed = std::random_device{}();
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    gen.seed(seed);
    std::cout << "using seed: " << seed << std::endl;

    //
    sizeMult = (std::min)(sizeMult, std::size_t(100));
    sizeMult = (std::max)(sizeMult, std::size_t(1));
    //
    // use a larger block size for Fermi and above, query default cuda target properties
    hpx::compute::cuda::target target(device);

    std::cout << "GPU Device " << target.native_handle().get_device() << ": \""
              << target.native_handle().processor_name() << "\" "
              << "with compute capability "
              << target.native_handle().processor_family() << "\n";

    int block_size = (target.native_handle().processor_family() < 2) ? 16 : 32;

    sMatrixSize matrix_size;
    matrix_size.uiWA = 2 * block_size * sizeMult;
    matrix_size.uiHA = 4 * block_size * sizeMult;
    matrix_size.uiWB = 2 * block_size * sizeMult;
    matrix_size.uiHB = 4 * block_size * sizeMult;
    matrix_size.uiWC = 2 * block_size * sizeMult;
    matrix_size.uiHC = 4 * block_size * sizeMult;

    printf("MatrixA(%u,%u), MatrixB(%u,%u), MatrixC(%u,%u)\n\n",
        matrix_size.uiWA, matrix_size.uiHA, matrix_size.uiWB, matrix_size.uiHB,
        matrix_size.uiWC, matrix_size.uiHC);

    matrixMultiply<float>(matrix_size, device, iterations);
    return hpx::finalize();
}

// -------------------------------------------------------------------------
int main(int argc, char** argv)
{
    printf("[HPX Matrix Multiply CUBLAS] - Starting...\n");

    using namespace hpx::program_options;
    options_description cmdline("usage: " HPX_APPLICATION_STRING " [options]");
    cmdline.add_options()("device",
        hpx::program_options::value<std::size_t>()->default_value(0),
        "Device to use")("sizemult",
        hpx::program_options::value<std::size_t>()->default_value(5),
        "Multiplier")("iterations",
        hpx::program_options::value<std::size_t>()->default_value(30),
        "iterations")("no-cpu",
        hpx::program_options::value<bool>()->default_value(false),
        "disable CPU validation to save time")("seed,s",
        hpx::program_options::value<unsigned int>(),
        "the random number generator seed to use for this run");

    return hpx::init(cmdline, argc, argv);
}
