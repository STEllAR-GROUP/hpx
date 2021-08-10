//  Copyright (c) 2018 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/preprocessor/seq/for_each.hpp>

#include <hpx/config.hpp>

#include <hpx/assert.hpp>
#include <hpx/async_cuda/cuda_exception.hpp>
#include <hpx/async_cuda/custom_gpu_api.hpp>

#include <iostream>

// Here is a trivial kernel that can be invoked on the GPU
template <typename T>
__global__ void trivial_kernel(T val)
{
    printf("hello from gpu with value %f\n", static_cast<double>(val));
}

// Copy Kernel for hip on rostam (printf in kernel not working)
template <typename T>
__global__ void copy_kernel(T* in, T* out)
{
    *out = *in;
}

// here is a wrapper that can call the kernel from C++ outside of the .cu file
template <typename T>
void cuda_trivial_kernel(T t, cudaStream_t stream)
{
#if !defined(HPX_HAVE_HIP)
    trivial_kernel<<<1, 1, 0, stream>>>(t);
    ::hpx::cuda::experimental::check_cuda_error(cudaDeviceSynchronize());
#else
    // Silence -Wuninitialized until [[uninitialized]] available
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuninitialized"
#endif
    T *d_in, *d_out;
    T* out = (T*) malloc(sizeof(T));
    ::hpx::cuda::experimental::check_cuda_error(
        cudaMalloc((void**) &d_in, sizeof(T)));
    ::hpx::cuda::experimental::check_cuda_error(
        cudaMalloc((void**) &d_out, sizeof(T)));
    ::hpx::cuda::experimental::check_cuda_error(
        cudaMemcpy(d_in, &t, sizeof(T), cudaMemcpyHostToDevice));
    copy_kernel<<<1, 1, 0, stream>>>(d_in, d_out);
    ::hpx::cuda::experimental::check_cuda_error(cudaDeviceSynchronize());
    ::hpx::cuda::experimental::check_cuda_error(
        cudaMemcpy(out, d_out, sizeof(T), cudaMemcpyDeviceToHost));
    HPX_ASSERT(*out == t);
    ::hpx::cuda::experimental::check_cuda_error(cudaFree(d_in));
    ::hpx::cuda::experimental::check_cuda_error(cudaFree(d_out));
    free(out);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#endif
}

// -------------------------------------------------------------------------
// To make the kernel visible for all template instantiations we might use
// we must instantiate them here first in the cuda compiled code.
// Use a simple boost preprocessor macro to allow any types to
// be added.

// list of types to generate code for
#define TYPES (double) (float)    //

#define GENERATE_SPECIALIZATIONS(_1, _2, elem)                                 \
    template __global__ void trivial_kernel<elem>(elem);                       \
    template __global__ void copy_kernel<elem>(elem*, elem*);                  \
    template void cuda_trivial_kernel<elem>(elem, cudaStream_t);               \
    //

BOOST_PP_SEQ_FOR_EACH(GENERATE_SPECIALIZATIONS, _, TYPES)
