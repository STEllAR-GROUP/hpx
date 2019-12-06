//  Copyright (c) 2018 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "cuda_runtime.h"
#include <boost/preprocessor/seq/for_each.hpp>

// Here is a trivial kernel that can be invoked on the GPU
template <typename T>
__global__ void trivial_kernel(T val) {
  printf("hello from gpu with value %f\n", static_cast<double>(val));
}

// here is a wrapper that can call the kernel from C++ outside of the .cu file
template <typename T>
void cuda_trivial_kernel(T t, cudaStream_t stream)
{
  trivial_kernel<<<1, 1, 0, stream>>>(t);
}

// -------------------------------------------------------------------------
// To make the kernel visible for all template instantiations we might use
// we must instantiate them here first in the cuda compiled code.
// Use a simple boost preprocessor macro to allow any types to
// be added.

// list of types to generate code for
#define TYPES (double)(float)                                             \
  //

#define GENERATE_SPECIALIZATIONS(_1, _2, elem)                            \
  template                                                                \
  __global__ void trivial_kernel<elem>(elem);                             \
  template                                                                \
  void cuda_trivial_kernel<elem>(elem, cudaStream_t);                     \
  //

BOOST_PP_SEQ_FOR_EACH(GENERATE_SPECIALIZATIONS, _, TYPES)
