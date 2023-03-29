//  Copyright (c) 2012 Maciej Brodowicz
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config/defines.hpp>

#if defined(DOXYGEN)
/// Returns the GCC version HPX is compiled with. Only set if compiled with GCC.
#define HPX_GCC_VERSION
/// Returns the Clang version HPX is compiled with. Only set if compiled with
/// Clang.
#define HPX_CLANG_VERSION
/// Returns the Intel Compiler version HPX is compiled with. Only set if
/// compiled with the Intel Compiler.
#define HPX_INTEL_VERSION
/// This macro is set if the compilation is with MSVC.
#define HPX_MSVC
/// This macro is set if the compilation is with Mingw.
#define HPX_MINGW
/// This macro is set if the compilation is for Windows.
#define HPX_WINDOWS
/// This macro is set if the compilation is for Intel Knights Landing.
#define HPX_NATIVE_MIC
#else

// clang-format off
#if defined(__GNUC__)

// macros to facilitate handling of compiler-specific issues
#  define HPX_GCC_VERSION (__GNUC__*10000 + __GNUC_MINOR__*100 + __GNUC_PATCHLEVEL__)

#  define HPX_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS 1

#  undef HPX_CLANG_VERSION
#  undef HPX_INTEL_VERSION

#else

#  undef HPX_GCC_VERSION

#endif

#if defined(__clang__)

#  define HPX_CLANG_VERSION \
 (__clang_major__*10000 + __clang_minor__*100 + __clang_patchlevel__)

#  undef HPX_INTEL_VERSION

#else

#  undef HPX_CLANG_VERSION

#endif

#if defined(__INTEL_COMPILER)
# define HPX_INTEL_VERSION __INTEL_COMPILER
# if defined(_WIN32) || (_WIN64)
#  define HPX_INTEL_WIN HPX_INTEL_VERSION
// suppress a couple of benign warnings
   // template parameter "..." is not used in declaring the parameter types of
   // function template "..."
#  pragma warning disable 488
   // invalid redeclaration of nested class
#  pragma warning disable 1170
   // decorated name length exceeded, name was truncated
#  pragma warning disable 2586
# endif
#else

#  undef HPX_INTEL_VERSION

#endif

// Identify if we compile for the MIC
#if defined(__MIC)
#   define HPX_NATIVE_MIC
#endif

#if defined(_MSC_VER)
#  define HPX_WINDOWS
#  define HPX_MSVC _MSC_VER
#  define HPX_MSVC_WARNING_PRAGMA
#  if defined(__NVCC__)
#    define HPX_MSVC_NVCC
#  endif
#  define HPX_CDECL __cdecl
#endif

#if defined(__MINGW32__)
#   define HPX_WINDOWS
#   define HPX_MINGW
#endif

// Detecting CUDA compilation mode
// Detecting NVCC
#if defined(__NVCC__) || defined(__CUDACC__)
// NVCC build version numbers can be high (without limit?) so we leave it out
// from the version definition
#  define HPX_CUDA_VERSION (__CUDACC_VER_MAJOR__*100 + __CUDACC_VER_MINOR__)
#  define HPX_COMPUTE_CODE
#  if defined(__CUDA_ARCH__)
     // nvcc compiling CUDA code, device mode.
#    define HPX_COMPUTE_DEVICE_CODE
#  endif
// Detecting Clang CUDA
#elif defined(__clang__) && defined(__CUDA__)
#  define HPX_COMPUTE_CODE
#  if defined(__CUDA_ARCH__)
     // clang compiling CUDA code, device mode.
#    define HPX_COMPUTE_DEVICE_CODE
#  endif
// Detecting HIPCC
#elif defined(__HIPCC__)
#  include <hip/hip_version.h>
#  define HPX_HIP_VERSION HIP_VERSION
#  if defined(__clang__)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wdeprecated-copy"
#    pragma clang diagnostic ignored "-Wunused-parameter"
#  endif
   // Not like nvcc, the __device__ __host__ function decorators are not defined
   // by the compiler
#  include <hip/hip_runtime_api.h>
#  if defined(__clang__)
#    pragma clang diagnostic pop
#  endif
#  define HPX_COMPUTE_CODE
#  if defined(__HIP_DEVICE_COMPILE__)
     // hipclang compiling CUDA/HIP code, device mode.
#    define HPX_COMPUTE_DEVICE_CODE
#  endif
// Detecting SYCL device pass
#elif defined(HPX_HAVE_SYCL)
#  if __HIPSYCL__
    // within hipsycl the macros are not defined without including the header
#    include <CL/sycl.hpp>
#  endif
#  if defined(__SYCL_DEVICE_ONLY__)
#    define HPX_COMPUTE_DEVICE_CODE
#  endif
#endif

#if !defined(HPX_COMPUTE_DEVICE_CODE)
#  define HPX_COMPUTE_HOST_CODE
#endif

#if defined(HPX_COMPUTE_CODE)
#define HPX_DEVICE __device__
#define HPX_HOST __host__
#define HPX_CONSTANT __constant__
#else
#define HPX_DEVICE
#define HPX_HOST
#define HPX_CONSTANT
#endif
#define HPX_HOST_DEVICE HPX_HOST HPX_DEVICE


#if !defined(HPX_CDECL)
#define HPX_CDECL
#endif

#if defined(HPX_HAVE_SANITIZERS)
#  if defined(__has_feature)
#    if __has_feature(address_sanitizer)
#      define HPX_HAVE_ADDRESS_SANITIZER
#    endif
#  elif defined(__SANITIZE_ADDRESS__)   // MSVC defines this
#    define HPX_HAVE_ADDRESS_SANITIZER
#  endif
#endif
// clang-format on

// alignment for lockfree datastructures
#ifdef _MSC_VER
#if defined(_M_IX86)
#define HPX_LOCKFREE_DCAS_ALIGNMENT __declspec(align(8))
#elif defined(_M_X64) || defined(_M_IA64)
#define HPX_LOCKFREE_DCAS_ALIGNMENT __declspec(align(16))
#define HPX_LOCKFREE_PTR_COMPRESSION 1
#endif

#endif /* _MSC_VER */

#ifdef __GNUC__
#if defined(__i386__) || defined(__ppc__)
#define HPX_LOCKFREE_DCAS_ALIGNMENT
#elif defined(__x86_64__)
#define HPX_LOCKFREE_DCAS_ALIGNMENT __attribute__((aligned(16)))
#define HPX_LOCKFREE_PTR_COMPRESSION 1
#elif defined(__alpha__)
// LATER: alpha may benefit from pointer compression.
//  but what is the maximum size of the address space?
#define HPX_LOCKFREE_DCAS_ALIGNMENT
#endif
#endif /* __GNUC__ */

#if !defined(HPX_LOCKFREE_DCAS_ALIGNMENT)
#define HPX_LOCKFREE_DCAS_ALIGNMENT
#endif

#endif    // defined(DOXYGEN)
