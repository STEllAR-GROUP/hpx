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
#  define HPX_COMPUTE_CODE
#  if defined(__CUDA_ARCH__)
     // nvcc compiling CUDA code, device mode.
#    define HPX_COMPUTE_DEVICE_CODE
#  else
     // nvcc compiling CUDA code, host mode.
#    define HPX_COMPUTE_HOST_CODE
#  endif
// Detecting Clang CUDA
#elif defined(__clang__) && defined(__CUDA__)
#  define HPX_COMPUTE_CODE
#  if defined(__CUDA_ARCH__)
     // clang compiling CUDA code, device mode.
#    define HPX_COMPUTE_DEVICE_CODE
#  else
     // clang compiling CUDA code, host mode.
#    define HPX_COMPUTE_HOST_CODE
#  endif
// Detecting HIPCC
#elif defined(__HIPCC__)
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
#  else
     // clang compiling CUDA/HIP code, host mode.
#    define HPX_COMPUTE_HOST_CODE
#  endif
#endif

#if defined(HPX_COMPUTE_DEVICE_CODE) || defined(HPX_COMPUTE_HOST_CODE)
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

#if defined(HPX_HAVE_SANITIZERS) && defined(__has_feature)
#  if __has_feature(address_sanitizer)
#    define HPX_HAVE_ADDRESS_SANITIZER
#  endif
#endif
// clang-format on
#endif
