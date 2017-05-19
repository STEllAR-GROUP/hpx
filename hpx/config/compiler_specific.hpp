//  Copyright (c) 2012 Maciej Brodowicz
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPILER_SPECIFIC_201204261048)
#define HPX_COMPILER_SPECIFIC_201204261048

#include <hpx/config/defines.hpp>

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

#if (defined(__NVCC__) || defined(__CUDACC__)) && defined(HPX_HAVE_CUDA)
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

#endif

