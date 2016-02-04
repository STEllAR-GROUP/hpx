//  Copyright (c) 2012 Maciej Brodowicz
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPILER_SPECIFIC_201204261048)
#define HPX_COMPILER_SPECIFIC_201204261048

#if defined(__GNUC__)

// macros to facilitate handling of compiler-specific issues
#  define HPX_GCC_VERSION (__GNUC__*10000 + __GNUC_MINOR__*100 + __GNUC_PATCHLEVEL__)

#  if HPX_GCC_VERSION >= 40600
#    define HPX_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS 1
#  endif

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
# endif
#else

#  undef HPX_INTEL_VERSION

#endif

// Identify if we compile for the MIC
#if defined(__MIC)
#   define HPX_NATIVE_MIC
#endif

#if defined(_MSC_VER)
#   define HPX_MSVC _MSC_VER
#   define HPX_WINDOWS
#endif

#if defined(__MINGW32__)
#   define HPX_WINDOWS
#endif

#endif

