//  Copyright (c) 2012 Maciej Brodowicz
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPILER_SPECIFIC_201204261048)
#define HPX_COMPILER_SPECIFIC_201204261048

#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER)

// macros to facilitate handling of compiler-specific issues
#  define HPX_GCC_VERSION (__GNUC__*10000 + __GNUC_MINOR__*100 + __GNUC__PATCHLEVEL__)

#  if HPX_GCC_VERSION >= 40600
#    define HPX_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS 1
#  endif

#  undef HPX_CLANG_VERSION

#elif defined(__clang__)

#  define HPX_CLANG_VERSION (__clang_major__*10000 + __clang_minor__*100 + __clang_patchlevel__)
#  undef HPX_GCC_VERSION

#else

#  undef HPX_GCC_VERSION
#  undef HPX_CLANG_VERSION

#endif

#endif

