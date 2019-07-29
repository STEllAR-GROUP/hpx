//  Copyright (c) 2007-2018 Hartmut Kaiser
//  Copyright (c) 2018 Thomas Heller
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_CONFIG_THREADS_STACK_HPP
#define HPX_CONFIG_THREADS_STACK_HPP

#include <hpx/config/debug.hpp>
#include <hpx/config/defines.hpp>
#include <hpx/config/compiler_specific.hpp>

///////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_THREADS_STACK_OVERHEAD)
#  if defined(HPX_DEBUG)
#    if defined(HPX_GCC_VERSION)
#      define HPX_THREADS_STACK_OVERHEAD 0x3000
#    else
#      define HPX_THREADS_STACK_OVERHEAD 0x2800
#    endif
#  else
#    if defined(HPX_INTEL_VERSION)
#      define HPX_THREADS_STACK_OVERHEAD 0x2800
#    else
#      if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 40900 && HPX_GCC_VERSION < 50000
#        define HPX_THREADS_STACK_OVERHEAD 0x1000
#      else
#        define HPX_THREADS_STACK_OVERHEAD 0x800
#      endif
#    endif
#  endif
#endif

#if !defined(HPX_SMALL_STACK_SIZE)
#  if defined(__has_feature)
#    if __has_feature(address_sanitizer)
#      define HPX_SMALL_STACK_SIZE  0x40000       // 256kByte
#    endif
#  endif
#endif

#if !defined(HPX_SMALL_STACK_SIZE)
#  if defined(HPX_WINDOWS) && !defined(HPX_HAVE_GENERIC_CONTEXT_COROUTINES)
#    define HPX_SMALL_STACK_SIZE    0x4000        // 16kByte
#  else
#    if defined(HPX_DEBUG)
#      define HPX_SMALL_STACK_SIZE  0x20000       // 128kByte
#    else
#      if defined(__powerpc__) || defined(__INTEL_COMPILER)
#         define HPX_SMALL_STACK_SIZE  0x20000       // 128kByte
#      else
#         define HPX_SMALL_STACK_SIZE  0x10000        // 64kByte
#      endif
#    endif
#  endif
#endif

#if !defined(HPX_MEDIUM_STACK_SIZE)
#  define HPX_MEDIUM_STACK_SIZE   0x0020000       // 128kByte
#endif
#if !defined(HPX_LARGE_STACK_SIZE)
#  define HPX_LARGE_STACK_SIZE    0x0200000       // 2MByte
#endif
#if !defined(HPX_HUGE_STACK_SIZE)
#  define HPX_HUGE_STACK_SIZE     0x2000000       // 32MByte
#endif

#endif
