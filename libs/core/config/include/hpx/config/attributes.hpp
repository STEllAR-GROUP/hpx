//  Copyright (c) 2017 Marcin Copik
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config/compiler_specific.hpp>
#include <hpx/config/defines.hpp>

#if defined(DOXYGEN)

/// Function attribute to tell compiler not to inline the function.
#define HPX_NOINLINE

/// Indicates that this data member need not have an address distinct from all
/// other non-static data members of its class.
///
/// For more details see
/// `https://en.cppreference.com/w/cpp/language/attributes/no_unique_address`__.
///
/// For details about the support on MSVC, see
/// `https://devblogs.microsoft.com/cppblog/msvc-cpp20-and-the-std-cpp20-switch/`__.
#define HPX_NO_UNIQUE_ADDRESS
#else

///////////////////////////////////////////////////////////////////////////////
// clang-format off
#if defined(HPX_MSVC)
#   define HPX_NOINLINE __declspec(noinline)
#elif defined(__GNUC__)
#  if defined(__NVCC__) || defined(__CUDACC__) || defined(__HIPCC__)
     // nvcc doesn't always parse __noinline
#    define HPX_NOINLINE __attribute__ ((noinline))
#  else
#    define HPX_NOINLINE __attribute__ ((__noinline__))
#  endif
#else
#  define HPX_NOINLINE
#endif

///////////////////////////////////////////////////////////////////////////////
// handle [[no_unique_address]]
#if defined(HPX_HAVE_MSVC_NO_UNIQUE_ADDRESS_ATTRIBUTE) ||                      \
    defined(HPX_HAVE_CXX20_NO_UNIQUE_ADDRESS_ATTRIBUTE)
#  if defined(HPX_MSVC)
#    define HPX_NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]
#  else
#    define HPX_NO_UNIQUE_ADDRESS [[no_unique_address]]
#  endif
#else
#   define HPX_NO_UNIQUE_ADDRESS
#endif

///////////////////////////////////////////////////////////////////////////////
// handle empty_bases
#if defined(_MSC_VER)
#  define HPX_EMPTY_BASES __declspec(empty_bases)
#else
#  define HPX_EMPTY_BASES
#endif

// clang-format on

#endif
