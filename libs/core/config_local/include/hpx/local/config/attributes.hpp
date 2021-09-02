//  Copyright (c) 2017 Marcin Copik
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config/defines.hpp>
#include <hpx/local/config/compiler_specific.hpp>

#if defined(DOXYGEN)

/// Function attribute to tell compiler not to inline the function.
#define HPX_NOINLINE

/// Function attribute to tell compiler that the function does not return.
#define HPX_NORETURN

/// Marks an entity as deprecated. The argument \c x specifies a custom message
/// that is included in the compiler warning. For more details see
/// `<https://en.cppreference.com/w/cpp/language/attributes/deprecated>`__.
#define HPX_LOCAL_DEPRECATED(x)

/// Indicates that the fall through from the previous case label is intentional
/// and should not be diagnosed by a compiler that warns on fallthrough. For
/// more details see
/// `<https://en.cppreference.com/w/cpp/language/attributes/fallthrough>`__.
#define HPX_FALLTHROUGH

/// If a function declared nodiscard or a function returning an enumeration or
/// class declared nodiscard by value is called from a discarded-value expression
/// other than a cast to void, the compiler is encouraged to issue a warning.
/// For more details see
/// `https://en.cppreference.com/w/cpp/language/attributes/nodiscard`__.
#define HPX_NODISCARD

/// Indicates that this data member need not have an address distinct from all
/// other non-static data members of its class.
/// For more details see
/// `https://en.cppreference.com/w/cpp/language/attributes/no_unique_address`__.
#define HPX_NO_UNIQUE_ADDRESS
#else

///////////////////////////////////////////////////////////////////////////////
// clang-format off
#if defined(HPX_MSVC)
#   define HPX_NOINLINE __declspec(noinline)
#elif defined(__GNUC__)
#   if defined(__NVCC__) || defined(__CUDACC__) || defined(__HIPCC__)
        // nvcc doesn't always parse __noinline
#       define HPX_NOINLINE __attribute__ ((noinline))
#   else
#       define HPX_NOINLINE __attribute__ ((__noinline__))
#   endif
#else
#   define HPX_NOINLINE
#endif

///////////////////////////////////////////////////////////////////////////////
// handle [[noreturn]]
#define HPX_NORETURN [[noreturn]]

///////////////////////////////////////////////////////////////////////////////
// handle [[deprecated]]
#if HPX_LOCAL_HAVE_DEPRECATION_WARNINGS && !defined(HPX_INTEL_VERSION)
#  define HPX_LOCAL_DEPRECATED_MSG \
   "This functionality is deprecated and will be removed in the future."
#  define HPX_LOCAL_DEPRECATED(x) [[deprecated(x)]]
#endif

#if !defined(HPX_LOCAL_DEPRECATED)
#  define HPX_LOCAL_DEPRECATED(x)
#endif

///////////////////////////////////////////////////////////////////////////////
// handle [[fallthrough]]
#define HPX_FALLTHROUGH [[fallthrough]]

///////////////////////////////////////////////////////////////////////////////
// handle empty_bases
#if defined(_MSC_VER)
#  define HPX_EMPTY_BASES __declspec(empty_bases)
#else
#  define HPX_EMPTY_BASES
#endif

///////////////////////////////////////////////////////////////////////////////
// handle [[nodiscard]]
#define HPX_NODISCARD [[nodiscard]]
#define HPX_NODISCARD_MSG(x) [[nodiscard(x)]]

///////////////////////////////////////////////////////////////////////////////
// handle [[no_unique_address]]
#if defined(HPX_HAVE_CXX20_NO_UNIQUE_ADDRESS_ATTRIBUTE)
#   define HPX_NO_UNIQUE_ADDRESS [[no_unique_address]]
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
