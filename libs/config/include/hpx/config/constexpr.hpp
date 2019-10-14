//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2015 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_CONFIG_CONSTEXPR_HPP
#define HPX_CONFIG_CONSTEXPR_HPP

#include <hpx/config/defines.hpp>

#if defined(DOXYGEN)
/// This macro evaluates to ``constexpr`` if the compiler supports it.
#define HPX_CONSTEXPR
/// This macro evaluates to ``constexpr`` if the compiler supports it, ``const``
/// otherwise.
#define HPX_CONSTEXPR_OR_CONST
/// This macro evaluates to ``constexpr`` if the compiler supports C++14
/// constexpr.
#define HPX_CXX14_CONSTEXPR
/// This macro evaluates to ``static :c:macro:HPX_CONSTEXPR_OR_CONST``.
#define HPX_STATIC_CONSTEXPR
#else

// clang-format off
#if defined(HPX_HAVE_CXX11_CONSTEXPR) && !defined(HPX_MSVC_NVCC) &&            \
    !(defined(__NVCC__) && defined(__clang__))
#   define HPX_CONSTEXPR constexpr
#   define HPX_CONSTEXPR_OR_CONST constexpr
#else
#   define HPX_CONSTEXPR
#   define HPX_CONSTEXPR_OR_CONST const
#endif

#ifdef HPX_HAVE_CXX14_CONSTEXPR
#   define HPX_CXX14_CONSTEXPR constexpr
#else
#   define HPX_CXX14_CONSTEXPR
#endif
// clang-format on

#define HPX_STATIC_CONSTEXPR static HPX_CONSTEXPR_OR_CONST
#endif

#endif
