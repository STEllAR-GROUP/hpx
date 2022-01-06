////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (C) 2007, 2008 Tim Blechmann
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#if defined(DOXYGEN)

/// Hint at the compiler that \c expr is likely to be true.
#define HPX_LIKELY(expr)
/// Hint at the compiler that \c expr is likely to be false.
#define HPX_UNLIKELY(expr)

#else

// clang-format off
#if defined(__GNUC__)
  #define HPX_LIKELY(expr)    __builtin_expect(static_cast<bool>(expr), true)
  #define HPX_UNLIKELY(expr)  __builtin_expect(static_cast<bool>(expr), false)
#else
  #define HPX_LIKELY(expr)    expr
  #define HPX_UNLIKELY(expr)  expr
#endif
#endif
// clang-format on
