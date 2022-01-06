////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

// clang-format off
#if defined(__GNUC__)
  #define HPX_SUPER_PURE  __attribute__((const))
  #define HPX_PURE        __attribute__((pure))
  #define HPX_HOT         __attribute__((hot))
  #define HPX_COLD        __attribute__((cold))
#else
  #define HPX_SUPER_PURE
  #define HPX_PURE
  #define HPX_HOT
  #define HPX_COLD
#endif
// clang-format on
