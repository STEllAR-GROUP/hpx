//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2015 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nodeprecated_macros

#pragma once

#include <hpx/config/defines.hpp>

/// This macro evaluates to ``constexpr`` if the compiler supports it.
///
/// This macro is deprecated. It is always replaced with the ``constexpr``
/// keyword. Prefer using ``constexpr`` directly instead.
#define HPX_CONSTEXPR constexpr

/// This macro evaluates to ``constexpr`` if the compiler supports it, ``const``
/// otherwise.
///
/// This macro is deprecated. It is always replaced with the ``constexpr``
/// keyword. Prefer using ``constexpr`` directly instead.
#define HPX_CONSTEXPR_OR_CONST constexpr

/// This macro evaluates to ``inline constexpr`` if the compiler supports it,
/// ``constexpr`` otherwise.
#ifdef HPX_HAVE_CXX17_INLINE_CONSTEXPR_VARIABLE
#define HPX_INLINE_CONSTEXPR_VARIABLE inline constexpr
#else
#define HPX_INLINE_CONSTEXPR_VARIABLE constexpr
#endif

/// This macro evaluates to ``static constexpr`` if the compiler supports it,
/// ``static const`` otherwise.
///
/// This macro is deprecated. It is always replaced with the ``static
/// constexpr`` keyword. Prefer using ``static constexpr`` directly instead.
#define HPX_STATIC_CONSTEXPR static constexpr
