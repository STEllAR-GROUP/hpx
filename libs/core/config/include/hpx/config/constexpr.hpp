//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2015 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nodeprecated_macros

#pragma once

#include <hpx/config/compiler_specific.hpp>
#include <hpx/config/defines.hpp>

/// This macro evaluates to ``inline constexpr`` for host code and
// ``device static const`` for device code
#if defined(HPX_COMPUTE_DEVICE_CODE)
#define HPX_HOST_DEVICE_INLINE_CONSTEXPR_VARIABLE HPX_DEVICE static const
#else
#define HPX_HOST_DEVICE_INLINE_CONSTEXPR_VARIABLE inline constexpr
#endif
