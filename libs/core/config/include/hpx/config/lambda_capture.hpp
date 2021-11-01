//  Copyright (c) 2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nodeprecated_macros

#pragma once

#include <hpx/config/defines.hpp>

#include <utility>

/// Evaluates to ``var = HPX_FORWARD(decltype(var), var)`` if the compiler
/// supports C++14 Lambdas. Defaults to ``var``.
///
/// This macro is deprecated. Prefer using ``var =
/// HPX_FORWARD(decltype((var)), var)`` directly instead.
#define HPX_CAPTURE_FORWARD(var) var = HPX_FORWARD(decltype((var)), var)

///  Evaluates to ``var = HPX_MOVE(var)`` if the compiler supports C++14
/// Lambdas. Defaults to `var`.
///
/// This macro is deprecated. Prefer using ``var = HPX_MOVE(var)`` directly
/// instead.
#define HPX_CAPTURE_MOVE(var) var = HPX_MOVE(var)
