//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

// hpxinspect:noinclude:HPX_PP_EXPAND

/// The HPX_PP_EXPAND macro performs a double macro-expansion on its argument.
/// \param X Token to be expanded twice
///
/// This macro can be used to produce a delayed preprocessor expansion.
///
/// Example:
/// \code
/// #define MACRO(a, b, c) (a)(b)(c)
/// #define ARGS() (1, 2, 3)
///
/// HPX_PP_EXPAND(MACRO ARGS()) // expands to (1)(2)(3)
/// \endcode
#define HPX_PP_EXPAND(X) X
