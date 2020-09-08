//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the HPX Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.HPX.org/LICENSE_1_0.txt)

/* Copyright (C) 2001
 * Housemarque Oy
 * http://www.housemarque.com
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 */

/* Revised by Paul Mensonides (2002) */

/* See http://www.boost.org for most recent version. */

// hpxinspect:noinclude:HPX_PP_STRINGIZE

#pragma once

#if defined(DOXYGEN)
/// The \a HPX_PP_STRINGIZE macro stringizes its argument after it has been expanded.
///
/// \param X The text to be converted to a string literal
///
/// The passed argument \c X will expand to \c "X". Note that the stringizing
/// operator (#) prevents arguments from expanding. This macro circumvents this
/// shortcoming.
#define HPX_PP_STRINGIZE(X)
#else

#include <hpx/preprocessor/config.hpp>

#if HPX_PP_CONFIG_FLAGS() & HPX_PP_CONFIG_MSVC()
#define HPX_PP_STRINGIZE(text) HPX_PP_STRINGIZE_A((text))
#define HPX_PP_STRINGIZE_A(arg) HPX_PP_STRINGIZE_I arg
#elif HPX_PP_CONFIG_FLAGS() & HPX_PP_CONFIG_MWCC()
#define HPX_PP_STRINGIZE(text) HPX_PP_STRINGIZE_OO((text))
#define HPX_PP_STRINGIZE_OO(par) HPX_PP_STRINGIZE_I##par
#else
#define HPX_PP_STRINGIZE(text) HPX_PP_STRINGIZE_I(text)
#endif

#define HPX_PP_STRINGIZE_I(text) #text

#endif
