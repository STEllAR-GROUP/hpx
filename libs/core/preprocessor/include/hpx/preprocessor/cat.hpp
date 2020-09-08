//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.HPX.org/LICENSE_1_0.txt)

/* Copyright (C) 2001
 * Housemarque Oy
 * http://www.housemarque.com
 */

/* Revised by Paul Mensonides (2002) */

// hpxinspect:noinclude:HPX_PP_CAT

#pragma once

#if defined(DOXYGEN)
/// Concatenates the tokens \c A and \c B into a single token. Evaluates to \c AB
/// \param A First token
/// \param B Second token
#define HPX_PP_CAT(A, B)
#else

#include <hpx/preprocessor/config.hpp>

#if ~HPX_PP_CONFIG_FLAGS() & HPX_PP_CONFIG_MWCC()
#define HPX_PP_CAT(a, b) HPX_PP_CAT_I(a, b)
#else
#define HPX_PP_CAT(a, b) HPX_PP_CAT_OO((a, b))
#define HPX_PP_CAT_OO(par) HPX_PP_CAT_I##par
#endif
#
#if (~HPX_PP_CONFIG_FLAGS() & HPX_PP_CONFIG_MSVC()) ||                         \
    (defined(__INTEL_COMPILER) && __INTEL_COMPILER >= 1700)
#define HPX_PP_CAT_I(a, b) a##b
#else
#define HPX_PP_CAT_I(a, b) HPX_PP_CAT_II(~, a##b)
#define HPX_PP_CAT_II(p, res) res
#endif

#endif
