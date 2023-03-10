//==============================================================================
//         Copyright 2003 - 2011  LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011  LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//  SPDX-License-Identifier: BSL-1.0
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
// modified to fit HPX macro naming scheme

// hpxinspect:noinclude:HPX_PP_STRIP_PARENS

#pragma once

#include <hpx/preprocessor/cat.hpp>

/// \cond NOINTERNAL
#define HPX_PP_DETAILS_APPLY(macro, args) HPX_PP_DETAILS_APPLY_I(macro, args)
#define HPX_PP_DETAILS_APPLY_I(macro, args) macro args
#define HPX_PP_DETAILS_STRIP_PARENS_I(...) 1, 1
#define HPX_PP_DETAILS_EVAL(test, x) HPX_PP_DETAILS_EVAL_I(test, x)
#define HPX_PP_DETAILS_EVAL_I(test, x)                                         \
    HPX_PP_DETAILS_MAYBE_STRIP_PARENS(HPX_PP_DETAILS_TEST_ARITY test, x)
#define HPX_PP_DETAILS_TEST_ARITY(...)                                         \
    HPX_PP_DETAILS_APPLY(HPX_PP_DETAILS_TEST_ARITY_I, (__VA_ARGS__, 2, 1, 0))
#define HPX_PP_DETAILS_TEST_ARITY_I(a, b, c, ...) c
#define HPX_PP_DETAILS_MAYBE_STRIP_PARENS(cond, x)                             \
    HPX_PP_DETAILS_MAYBE_STRIP_PARENS_I(cond, x)
#define HPX_PP_DETAILS_MAYBE_STRIP_PARENS_I(cond, x)                           \
    HPX_PP_CAT(HPX_PP_DETAILS_MAYBE_STRIP_PARENS_, cond)(x)
#define HPX_PP_DETAILS_MAYBE_STRIP_PARENS_1(x) x
#define HPX_PP_DETAILS_MAYBE_STRIP_PARENS_2(x)                                 \
    HPX_PP_DETAILS_APPLY(HPX_PP_DETAILS_MAYBE_STRIP_PARENS_2_I, x)
#define HPX_PP_DETAILS_MAYBE_STRIP_PARENS_2_I(...) __VA_ARGS__
/// \endcond

//==============================================================================
/*!
 * For any symbol \c X, this macro returns the same symbol from which potential
 * outer parens have been removed. If no outer parens are found, this macros
 * evaluates to \c X itself without error.
 *
 * The original implementation of this macro is from Steven Watanbe as shown
 * in http://boost.2283326.n4.nabble.com/preprocessor-removing-parentheses-td2591973.html#a2591976
 *
 * \param X Symbol to strip parens from
 *
 * \par Example Usage:
 *
 * \code
 * HPX_PP_STRIP_PARENS(no_parens)
 * HPX_PP_STRIP_PARENS((with_parens))
 * \endcode
 *
 * This produces the following output
 * \code
 * no_parens
 * with_parens
 * \endcode
 */
//==============================================================================
#define HPX_PP_STRIP_PARENS(X)                                                 \
    HPX_PP_DETAILS_EVAL((HPX_PP_DETAILS_STRIP_PARENS_I X), X)                  \
    /**/
