//==============================================================================
//         Copyright 2003 - 2011  LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011  LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
// modified to fit HPX macro nameing scheme
#ifndef HPX_UTIL_DETAIL_STRIP_HPP_INCLUDED
#define HPX_UTIL_DETAIL_STRIP_HPP_INCLUDED

#include <boost/preprocessor/cat.hpp>

/*!
 * \file
 * \brief Defines the HPX_UTIL_STRIP macro
 */

#define HPX_UTIL_DETAILS_APPLY(macro, args)   HPX_UTIL_DETAILS_APPLY_I(macro, args)
#define HPX_UTIL_DETAILS_APPLY_I(macro, args) macro args
#define HPX_UTIL_DETAILS_STRIP_PARENS_I(...) 1,1
#define HPX_UTIL_DETAILS_EVAL(test, x) HPX_UTIL_DETAILS_EVAL_I(test, x)
#define HPX_UTIL_DETAILS_EVAL_I(test, x) HPX_UTIL_DETAILS_MAYBE_STRIP_PARENS(HPX_UTIL_DETAILS_TEST_ARITY test, x)
#define HPX_UTIL_DETAILS_TEST_ARITY(...) HPX_UTIL_DETAILS_APPLY(HPX_UTIL_DETAILS_TEST_ARITY_I, (__VA_ARGS__, 2, 1, 0))
#define HPX_UTIL_DETAILS_TEST_ARITY_I(a,b,c,...) c
#define HPX_UTIL_DETAILS_MAYBE_STRIP_PARENS(cond, x) HPX_UTIL_DETAILS_MAYBE_STRIP_PARENS_I(cond, x)
#define HPX_UTIL_DETAILS_MAYBE_STRIP_PARENS_I(cond, x) BOOST_PP_CAT(HPX_UTIL_DETAILS_MAYBE_STRIP_PARENS_, cond)(x)
#define HPX_UTIL_DETAILS_MAYBE_STRIP_PARENS_1(x) x
#define HPX_UTIL_DETAILS_MAYBE_STRIP_PARENS_2(x) HPX_UTIL_DETAILS_APPLY(HPX_UTIL_DETAILS_MAYBE_STRIP_PARENS_2_I, x)
#define HPX_UTIL_DETAILS_MAYBE_STRIP_PARENS_2_I(...) __VA_ARGS__

//==============================================================================
/*!
 * \ingroup preprocessor
 * For any symbol \c X, this macro returns the same symbol from which potential
 * outer parens have been removed. If no outer parens are found, this macros
 * evaluates to \c X itself without error.
 *
 * The original implementation of this macro is from Steven Watanbe as shown
 * in http://article.gmane.org/gmane.comp.lib.boost.user/61011
 *
 * \param X Symbol to strip parens from
 *
 * \par Example Usage:
 *
 * \include pp_strip.cpp
 *
 * This produces the following output
 * \code
 * (no parens)
 * (with parens)
 * \endcode
 */
//==============================================================================
#define HPX_UTIL_STRIP(X)                                                     \
    HPX_UTIL_DETAILS_EVAL((HPX_UTIL_DETAILS_STRIP_PARENS_I X), X)             \
/**/

#endif
