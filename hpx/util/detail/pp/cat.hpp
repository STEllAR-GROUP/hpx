//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the HPX Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.HPX.org/LICENSE_1_0.txt)

/* Copyright (C) 2001
 * Housemarque Oy
 * http://www.housemarque.com
 */

/* Revised by Paul Mensonides (2002) */

#ifndef HPX_UTIL_DETAIL_CAT_HPP_INCLUDED
#define HPX_UTIL_DETAIL_CAT_HPP_INCLUDED

#include <hpx/util/detail/pp/config.hpp>

# if ~HPX_PP_CONFIG_FLAGS() & HPX_PP_CONFIG_MWCC()
#    define HPX_PP_CAT(a, b) HPX_PP_CAT_I(a, b)
# else
#    define HPX_PP_CAT(a, b) HPX_PP_CAT_OO((a, b))
#    define HPX_PP_CAT_OO(par) HPX_PP_CAT_I ## par
# endif
#
# if (~HPX_PP_CONFIG_FLAGS() & HPX_PP_CONFIG_MSVC()) || (defined(__INTEL_COMPILER) && __INTEL_COMPILER >= 1700)
#    define HPX_PP_CAT_I(a, b) a ## b
# else
#    define HPX_PP_CAT_I(a, b) HPX_PP_CAT_II(~, a ## b)
#    define HPX_PP_CAT_II(p, res) res
# endif

#endif

