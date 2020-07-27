//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config/attributes.hpp>
#include <hpx/config/defines.hpp>
#include <hpx/config/version.hpp>
#include <hpx/preprocessor/cat.hpp>
#include <hpx/preprocessor/expand.hpp>

///////////////////////////////////////////////////////////////////////////////
// Deprecate a given functionality starting HPX V1.4
#if !defined(HPX_HAVE_DEPRECATION_WARNINGS_V1_4)
#define HPX_HAVE_DEPRECATION_WARNINGS_V1_4 1
#endif

#if (HPX_VERSION_FULL >= 010400) && (HPX_HAVE_DEPRECATION_WARNINGS_V1_4 != 0)
#define HPX_DEPRECATED_MSG_V1_4                                                \
    "This functionality is deprecated starting HPX V1.4 and will be removed "  \
    "in the future. You can define HPX_HAVE_DEPRECATION_WARNINGS_V1_4=0 to "   \
    "acknowledge that you have received this warning."
#define HPX_DEPRECATED_V1_4(x)                                                 \
    HPX_DEPRECATED(x " (" HPX_PP_EXPAND(HPX_DEPRECATED_MSG_V1_4) ")")
#endif

#if !defined(HPX_DEPRECATED_V1_4)
#define HPX_DEPRECATED_V1_4(x)
#endif

///////////////////////////////////////////////////////////////////////////////
// Deprecate a given functionality starting HPX V1.5
#if !defined(HPX_HAVE_DEPRECATION_WARNINGS_V1_5)
#define HPX_HAVE_DEPRECATION_WARNINGS_V1_5 1
#endif

#if (HPX_VERSION_FULL >= 010500) && (HPX_HAVE_DEPRECATION_WARNINGS_V1_5 != 0)
#define HPX_DEPRECATED_MSG_V1_5                                                \
    "This functionality is deprecated starting HPX V1.5 and will be removed "  \
    "in the future. You can define HPX_HAVE_DEPRECATION_WARNINGS_V1_5=0 to "   \
    "acknowledge that you have received this warning."
#define HPX_DEPRECATED_V1_5(x)                                                 \
    HPX_DEPRECATED(x " (" HPX_PP_EXPAND(HPX_DEPRECATED_MSG_V1_5) ")")
#endif

#if !defined(HPX_DEPRECATED_V1_5)
#define HPX_DEPRECATED_V1_5(x)
#endif

///////////////////////////////////////////////////////////////////////////////
// Deprecate a given functionality starting HPX V1.6
#if !defined(HPX_HAVE_DEPRECATION_WARNINGS_V1_6)
#define HPX_HAVE_DEPRECATION_WARNINGS_V1_6 0
#endif

#if (HPX_VERSION_FULL >= 010600) && (HPX_HAVE_DEPRECATION_WARNINGS_V1_6 != 0)
#define HPX_DEPRECATED_MSG_V1_6                                                \
    "This functionality is deprecated starting HPX V1.6 and will be removed "  \
    "in the future. You can define HPX_HAVE_DEPRECATION_WARNINGS_V1_6=0 to "   \
    "acknowledge that you have received this warning."
#define HPX_DEPRECATED_V1_6(x)                                                 \
    HPX_DEPRECATED(x " (" HPX_PP_EXPAND(HPX_DEPRECATED_MSG_V1_6) ")")
#endif

#if !defined(HPX_DEPRECATED_V1_6)
#define HPX_DEPRECATED_V1_6(x)
#endif

///////////////////////////////////////////////////////////////////////////////
// Deprecate a given functionality starting at the given version of HPX
#define HPX_DEPRECATED_V(major, minor, x)                                      \
    HPX_PP_CAT(HPX_PP_CAT(HPX_PP_CAT(HPX_DEPRECATED_V, major), _), minor)(x)
