//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config/defines.hpp>
#include <hpx/local/config/attributes.hpp>
#include <hpx/local/config/version.hpp>
#include <hpx/preprocessor/cat.hpp>
#include <hpx/preprocessor/expand.hpp>

///////////////////////////////////////////////////////////////////////////////
// Deprecate a given functionality starting HPX V1.4
#if !defined(HPX_LOCAL_HAVE_DEPRECATION_WARNINGS_V1_4)
#define HPX_LOCAL_HAVE_DEPRECATION_WARNINGS_V1_4 1
#endif

#if (HPX_LOCAL_VERSION_FULL >= 0x010400) &&                                    \
    (HPX_LOCAL_HAVE_DEPRECATION_WARNINGS_V1_4 != 0)
#define HPX_LOCAL_DEPRECATED_MSG_V1_4                                          \
    "This functionality is deprecated starting HPX V1.4 and will be removed "  \
    "in the future. You can define "                                           \
    "HPX_LOCAL_HAVE_DEPRECATION_WARNINGS_V1_4=0 to "                           \
    "acknowledge that you have received this warning."
#define HPX_LOCAL_DEPRECATED_V1_4(x)                                           \
    HPX_LOCAL_DEPRECATED(                                                      \
        x " (" HPX_PP_EXPAND(HPX_LOCAL_DEPRECATED_MSG_V1_4) ")")
#endif

#if !defined(HPX_LOCAL_DEPRECATED_V1_4)
#define HPX_LOCAL_DEPRECATED_V1_4(x)
#endif

///////////////////////////////////////////////////////////////////////////////
// Deprecate a given functionality starting HPX V1.5
#if !defined(HPX_LOCAL_HAVE_DEPRECATION_WARNINGS_V1_5)
#define HPX_LOCAL_HAVE_DEPRECATION_WARNINGS_V1_5 1
#endif

#if (HPX_LOCAL_VERSION_FULL >= 0x010500) &&                                    \
    (HPX_LOCAL_HAVE_DEPRECATION_WARNINGS_V1_5 != 0)
#define HPX_LOCAL_DEPRECATED_MSG_V1_5                                          \
    "This functionality is deprecated starting HPX V1.5 and will be removed "  \
    "in the future. You can define "                                           \
    "HPX_LOCAL_HAVE_DEPRECATION_WARNINGS_V1_5=0 to "                           \
    "acknowledge that you have received this warning."
#define HPX_LOCAL_DEPRECATED_V1_5(x)                                           \
    HPX_LOCAL_DEPRECATED(                                                      \
        x " (" HPX_PP_EXPAND(HPX_LOCAL_DEPRECATED_MSG_V1_5) ")")
#endif

#if !defined(HPX_LOCAL_DEPRECATED_V1_5)
#define HPX_LOCAL_DEPRECATED_V1_5(x)
#endif

///////////////////////////////////////////////////////////////////////////////
// Deprecate a given functionality starting HPX V1.6
#if !defined(HPX_LOCAL_HAVE_DEPRECATION_WARNINGS_V1_6)
#define HPX_LOCAL_HAVE_DEPRECATION_WARNINGS_V1_6 1
#endif

#if (HPX_LOCAL_VERSION_FULL >= 0x010600) &&                                    \
    (HPX_LOCAL_HAVE_DEPRECATION_WARNINGS_V1_6 != 0)
#define HPX_LOCAL_DEPRECATED_MSG_V1_6                                          \
    "This functionality is deprecated starting HPX V1.6 and will be removed "  \
    "in the future. You can define "                                           \
    "HPX_LOCAL_HAVE_DEPRECATION_WARNINGS_V1_6=0 to "                           \
    "acknowledge that you have received this warning."
#define HPX_LOCAL_DEPRECATED_V1_6(x)                                           \
    HPX_LOCAL_DEPRECATED(                                                      \
        x " (" HPX_PP_EXPAND(HPX_LOCAL_DEPRECATED_MSG_V1_6) ")")
#endif

#if !defined(HPX_LOCAL_DEPRECATED_V1_6)
#define HPX_LOCAL_DEPRECATED_V1_6(x)
#endif

///////////////////////////////////////////////////////////////////////////////
// Deprecate a given functionality starting HPX V1.7
#if !defined(HPX_LOCAL_HAVE_DEPRECATION_WARNINGS_V1_7)
#define HPX_LOCAL_HAVE_DEPRECATION_WARNINGS_V1_7 1
#endif

#if (HPX_LOCAL_VERSION_FULL >= 0x010700) &&                                    \
    (HPX_LOCAL_HAVE_DEPRECATION_WARNINGS_V1_7 != 0)
#define HPX_LOCAL_DEPRECATED_MSG_V1_7                                          \
    "This functionality is deprecated starting HPX V1.7 and will be removed "  \
    "in the future. You can define "                                           \
    "HPX_LOCAL_HAVE_DEPRECATION_WARNINGS_V1_7=0 to "                           \
    "acknowledge that you have received this warning."
#define HPX_LOCAL_DEPRECATED_V1_7(x)                                           \
    HPX_LOCAL_DEPRECATED(                                                      \
        x " (" HPX_PP_EXPAND(HPX_LOCAL_DEPRECATED_MSG_V1_7) ")")
#endif

#if !defined(HPX_LOCAL_DEPRECATED_V1_7)
#define HPX_LOCAL_DEPRECATED_V1_7(x)
#endif

///////////////////////////////////////////////////////////////////////////////
// Deprecate a given functionality starting HPX V1.8
#if !defined(HPX_LOCAL_HAVE_DEPRECATION_WARNINGS_V1_8)
#define HPX_LOCAL_HAVE_DEPRECATION_WARNINGS_V1_8 1
#endif

#if (HPX_LOCAL_VERSION_FULL >= 0x010800) &&                                    \
    (HPX_LOCAL_HAVE_DEPRECATION_WARNINGS_V1_8 != 0)
#define HPX_LOCAL_DEPRECATED_MSG_V1_8                                          \
    "This functionality is deprecated starting HPX V1.8 and will be removed "  \
    "in the future. You can define "                                           \
    "HPX_LOCAL_HAVE_DEPRECATION_WARNINGS_V1_8=0 to "                           \
    "acknowledge that you have received this warning."
#define HPX_LOCAL_DEPRECATED_V1_8(x)                                           \
    HPX_LOCAL_DEPRECATED(                                                      \
        x " (" HPX_PP_EXPAND(HPX_LOCAL_DEPRECATED_MSG_V1_8) ")")
#endif

#if !defined(HPX_LOCAL_DEPRECATED_V1_8)
#define HPX_LOCAL_DEPRECATED_V1_8(x)
#endif

///////////////////////////////////////////////////////////////////////////////
// Deprecate a given functionality starting at the given version of HPX
#define HPX_LOCAL_DEPRECATED_V(major, minor, x)                                \
    HPX_PP_CAT(                                                                \
        HPX_PP_CAT(HPX_PP_CAT(HPX_LOCAL_DEPRECATED_V, major), _), minor)       \
    (x)
