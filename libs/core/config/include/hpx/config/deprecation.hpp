//  Copyright (c) 2020-2022 Hartmut Kaiser
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
// Deprecate a given functionality starting HPX V1.8
#if !defined(HPX_HAVE_DEPRECATION_WARNINGS_V1_8)
#define HPX_HAVE_DEPRECATION_WARNINGS_V1_8 1
#endif

#if (HPX_VERSION_FULL >= 0x010800) && (HPX_HAVE_DEPRECATION_WARNINGS_V1_8 != 0)
#define HPX_DEPRECATED_MSG_V1_8                                                \
    "This functionality is deprecated starting HPX V1.8 and will be removed "  \
    "in the future. You can define HPX_HAVE_DEPRECATION_WARNINGS_V1_8=0 to "   \
    "acknowledge that you have received this warning."
#define HPX_DEPRECATED_V1_8(x)                                                 \
    [[deprecated(x " (" HPX_PP_EXPAND(HPX_DEPRECATED_MSG_V1_8) ")")]]
#endif

#if !defined(HPX_DEPRECATED_V1_8)
#define HPX_DEPRECATED_V1_8(x)
#endif

///////////////////////////////////////////////////////////////////////////////
// Deprecate a given functionality starting HPX V1.9
#if !defined(HPX_HAVE_DEPRECATION_WARNINGS_V1_9)
#define HPX_HAVE_DEPRECATION_WARNINGS_V1_9 1
#endif

#if (HPX_VERSION_FULL >= 0x010900) && (HPX_HAVE_DEPRECATION_WARNINGS_V1_9 != 0)
#define HPX_DEPRECATED_MSG_V1_9                                                \
    "This functionality is deprecated starting HPX V1.9 and will be removed "  \
    "in the future. You can define HPX_HAVE_DEPRECATION_WARNINGS_V1_9=0 to "   \
    "acknowledge that you have received this warning."
#define HPX_DEPRECATED_V1_9(x)                                                 \
    [[deprecated(x " (" HPX_PP_EXPAND(HPX_DEPRECATED_MSG_V1_9) ")")]]
#endif

#if !defined(HPX_DEPRECATED_V1_9)
#define HPX_DEPRECATED_V1_9(x)
#endif

///////////////////////////////////////////////////////////////////////////////
// Deprecate a given functionality starting HPX V1.10
#if !defined(HPX_HAVE_DEPRECATION_WARNINGS_V1_10)
#define HPX_HAVE_DEPRECATION_WARNINGS_V1_10 1
#endif

#if (HPX_VERSION_FULL >= 0x011000) && (HPX_HAVE_DEPRECATION_WARNINGS_V1_10 != 0)
#define HPX_DEPRECATED_MSG_V1_10                                               \
    "This functionality is deprecated starting HPX V1.10 and will be removed " \
    "in the future. You can define HPX_HAVE_DEPRECATION_WARNINGS_V1_9=0 to "   \
    "acknowledge that you have received this warning."
#define HPX_DEPRECATED_V1_10(x)                                                \
    [[deprecated(x " (" HPX_PP_EXPAND(HPX_DEPRECATED_MSG_V1_10) ")")]]
#endif

#if !defined(HPX_DEPRECATED_V1_10)
#define HPX_DEPRECATED_V1_10(x)
#endif

///////////////////////////////////////////////////////////////////////////////
// Deprecate a given functionality starting at the given version of HPX
#define HPX_DEPRECATED_V(major, minor, x)                                      \
    HPX_PP_CAT(HPX_PP_CAT(HPX_PP_CAT(HPX_DEPRECATED_V, major), _), minor)(x)
