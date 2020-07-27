//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config/attributes.hpp>
#include <hpx/config/defines.hpp>
#include <hpx/config/version.hpp>

///////////////////////////////////////////////////////////////////////////////
// Deprecate a given functionality starting HPX V1.5
#if HPX_VERSION_FULL >= 010500 && defined(HPX_HAVE_DEPRECATION_WARNINGS_V1_5)
#define HPX_DEPRECATED_MSG_V1_5                                                \
    "This functionality is deprecated starting HPX V1.5 and will be removed "  \
    "in the future. Disable this warning by adding this to your cmake "        \
    "invocation: -DHPX_WITH_DEPRECATION_WARNINGS_V1_5=Off"
#define HPX_DEPRECATED_V1_5(x)                                                 \
    HPX_DEPRECATED(x " (" HPX_DEPRECATED_MSG_V1_5 ")")
#endif

#if !defined(HPX_DEPRECATED_V1_5)
#define HPX_DEPRECATED_V1_5(x)
#endif

///////////////////////////////////////////////////////////////////////////////
// Deprecate a given functionality starting HPX V1.6
#if HPX_VERSION_FULL >= 010600 && defined(HPX_HAVE_DEPRECATION_WARNINGS_V1_6)
#define HPX_DEPRECATED_MSG_V1_6                                                \
    "This functionality is deprecated starting HPX V1.6 and will be removed "  \
    "in the future. Disable this warning by adding this to your cmake "        \
    "invocation: -DHPX_WITH_DEPRECATION_WARNINGS_V1_6=Off"
#define HPX_DEPRECATED_V1_6(x)                                                 \
    HPX_DEPRECATED(x " (" HPX_DEPRECATED_MSG_V1_6 ")")
#endif

#if !defined(HPX_DEPRECATED_V1_6)
#define HPX_DEPRECATED_V1_6(x)
#endif
