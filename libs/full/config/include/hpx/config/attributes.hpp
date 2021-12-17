//  Copyright (c) 2017 Marcin Copik
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config/attributes.hpp>

#if HPX_HAVE_DEPRECATION_WARNINGS && !defined(HPX_INTEL_VERSION)
#define HPX_DEPRECATED_MSG                                                     \
    "This functionality is deprecated and will be removed in the future."
#define HPX_DEPRECATED(x) [[deprecated(x)]]
#endif

#if !defined(HPX_DEPRECATED)
#define HPX_DEPRECATED(x)
#endif
