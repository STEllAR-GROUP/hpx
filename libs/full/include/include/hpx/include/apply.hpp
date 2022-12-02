//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/include/post.hpp>

#if HPX_HAVE_DEPRECATION_WARNINGS
#if defined(HPX_MSVC)
#pragma message(                                                               \
    "The header hpx/include/apply.hpp is deprecated, "                         \
    "please include hpx/include/post.hpp instead."                             \
    "To disable this warning, please define HPX_HAVE_DEPRECATION_WARNINGS=0.")
#else
#warning "The header hpx/include/apply.hpp is deprecated, " \
    "please include hpx/include/post.hpp instead" \
    "To disable this warning, please define HPX_HAVE_DEPRECATION_WARNINGS=0."
#endif
#endif
