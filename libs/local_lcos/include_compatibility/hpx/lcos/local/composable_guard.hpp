//  Copyright (c) 2019 Ste||ar Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/local_lcos/config/defines.hpp>
#include <hpx/local_lcos/composable_guard.hpp>

#if HPX_LOCAL_LCOS_HAVE_DEPRECATION_WARNINGS
#if defined(HPX_MSVC)
#pragma message(                                                               \
    "The header hpx/lcos/local/composable_guard.hpp is deprecated, \
    please include hpx/local_lcos/composable_guard.hpp instead")
#else
#warning "The header hpx/lcos/local/composable_guard.hpp is deprecated, \
    please include hpx/local_lcos/composable_guard.hpp instead"
#endif
#endif
