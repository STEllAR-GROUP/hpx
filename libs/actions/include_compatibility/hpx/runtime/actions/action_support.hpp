//  Copyright (c) 2019-2020 Ste||ar Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions/config/defines.hpp>
#include <hpx/actions/action_support.hpp>

#if HPX_ACTIONS_HAVE_DEPRECATION_WARNINGS
#if defined(HPX_MSVC)
#pragma message(                                                               \
    "The header hpx/runtime/actions/action_support.hpp is deprecated, \
    please include hpx/actions/action_support.hpp instead")
#else
#warning "The header hpx/runtime/actions/action_support.hpp is deprecated, \
    please include hpx/actions/action_support.hpp instead"
#endif
#endif
