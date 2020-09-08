//  Copyright (c) 2020 Ste||ar Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions_base/config/defines.hpp>
#include <hpx/actions_base/preassigned_action_id.hpp>

#if HPX_ACTIONS_BASE_HAVE_DEPRECATION_WARNINGS
#if defined(HPX_MSVC)
#pragma message(                                                               \
    "The header hpx/runtime/actions/preassigned_action_id.hpp is deprecated, \
    please include hpx/actions_base/preassigned_action_id.hpp instead")
#else
#warning                                                                       \
    "The header hpx/runtime/actions/preassigned_action_id.hpp is deprecated, \
    please include hpx/actions_base/preassigned_action_id.hpp instead"
#endif
#endif
