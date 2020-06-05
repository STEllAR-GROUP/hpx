//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution/config/defines.hpp>
#include <hpx/execution/detail/sync_launch_policy_dispatch.hpp>

#if HPX_EXECUTION_HAVE_DEPRECATION_WARNINGS
#if defined(HPX_MSVC)
#pragma message(                                                               \
    "The header hpx/sync_launch_policy_dispatch.hpp is deprecated, please \
    include hpx/execution/detail/sync_launch_policy_dispatch.hpp instead")
#else
#warning                                                                       \
    "The header hpx/sync_launch_policy_dispatch.hpp is deprecated, please \
    include hpx/execution/detail/sync_launch_policy_dispatch.hpp instead"
#endif
#endif
