//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_base/config/defines.hpp>
#include <hpx/async_base/traits/is_launch_policy.hpp>

#if HPX_ASYNC_BASE_HAVE_DEPRECATION_WARNINGS
#if defined(HPX_MSVC)
#pragma message("The header hpx/traits/is_launch_policy.hpp is deprecated, \
    please include hpx/async_base/traits/is_launch_policy.hpp instead")
#else
#warning "The header hpx/traits/is_launch_policy.hpp is deprecated, \
    please include hpx/async_base/traits/is_launch_policy.hpp instead"
#endif
#endif
