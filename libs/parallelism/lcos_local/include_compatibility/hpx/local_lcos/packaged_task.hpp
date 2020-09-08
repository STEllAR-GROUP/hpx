//  Copyright (c) 2019 Ste||ar Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/lcos_local/config/defines.hpp>
#include <hpx/lcos_local/packaged_task.hpp>

#if HPX_LCOS_LOCAL_HAVE_DEPRECATION_WARNINGS
#if defined(HPX_MSVC)
#pragma message("The header hpx/local_lcos/packaged_task.hpp is deprecated, \
    please include hpx/lcos_local/packaged_task.hpp instead")
#else
#warning "The header hpx/local_lcos/packaged_task.hpp is deprecated, \
    please include hpx/lcos_local/packaged_task.hpp instead"
#endif
#endif
