//  Copyright (c) 2019 Ste||ar Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/synchronization/config/defines.hpp>
#include <hpx/shared_mutex.hpp>

#if HPX_SYNCHRONIZATION_HAVE_DEPRECATION_WARNINGS
#if defined(HPX_MSVC)
#pragma message("The header hpx/lcos/local/shared_mutex.hpp is deprecated, \
    please include hpx/shared_mutex.hpp instead")
#else
#warning "The header hpx/lcos/local/shared_mutex.hpp is deprecated, \
    please include hpx/shared_mutex.hpp instead"
#endif
#endif
