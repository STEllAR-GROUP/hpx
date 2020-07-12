//  Copyright (c) 2019 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/timing/config/defines.hpp>
#include <hpx/chrono.hpp>

#if HPX_TIMING_HAVE_DEPRECATION_WARNINGS
#if defined(HPX_MSVC)
#pragma message(                                                               \
    "The header hpx/util/high_resolution_clock.hpp is deprecated, \
    please include hpx/chrono.hpp instead")
#else
#warning "The header hpx/util/high_resolution_clock.hpp is deprecated, \
    please include hpx/chrono.hpp instead"
#endif
#endif
