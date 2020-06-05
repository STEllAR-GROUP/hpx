//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/timing/config/defines.hpp>
#include <hpx/timing/tick_counter.hpp>

#if HPX_TIMING_HAVE_DEPRECATION_WARNINGS
#if defined(HPX_MSVC)
#pragma message("The header hpx/util/tick_counter.hpp is deprecated, \
    please include hpx/timing/tick_counter.hpp instead")
#else
#warning "The header hpx/util/tick_counter.hpp is deprecated, \
    please include hpx/timing/tick_counter.hpp instead"
#endif
#endif
