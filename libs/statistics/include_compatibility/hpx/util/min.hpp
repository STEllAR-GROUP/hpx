//  Copyright (c) 2019 Auriane Reverdell
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/statistics/config/defines.hpp>
#include <hpx/modules/statistics.hpp>

#if HPX_STATISTICS_HAVE_DEPRECATION_WARNINGS
#if defined(HPX_MSVC)
#pragma message("The header hpx/util/min.hpp is deprecated, \
    please include hpx/modules/statistics.hpp instead")
#else
#warning "The header hpx/util/min.hpp is deprecated, \
    please include hpx/modules/statistics.hpp instead"
#endif
#endif
