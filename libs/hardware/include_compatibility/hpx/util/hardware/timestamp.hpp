//  Copyright (c) 2019 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/hardware/config/defines.hpp>
#include <hpx/modules/hardware.hpp>

#if HPX_HARDWARE_HAVE_DEPRECATION_WARNINGS
#if defined(HPX_MSVC)
#pragma message(                                                               \
    "The header hpx/util/hardware/timestamp.hpp is deprecated,                 \
    please include hpx/modules/hardware.hpp instead")
#else
#warning                                                                       \
    "The header hpx/util/hardware/timestamp.hpp is deprecated,                 \
    please include hpx/modules/hardware.hpp instead"
#endif
#endif
