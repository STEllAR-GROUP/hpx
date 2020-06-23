//  Copyright (c) 2019 STE||AR Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/runtime_configuration/config/defines.hpp>
#include <hpx/modules/runtime_configuration.hpp>

#if HPX_RUNTIME_CONFIGURATION_HAVE_DEPRECATION_WARNINGS
#if defined(HPX_MSVC)
#pragma message(                                                               \
    "The header hpx/runtime_configuration/runtime_mode.hpp is deprecated, \
    please include hpx/modules/runtime_configuration.hpp instead")
#else
#warning                                                                       \
    "The header hpx/runtime_configuration/runtime_mode.hpp is deprecated, \
    please include hpx/modules/runtime_configuration.hpp instead"
#endif
#endif
