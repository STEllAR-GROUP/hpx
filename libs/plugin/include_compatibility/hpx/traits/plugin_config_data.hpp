//  Copyright (c) 2019 Ste||ar Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/plugin/config/defines.hpp>
#include <hpx/modules/plugin.hpp>

#if HPX_PLUGIN_HAVE_DEPRECATION_WARNINGS
#if defined(HPX_MSVC)
#pragma message("The header hpx/traits/plugin_config_data.hpp is deprecated, \
    please include hpx/modules/plugin.hpp instead")
#else
#warning "The header hpx/traits/plugin_config_data.hpp is deprecated, \
    please include hpx/modules/plugin.hpp instead"
#endif
#endif
