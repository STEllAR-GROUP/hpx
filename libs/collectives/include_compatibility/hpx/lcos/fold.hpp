//  Copyright (c) 2019 Ste||ar Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/collectives/config/defines.hpp>
#include <hpx/include/lcos.hpp>

#if HPX_COLLECTIVES_HAVE_DEPRECATION_WARNINGS
#if defined(HPX_MSVC)
#pragma message("The header hpx/lcos/fold.hpp is deprecated, \
    please include hpx/include/lcos.hpp instead")
#else
#warning "The header hpx/lcos/fold.hpp is deprecated, \
    please include hpx/include/lcos.hpp instead"
#endif
#endif
