//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/lcos_distributed/config/defines.hpp>
#include <hpx/channel.hpp>

#if HPX_LCOS_DISTRIBUTED_HAVE_DEPRECATION_WARNINGS
#if defined(HPX_MSVC)
#pragma message("The header hpx/lcos/channel.hpp is deprecated, \
    please include hpx/channel.hpp instead")
#else
#warning "The header hpx/lcos/channel.hpp is deprecated, \
    please include hpx/channel.hpp instead"
#endif
#endif
