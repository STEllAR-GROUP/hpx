//  Copyright (c) 2020 Ste||ar Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_combinators/config/defines.hpp>
#include <hpx/future.hpp>

#if HPX_ASYNC_COMBINATORS_HAVE_DEPRECATION_WARNINGS
#if defined(HPX_MSVC)
#pragma message("The header hpx/lcos/when_each.hpp is deprecated, \
    please include hpx/future.hpp instead")
#else
#warning "The header hpx/lcos/when_each.hpp is deprecated, \
    please include hpx/future.hpp instead"
#endif
#endif
