//  Copyright (c) 2019 Ste||ar Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/static_reinit/config/defines.hpp>
#include <hpx/modules/static_reinit.hpp>

#if HPX_FUNCTIONAL_HAVE_DEPRECATION_WARNINGS
#if defined(HPX_MSVC)
#pragma message(                                                               \
    "The header hpx/util/reinitializable_static.hpp is deprecated, \
    please include hpx/modules/static_reinit.hpp instead")
#else
#warning "The header hpx/util/reinitializable_static.hpp is deprecated, \
    please include hpx/modules/static_reinit.hpp instead"
#endif
#endif
