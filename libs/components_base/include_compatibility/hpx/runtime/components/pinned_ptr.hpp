//  Copyright (c) 2019-2020 Ste||ar Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/components_base/config/defines.hpp>
#include <hpx/components_base/pinned_ptr.hpp>

#if HPX_COMPONENTS_BASE_HAVE_DEPRECATION_WARNINGS
#if defined(HPX_MSVC)
#pragma message(                                                               \
    "The header hpx/runtime/components/pinned_ptr.hpp is deprecated, \
    please include hpx/components_base/pinned_ptr.hpp instead")
#else
#warning                                                                       \
    "The header hpx/runtime/components_base/pinned_ptr.hpp is deprecated, \
    please include hpx/components_base/pinned_ptr.hpp instead"
#endif
#endif
