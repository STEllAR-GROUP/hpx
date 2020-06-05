//  Copyright (c) 2019 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/schedulers/config/defines.hpp>
#include <hpx/schedulers/deadlock_detection.hpp>

#if HPX_SCHEDULERS_HAVE_DEPRECATION_WARNINGS
#if defined(HPX_MSVC)
#pragma message(                                                               \
    "The header hpx/runtime/threads/policies/deadlock_detection.hpp \
    is deprecated, please include \
    hpx/schedulers/deadlock_detection.hpp instead")
#else
#warning "The header hpx/runtime/threads/policies/deadlock_detection.hpp \
    is deprecated, please include \
    hpx/schedulers/deadlock_detection.hpp instead"
#endif
#endif
