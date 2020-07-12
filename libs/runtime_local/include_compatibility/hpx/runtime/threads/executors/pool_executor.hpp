//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/runtime_local/config/defines.hpp>
#include <hpx/include/parallel_executors.hpp>

#if HPX_RUNTIME_LOCAL_HAVE_DEPRECATION_WARNINGS
#if defined(HPX_MSVC)
#pragma message(                                                               \
    "The header hpx/runtime/threads/executors/pool_executor.hpp is deprecated, \
    please include hpx/include/parallel_executors.hpp instead")
#else
#warning                                                                       \
    "The header hpx/runtime/threads/executors/pool_executor.hpp is deprecated, \
    please include hpx/include/parallel_executors.hpp instead"
#endif
#endif
