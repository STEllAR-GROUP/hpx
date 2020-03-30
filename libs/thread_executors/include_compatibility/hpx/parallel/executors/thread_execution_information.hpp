//  Copyright (c) 2019 Ste||ar Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/thread_executors/config/defines.hpp>
#include <hpx/thread_executors/executors/thread_execution_information.hpp>

#if defined(HPX_THREAD_EXECUTORS_HAVE_DEPRECATION_WARNINGS)
#if defined(HPX_MSVC)
#pragma message(                                                               \
    "The header hpx/parallel/executors/thread_execution_information.hpp is \
    deprecated, please include \
    hpx/thread_executors/thread_execution_information.hpp instead")
#else
#warning                                                                       \
    "The header hpx/parallel/executors/thread_execution_information.hpp is \
    deprecated, please include \
    hpx/thread_executors/thread_execution_information.hpp instead"
#endif
#endif
