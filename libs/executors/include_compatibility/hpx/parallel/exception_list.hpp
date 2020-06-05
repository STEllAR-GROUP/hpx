//  Copyright (c) 2019 Ste||ar Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/executors/config/defines.hpp>
#include <hpx/executors/exception_list.hpp>

#if HPX_EXECUTORS_HAVE_DEPRECATION_WARNINGS
#if defined(HPX_MSVC)
#pragma message("The header hpx/parallel/exception_list.hpp is deprecated, \
    please include hpx/executors/exception_list.hpp instead")
#else
#warning "The header hpx/parallel/exception_list.hpp is deprecated, \
    please include hpx/executors/exception_list.hpp instead"
#endif
#endif
