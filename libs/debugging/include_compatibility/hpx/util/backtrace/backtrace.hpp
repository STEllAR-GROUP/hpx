//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/debugging/config/defines.hpp>
#include <hpx/debugging/backtrace/backtrace.hpp>

#if HPX_DEBUGGING_HAVE_DEPRECATION_WARNINGS
#if defined(HPX_MSVC)
#pragma message("The header hpx/util/backtrace/backtrace.hpp is deprecated, \
    please include hpx/debugging/backtrace/backtrace.hpp instead")
#else
#warning "The header hpx/util/backtrace/backtrace.hpp is deprecated, \
    please include hpx/debugging/backtrace/backtrace.hpp instead"
#endif
#endif
