//  Copyright (c) 2019 STE||AR Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/command_line_handling/config/defines.hpp>
#include <hpx/modules/command_line_handling.hpp>

#if HPX_COMMAND_LINE_HANDLING_HAVE_DEPRECATION_WARNINGS
#if defined(HPX_MSVC)
#pragma message(                                                               \
    "The header hpx/util/command_line_handling.hpp is deprecated, \
    please include hpx/modules/command_line_handling.hpp instead")
#else
#warning "The header hpx/util/command_line_handling.hpp is deprecated, \
    please include hpx/modules/command_line_handling.hpp instead"
#endif
#endif
