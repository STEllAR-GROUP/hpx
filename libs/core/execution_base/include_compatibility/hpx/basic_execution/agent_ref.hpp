//  Copyright (c) 2019 Ste||ar Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution_base/config/defines.hpp>
#include <hpx/modules/execution_base.hpp>

#if HPX_EXECUTION_BASE_HAVE_DEPRECATION_WARNINGS
#if defined(HPX_MSVC)
#pragma message("The header hpx/basic_execution/agent_ref.hpp is deprecated, \
    please include hpx/modules/execution_base.hpp instead")
#else
#warning "The header hpx/basic_execution/agent_ref.hpp is deprecated, \
    please include hpx/modules/execution_base.hpp instead"
#endif
#endif
