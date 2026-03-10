//  Copyright (c) 2020-2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CXX_MODULES) && !defined(HPX_FULL_EXPORTS) &&             \
    !defined(HPX_COMPILE_BMI)
import HPX.Full;
#else

#include <hpx/modules/naming_base.hpp>
#endif
