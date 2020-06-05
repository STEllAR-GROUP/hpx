//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/testing/config/defines.hpp>
#include <hpx/modules/testing.hpp>

#if HPX_TESTING_HAVE_DEPRECATION_WARNINGS
#if defined(HPX_MSVC)
#pragma message("The header hpx/testing.hpp is deprecated, \
    please include hpx/modules/testing.hpp instead")
#else
#warning "The header hpx/testing.hpp is deprecated, \
    please include hpx/modules/testing.hpp instead"
#endif
#endif
