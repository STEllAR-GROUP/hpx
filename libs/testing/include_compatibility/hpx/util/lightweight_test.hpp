//  Copyright (c) 2019 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/testing/config/defines.hpp>
#include <hpx/testing.hpp>

#if defined(HPX_TESTING_HAVE_DEPRECATION_WARNINGS)
#if defined(HPX_MSVC)
#pragma message(                                                               \
    "The header hpx/util/lightweight_test.hpp is deprecated,                   \
    please include hpx/testing.hpp instead")
#else
#warning                                                                       \
    "The header hpx/util/lightweight_test.hpp is deprecated,                   \
    please include hpx/testing.hpp instead"
#endif
#endif
