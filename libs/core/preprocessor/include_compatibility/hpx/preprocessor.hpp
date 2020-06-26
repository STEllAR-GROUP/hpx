//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/preprocessor/config/defines.hpp>
#include <hpx/modules/preprocessor.hpp>

#if HPX_PREPROCESSOR_HAVE_DEPRECATION_WARNINGS
#if defined(_MSC_VER)
#pragma message("The header hpx/preprocessor.hpp is deprecated, \
    please include hpx/modules/preprocessor.hpp instead")
#else
#warning "The header hpx/preprocessor.hpp is deprecated, \
    please include hpx/modules/preprocessor.hpp instead"
#endif
#endif
