//  Copyright (c) 2019 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/preprocessor/config/defines.hpp>
#include <hpx/preprocessor/cat.hpp>

#if defined(HPX_PREPROCESSOR_HAVE_DEPRECATION_WARNINGS)
#if defined(_MSC_VER)
#pragma message(                                                               \
    "The header hpx/util/detail/pp/cat.hpp is deprecated,                      \
    please include hpx/preprocessor/cat.hpp instead")
#else
#warning                                                                       \
    "The header hpx/util/detail/pp/cat.hpp is deprecated,                      \
    please include hpx/preprocessor/cat.hpp instead"
#endif
#endif
