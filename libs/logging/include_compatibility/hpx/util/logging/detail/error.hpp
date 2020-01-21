//  Copyright (c) 2019 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/logging/config/defines.hpp>
#include <hpx/logging/detail/error.hpp>

#if defined(HPX_LOGGING_HAVE_DEPRECATION_WARNINGS)
#    if defined(HPX_MSVC)
#        pragma message(                                                       \
            "The header hpx/util/logging/detail/error.hpp is deprecated,   \
            please include hpx/logging/detail/error.hpp instead")
#    else
#        warning                                                               \
            "The header hpx/util/logging/detail/error.hpp is deprecated,       \
            please include hpx/logging/detail/error.hpp instead"
#    endif
#endif
