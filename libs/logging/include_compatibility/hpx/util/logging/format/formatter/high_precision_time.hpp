//  Copyright (c) 2019 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/logging/config/defines.hpp>
#include <hpx/logging/format/formatters.hpp>

#if HPX_LOGGING_HAVE_DEPRECATION_WARNINGS
#if defined(HPX_MSVC)
#pragma message(                                                               \
    "The header hpx/util/logging/format/formatter/high_precision_time.hpp is deprecated, \
    please include hpx/logging/format/formatters.hpp instead")
#else
#warning                                                                       \
    "The header hpx/util/logging/format/formatter/high_precision_time.hpp is deprecated, \
    please include hpx/logging/format/formatters.hpp instead"
#endif
#endif
