//  Copyright (c) 2025 Haokun Wu
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

module;

// Include all system headers in global module fragment to prevent ODR violations
#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>

// Define module-specific macros before including config
#define HPX_BUILD_MODULE
#include <hpx/config.hpp>

export module HPX.Core;

#include <hpx/modules/version.hpp>