//  Copyright (c) 2025 Haokun Wu
//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config/defines.hpp>

// Include all standard library headers in global module fragment to prevent ODR
// violations.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(HPX_HAVE_CXX20_STD_ENDIAN)
#include <bit>
#endif
