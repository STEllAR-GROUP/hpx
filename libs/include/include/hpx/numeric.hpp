//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/parallel/numeric.hpp>

#include <hpx/parallel/container_numeric.hpp>

namespace hpx {
    using hpx::parallel::adjacent_difference;
    using hpx::parallel::exclusive_scan;
    using hpx::parallel::inclusive_scan;
    using hpx::parallel::transform_exclusive_scan;
    using hpx::parallel::transform_inclusive_scan;
}    // namespace hpx
