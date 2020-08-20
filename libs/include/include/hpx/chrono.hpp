//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/timing/high_resolution_clock.hpp>
#include <hpx/timing/high_resolution_timer.hpp>
#include <hpx/timing/steady_clock.hpp>

namespace hpx { namespace chrono {
    using hpx::util::high_resolution_clock;
    using hpx::util::high_resolution_timer;
    using hpx::util::steady_time_point;
}}    // namespace hpx::chrono
