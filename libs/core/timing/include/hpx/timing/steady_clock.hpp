//  Copyright (c) 2007-2023 Hartmut Kaiser
//  Copyright (c) 2014-2016 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nodeprecatedinclude:boost/chrono/chrono.hpp
// hpxinspect:nodeprecatedname:boost::chrono

#pragma once

#include <hpx/config.hpp>

#include <chrono>

namespace hpx::chrono {

    HPX_CXX_EXPORT using std::chrono::steady_clock;

    using steady_time_point = std::chrono::steady_clock::time_point;
    using steady_duration = std::chrono::steady_clock::duration;

    HPX_CXX_EXPORT inline constexpr steady_duration null_duration =
        steady_duration::zero();

}    // namespace hpx::chrono
