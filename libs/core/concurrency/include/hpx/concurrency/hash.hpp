//  Copyright (c) 2026 Pratyksh Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <functional>

namespace hpx::concurrent {

    template <typename T>
    using tbb_hash = std::hash<T>;

    template <typename T>
    using tbb_hasher = std::hash<T>;

}    // namespace hpx::concurrent
