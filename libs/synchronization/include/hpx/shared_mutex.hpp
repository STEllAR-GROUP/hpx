//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/synchronization/lock_types.hpp>
#include <hpx/synchronization/shared_mutex.hpp>

namespace hpx {
    using hpx::lcos::local::shared_mutex;
    using hpx::lcos::local::upgrade_lock;
    using hpx::lcos::local::upgrade_to_unique_lock;
}    // namespace hpx
