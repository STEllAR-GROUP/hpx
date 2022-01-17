//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/synchronization/mutex.hpp>
#include <hpx/synchronization/no_mutex.hpp>
#include <hpx/synchronization/once.hpp>
#include <hpx/synchronization/recursive_mutex.hpp>
#include <hpx/thread_support/unlock_guard.hpp>

namespace hpx {
    using hpx::lcos::local::call_once;
    using hpx::lcos::local::mutex;
    using hpx::lcos::local::no_mutex;
    using hpx::lcos::local::once_flag;
    using hpx::lcos::local::recursive_mutex;
    using hpx::lcos::local::spinlock;
    using hpx::lcos::local::timed_mutex;
    using hpx::util::unlock_guard;
}    // namespace hpx
