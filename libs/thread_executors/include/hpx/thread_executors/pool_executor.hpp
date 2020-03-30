//  Copyright (c)      2017 John Biddiscombe
//  Copyright (c)      2017 Shoshana Jakobovits
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY)
#include <hpx/executors/pool_executor.hpp>

namespace hpx { namespace threads { namespace executors {
    using pool_executor = parallel::execution::pool_executor;
}}}    // namespace hpx::threads::executors

#endif
