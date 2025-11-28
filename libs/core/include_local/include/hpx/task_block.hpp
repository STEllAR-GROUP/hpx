//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/parallel/task_block.hpp>

namespace hpx {

    HPX_CXX_EXPORT using task_cancelled_exception =
        hpx::experimental::task_canceled_exception;
    HPX_CXX_EXPORT using hpx::experimental::define_task_block;
    HPX_CXX_EXPORT using hpx::experimental::define_task_block_restore_thread;
    HPX_CXX_EXPORT using hpx::experimental::task_block;
}    // namespace hpx
