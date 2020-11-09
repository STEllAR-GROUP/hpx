//  Copyright (c) 2014-2016 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/modules/threading_base.hpp>

#include <string>
#include <vector>

namespace hpx { namespace util { namespace debug {

    // ------------------------------------------------------------------------
    // return a vector of suspended/other task Ids
    HPX_EXPORT std::vector<hpx::threads::thread_id_type> get_task_ids(
        hpx::threads::thread_schedule_state state =
            hpx::threads::thread_schedule_state::suspended);

    // ------------------------------------------------------------------------
    // return a vector of thread data structure pointers for suspended tasks
    HPX_EXPORT std::vector<hpx::threads::thread_data*> get_task_data(
        hpx::threads::thread_schedule_state state =
            hpx::threads::thread_schedule_state::suspended);

    // ------------------------------------------------------------------------
    // return string containing the stack backtrace for suspended tasks
    HPX_EXPORT std::string suspended_task_backtraces();

}}}    // namespace hpx::util::debug
