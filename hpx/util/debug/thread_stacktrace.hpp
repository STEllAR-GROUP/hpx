//  Copyright (c) 2014-2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DEBUG_THREAD_STACKS_HPP
#define HPX_UTIL_DEBUG_THREAD_STACKS_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/threads/thread_data_fwd.hpp>
#include <hpx/runtime/threads/thread_enums.hpp>

#include <vector>
#include <string>

namespace hpx {
namespace util {
namespace debug {

    // ------------------------------------------------------------------------
    // return a vector of suspended/other task Ids
    HPX_EXPORT std::vector<hpx::threads::thread_id_type> get_task_ids(
        hpx::threads::thread_state_enum state = hpx::threads::suspended);

    // ------------------------------------------------------------------------
    // return a vector of thread data structure pointers for suspended tasks
    HPX_EXPORT std::vector<hpx::threads::thread_data*> get_task_data(
        hpx::threads::thread_state_enum state = hpx::threads::suspended);

    // ------------------------------------------------------------------------
    // return string containing the stack backtrace for suspended tasks
    HPX_EXPORT std::string suspended_task_backtraces();

}}}

#endif

