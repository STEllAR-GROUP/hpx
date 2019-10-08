//  Copyright (c) 2007-2013 Kevin Huck
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <hpx/config.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/util/apex.hpp>

#include <cstddef>
#include <cstdint>

namespace hpx { namespace util {

#ifdef HPX_HAVE_APEX

    static enable_parent_task_handler_type enable_parent_task_handler;

    void set_enable_parent_task_handler(enable_parent_task_handler_type f)
    {
        enable_parent_task_handler = f;
    }

    apex_task_wrapper apex_new_task(thread_description const& description,
        threads::thread_id_type const& parent_task)
    {
        apex_task_wrapper parent_wrapper = nullptr;
        // Parent pointers aren't reliable in distributed runs.
        if (parent_task != nullptr && enable_parent_task_handler &&
            enable_parent_task_handler())
        {
            parent_wrapper = parent_task.get()->get_apex_data();
        }

        if (description.kind() == thread_description::data_type_description)
        {
            return apex::new_task(
                description.get_description(), UINTMAX_MAX, parent_wrapper);
        }
        else
        {
            HPX_ASSERT(
                description.kind() == thread_description::data_type_address);
            return apex::new_task(
                description.get_address(), UINTMAX_MAX, parent_wrapper);
        }
    }

#endif

}}    // namespace hpx::util
