//  Copyright (c) 2007-2013 Kevin Huck
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <hpx/config.hpp>
#include <hpx/assertion.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/util/external_timer.hpp>

#include <cstddef>
#include <cstdint>

/* Static object instantiation */
// static hpx::util::external_timer::timer_interface hpx::util::external_timer::timer();

namespace hpx { namespace util {

#ifdef HPX_HAVE_APEX

    static enable_parent_task_handler_type enable_parent_task_handler;

    void set_enable_parent_task_handler(enable_parent_task_handler_type f)
    {
        enable_parent_task_handler = f;
    }

    namespace external_timer {

    std::shared_ptr<task_wrapper> new_task(thread_description const& description,
        std::uint32_t parent_locality_id,
        threads::thread_id_type const& parent_task)
    {
        std::shared_ptr<task_wrapper> parent_wrapper = nullptr;
        // Parent pointers aren't reliable in distributed runs.
        if (parent_task != nullptr && enable_parent_task_handler &&
            enable_parent_task_handler())
        {
            parent_wrapper = get_thread_id_data(parent_task)->get_timer_data();
        }

        if (description.kind() == thread_description::data_type_description)
        {
            return timer.new_task(
                description.get_description(), UINTMAX_MAX, parent_wrapper);
        }
        else
        {
            HPX_ASSERT(
                description.kind() == thread_description::data_type_address);
            return timer.new_task(
                description.get_address(), UINTMAX_MAX, parent_wrapper);
        }
    }
    } // namespace hpx::util::external_timer

#endif

}}    // namespace hpx::util
