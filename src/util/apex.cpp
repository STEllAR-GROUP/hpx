//  Copyright (c) 2007-2013 Kevin Huck
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <hpx/config.hpp>
#include <hpx/util/apex.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/threads/thread_data.hpp>

namespace hpx { namespace util
{
    void * apex_new_task(thread_description const& description,
           threads::thread_id_type const& parent_task)
    {
        apex::task_wrapper* parent_wrapper = nullptr;
        if (parent_task != nullptr) {
            parent_wrapper = (apex::task_wrapper*)(parent_task.get()->get_apex_data());
        }
        if (description.kind() ==
                thread_description::data_type_description) {
            return (void*)apex::new_task(description.get_description(),
                UINTMAX_MAX, parent_wrapper);
        } else {
            return (void*)apex::new_task(description.get_address(),
                UINTMAX_MAX, parent_wrapper);
        }
    }
}}

