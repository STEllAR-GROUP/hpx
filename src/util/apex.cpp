//  Copyright (c) 2007-2013 Kevin Huck
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <hpx/config.hpp>
#include <hpx/util/apex.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/find_localities.hpp>
#include <cstdint>

namespace hpx { namespace util
{
#ifdef HPX_HAVE_APEX
    apex_task_wrapper apex_new_task(
           thread_description const& description,
           std::uint32_t parent_locality_id,
           threads::thread_id_type const& parent_task)
    {
        static std::uint32_t num_localities = hpx::get_initial_num_localities();
        apex_task_wrapper parent_wrapper = nullptr;
        // Parent pointers aren't reliable in distributed runs.
        if (parent_task != nullptr &&
            num_localities == 1
            /*hpx::get_locality_id() == parent_locality_id*/) {
            parent_wrapper = parent_task.get()->get_apex_data();
        }
        if (description.kind() ==
                thread_description::data_type_description) {
            return apex::new_task(description.get_description(),
                UINTMAX_MAX, parent_wrapper);
        } else {
            return apex::new_task(description.get_address(),
                UINTMAX_MAX, parent_wrapper);
        }
    }

#endif
}}

