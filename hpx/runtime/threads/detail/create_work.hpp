//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_THREADS_DETAIL_CREATE_WORK_JAN_13_2013_0526PM)
#define HPX_RUNTIME_THREADS_DETAIL_CREATE_WORK_JAN_13_2013_0526PM

#include <hpx/config.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/thread_init_data.hpp>
#include <hpx/runtime/threads/policies/scheduler_base.hpp>
#include <hpx/util/logging.hpp>

namespace hpx { namespace threads { namespace detail
{
    inline void create_work(policies::scheduler_base* scheduler,
        thread_init_data& data,
        thread_state_enum initial_state = threads::pending,
        error_code& ec = throws)
    {
        // verify parameters
        switch (initial_state) {
        case pending:
        case suspended:
            break;

        default:
            {
                std::ostringstream strm;
                strm << "invalid initial state: "
                     << get_thread_state_name(initial_state);
                HPX_THROWS_IF(ec, bad_parameter,
                    "thread::detail::create_work",
                    strm.str());
                return;
            }
        }

#ifdef HPX_HAVE_THREAD_DESCRIPTION
        if (!data.description)
        {
            HPX_THROWS_IF(ec, bad_parameter,
                "thread::detail::create_work", "description is NULL");
            return;
        }
#endif

        LTM_(info)
            << "create_work: initial_state("
            << get_thread_state_name(initial_state) << "), thread_priority("
            << get_thread_priority_name(data.priority)
#ifdef HPX_HAVE_THREAD_DESCRIPTION
            << "), description(" << data.description
#endif
            << ")";

        thread_self* self = get_self_ptr();

#ifdef HPX_HAVE_THREAD_PARENT_REFERENCE
        if (0 == data.parent_id) {

            if (self)
            {
                data.parent_id = threads::get_self_id().get();
                data.parent_phase = self->get_thread_phase();
            }
        }
        if (0 == data.parent_locality_id)
            data.parent_locality_id = get_locality_id();
#endif

        if (0 == data.scheduler_base)
            data.scheduler_base = scheduler;

        // Pass critical priority from parent to child.
        if (self)
        {
            if (thread_priority_critical == threads::get_self_id()->get_priority())
                data.priority = thread_priority_critical;
        }

        // create the new thread
        if (thread_priority_critical == data.priority ||
            thread_priority_boost == data.priority)
        {
            // For critical priority threads, create the thread immediately.
            scheduler->create_thread(data, 0, initial_state, true, ec,
                data.num_os_thread);
        }
        else {
            // Create a task description for the new thread.
            scheduler->create_thread(data, 0, initial_state, false, ec,
                data.num_os_thread);
        }
    }
}}}

#endif

