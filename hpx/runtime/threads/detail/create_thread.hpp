//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_THREADS_DETAIL_CREATE_THREAD_JAN_13_2013_0439PM)
#define HPX_RUNTIME_THREADS_DETAIL_CREATE_THREAD_JAN_13_2013_0439PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/thread_init_data.hpp>
#include <hpx/runtime/threads/policies/scheduler_base.hpp>
#include <hpx/util/logging.hpp>

namespace hpx { namespace threads { namespace detail
{
    inline threads::thread_id_type create_thread(
        policies::scheduler_base* scheduler,
        thread_init_data& data, thread_state_enum initial_state = pending,
        bool run_now = true, error_code& ec = throws)
    {
        // verify parameters
        switch (initial_state) {
        case pending:
        case suspended:
            break;

        default:
            {
                hpx::util::osstream strm;
                strm << "invalid initial state: "
                     << get_thread_state_name(initial_state);
                HPX_THROWS_IF(ec, bad_parameter,
                    "threads::detail::create_thread",
                    hpx::util::osstream_get_string(strm));
                return invalid_thread_id;
            }
        }

#if HPX_THREAD_MAINTAIN_DESCRIPTION
        if (0 == data.description)
        {
            HPX_THROWS_IF(ec, bad_parameter,
                "threads::detail::create_thread", "description is NULL");
            return invalid_thread_id;
        }
#endif

#if HPX_THREAD_MAINTAIN_PARENT_REFERENCE
        if (0 == data.parent_id) {
            thread_self* self = get_self_ptr();
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

        // create the new thread
        thread_id_type newid = scheduler->create_thread(
            data, initial_state, run_now, ec, data.num_os_thread);

        LTM_(info) << "register_thread(" << newid << "): initial_state("
                   << get_thread_state_name(initial_state) << "), "
                   << "run_now(" << (run_now ? "true" : "false")
#if HPX_THREAD_MAINTAIN_DESCRIPTION
                   << "), description(" << data.description
#endif
                   << ")";

        return newid;
    }
}}}

#endif

