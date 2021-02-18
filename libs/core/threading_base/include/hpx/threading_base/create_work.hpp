//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/threading_base/scheduler_base.hpp>
#include <hpx/threading_base/thread_data.hpp>
#include <hpx/threading_base/thread_init_data.hpp>

#include <sstream>

namespace hpx { namespace threads { namespace detail {
    inline void create_work(policies::scheduler_base* scheduler,
        thread_init_data& data, error_code& ec = throws)
    {
        // verify parameters
        switch (data.initial_state)
        {
        case thread_schedule_state::pending:
        case thread_schedule_state::pending_do_not_schedule:
        case thread_schedule_state::pending_boost:
        case thread_schedule_state::suspended:
            break;

        default:
        {
            std::ostringstream strm;
            strm << "invalid initial state: "
                 << get_thread_state_name(data.initial_state);
            HPX_THROWS_IF(
                ec, bad_parameter, "thread::detail::create_work", strm.str());
            return;
        }
        }

#ifdef HPX_HAVE_THREAD_DESCRIPTION
        if (!data.description)
        {
            HPX_THROWS_IF(ec, bad_parameter, "thread::detail::create_work",
                "description is nullptr");
            return;
        }
#endif

        LTM_(info) << "create_work: initial_state("
                   << get_thread_state_name(data.initial_state)
                   << "), thread_priority("
                   << get_thread_priority_name(data.priority)
#ifdef HPX_HAVE_THREAD_DESCRIPTION
                   << "), description(" << data.description
#endif
                   << ")";

        thread_self* self = get_self_ptr();

#ifdef HPX_HAVE_THREAD_PARENT_REFERENCE
        if (nullptr == data.parent_id)
        {
            if (self)
            {
                data.parent_id = threads::get_self_id();
                data.parent_phase = self->get_thread_phase();
            }
        }
        if (0 == data.parent_locality_id)
            data.parent_locality_id = detail::get_locality_id(hpx::throws);
#endif

        if (nullptr == data.scheduler_base)
            data.scheduler_base = scheduler;

        // Pass critical priority from parent to child.
        if (self)
        {
            if (data.priority == thread_priority::default_ &&
                thread_priority::high_recursive ==
                    threads::get_self_id_data()->get_priority())
            {
                data.priority = thread_priority::high_recursive;
            }
        }

        // create the new thread
        if (data.priority == thread_priority::default_)
            data.priority = thread_priority::normal;

        data.run_now = (thread_priority::high == data.priority ||
            thread_priority::high_recursive == data.priority ||
            thread_priority::boost == data.priority);

        scheduler->create_thread(data, nullptr, ec);

        // NOTE: Don't care if the hint is a NUMA hint, just want to wake up a
        // thread.
        scheduler->do_some_work(data.schedulehint.hint);
    }
}}}    // namespace hpx::threads::detail
