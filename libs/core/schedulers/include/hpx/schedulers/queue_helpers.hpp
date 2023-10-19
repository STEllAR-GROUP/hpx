////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2017-2018 John Biddiscombe
//  Copyright (c) 2007-2022 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/schedulers/deadlock_detection.hpp>
#include <hpx/threading_base/thread_data.hpp>
#include <hpx/type_support/unused.hpp>

#include <cstddef>
#include <cstdint>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::threads::policies {

    ///////////////////////////////////////////////////////////////////////////////
    namespace detail {

        ///////////////////////////////////////////////////////////////////////////
        // debug helper function, logs all suspended threads this returns true
        // if all threads in the map are currently suspended
        template <typename Map>
        bool dump_suspended_threads([[maybe_unused]] std::size_t num_thread,
            [[maybe_unused]] Map& tm,
            [[maybe_unused]] std::int64_t& idle_loop_count,
            [[maybe_unused]] bool running)
        {
#if !defined(HPX_HAVE_THREAD_MINIMAL_DEADLOCK_DETECTION)
            return false;
#else
            if (!get_minimal_deadlock_detection_enabled())
                return false;

            // attempt to output possibly deadlocked threads occasionally only
            if (HPX_LIKELY((idle_loop_count++ % HPX_IDLE_LOOP_COUNT_MAX) != 0))
                return false;

            bool result = false;
            bool collect_suspended = true;

            bool logged_headline = false;
            typename Map::const_iterator end = tm.end();
            for (typename Map::const_iterator it = tm.begin(); it != end; ++it)
            {
                threads::thread_data const* thrd = get_thread_id_data(*it);
                threads::thread_schedule_state const state =
                    thrd->get_state().state();
                threads::thread_schedule_state const marked_state =
                    thrd->get_marked_state();

                if (state != marked_state)
                {
                    // log each thread only once
                    if (!logged_headline)
                    {
                        if (running)
                        {
                            LTM_(warning).format("Listing suspended threads "
                                                 "while queue ({}) is empty:",
                                num_thread);
                        }
                        else
                        {
                            LHPX_CONSOLE_(hpx::util::logging::level::warning)
                                .format("  [TM] Listing suspended threads "
                                        "while queue ({}) is empty:\n",
                                    num_thread);
                        }
                        logged_headline = true;
                    }

                    if (running)
                    {
                        LTM_(warning)
                            .format("queue({}): {}({:08x}.{:02x}/{:08x})",
                                num_thread, get_thread_state_name(state), *it,
                                thrd->get_thread_phase(),
                                thrd->get_component_id())
#ifdef HPX_HAVE_THREAD_PARENT_REFERENCE
                            .format(" P{:08x}", thrd->get_parent_thread_id())
#endif
                            .format(": {}: {}", thrd->get_description(),
                                thrd->get_lco_description());
                    }
                    else
                    {
                        LHPX_CONSOLE_(hpx::util::logging::level::warning)
                            .format("queue({}): {}({:08x}.{:02x}/{:08x})",
                                num_thread, get_thread_state_name(state), *it,
                                thrd->get_thread_phase(),
                                thrd->get_component_id())
#ifdef HPX_HAVE_THREAD_PARENT_REFERENCE
                            .format(" P{:08x}", thrd->get_parent_thread_id())
#endif
                            .format(": {}: {}", thrd->get_description(),
                                thrd->get_lco_description());
                    }
                    thrd->set_marked_state(state);

                    // result should be true if we found only suspended threads
                    if (collect_suspended)
                    {
                        switch (state)
                        {
                        case threads::thread_schedule_state::suspended:
                            result = true;    // at least one is suspended
                            break;

                        case threads::thread_schedule_state::pending:
                            [[fallthrough]];
                        case threads::thread_schedule_state::active:
                            // one is active, no deadlock (yet)
                            result = false;
                            collect_suspended = false;
                            break;

                        default:
                            // If the thread is terminated we don't care too
                            // much anymore.
                            break;
                        }
                    }
                }
            }
            return result;
#endif
        }
    }    // namespace detail
}    // namespace hpx::threads::policies
