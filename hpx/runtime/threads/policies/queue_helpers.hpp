////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_F0153C92_99B1_4F31_8FA9_4208DB2F26CE)
#define HPX_F0153C92_99B1_4F31_8FA9_4208DB2F26CE

#include <hpx/config.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/runtime/threads/thread_data.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace policies
{

    struct add_new_tag {};

#ifdef HPX_HAVE_THREAD_MINIMAL_DEADLOCK_DETECTION
    ///////////////////////////////////////////////////////////////////////////
    // We globally control whether to do minimal deadlock detection using this
    // global bool variable. It will be set once by the runtime configuration
    // startup code
    extern bool minimal_deadlock_detection;
#endif

///////////////////////////////////////////////////////////////////////////////
namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    // debug helper function, logs all suspended threads
    // this returns true if all threads in the map are currently suspended
    template <typename Map>
    bool dump_suspended_threads(std::size_t num_thread,
        Map& tm, boost::int64_t& idle_loop_count, bool running) HPX_COLD;

    template <typename Map>
    bool dump_suspended_threads(std::size_t num_thread,
        Map& tm, boost::int64_t& idle_loop_count, bool running)
    {
#ifndef HPX_HAVE_THREAD_MINIMAL_DEADLOCK_DETECTION
        HPX_UNUSED(tm);
        HPX_UNUSED(idle_loop_count);
        HPX_UNUSED(running); //-V601
        return false;
#else
        if (!minimal_deadlock_detection)
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
            threads::thread_data const* thrd = (*it).get();
            threads::thread_state_enum state = thrd->get_state().state();
            threads::thread_state_enum marked_state = thrd->get_marked_state();

            if (state != marked_state) {
                // log each thread only once
                if (!logged_headline) {
                    if (running) {
                        LTM_(error) //-V128
                            << "Listing suspended threads while queue ("
                            << num_thread << ") is empty:";
                    }
                    else {
                        LHPX_CONSOLE_(hpx::util::logging::level::error) //-V128
                            << "  [TM] Listing suspended threads while queue ("
                            << num_thread << ") is empty:\n";
                    }
                    logged_headline = true;
                }

                if (running) {
                    LTM_(error) << "queue(" << num_thread << "): " //-V128
                                << get_thread_state_name(state)
                                << "(" << std::hex << std::setw(8)
                                    << std::setfill('0') << (*it).get()
                                << "." << std::hex << std::setw(2)
                                    << std::setfill('0') << thrd->get_thread_phase()
                                << "/" << std::hex << std::setw(8)
                                    << std::setfill('0') << thrd->get_component_id()
                                << ")"
#ifdef HPX_HAVE_THREAD_PARENT_REFERENCE
                                << " P" << std::hex << std::setw(8)
                                    << std::setfill('0') << thrd->get_parent_thread_id()
#endif
                                << ": " << thrd->get_description()
                                << ": " << thrd->get_lco_description();
                }
                else {
                    LHPX_CONSOLE_(hpx::util::logging::level::error) << "  [TM] " //-V128
                                << "queue(" << num_thread << "): "
                                << get_thread_state_name(state)
                                << "(" << std::hex << std::setw(8)
                                    << std::setfill('0') << (*it).get()
                                << "." << std::hex << std::setw(2)
                                    << std::setfill('0') << thrd->get_thread_phase()
                                << "/" << std::hex << std::setw(8)
                                    << std::setfill('0') << thrd->get_component_id()
                                << ")"
#ifdef HPX_HAVE_THREAD_PARENT_REFERENCE
                                << " P" << std::hex << std::setw(8)
                                    << std::setfill('0') << thrd->get_parent_thread_id()
#endif
                                << ": " << thrd->get_description()
                                << ": " << thrd->get_lco_description() << "\n";
                }
                thrd->set_marked_state(state);

                // result should be true if we found only suspended threads
                if (collect_suspended) {
                    switch(state) {
                    case threads::suspended:
                        result = true;    // at least one is suspended
                        break;

                    case threads::pending:
                    case threads::active:
                        result = false;   // one is active, no deadlock (yet)
                        collect_suspended = false;
                        break;

                    default:
                        // If the thread is terminated we don't care too much
                        // anymore.
                        break;
                    }
                }
            }
        }
        return result;
#endif
    }
}

}}}

#endif // HPX_F0153C92_99B1_4F31_8FA9_4208DB2F26CE

