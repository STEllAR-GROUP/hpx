////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_F0153C92_99B1_4F31_8FA9_4208DB2F26CE)
#define HPX_F0153C92_99B1_4F31_8FA9_4208DB2F26CE

#include <hpx/config.hpp>
#include <hpx/util/try_lock_wrapper.hpp>
#include <hpx/util/logging.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace policies
{

struct add_new_tag {};

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
#if !HPX_THREAD_MINIMAL_DEADLOCK_DETECTION
        (void)tm;
        (void)idle_loop_count;
        (void)running;
        return false;
#else
        if (HPX_LIKELY(idle_loop_count++ < HPX_IDLE_LOOP_COUNT_MAX))
            return false;

        // reset idle loop count
        idle_loop_count = 0;

        bool result = false;
        bool collect_suspended = true;

        bool logged_headline = false;
        typename Map::const_iterator end = tm.end();
        for (typename Map::const_iterator it = tm.begin(); it != end; ++it)
        {
            threads::thread_data const* thrd = (*it).second;
            threads::thread_state state = thrd->get_state();
            threads::thread_state marked_state = thrd->get_marked_state();

            if (state != marked_state) {
                // log each thread only once
                if (!logged_headline) {
                    if (running) {
                        LTM_(error)
                            << "Listing suspended threads while queue ("
                            << num_thread << ") is empty:";
                    }
                    else {
                        LHPX_CONSOLE_(boost::logging::level::error)
                            << "  [TM] Listing suspended threads while queue ("
                            << num_thread << ") is empty:\n";
                    }
                    logged_headline = true;
                }

                if (running) {
                    LTM_(error) << "queue(" << num_thread << "): "
                                << get_thread_state_name(state)
                                << "(" << std::hex << std::setw(8)
                                    << std::setfill('0') << (*it).first
                                << "." << std::hex << std::setw(2)
                                    << std::setfill('0') << thrd->get_thread_phase()
                                << "/" << std::hex << std::setw(8)
                                    << std::setfill('0') << thrd->get_component_id()
                                << ")"
#if HPX_THREAD_MAINTAIN_PARENT_REFERENCE
                                << " P" << std::hex << std::setw(8)
                                    << std::setfill('0') << thrd->get_parent_thread_id()
#endif
                                << ": " << thrd->get_description()
                                << ": " << thrd->get_lco_description();
                }
                else {
                    LHPX_CONSOLE_(boost::logging::level::error) << "  [TM] "
                                << "queue(" << num_thread << "): "
                                << get_thread_state_name(state)
                                << "(" << std::hex << std::setw(8)
                                    << std::setfill('0') << (*it).first
                                << "." << std::hex << std::setw(2)
                                    << std::setfill('0') << thrd->get_thread_phase()
                                << "/" << std::hex << std::setw(8)
                                    << std::setfill('0') << thrd->get_component_id()
                                << ")"
#if HPX_THREAD_MAINTAIN_PARENT_REFERENCE
                                << " P" << std::hex << std::setw(8)
                                    << std::setfill('0') << thrd->get_parent_thread_id()
#endif
                                << ": " << thrd->get_description()
                                << ": " << thrd->get_lco_description() << "\n";
                }
                thrd->set_marked_state(state);

                // result should be true if we found only suspended threads
                if (collect_suspended) {
                    switch(state.get_state()) {
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

