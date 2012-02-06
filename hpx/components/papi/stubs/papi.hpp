//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c) 2011-2012 Maciej Brodowicz
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PERFORMANCE_COUNTERS_PAPI_STUBS_PAPI_COUNTER_201111181443)
#define HPX_PERFORMANCE_COUNTERS_PAPI_STUBS_PAPI_COUNTER_201111181443

#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/lcos/eager_future.hpp>

#include <hpx/components/papi/server/papi.hpp>


namespace hpx { namespace performance_counters { namespace papi { namespace stubs
{
    using namespace hpx;
    using namespace hpx::performance_counters;

    struct papi_counter: components::stub_base<papi::server::papi_counter>
    {
        static lcos::promise<bool> set_event_async(naming::id_type const& gid,
                                                   int event, bool activate)
        {
            typedef server::papi_counter::set_event_action action_type;
            return lcos::eager_future<action_type>(gid, event, activate);
        }

        static bool set_event(naming::id_type const& gid, int event, bool activate)
        {
            return set_event_async(gid, event, activate).get();
        }

        static lcos::promise<bool> start_async(naming::id_type const& gid)
        {
            typedef server::papi_counter::start_action action_type;
            return lcos::eager_future<action_type>(gid);
        }

        static bool start(naming::id_type gid)
        {
            return start_async(gid).get();
        }

        static lcos::promise<bool> stop_async(naming::id_type const& gid)
        {
            typedef server::papi_counter::stop_action action_type;
            return lcos::eager_future<action_type>(gid);
        }

        static bool stop(naming::id_type gid)
        {
            return stop_async(gid).get();
        }

        static lcos::promise<bool>
        enable_multiplexing_async(naming::id_type const& gid, long ival)
        {
            typedef server::papi_counter::enable_multiplexing_action action_type;
            return lcos::eager_future<action_type>(gid, ival);
        }

        static bool enable_multiplexing(naming::id_type gid, long ival)
        {
            return enable_multiplexing_async(gid, ival).get();
        }

    };

}}}}

#endif
