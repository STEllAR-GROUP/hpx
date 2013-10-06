//  Copyright (c) 2008 Anshul Tandon
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

#include <boost/thread.hpp>
#include <boost/config.hpp>

#include <hpx/state.hpp>
#include <hpx/runtime/actions/action_manager.hpp>
#include <hpx/runtime/actions/manage_object_action.hpp>
#include <hpx/runtime/parcelset/parcelhandler.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/util/stringstream.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    ///////////////////////////////////////////////////////////////////////////
    // Call-back function for parcelHandler to call when new parcels are received
    void action_manager::fetch_parcel(
        parcelset::parcelhandler& parcel_handler,
        naming::gid_type const& parcel_id)
    {
        parcelset::parcel p;
        if (!parcel_handler.get_parcel(p, parcel_id))
            return;

        while (threads::threadmanager_is(starting))
        {
            boost::this_thread::sleep(boost::get_system_time() +
                boost::posix_time::milliseconds(HPX_NETWORK_RETRIES_SLEEP));
        }

        // Give up if we're shutting down.
        if (threads::threadmanager_is(stopping))
        {
            LPT_(debug) << "action_manager: fetch_parcel: dropping late "
                            "parcel " << p;
            return;
        }

        // write this parcel to the log
        LPT_(debug) << "action_manager: fetch_parcel: " << p;

        appl_.schedule_action(p);
    }

    // Invoked by the Thread Manager when it is running out of work-items
    // and needs something to execute on a specific starving resources
    // specified as the argument
//     void action_manager::fetch_parcel (naming::id_type const& resourceID)
//     {
//     }

///////////////////////////////////////////////////////////////////////////////
}}
