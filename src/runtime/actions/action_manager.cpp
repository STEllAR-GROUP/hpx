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
        parcelset::parcelhandler& parcel_handler, naming::address const& dest)
    {
        parcelset::parcel p;
        if (parcel_handler.get_parcel(p))  // if new parcel is found
        {
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

            // decode the local virtual address of the parcel
            naming::address addr = p.get_destination_addr();
            naming::address::address_type lva = addr.address_;

            // by convention, a zero address references the local runtime support
            // component
            if (0 == lva)
                lva = appl_.get_runtime_support_raw_gid().get_lsb();

            // decode the action-type in the parcel
            action_type act = p.get_action();
            continuation_type cont = p.get_continuation();

            // make sure the component_type of the action matches the component
            // type in the destination address
            if (HPX_UNLIKELY(!components::types_are_compatible(
                dest.type_, act->get_component_type())))
            {
                hpx::util::osstream strm;
                strm << " types are not compatible: destination_type("
                     << dest.type_ << ") action_type("
                     << act->get_component_type()
                     << ") parcel ("  << p << ")";
                HPX_THROW_EXCEPTION(bad_component_type,
                    "action_manager::fetch_parcel",
                    hpx::util::osstream_get_string(strm));
            }

            // either directly execute the action or create a new thread
            if (actions::base_action::direct_action == act->get_action_type())
            {
                // direct execution of the action
                if (!cont) {
                // no continuation is to be executed
                    act->get_thread_function(lva)(threads::wait_signaled);
                }
                else {
                // this parcel carries a continuation, we execute a wrapper
                // handling all related functionality
                    act->get_thread_function(cont, lva)(threads::wait_signaled);
                }
            }
            else {
                // dispatch action, register work item either with or without
                // continuation support
                if (!cont) {
                // no continuation is to be executed, register the plain action
                // and the local-virtual address with the TM only
                    threads::thread_init_data data;
                    appl_.get_thread_manager().register_work(
                        act->get_thread_init_data(lva, data), threads::pending);
                }
                else {
                // this parcel carries a continuation, register a wrapper which
                // first executes the original thread function as required by
                // the action and triggers the continuations afterwards
                    threads::thread_init_data data;
                    appl_.get_thread_manager().register_work(
                        act->get_thread_init_data(cont, lva, data),
                        threads::pending);
                }
            }
        }
    }

    // Invoked by the Thread Manager when it is running out of work-items
    // and needs something to execute on a specific starving resources
    // specified as the argument
    void action_manager::fetch_parcel (naming::id_type const& resourceID)
    {

    }

///////////////////////////////////////////////////////////////////////////////
}}
