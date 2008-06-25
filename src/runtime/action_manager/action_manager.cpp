//  Copyright (c) 2008 Anshul Tandon
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

#include <hpx/runtime/action_manager/action_manager.hpp>
#include <hpx/runtime/parcelset/parcelhandler.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/components/action.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace action_manager
{
    // Call-back function for parcelHandler to call when new parcels are received
    void action_manager::fetch_new_parcel (
        parcelset::parcelhandler& parcel_handler_, naming::address const& dest)
    {
        parcelset::parcel p;
        if (parcel_handler_.get_parcel(p))  // if new parcel is found
        {
            // decode the local virtual address of the parcel
            naming::address addr = p.get_destination_addr();
            naming::address::address_type lva = addr.address_;

            // decode the action-type in the parcel
            components::action_type act = p.get_action();

            components::continuation_type cont = p.get_continuation();
            if (!cont) {
                // no continuation is to be executed, register the plain action 
                // and the local-virtual address with the TM only
                applier_.get_thread_manager().register_work(
                    act->get_thread_function(applier_, lva));
            }
            else {
                // this parcel carries a continuation, register a wrapper which
                // first executes the original thread function as required by 
                // the action and triggers the continuations afterwards
                applier_.get_thread_manager().register_work(
                    act->get_thread_function(cont, applier_, lva));
            }
        }
    }

    // Invoked by the Thread Manager when it is running out of work-items 
    // and needs something to execute on a specific starving resources 
    // specified as the argument
    void action_manager::fetch_parcel (naming::id_type resourceID)
    {

    }
    
///////////////////////////////////////////////////////////////////////////////
}}
