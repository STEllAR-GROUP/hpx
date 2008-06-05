//  Copyright (c) 2008 Anshul Tandon
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/action_manager/action_manager.hpp>
#include <hpx/components/action.hpp>
#include <hpx/parcelset/parcelhandler.hpp>
#include <hpx/applier/applier.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace action_manager
{
    // Call-back function for parcelHandler to call when new parcels are received
    void action_manager::fetchNewParcel (parcelset::parcelhandler& parcel_handler_, 
        naming::address const&)
    {
        parcelset::parcel p;
        if (parcel_handler_.get_parcel(p))            // if new parcel is found
        {
            // decode the local virtual address of the parcel
            naming::address addr = p.get_destination_addr();
            naming::address::address_type lva = addr.address_;

            // decode the action-type in the parcel
            components::action_type act = p.get_action();

            // register the action and the local-virtual address with the TM
            thread_manager_.register_work(act->get_thread_function(applier_, lva));
        }
    }

    // Invoked by the Thread Manager when it is running out of work-items 
    // and needs something to execute on a specific starving resources 
    // specified as the argument
    void action_manager::fetchParcel (naming::id_type resourceID)
    {

    }
    
///////////////////////////////////////////////////////////////////////////////
}}
