//  Copyright (c) 2007-2008 Anshul Tandon
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_ACTION_MANAGER_ACTION_MANAGER_JUN_03_2008_0445PM)
#define HPX_ACTION_MANAGER_ACTION_MANAGER_JUN_03_2008_0445PM

#include <hpx/naming.hpp>
#include <hpx/parcelset.hpp>
#include <hpx/threadmanager.hpp>
#include <hpx/applier.hpp>

namespace hpx { namespace action_manager
{
    class action_manager
    {
    public:
        // Constructor
        action_manager(parcelset::parcelhandler& ph, applier::applier& appl)
            : parcel_handler_(ph), applier_(appl)
        {
            // Need to register the call-back function in parcelHandler so that
            // when a new parcel is received, it calls action_manager's fetchNewParcel()
            parcel_handler_.register_event_handler(boost::bind(
                &hpx::action_manager::action_manager::fetchNewParcel, this, 
                _1, _2), conn_);
        }

        // Call-back function for parcelHandler to call when new parcels are received
        void fetchNewParcel (parcelset::parcelhandler& parcel_handler_, 
            naming::address const&);

        // Invoked by the Thread Manager when it is running out of work-items 
        // and needs something to execute on a specific starving resources 
        // specified as the argument
        void fetchParcel (naming::id_type resourceID);

        // Invoked by the Applier when it has a local action to be executed
//        void fetchNewAction ();

        // Invoked during run-time or setup-time to add a new resource and its 
        // associated functions
//        void addResource (naming::id_type resourceGUID, 
//            boost::tuple resourceExecuteFunctions);

        // Invoked during run-time or setup-time to remove an existing resource
        // and its associated functions
//        void removeResource (naming::id_type resourceGUID);

        ~action_manager()
        {
        }

    private:
        parcelset::parcelhandler& parcel_handler_;
        applier::applier& applier_;

        // this scoped connection instance ensures the event handler to be 
        // automatically unregistered whenever it gets destroyed
        parcelset::parcelhandler::scoped_connection_type conn_;
    };
}}

#endif
