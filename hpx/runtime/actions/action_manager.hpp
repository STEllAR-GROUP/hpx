//  Copyright (c) 2007-2008 Anshul Tandon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_ACTION_MANAGER_ACTION_MANAGER_JUN_03_2008_0445PM)
#define HPX_ACTION_MANAGER_ACTION_MANAGER_JUN_03_2008_0445PM

#include <boost/noncopyable.hpp>

#include <hpx/include/naming.hpp>
#include <hpx/include/parcelset.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime/applier/applier.hpp>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace actions
{
    class HPX_EXPORT action_manager : private boost::noncopyable
    {
    public:
        // Constructor
        action_manager(applier::applier& appl)
          : appl_(appl)
        {
            // Need to register the call-back function in parcelHandler so that
            // when a new parcel is received, it calls action_manager's
            // fetchNewParcel()
            appl.get_parcel_handler().register_event_handler(boost::bind(
                    &action_manager::fetch_parcel, this, _1, _2), conn_);
        }

        // Call-back function for parcelHandler to call when new parcels are received
        void fetch_parcel (parcelset::parcelhandler& parcel_handler,
            naming::gid_type const& parcel_id);

        // Invoked by the Thread Manager when it is running out of work-items
        // and needs something to execute on a specific starving resources
        // specified as the argument
        //void fetch_parcel (naming::id_type const& resourceID);

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
        {}

    private:
        // this scoped connection instance ensures the event handler to be
        // automatically unregistered whenever it gets destroyed
        parcelset::parcelhandler::scoped_connection_type conn_;

        applier::applier& appl_;
    };
}}

#include <hpx/config/warnings_suffix.hpp>

#endif
