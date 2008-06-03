//  Copyright (c) 2007-2008 Anshul Tandon
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_ACTION_MANAGER_ACTION_MANAGER_JUN_03_2008_0445PM)
#define HPX_ACTION_MANAGER_ACTION_MANAGER_JUN_03_2008_0445PM

#include <hpx/naming.hpp>
#include <hpx/parcelset.hpp>
#include <hpx/threadmanager.hpp>

namespace hpx { namespace action_manager
{
    // How are the arguments of an action stored in a parcel??
/*    class meta_action
    {
    public:
        meta_action (void) {}
        meta_action (naming::id_type dest_, components::action_type:: action_, continuation cont_);
        ~meta_action (void) {}

        naming::id_type getDestination (void)
        {
            return destination_;
        }
        components::action_type getAction (void)
        {
            return action_;
        }
        continuation getContinuation(void)
        {
            return cont_;
        }

    private:
        naming::id_type destination_;
        // action_ holds the entire action + arguments package
        components::action_type action_;
        continuation cont_;
    };
*/
    class action_manager
    {
    public:
        // Constructor
        action_manager(parcelset::parcelhandler& ph, threadmanager::threadmanager& tm)
            : pHandler(ph), tManager(tm)
        {
            // Need to register the call-back function in parcelHandler so that
            // when a new parcel is received, it calls action_manager's fetchNewParcel()
            pHandler.register_event_handler(boost::bind(
                &hpx::action_manager::action_manager::fetchNewParcel, this, 
                _1, _2), conn_);
        }

        // Call-back function for parcelHandler to call when new parcels are received
        void fetchNewParcel (parcelset::parcelhandler& pHandler, naming::address const&);

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
        parcelset::parcelhandler& pHandler;
        threadmanager::threadmanager& tManager;
        
        // this scoped connection instance ensures the event handler to be 
        // automatically unregistered whenever it gets destroyed
        parcelset::parcelhandler::scoped_connection_type conn_;

        // The following mappings are not needed in the HPX implementation.
        // They are kept for testing purposes.

        // First item is the name of the action,
        // second item is the pointer to the function itself
        //typedef std::map<std::string, function_name> action_list;
        // First item is the GUID of the resource, second item is the 
        //typedef std::map<naming::id_type, action_list> registry;

        // The two typedefs emulate a nested dictionary
        //registry
        //{
        //    v1 : 
        //    {
        //        "init" : init(),
        //        "print" : print(),
        //        "visit" : visit()
        //    }
        //    v2 :
        //    {
        //        "init" : init(),
        //        "print" : print(),
        //        "visit" : visit()
        //    }
        //}
    };
}}

#endif
