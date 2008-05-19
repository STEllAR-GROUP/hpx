//  Copyright (c) 2007-2008 Anshul Tandon
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/naming.hpp>
#include <hpx/parcelset.hpp>
// #include <hpx/threadmanager.hpp>
// <waiting on CND's implementation of the TM API>

#include <boost/tuple/tuple.hpp>

namespace hpx { namespace action_manager
{
    // How are the arguments of an action stored in a parcel??
    class meta_action
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

    class action_manager
    {
    public:
        // Invoked by the Parcel Handler when PH has new parcels to be executed 
        void fetchNewParcel ();

        // Invoked by the Thread Manager when it is running out of work-items 
        // and needs something to execute on a specific starving resources 
        // specified as the argument
        void fetchNewParcel (naming::id_type resourceID);
        
        // Invoked by the Applier when it has a local action to be executed
        void fetchNewAction ();

        // Invoked during run-time or setup-time to add a new resource and its 
        // associated functions
        void addResource (naming::id_type resourceGUID, 
            boost::tuple resourceExecuteFunctions);
        
        // Invoked during run-time or setup-time to remove an existing resource
        // and its associated functions
        void removeResource (naming::id_type resourceGUID);

    private:
        // The following mappings are not needed in the HPX implementation. They are kept for testing purposes.

        // First item is the name of the action, second item is the pointer to the function itself
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
