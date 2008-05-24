//  Copyright (c) 2007-2008 Anshul Tandon
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/naming.hpp>
#include <hpx/parcelset.hpp>
#include <hpx/action_manager.hpp>

#include <boost/tuple/tuple.hpp>

namespace hpx { namespace applier
{
    class applier
    {
    public:
        // Invoked by a running PX-thread to apply an action to any resource
        void apply (naming::id_type resourceGUID, 
            components::action_type action, continuation cont);
        // apply (resource_GUID, thread_actions, thread_args, thread_cont)

        // Invoked by a running PX-thread to determine whether a resource is 
        // local or remote
        bool isLocal (naming::id_type resourceGUID);

        // Invoked by the AM to request the new meta-action to execute locally
        action_manager::meta_action getAction ()
    };
}}