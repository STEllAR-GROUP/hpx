//  Copyright (c) 2007-2008 Anshul Tandon
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_APPLIER_APPLIER_JUN_03_2008_0438PM)
#define HPX_APPLIER_APPLIER_JUN_03_2008_0438PM

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
        template <typename Action>
        void apply (naming::id_type gid);

        template <typename Action, typename Arg0>
        void apply (naming::id_type gid, Arg0 const& arg0);

        template <typename Action, typename Arg0, typename Arg1>
        void apply (naming::id_type gid, Arg0 const& arg0, 
            Arg1 const& arg1);

        // Invoked by a running PX-thread to determine whether a resource is 
        // local or remote
//        bool isLocal (naming::id_type resourceGUID);

        // Invoked by the AM to request the new meta-action to execute locally
//        action_manager::meta_action getAction ()
    };
}}

#endif
