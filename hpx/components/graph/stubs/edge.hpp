//  Copyright (c) 2007-2009 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUBS_EDGE_AUG_28_2009_0447PM)
#define HPX_COMPONENTS_STUBS_EDGE_AUG_28_2009_0447PM

#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/lcos/eager_future.hpp>

#include <hpx/components/graph/server/edge.hpp>

namespace hpx { namespace components { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a stubs#edge class is the client side representation of all
    /// \a server#edge components
    struct edge : stub_base<server::edge>
    {
        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        typedef server::edge::edge_snapshot_type edge_snapshot_type;

        /// Initialize the server#edge instance
        /// with the given \a gid
        static int init(naming::id_type gid, naming::id_type source, naming::id_type target, int label)
        {
            typedef server::edge::init_action action_type;
            return lcos::eager_future<action_type>(gid, source, target, label).get();
        }

        static edge_snapshot_type get_snapshot(naming::id_type gid)
        {
            typedef server::edge::get_snapshot_action action_type;
            return lcos::eager_future<action_type>(gid).get();
        }
    };

}}}

#endif
