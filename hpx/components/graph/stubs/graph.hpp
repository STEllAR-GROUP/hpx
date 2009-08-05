//  Copyright (c) 2007-2009 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUBS_GRAPH_JUN_09_2008_0458PM)
#define HPX_COMPONENTS_STUBS_GRAPH_JUN_09_2008_0458PM

#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/lcos/eager_future.hpp>

#include <hpx/components/graph/server/graph.hpp>

namespace hpx { namespace components { namespace stubs
{
    typedef int count_t;
    
    ///////////////////////////////////////////////////////////////////////////
    /// The \a stubs#graph class is the client side representation of all
    /// \a server#graph components
    struct graph : stub_base<server::graph>
    {
        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the server#graph instance 
        /// with the given \a gid
        static int init(naming::id_type gid, count_t order)
        {
            typedef server::graph::init_action action_type;
            return lcos::eager_future<action_type>(gid, order).get();
        }

        static int order(naming::id_type gid)
        {
            typedef server::graph::order_action action_type;
            return lcos::eager_future<action_type>(gid).get();
        }

        static int size(naming::id_type gid)
        {
            typedef server::graph::size_action action_type;
            return lcos::eager_future<action_type>(gid).get();
        }

        static int add_edge(naming::id_type gid, naming::id_type u, naming::id_type v, int w)
        {
        	typedef server::graph::add_edge_action action_type;
        	return lcos::eager_future<action_type>(gid, u, v, w).get();
        }

        static naming::id_type vertex_name(naming::id_type gid, int id)
        {
        	typedef server::graph::vertex_name_action action_type;
        	return lcos::eager_future<action_type>(gid, id).get();
        }
    };

}}}

#endif
