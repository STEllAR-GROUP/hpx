//  Copyright (c) 2007-2009 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUBS_VERTEX_JUN_09_2008_0458PM)
#define HPX_COMPONENTS_STUBS_VERTEX_JUN_09_2008_0458PM

#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/lcos/eager_future.hpp>

#include <hpx/components/vertex/server/vertex.hpp>

//typedef std::vector<std::pair<hpx::naming::id_type, int> > partial_edge_set_type;
typedef hpx::components::server::vertex::partial_edge_set_type partial_edge_set_type;

namespace hpx { namespace components { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a stubs#vertex class is the client side representation of all
    /// \a server#vertex components
    struct vertex : stub_base<server::vertex>
    {
        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the server#vertex instance
        /// with the given \a gid
        static int init(naming::id_type gid, int label)
        {
            typedef server::vertex::init_action action_type;
            return lcos::eager_future<action_type>(gid, label).get();
        }

        static int label(naming::id_type gid)
        {
            typedef server::vertex::label_action action_type;
            return lcos::eager_future<action_type>(gid).get();
        }

        static int add_edge(naming::id_type gid, naming::id_type v_g, int label)
        {
            typedef server::vertex::add_edge_action action_type;
            return lcos::eager_future<action_type>(gid, v_g, label).get();
        }

        static partial_edge_set_type
        out_edges(naming::id_type gid)
        {
            typedef server::vertex::out_edges_action action_type;
            return lcos::eager_future<action_type>(gid).get();
        }
    };

}}}

#endif
