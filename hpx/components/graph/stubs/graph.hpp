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
    ///////////////////////////////////////////////////////////////////////////
    /// The \a stubs#graph class is the client side representation of all
    /// \a server#graph components
    struct graph : stub_base<server::graph>
    {
        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the server#graph instance 
        /// with the given \a gid
        static void init(naming::id_type gid) 
        {
            applier::apply<server::graph::init_action>(gid);
        }
    };

}}}

#endif
