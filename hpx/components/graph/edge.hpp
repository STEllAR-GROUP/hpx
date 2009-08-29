//  Copyright (c) 2007-2009 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_EDGE_AUG_28_2009_0455PM)
#define HPX_COMPONENTS_EDGE_AUG_28_2009_0455PM

#include <hpx/runtime.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include <hpx/components/graph/stubs/edge.hpp>

namespace hpx { namespace components 
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a edge class is the client side representation of a
    /// specific \a server#edge component
    class edge
      : public client_base<edge, stubs::edge>
    {
        typedef client_base<edge, stubs::edge> base_type;

    public:
        /// Create a client side representation for the existing
        /// \a server#edge instance with the given global id \a gid.
        edge(naming::id_type gid, bool freeonexit = true)
          : base_type(gid, freeonexit)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component
        typedef base_type::edge_snapshot_type edge_snapshot_type;

        /// Initialize the edge
        int init(naming::id_type source, naming::id_type target, int label)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::init(gid_, source, target, label);
        }

        edge_snapshot_type get_snapshot(void)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_snapshot(gid_);
        }
    };
    
}}

#endif
