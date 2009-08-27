//  Copyright (c) 2009-2010 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SSCA2_AUG_14_2009_1044AM)
#define HPX_COMPONENTS_SSCA2_AUG_14_2009_1044AM

#include <hpx/runtime.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include "stubs/ssca2.hpp"

namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a ssca2 class is the client side representation of a
    /// specific \a server#ssca2 component
    class ssca2
      : public client_base<ssca2, stubs::ssca2>
    {
    private:
        typedef client_base<ssca2, stubs::ssca2> base_type;

    public:
        ssca2(naming::id_type gid, bool freeonexit = true)
          : base_type(gid, freeonexit)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        int
        large_set(naming::id_type G, naming::id_type edge_set)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::large_set(gid_, G, edge_set);
        }

        int
        large_set_local(locality_result local_set,
                        naming::id_type edge_set,
                        naming::id_type local_max_lco,
                        naming::id_type global_max_lco)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::large_set_local(
                gid_, local_set, edge_set, local_max_lco, global_max_lco);
        }

        int
        extract(naming::id_type edge_set, naming::id_type subgraphs)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::extract(gid_, edge_set, subgraphs);
        }

        int
        extract_local(naming::id_type local_edge_set, naming::id_type subgraphs)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::extract_local(gid_, local_edge_set, subgraphs);
        }

        int
        extract_subgraph(naming::id_type H, naming::id_type pmap, naming::id_type source, naming::id_type target, int d)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::extract_subgraph(gid_, H, pmap, source, target, d);
        }

        int
        init_props_map(naming::id_type P, naming::id_type G)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::init_props_map(gid_, P, G);
        }

        int
        init_props_map_local(naming::id_type local_props, locality_result local_vertices)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::init_props_map_local(gid_, local_props, local_vertices);
        }
    };
    
}}

#endif
